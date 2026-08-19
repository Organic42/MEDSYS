"""
VR-Segmentation web app — drag-and-drop DICOM zip -> segmented images + 3D models.

Dev:   python app.py
Prod:  set REDIS_URL + run a separate `python worker.py` (see docker-compose.yml)
"""
import os
import json
import uuid
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles

import config
import jobstore
import tasks

app = FastAPI(title="VR-Segmentation")

# ── job dispatch: Redis/RQ in prod, bounded thread pool in dev ────────────
_executor = None
_queue = None

if config.REDIS_URL:
    from redis import Redis
    from rq import Queue
    _queue = Queue('segmentation', connection=Redis.from_url(config.REDIS_URL),
                   default_timeout=config.JOB_TIMEOUT)
else:
    _executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)


def _dispatch(job_id, extract_dir, name, modality, no_medsam, engine):
    args = (job_id, extract_dir, name, modality, no_medsam, engine)
    if _queue is not None:
        _queue.enqueue('tasks.run_segmentation', *args,
                       job_timeout=config.JOB_TIMEOUT, job_id=job_id)
    else:
        _executor.submit(tasks.run_segmentation, *args)


@app.on_event('startup')
def _startup():
    jobstore.init()
    jobstore.prune(config.JOB_RETENTION_DAYS)


# ── API ───────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/ready')
def ready():
    """Readiness: DB reachable and (if configured) Redis reachable."""
    detail = {'db': False, 'queue': 'inprocess'}
    try:
        jobstore.get('__probe__')
        detail['db'] = True
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f'db not ready: {e}')
    if _queue is not None:
        try:
            _queue.connection.ping()
            detail['queue'] = 'redis'
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f'redis not ready: {e}')
    return {'status': 'ready', **detail}


@app.post('/api/upload')
async def upload(file: UploadFile = File(...), name: str = Form(None),
                 modality: str = Form(None), no_medsam: str = Form(None),
                 engine: str = Form('heuristic')):
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(400, "Please upload a .zip of DICOM files")

    job_id = uuid.uuid4().hex[:12]
    ds_name = (name or os.path.splitext(file.filename)[0]).strip()
    ds_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in ds_name) or job_id

    work = os.path.join(config.UPLOAD_ROOT, job_id)
    os.makedirs(work, exist_ok=True)
    zip_path = os.path.join(work, 'data.zip')

    # stream to disk with a size cap
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with open(zip_path, 'wb') as f:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                f.close()
                shutil.rmtree(work, ignore_errors=True)
                raise HTTPException(413, f'Upload exceeds {config.MAX_UPLOAD_MB} MB limit')
            f.write(chunk)

    extract_dir = os.path.join(work, 'dicom')
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(400, f"Bad zip: {e}")

    jobstore.create(job_id, ds_name, engine=engine)
    _dispatch(job_id, extract_dir, ds_name, modality, bool(no_medsam), engine)
    return {'job_id': job_id, 'name': ds_name}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    job = jobstore.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


@app.get('/api/device')
def device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return {'device': 'cuda', 'name': torch.cuda.get_device_name(0)}
    except Exception:
        pass
    return {'device': 'cpu', 'name': 'CPU'}


def _list_assets(ds_dir):
    pngs, meshes = [], []
    for f in sorted(os.listdir(ds_dir)):
        if f.lower().endswith('.png'):
            pngs.append(f)
        elif f.lower().endswith('.stl'):
            meshes.append(f)
    report = None
    rp = os.path.join(ds_dir, 'report.json')
    if os.path.exists(rp):
        try:
            report = json.load(open(rp))
        except Exception:
            pass
    sub = os.path.join(ds_dir, 'meshes')
    if os.path.isdir(sub):
        for f in sorted(os.listdir(sub)):
            if f.lower().endswith('.stl'):
                meshes.append(f'meshes/{f}')
    return pngs, meshes, report


@app.get('/api/datasets')
def list_datasets():
    out = []
    if os.path.isdir(config.OUTPUT_ROOT):
        for name in sorted(os.listdir(config.OUTPUT_ROOT)):
            d = os.path.join(config.OUTPUT_ROOT, name)
            if not os.path.isdir(d):
                continue
            pngs, meshes, report = _list_assets(d)
            if pngs or meshes:
                out.append({'name': name, 'images': pngs, 'meshes': meshes,
                            'report': report})
    return {'datasets': out}


# ── Neuroplasticity Explorer ─────────────────────────────────────────────

@app.get('/api/brain/regions')
def brain_regions():
    """Region catalog — the frontend builds its 3D scene from this."""
    import brain_knowledge
    return {'regions': brain_knowledge.REGIONS,
            'suggestions': brain_knowledge.SUGGESTIONS,
            'engine': 'hybrid' if brain_knowledge.claude_available() else 'curated'}


@app.post('/api/brain/ask')
async def brain_ask(question: str = Form(...)):
    import brain_knowledge
    q = (question or '').strip()
    if not q:
        raise HTTPException(400, 'Ask a question first')
    if len(q) > 500:
        raise HTTPException(400, 'Question too long (500 char max)')
    return brain_knowledge.answer(q)


# static: serve generated outputs and the frontend
app.mount('/output', StaticFiles(directory=config.OUTPUT_ROOT), name='output')
app.mount('/', StaticFiles(directory=config.WEB_DIR, html=True), name='web')


if __name__ == '__main__':
    import uvicorn
    print(f"VR-Segmentation -> http://{config.HOST}:{config.PORT}  "
          f"(queue: {'redis' if config.REDIS_URL else 'in-process'})")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level='info')
