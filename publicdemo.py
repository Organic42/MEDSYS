"""
Public demo mode (VRSEG_PUBLIC_DEMO=1), for hosting MEDSYS where anyone can
try it, such as a Hugging Face Space.

  - Each browser gets a random visitor id; it sees only the scans it uploaded
    and the bundled samples, never another visitor's.
  - Uploaded DICOMs are deleted as soon as a job finishes, and results after
    VRSEG_RETENTION_MIN minutes.
  - Dataset names carry the random job id, so result URLs cannot be guessed.
  - At most VRSEG_MAX_QUEUE jobs wait or run at once.
  - Sample scans in samples/ are processed at start-up and shown to everyone.

Without the flag nothing here runs and the app behaves as before.
"""
import hashlib
import os
import shutil
import threading
import time
import zipfile

import config
import jobstore

OWNER_FILE = '.owner'
SAMPLE = 'sample'
# Bundled sample scans: zip in samples/ -> dataset name shown to every visitor
SAMPLES = {'sample_chest_ct.zip': 'Sample_chest_CT'}


def owner_tag(visitor):
    """Stored form of a visitor id: a hash, so the folder never holds the id itself."""
    visitor = (visitor or '').strip()
    if not visitor or len(visitor) > 100:
        return None
    return hashlib.sha256(visitor.encode()).hexdigest()[:20]


def mark(ds_dir, owner):
    os.makedirs(ds_dir, exist_ok=True)
    with open(os.path.join(ds_dir, OWNER_FILE), 'w', encoding='utf-8') as fh:
        fh.write(owner)


def owner_of(ds_dir):
    try:
        with open(os.path.join(ds_dir, OWNER_FILE), encoding='utf-8') as fh:
            return fh.read().strip()
    except OSError:
        return None


def visible(ds_dir, visitor):
    """Whether a visitor may list this dataset (always, outside public mode)."""
    if not config.PUBLIC_DEMO:
        return True
    owner = owner_of(ds_dir)
    return owner == SAMPLE or (owner is not None and owner == owner_tag(visitor))


def busy():
    """True when the queue is full."""
    return jobstore.count_active() >= config.MAX_QUEUE


# ── retention ────────────────────────────────────────────────────────────

def _inside(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return os.path.commonpath([path, root]) == root and path != root


def remove_upload(input_dir):
    """Delete a finished job's uploaded files (the upload folder holding input_dir)."""
    work = os.path.dirname(os.path.abspath(input_dir))
    if _inside(work, config.UPLOAD_ROOT):
        shutil.rmtree(work, ignore_errors=True)


def cleanup(now=None):
    """Delete visitor results and uploads older than the retention period."""
    now = now or time.time()
    cutoff = now - config.RETENTION_MIN * 60
    removed = 0
    for root, keep_samples in ((config.OUTPUT_ROOT, True), (config.UPLOAD_ROOT, False)):
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path) or (keep_samples and owner_of(path) == SAMPLE):
                continue
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    jobstore.delete_older_than(cutoff)
    return removed


def _cleanup_loop():
    while True:
        try:
            cleanup()
        except Exception as e:  # noqa: BLE001
            print(f'[public-demo] cleanup failed: {e}', flush=True)
        time.sleep(300)


# ── start-up ─────────────────────────────────────────────────────────────

def start(dispatch):
    """Start the cleanup loop and process the bundled samples once."""
    threading.Thread(target=_cleanup_loop, daemon=True, name='public-demo-cleanup').start()
    samples_dir = os.path.join(config.BUNDLE_DIR, 'samples')
    for zip_name, ds_name in SAMPLES.items():
        src = os.path.join(samples_dir, zip_name)
        out = os.path.join(config.OUTPUT_ROOT, ds_name)
        already_built = os.path.isdir(out) and any(f.endswith('.stl') for f in os.listdir(out))
        if not os.path.exists(src) or already_built:
            continue
        job_id = f'sample{int(time.time())}'
        work = os.path.join(config.UPLOAD_ROOT, job_id)
        extract = os.path.join(work, 'dicom')
        os.makedirs(extract, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(extract)
        mark(out, SAMPLE)
        jobstore.create(job_id, ds_name, engine='heuristic')
        dispatch(job_id, extract, ds_name, None, True, 'heuristic')
        print(f'[public-demo] processing sample {ds_name}', flush=True)
