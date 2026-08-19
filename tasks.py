"""The heavy job: run the segmentation pipeline as a subprocess and stream
status into the job store. Shared by the in-process pool and the RQ worker."""
import os
import sys
import subprocess

import config
import jobstore

PIPELINE = os.path.join(config.PROJECT_DIR, 'segmentation_pipeline.py')


def run_segmentation(job_id, input_dir, name, modality=None,
                     no_medsam=False, engine='heuristic'):
    jobstore.update(job_id, status='running', message='Starting pipeline...')
    cmd = [sys.executable, '-u', PIPELINE, '--input', input_dir, '--name', name]
    if modality:
        cmd += ['--modality', modality]
    if no_medsam:
        cmd += ['--no-medsam']
    if engine == 'totalseg':
        cmd += ['--engine', 'totalseg']

    try:
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', errors='replace', env=env)
        log = []
        for line in proc.stdout:
            line = line.rstrip()
            if not line or line.startswith('Iteration:'):
                continue
            log.append(line)
            log = log[-40:]
            fields = {'log': log}
            if line.lstrip().startswith('['):
                fields['message'] = line.strip()
            jobstore.update(job_id, **fields)
        proc.wait()
        if proc.returncode == 0:
            jobstore.update(job_id, status='done', message='Complete')
        else:
            jobstore.update(job_id, status='error',
                            message=f'Pipeline exited with code {proc.returncode}')
    except Exception as e:  # noqa: BLE001
        jobstore.update(job_id, status='error', message=str(e))
