"""Environment-driven configuration for the MEDSYS service."""
import os
import sys

# Running as a PyInstaller-frozen .exe? Two different base paths matter then:
#   - RUNTIME_DIR: where the .exe itself lives — writable data (output,
#     uploads, job DB) goes next to it so it's easy to find and survives
#     re-launching the app.
#   - BUNDLE_DIR: PyInstaller's extracted resource dir (sys._MEIPASS) — where
#     bundled read-only assets (web/) were placed at build time.
# In normal `python app.py` use, both are just this file's directory.
FROZEN = bool(getattr(sys, 'frozen', False))
if FROZEN:
    RUNTIME_DIR = os.path.dirname(os.path.abspath(sys.executable))
    BUNDLE_DIR = getattr(sys, '_MEIPASS', RUNTIME_DIR)
    EXE_PATH = os.path.abspath(sys.executable)
else:
    RUNTIME_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = RUNTIME_DIR
    EXE_PATH = None

PROJECT_DIR = RUNTIME_DIR  # kept for backward compatibility with existing call sites


def _path(env, default):
    return os.path.abspath(os.environ.get(env, os.path.join(RUNTIME_DIR, default)))


# Storage locations (override in containers via env)
OUTPUT_ROOT = _path('VRSEG_OUTPUT', 'output')
UPLOAD_ROOT = _path('VRSEG_UPLOADS', '_uploads')
DB_PATH = _path('VRSEG_DB', 'jobs.db')
WEB_DIR = os.path.join(BUNDLE_DIR, 'web')

# Concurrency — segmentation is heavy, so serialize by default
MAX_WORKERS = int(os.environ.get('VRSEG_MAX_WORKERS', '1'))

# Upload guardrails
MAX_UPLOAD_MB = int(os.environ.get('VRSEG_MAX_UPLOAD_MB', '4096'))

# Job timeout (seconds) for the Redis/RQ worker path
JOB_TIMEOUT = int(os.environ.get('VRSEG_JOB_TIMEOUT', '3600'))

# Public demo mode (publicdemo.py): per-visitor datasets, uploads deleted after
# each job, results after RETENTION_MIN minutes, at most MAX_QUEUE jobs at once.
PUBLIC_DEMO = os.environ.get('VRSEG_PUBLIC_DEMO', '').lower() in ('1', 'true', 'on', 'yes')
RETENTION_MIN = int(os.environ.get('VRSEG_RETENTION_MIN', '60'))
MAX_QUEUE = int(os.environ.get('VRSEG_MAX_QUEUE', '3'))

# Accuracy estimate for every mesh (accuracy.py): compares each delivered mesh
# with the raw segmentation surface, adding a fraction of a second per mesh.
ACCURACY = os.environ.get('VRSEG_ACCURACY', 'on').lower() != 'off'

# Persistent GPU worker (gpu_worker.py): 'auto' starts it when a CUDA GPU is
# present and keeps the models loaded between jobs; 'off' runs every GPU step
# inside the job's own process, as before.
GPU_WORKER = os.environ.get('VRSEG_GPU_WORKER_MODE', 'auto').lower()

# Production queue: set REDIS_URL to use Redis+RQ; otherwise an in-process pool
REDIS_URL = os.environ.get('VRSEG_REDIS_URL') or os.environ.get('REDIS_URL') or ''

# Server bind. Prefer the generic $PORT most PaaS platforms inject
# (Render, Railway, Heroku-style, Hugging Face Spaces); fall back to
# VRSEG_PORT, then 8000 for local/VPS/docker-compose use.
HOST = os.environ.get('VRSEG_HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT') or os.environ.get('VRSEG_PORT') or '8000')

# Retain finished job rows this many days (0 = forever)
JOB_RETENTION_DAYS = int(os.environ.get('VRSEG_JOB_RETENTION_DAYS', '0'))

for _d in (OUTPUT_ROOT, UPLOAD_ROOT):
    os.makedirs(_d, exist_ok=True)
