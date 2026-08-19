"""Environment-driven configuration for the VR-Segmentation service."""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(env, default):
    return os.path.abspath(os.environ.get(env, os.path.join(PROJECT_DIR, default)))


# Storage locations (override in containers via env)
OUTPUT_ROOT = _path('VRSEG_OUTPUT', 'output')
UPLOAD_ROOT = _path('VRSEG_UPLOADS', '_uploads')
DB_PATH = _path('VRSEG_DB', 'jobs.db')
WEB_DIR = os.path.join(PROJECT_DIR, 'web')

# Concurrency — segmentation is heavy, so serialize by default
MAX_WORKERS = int(os.environ.get('VRSEG_MAX_WORKERS', '1'))

# Upload guardrails
MAX_UPLOAD_MB = int(os.environ.get('VRSEG_MAX_UPLOAD_MB', '4096'))

# Job timeout (seconds) for the Redis/RQ worker path
JOB_TIMEOUT = int(os.environ.get('VRSEG_JOB_TIMEOUT', '3600'))

# Production queue: set REDIS_URL to use Redis+RQ; otherwise an in-process pool
REDIS_URL = os.environ.get('VRSEG_REDIS_URL') or os.environ.get('REDIS_URL') or ''

# Server bind
HOST = os.environ.get('VRSEG_HOST', '127.0.0.1')
PORT = int(os.environ.get('VRSEG_PORT', '8000'))

# Retain finished job rows this many days (0 = forever)
JOB_RETENTION_DAYS = int(os.environ.get('VRSEG_JOB_RETENTION_DAYS', '0'))

for _d in (OUTPUT_ROOT, UPLOAD_ROOT):
    os.makedirs(_d, exist_ok=True)
