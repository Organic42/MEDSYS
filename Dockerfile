# VR-Segmentation service image (API + worker share this image).
# CPU by default. For GPU, see the note at the bottom.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PYVISTA_OFF_SCREEN=true \
    VRSEG_HOST=0.0.0.0 \
    VRSEG_OUTPUT=/data/output \
    VRSEG_UPLOADS=/data/uploads \
    VRSEG_DB=/data/jobs.db

# System libs needed by VTK/PyVista (headless GL), SimpleITK, and rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 libxrender1 libxext6 libsm6 \
        xvfb libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
# NOTE: requirements-full.txt pulls torch + TotalSegmentator (multi-GB).
# For a CPU-only image you may pin the CPU torch wheel to shrink it:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-full.txt .
RUN pip install --upgrade pip && pip install -r requirements-full.txt

COPY . .

# /data holds outputs, uploads, and the job DB (mount a volume here)
RUN mkdir -p /data/output /data/uploads
VOLUME ["/data"]
EXPOSE 8000

# xvfb-run gives PyVista a virtual display for off-screen rendering.
# Default command runs the API; the worker service overrides it (see compose).
CMD ["xvfb-run", "-a", "python", "app.py"]

# ── GPU image ────────────────────────────────────────────────────────────
# Build on a CUDA base instead and install the matching torch wheel, e.g.:
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#   (install python3 + the cu121 torch wheel)
# Then run the worker with `--gpus all` (compose: see the `gpu` profile).
