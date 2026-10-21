"""
Persistent GPU worker: keeps PyTorch, CUDA and the segmentation models loaded
between jobs, so a job pays only for its own inference.

Each job still runs segmentation_pipeline.py in its own process; only the GPU
steps (TotalSegmentator and MedSAM refinement) are sent here. The worker

  - is started once by the app (or the RQ worker) and exits when it finds no
    CUDA GPU, so CPU machines are unaffected;
  - keeps nnU-Net networks and the MedSAM model in GPU memory after first use;
  - runs nnU-Net sequentially in its own process and saves masks with
    threads, avoiding the process pools that cost about 20 s per job on
    Windows (each pool worker re-imports PyTorch);
  - serves one request at a time over a local, authenticated socket.

If the worker is not running, or a request fails, the pipeline runs the step
in its own process exactly as before, so results never depend on it.

    python gpu_worker.py --serve      (normally started by app.py or worker.py)
"""
import os
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener

import config

ADDRESS_ENV, KEY_ENV, PARENT_ENV = 'VRSEG_GPU_WORKER', 'VRSEG_GPU_WORKER_KEY', 'VRSEG_GPU_WORKER_PARENT'
READY, UNAVAILABLE = 'GPU_WORKER_READY', 'GPU_WORKER_UNAVAILABLE'


# ── client side (used by segmentation_pipeline.py) ──────────────────────

def _request(message, timeout=config.JOB_TIMEOUT):
    """Send one request to the worker; None when no worker is configured."""
    address = os.environ.get(ADDRESS_ENV)
    if not address:
        return None
    host, port = address.rsplit(':', 1)
    try:
        with Client((host, int(port)), authkey=bytes.fromhex(os.environ[KEY_ENV])) as conn:
            conn.send(message)
            if not conn.poll(timeout):
                return {'ok': False, 'error': f'no reply within {timeout} s'}
            return conn.recv()
    except (OSError, EOFError, KeyError, ValueError, AuthenticationError) as e:
        return {'ok': False, 'error': f'worker unreachable: {e}'}


def run_totalseg(nifti, seg_dir, fast=True, task='total'):
    """TotalSegmentator on the warm worker. True on success; False means the
    caller should run it in-process (no worker, or the worker failed)."""
    reply = _request({'op': 'totalseg', 'input': os.path.abspath(nifti),
                      'output': os.path.abspath(seg_dir), 'fast': fast, 'task': task})
    if reply is None:
        return False
    if not reply.get('ok'):
        print(f"  GPU worker failed ({reply.get('error')}); running in this process", flush=True)
        return False
    print(f"  Ran on the persistent GPU worker in {reply['seconds']:.1f} s", flush=True)
    return True


def medsam_refine(norm, bet_mask, checkpoint):
    """MedSAM refinement on the warm worker; None means run it in-process."""
    import tempfile

    import numpy as np
    if not os.environ.get(ADDRESS_ENV):
        return None
    with tempfile.TemporaryDirectory(dir=config.UPLOAD_ROOT) as tmp:
        src, dst = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
        np.savez(src, norm=norm, bet=bet_mask)
        reply = _request({'op': 'medsam', 'input': src, 'output': dst, 'checkpoint': checkpoint})
        if not reply or not reply.get('ok'):
            if reply:
                print(f"  GPU worker failed ({reply.get('error')}); running in this process", flush=True)
            return None
        print(f"  Ran on the persistent GPU worker in {reply['seconds']:.1f} s", flush=True)
        with np.load(dst) as z:
            return z['mask']


def status():
    """For /api/capabilities: 'warm', 'starting' or 'off'. Reads this
    process's state only, so it never waits behind a running job."""
    if _process is not None and _process.poll() is not None:
        return 'off'                                        # started, then exited
    if os.environ.get(ADDRESS_ENV):
        return 'warm'
    return 'starting' if _process is not None else 'off'


# ── starting the worker (used by app.py and worker.py) ───────────────────

_process = None
# Routine lines from nnU-Net that say nothing about the worker's health
_QUIET = ('perform_everything_on_device', 'nnUNet_raw is not defined', 'nnUNet_preprocessed is not defined',
          'nnUNet_results is not defined')


def _log(message):
    print(message, flush=True)


def ensure_started(log=_log, wait=False):
    """Start the worker in the background unless disabled or already set up.

    The address and key go into os.environ, which every job subprocess
    inherits. Jobs that start before the worker is ready simply run the old
    in-process path.
    """
    global _process
    if os.environ.get(ADDRESS_ENV) or config.GPU_WORKER == 'off' or _process is not None:
        return
    import importlib.util
    if importlib.util.find_spec('torch') is None:          # lite builds: nothing to serve
        return
    key = secrets.token_hex(16)
    cmd = ([config.EXE_PATH, '--gpu-worker'] if config.FROZEN
           else [sys.executable, '-u', os.path.abspath(__file__), '--serve'])
    env = dict(os.environ, **{KEY_ENV: key, PARENT_ENV: str(os.getpid())}, PYTHONIOENCODING='utf-8')
    _process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', env=env)
    ready = threading.Event()

    def follow():
        for line in _process.stdout:
            line = line.rstrip()
            if line.startswith(READY):
                os.environ[KEY_ENV] = key
                os.environ[ADDRESS_ENV] = line.split()[1]
                log(f'[gpu-worker] warm on {line.split(maxsplit=2)[2]} at {line.split()[1]}')
                ready.set()
            elif line.startswith(UNAVAILABLE):
                log(f'[gpu-worker] not started: {line.split(maxsplit=1)[1]}')
                ready.set()
            elif line and not line.startswith(_QUIET):
                log(f'[gpu-worker] {line}')
        ready.set()

    threading.Thread(target=follow, daemon=True, name='gpu-worker-log').start()
    if wait:
        ready.wait(timeout=180)


def stop():
    global _process
    if _process is not None and _process.poll() is None:
        _process.terminate()
    _process = None
    os.environ.pop(ADDRESS_ENV, None)


# ── server side ──────────────────────────────────────────────────────────

def install_fast_path():
    """Patch TotalSegmentator's nnU-Net calls for a long-lived process.

    nnU-Net's predict_from_files spawns preprocessing and export workers, and
    TotalSegmentator saves one mask per class through a process Pool; on
    Windows every spawned process re-imports PyTorch. Here prediction runs
    through nnU-Net's own sequential path, networks loaded from a model
    folder are kept and reused, and masks are saved by a thread pool. The
    computation is unchanged, so the masks are identical.
    """
    from multiprocessing.pool import ThreadPool

    # nnU-Net reads its model folders from the environment when first imported;
    # TotalSegmentator sets them just before its own first import, so do the same.
    from totalsegmentator.config import setup_nnunet
    setup_nnunet()
    import totalsegmentator.nnunet as ts_nnunet
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    ts_nnunet.Pool = ThreadPool

    def predict_from_files(self, list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,
                           save_probabilities=False, overwrite=True, num_processes_preprocessing=None,
                           num_processes_segmentation_export=None, folder_with_segs_from_prev_stage=None,
                           num_parts=1, part_id=0):
        return self.predict_from_files_sequential(
            list_of_lists_or_source_folder, output_folder_or_list_of_truncated_output_files,
            save_probabilities=save_probabilities, overwrite=overwrite,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage)

    loaded = {}
    original_init = nnUNetPredictor.initialize_from_trained_model_folder
    fields = ('plans_manager', 'configuration_manager', 'list_of_parameters', 'network', 'dataset_json',
              'trainer_name', 'allowed_mirroring_axes', 'label_manager')

    def initialize(self, model_training_output_dir, use_folds, checkpoint_name='checkpoint_final.pth'):
        key = (os.path.abspath(model_training_output_dir), str(use_folds), checkpoint_name)
        if key not in loaded:
            original_init(self, model_training_output_dir, use_folds, checkpoint_name)
            loaded[key] = {f: getattr(self, f) for f in fields}
        for f, v in loaded[key].items():
            setattr(self, f, v)

    nnUNetPredictor.predict_from_files = predict_from_files
    nnUNetPredictor.initialize_from_trained_model_folder = initialize
    return loaded


def _exit_with_parent(pid):
    """Exit when the process that started us exits.

    Not by reading stdin: on Windows a thread blocked reading a pipe stalls
    every child process that inherits it, including those nnU-Net spawns.
    """
    if os.name == 'nt':
        import _winapi
        try:
            handle = _winapi.OpenProcess(0x00100000, False, pid)      # SYNCHRONIZE
        except OSError:
            os._exit(0)
        _winapi.WaitForSingleObject(handle, _winapi.INFINITE)
        os._exit(0)
    while os.getppid() == pid:
        time.sleep(2)
    os._exit(0)


def serve():
    try:
        import torch
        if not torch.cuda.is_available():
            print(f'{UNAVAILABLE} no CUDA GPU', flush=True)
            return
        torch.zeros(1, device='cuda')                       # create the CUDA context now
        device_name = torch.cuda.get_device_name(0)
        from totalsegmentator.python_api import totalsegmentator
    except Exception as e:  # noqa: BLE001
        print(f'{UNAVAILABLE} {e}', flush=True)
        return
    loaded_networks = install_fast_path()
    sam_models = {}

    parent = int(os.environ.get(PARENT_ENV, '0'))
    if parent:
        threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()

    def handle(msg):
        op = msg.get('op')
        if op == 'ping':
            return {'device': device_name, 'networks': len(loaded_networks), 'medsam': bool(sam_models)}
        if op == 'totalseg':
            os.makedirs(msg['output'], exist_ok=True)
            totalsegmentator(msg['input'], msg['output'], fast=msg['fast'], ml=False, device='gpu',
                             task=msg['task'], quiet=True)
            return {}
        if op == 'medsam':
            import numpy as np

            import medsam
            if msg['checkpoint'] not in sam_models:
                sam_models[msg['checkpoint']] = medsam.load(msg['checkpoint'])
            with np.load(msg['input']) as z:
                mask = medsam.refine(sam_models[msg['checkpoint']], z['norm'], z['bet'])
            np.savez(msg['output'], mask=mask)
            return {}
        raise ValueError(f'unknown op {op!r}')

    with Listener(('127.0.0.1', 0), authkey=bytes.fromhex(os.environ[KEY_ENV])) as listener:
        host, port = listener.address
        print(f'{READY} {host}:{port} {device_name}', flush=True)
        while True:
            try:
                conn = listener.accept()
            except Exception:  # noqa: BLE001  (a client with the wrong key, or a dropped connection)
                continue
            with conn:
                try:
                    msg = conn.recv()
                    t = time.perf_counter()
                    reply = {'ok': True, **handle(msg), 'seconds': time.perf_counter() - t}
                except Exception as e:  # noqa: BLE001
                    reply = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
                    print(f"request {msg.get('op') if isinstance(msg, dict) else '?'} failed: {reply['error']}",
                          flush=True)
                try:
                    conn.send(reply)
                except Exception:  # noqa: BLE001
                    pass


if __name__ == '__main__':
    if '--serve' in sys.argv:
        serve()
    else:
        print(__doc__)
