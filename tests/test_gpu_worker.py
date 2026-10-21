"""Tests for the persistent GPU worker's client side and fallbacks.

The worker itself needs CUDA and TotalSegmentator; these tests check what a
CPU-only CI machine can: that the pipeline falls back cleanly when there is
no worker, and that the request protocol works against a stand-in server.
"""
import os
import secrets
import sys
import threading
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_worker  # noqa: E402


@pytest.fixture
def no_worker(monkeypatch):
    for key in (gpu_worker.ADDRESS_ENV, gpu_worker.KEY_ENV):
        monkeypatch.delenv(key, raising=False)


def test_without_a_worker_every_step_runs_in_process(no_worker):
    assert gpu_worker.status() == 'off'
    assert gpu_worker.run_totalseg('ct.nii.gz', 'seg') is False
    assert gpu_worker.medsam_refine(np.zeros((2, 4, 4)), np.zeros((2, 4, 4), bool), 'ckpt') is None


def test_an_unreachable_worker_falls_back(monkeypatch):
    with Listener(('127.0.0.1', 0)) as probe:      # find a free port, then close it
        port = probe.address[1]
    monkeypatch.setenv(gpu_worker.ADDRESS_ENV, f'127.0.0.1:{port}')
    monkeypatch.setenv(gpu_worker.KEY_ENV, secrets.token_hex(16))
    assert gpu_worker.run_totalseg('ct.nii.gz', 'seg') is False


def serve_once(replies):
    """A stand-in worker answering one request per reply, with a random key."""
    key = secrets.token_hex(16)
    listener = Listener(('127.0.0.1', 0), authkey=bytes.fromhex(key))
    received = []

    def run():
        for reply in replies:
            try:
                conn = listener.accept()
            except AuthenticationError:
                continue                            # a client with the wrong key
            with conn:
                received.append(conn.recv())
                conn.send(reply)
        listener.close()

    threading.Thread(target=run, daemon=True).start()
    host, port = listener.address
    return f'{host}:{port}', key, received


def test_requests_reach_the_worker(monkeypatch, tmp_path):
    address, key, received = serve_once([{'ok': True, 'seconds': 0.5}, {'ok': False, 'error': 'boom'}])
    monkeypatch.setenv(gpu_worker.ADDRESS_ENV, address)
    monkeypatch.setenv(gpu_worker.KEY_ENV, key)
    assert gpu_worker.run_totalseg('ct.nii.gz', str(tmp_path / 'seg'), fast=True, task='total') is True
    assert received[0]['op'] == 'totalseg' and received[0]['fast'] is True and received[0]['task'] == 'total'
    assert os.path.isabs(received[0]['input']) and os.path.isabs(received[0]['output'])
    assert gpu_worker.run_totalseg('ct.nii.gz', str(tmp_path / 'seg')) is False      # failure -> fall back


def test_a_wrong_key_is_refused(monkeypatch):
    address, _, received = serve_once([{'ok': True, 'seconds': 0.1}])
    monkeypatch.setenv(gpu_worker.ADDRESS_ENV, address)
    monkeypatch.setenv(gpu_worker.KEY_ENV, secrets.token_hex(16))                # not the server's key
    assert gpu_worker.run_totalseg('ct.nii.gz', 'seg') is False
    assert received == []


def test_disabled_by_configuration(monkeypatch, no_worker):
    monkeypatch.setattr(gpu_worker.config, 'GPU_WORKER', 'off')
    monkeypatch.setattr(gpu_worker, '_process', None)
    gpu_worker.ensure_started()
    assert gpu_worker._process is None and gpu_worker.status() == 'off'


FAST_PATH_CHECK = """
import sys
sys.path.insert(0, {root!r})
import torch  # the worker's import order: PyTorch first
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
calls = []

def fake_init(self, folder, folds, checkpoint_name='checkpoint_final.pth'):
    calls.append(folder)
    for f in ('plans_manager', 'configuration_manager', 'list_of_parameters', 'network', 'dataset_json',
              'trainer_name', 'allowed_mirroring_axes', 'label_manager'):
        setattr(self, f, f + '-' + str(len(calls)))

nnUNetPredictor.initialize_from_trained_model_folder = fake_init
import gpu_worker
from multiprocessing.pool import ThreadPool
import totalsegmentator.nnunet as ts_nnunet
loaded = gpu_worker.install_fast_path()
a, b = object.__new__(nnUNetPredictor), object.__new__(nnUNetPredictor)
a.initialize_from_trained_model_folder('model', [0])
b.initialize_from_trained_model_folder('model', [0])
assert calls == ['model'] and b.network == 'network-1' and len(loaded) == 1, (calls, b.network)
assert ts_nnunet.Pool is ThreadPool
assert nnUNetPredictor.predict_from_files.__name__ == 'predict_from_files'
print('OK')
"""


def test_fast_path_reuses_loaded_networks():
    """Run in a fresh process with the worker's import order: importing
    TotalSegmentator after this test session's other native libraries can
    crash on Windows (a pyarrow DLL clash), which the worker never meets."""
    import importlib.util
    import subprocess
    if not (importlib.util.find_spec('totalsegmentator') and importlib.util.find_spec('nnunetv2')):
        pytest.skip('TotalSegmentator is not installed')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, '-c', FAST_PATH_CHECK.format(root=root)],
                            capture_output=True, text=True, timeout=300)
    assert result.stdout.strip().endswith('OK'), result.stderr[-2000:]
