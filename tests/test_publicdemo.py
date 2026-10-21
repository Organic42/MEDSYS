"""Tests for public demo mode (publicdemo.py)."""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import jobstore  # noqa: E402
import publicdemo  # noqa: E402


@pytest.fixture
def demo(tmp_path, monkeypatch):
    out, up = tmp_path / 'output', tmp_path / 'uploads'
    out.mkdir(), up.mkdir()
    monkeypatch.setattr(config, 'PUBLIC_DEMO', True)
    monkeypatch.setattr(config, 'OUTPUT_ROOT', str(out))
    monkeypatch.setattr(config, 'UPLOAD_ROOT', str(up))
    monkeypatch.setattr(config, 'DB_PATH', str(tmp_path / 'jobs.db'))
    monkeypatch.setattr(config, 'RETENTION_MIN', 60)
    jobstore.init()
    return out, up


def test_visitors_see_only_their_own_scans_and_the_samples(demo):
    out, _ = demo
    publicdemo.mark(str(out / 'mine'), publicdemo.owner_tag('alice'))
    publicdemo.mark(str(out / 'sample'), publicdemo.SAMPLE)
    (out / 'unowned').mkdir()
    seen = lambda v: sorted(n for n in os.listdir(out) if publicdemo.visible(str(out / n), v))  # noqa: E731
    assert seen('alice') == ['mine', 'sample']
    assert seen('bob') == ['sample'] and seen(None) == ['sample']


def test_owner_tag_hides_the_visitor_id():
    tag = publicdemo.owner_tag('alice')
    assert tag != 'alice' and len(tag) == 20 and tag == publicdemo.owner_tag(' alice ')
    assert publicdemo.owner_tag('') is None and publicdemo.owner_tag('x' * 200) is None


def test_outside_demo_mode_everything_is_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'PUBLIC_DEMO', False)
    assert publicdemo.visible(str(tmp_path), None)


def test_cleanup_removes_old_results_and_uploads_but_keeps_samples(demo):
    out, up = demo
    old = time.time() - 2 * 3600
    for d in ('old_scan', 'sample', 'new_scan'):
        publicdemo.mark(str(out / d), publicdemo.SAMPLE if d == 'sample' else 'someone')
    (up / 'old_upload').mkdir()
    for p in (out / 'old_scan', out / 'sample', up / 'old_upload'):
        os.utime(p, (old, old))
    publicdemo.cleanup()
    assert sorted(os.listdir(out)) == ['new_scan', 'sample'] and os.listdir(up) == []


def test_uploads_are_removed_only_inside_the_upload_folder(demo, tmp_path):
    _, up = demo
    (up / 'job1' / 'dicom').mkdir(parents=True)
    publicdemo.remove_upload(str(up / 'job1' / 'dicom'))
    assert not (up / 'job1').exists()
    outside = tmp_path / 'elsewhere' / 'dicom'
    outside.mkdir(parents=True)
    publicdemo.remove_upload(str(outside))             # never deletes outside UPLOAD_ROOT
    assert outside.exists()


def test_queue_limit(demo, monkeypatch):
    monkeypatch.setattr(config, 'MAX_QUEUE', 2)
    assert not publicdemo.busy()
    jobstore.create('a', 'a'); jobstore.create('b', 'b')
    assert publicdemo.busy()
    jobstore.update('a', status='done')
    assert not publicdemo.busy()
