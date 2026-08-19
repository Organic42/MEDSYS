"""SQLite-backed job store — survives restarts and is shared across processes
(API + worker). Replaces the old in-memory dict so jobs aren't lost on reload."""
import json
import time
import sqlite3
import threading
from datetime import datetime

import config

_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA busy_timeout=30000')
    return c


def init():
    with _lock, _conn() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            name TEXT,
            status TEXT,
            message TEXT,
            log TEXT,
            engine TEXT,
            created TEXT,
            updated REAL)''')


def create(job_id, name, engine='heuristic'):
    with _lock, _conn() as c:
        c.execute('INSERT OR REPLACE INTO jobs(id,name,status,message,log,engine,created,updated)'
                  ' VALUES(?,?,?,?,?,?,?,?)',
                  (job_id, name, 'queued', 'Queued', '[]', engine,
                   datetime.now().isoformat(timespec='seconds'), time.time()))


def update(job_id, **fields):
    if not fields:
        return
    if 'log' in fields and not isinstance(fields['log'], str):
        fields['log'] = json.dumps(fields['log'][-40:])
    fields['updated'] = time.time()
    cols = ', '.join(f'{k}=?' for k in fields)
    with _lock, _conn() as c:
        c.execute(f'UPDATE jobs SET {cols} WHERE id=?',
                  (*fields.values(), job_id))


def get(job_id):
    with _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d['log'] = json.loads(d.get('log') or '[]')
    except Exception:
        d['log'] = []
    return d


def prune(days):
    """Delete finished jobs older than `days` (housekeeping)."""
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    with _lock, _conn() as c:
        c.execute("DELETE FROM jobs WHERE updated < ? AND status IN ('done','error')",
                  (cutoff,))
