"""RQ worker entry point (production path). Started by the worker container.

    REDIS_URL=redis://redis:6379/0 python worker.py
"""
import config
import jobstore


def main():
    if not config.REDIS_URL:
        raise SystemExit("Set REDIS_URL to run the RQ worker.")
    from redis import Redis
    from rq import Queue, Worker

    jobstore.init()
    conn = Redis.from_url(config.REDIS_URL)
    queue = Queue('segmentation', connection=conn)
    print(f"[worker] listening on {config.REDIS_URL} queue=segmentation", flush=True)
    Worker([queue], connection=conn).work(with_scheduler=False)


if __name__ == '__main__':
    main()
