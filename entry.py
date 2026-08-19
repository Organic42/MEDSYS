"""
Single entry point for the frozen MEDSYS.exe.

A frozen exe has no separate `python` interpreter to shell out to, so the
app can't launch `python segmentation_pipeline.py ...` as a worker subprocess
the way it does when run from source. Instead, tasks.py re-invokes THIS exe
with a `--run-pipeline` flag; we detect that here and dispatch straight into
segmentation_pipeline.main() instead of starting the web server.

This must be the PyInstaller entry script (the file passed to `pyinstaller`).
"""
import sys

# A frozen exe's stdout, when piped (as it always is when self-invoked as a
# subprocess), isn't guaranteed to be UTF-8 on Windows — it can fall back to
# the system codepage (cp1252) and crash on the first emoji/unicode symbol
# anywhere in our own logging or a third-party library's warnings. Force it
# so a stray character degrades gracefully instead of taking down the job.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--run-pipeline':
        # Strip our dispatch flag; argparse in segmentation_pipeline.main()
        # reads the rest of sys.argv as if it were invoked directly.
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        import segmentation_pipeline
        segmentation_pipeline.main()
    else:
        import launcher
        launcher.main()


if __name__ == '__main__':
    main()
