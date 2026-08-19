"""Desktop launcher: starts the MEDSYS server and opens it in the browser.

Used both when running `python launcher.py` directly and as the normal
(non `--run-pipeline`) path inside the frozen .exe via entry.py.
"""
import os
import sys
import time
import socket
import threading
import webbrowser
import urllib.request

import config


def _port_is_open(host, port, timeout=0.4):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_and_open_browser(host, port, max_wait=30):
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if _port_is_open('127.0.0.1', port):
            try:
                urllib.request.urlopen(url + '/health', timeout=1)
                break
            except Exception:
                pass
        time.sleep(0.3)
    print(f"\nMEDSYS is ready -> {url}\nOpening in your browser...\n", flush=True)
    webbrowser.open(url)


def main():
    print("=" * 60)
    print("  MEDSYS — DICOM to 3D  (standalone desktop build)")
    print("=" * 60)
    print(f"  Data folder: {config.RUNTIME_DIR}")
    print(f"  Starting server on port {config.PORT}...")
    print("  Close this window to stop the server.")
    print("=" * 60, flush=True)

    threading.Thread(target=_wait_and_open_browser,
                     args=(config.HOST, config.PORT), daemon=True).start()

    import uvicorn
    from app import app
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level='info')


if __name__ == '__main__':
    main()
