"""Start the server on a free local port and open the browser once it answers."""
import os
import socket
import threading
import time
import urllib.request
import webbrowser

import uvicorn


def free_port(preferred=8787):
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise SystemExit("no free port")


def main():
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    server = uvicorn.Server(uvicorn.Config("server.app:app", host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(600):
        try:
            urllib.request.urlopen(url + "api/health", timeout=1).read()
            break
        except Exception:
            time.sleep(0.2)
    print(f"星空的时间机器：{url}  （关闭此窗口即停止）", flush=True)
    if os.environ.get("STM_NO_BROWSER") != "1":
        webbrowser.open(url)
    try:
        while not server.should_exit:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True


if __name__ == "__main__":
    os.environ.setdefault("STM_OFFLINE", "1")
    main()
