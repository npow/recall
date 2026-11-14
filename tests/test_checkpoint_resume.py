from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from scripts.prepare_matrix import parallel_download


def test_parallel_http_resume_preserves_prefix_and_fetches_only_missing_bytes(tmp_path):
    payload = bytes(range(256)) * 256
    prefix = 12345
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((start, end))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(payload[start:end + 1])
        def log_message(self, *unused):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    temporary = tmp_path / "checkpoint.part"
    temporary.write_bytes(payload[:prefix])
    try:
        parallel_download(f"http://127.0.0.1:{server.server_port}/checkpoint", temporary, len(payload), workers=4)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    assert temporary.read_bytes() == payload
    assert requests == [(prefix, len(payload) - 1)]
