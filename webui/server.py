# -*- coding: utf-8 -*-
"""fpverify 本地检测台：给"只有中转站 key"的用户的图形界面。

    python -X utf8 -m webui.server            # 默认 http://127.0.0.1:8765
    python -X utf8 webui/server.py --port 9000 --library refs

安全边界：服务只绑定 127.0.0.1；探针请求从**你自己的机器**直连中转站；
API key 只存在本进程内存里，不写盘、不上传任何后台。指纹库是随仓库分发的
公开数据（更新 = git pull）。纯标准库实现，identify 依赖 httpx（探针请求）。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpverify.endpoints import HTTPEndpoint
from fpverify.library import Library, default_library_path, identify

INDEX = Path(__file__).with_name("index.html")

_state_lock = threading.Lock()
_state = {"running": False, "log": [], "result": None, "error": None}
_lib_path: Path = default_library_path()


def _log(line: str):
    with _state_lock:
        _state["log"].append(line)
        if len(_state["log"]) > 400:
            del _state["log"][:100]


def _run_job(params: dict):
    ep = None
    try:
        lib = Library.load(_lib_path)
        ep = HTTPEndpoint(params["base_url"], params["api_key"], params["model"],
                          timeout=float(params.get("timeout", 45)))
        claimed = params.get("claimed") or params["model"]
        channel = params.get("channel", "api")
        spc = max(2, min(40, int(params.get("samples", 8))))
        _log(f"开始识别：声称『{claimed}』，频道 {channel}，{spc} 样本/cell")
        res = identify(
            ep, lib, claimed, channel=channel, samples_per_cell=spc,
            progress=lambda n, m, w: _log(
                f"进度 {n}/{m}" + (f"　财富={w:.3g}" if w is not None else "")))
        d = res.to_dict()
        d["ranking_display"] = [
            [(lib.get(mid).model if lib.get(mid) else mid), dist, mid == res.claimed_entry]
            for mid, dist in res.ranking]
        with _state_lock:
            _state["result"] = d
        _log(f"完成：{res.verdict}")
    except Exception as e:  # 展示给本地用户，而非静默
        with _state_lock:
            _state["error"] = f"{type(e).__name__}: {e}"
        _log(f"出错：{e}")
    finally:
        if ep is not None:
            try:
                ep.close()
            except Exception:
                pass
        with _state_lock:
            _state["running"] = False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/library":
            try:
                lib = Library.load(_lib_path)
                self._json({
                    "root": str(lib.root),
                    "updated_at": lib.meta.get("updated_at", ""),
                    "channels": lib.meta.get("channels", {}),
                    "entries": [{"id": e.id, "model": e.model, "family": e.family,
                                 "channel": e.channel, "enrolled_at": e.enrolled_at,
                                 "samples_per_cell": e.samples_per_cell, "note": e.note}
                                for e in lib.entries],
                })
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif self.path == "/api/status":
            with _state_lock:
                self._json({k: _state[k] for k in ("running", "log", "result", "error")})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/models":
            # 代拉中转站的 /v1/models（请求同样从本机发出，key 不出机器）
            try:
                n = int(self.headers.get("Content-Length", 0))
                p = json.loads(self.rfile.read(n).decode("utf-8"))
                base = str(p.get("base_url", "")).rstrip("/")
                key = str(p.get("api_key", ""))
                if not base or not key:
                    self._json({"error": "需要 base_url 和 api_key"}, 400)
                    return
                import httpx
                r = httpx.get(f"{base}/models",
                              headers={"Authorization": f"Bearer {key}"}, timeout=30)
                if r.status_code != 200:
                    self._json({"error": f"HTTP {r.status_code}: {r.text[:200]}"}, 502)
                    return
                ids = sorted(m.get("id", "") for m in (r.json().get("data") or []) if m.get("id"))
                self._json({"models": ids})
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            return
        if self.path != "/api/run":
            self._json({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            params = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            self._json({"error": "请求体不是合法 JSON"}, 400)
            return
        missing = [k for k in ("base_url", "api_key", "model") if not str(params.get(k, "")).strip()]
        if missing:
            self._json({"error": f"缺少字段: {', '.join(missing)}"}, 400)
            return
        with _state_lock:
            if _state["running"]:
                self._json({"error": "已有检测在运行，请等它结束"}, 409)
                return
            _state.update({"running": True, "log": [], "result": None, "error": None})
        threading.Thread(target=_run_job, args=(params,), daemon=True).start()
        self._json({"ok": True})

    def log_message(self, fmt, *args):  # 静音默认访问日志
        pass


def main(argv=None) -> int:
    global _lib_path
    ap = argparse.ArgumentParser(description="fpverify 本地检测台（key 不出本机）")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--library", help="指纹库目录（缺省 = 仓库自带 refs/）")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.library:
        _lib_path = Path(args.library)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"fpverify 检测台: {url}   （Ctrl+C 退出；key 只在本机内存）")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
