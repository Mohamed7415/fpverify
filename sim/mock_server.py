# -*- coding: utf-8 -*-
"""OpenAI 兼容的模拟中转站 HTTP 服务器（红队演示用）。

把 sim.adversaries 的九类对手挂到真实的 /v1/chat/completions 上，
用于端到端验证 fpverify 的 HTTP 链路（enroll → audit）：

  py -3.13 -X utf8 sim/mock_server.py --port 18801 --kind honest --model gpt-4o
  py -3.13 -X utf8 sim/mock_server.py --port 18802 --kind swap   --model gpt-4o

要点：
- 无论真身是谁，响应 model 字段永远回显请求方宣称的模型名（模拟中转站撒谎）；
- 对手内部的仿真延迟按 --latency-scale 折算成真实 sleep（默认 0.05，
  即"解码 0.8s"折算成 40ms），保持缓存对手的相对延迟签名又不拖慢演示。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.adversaries import ADVERSARY_KINDS, make_endpoint


def make_handler(endpoint, latency_scale: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self.path.rstrip("/").endswith("chat/completions"):
                self._send(404, b'{"error":"unknown path"}')
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                self._send(400, b'{"error":"bad json"}')
                return

            system, user = "", ""
            for m in data.get("messages", []):
                c = m.get("content", "")
                if not isinstance(c, str):
                    c = " ".join(str(x.get("text", "")) for x in c if isinstance(x, dict))
                if m.get("role") == "system":
                    system = c
                elif m.get("role") == "user":
                    user = c

            ans = endpoint.ask(system, user)
            if ans.latency and latency_scale > 0:
                time.sleep(ans.latency * latency_scale)

            resp = {
                "id": f"chatcmpl-mock-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                # 中转站的谎言：永远回显宣称的模型名
                "model": data.get("model", ans.model_field or "unknown"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ans.text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 40, "completion_tokens": 3, "total_tokens": 43},
            }
            self._send(200, json.dumps(resp, ensure_ascii=False).encode("utf-8"))

    return Handler


def main():
    ap = argparse.ArgumentParser(description="模拟中转站（OpenAI 兼容）")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--kind", choices=ADVERSARY_KINDS, required=True,
                    help="对手类型：honest/drift/quantized/swap/pin/filter_en/true_random/cache/partial_mimic")
    ap.add_argument("--model", default="gpt-4o", help="端点宣称提供的模型（真身由 --kind 决定）")
    ap.add_argument("--actual", default="cheap-7b", help="掺水对手实际使用的廉价模型")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-scale", type=float, default=0.05,
                    help="仿真延迟折算比例（0=不 sleep）")
    args = ap.parse_args()

    kw = {}
    if args.kind in ("swap", "pin", "filter_en", "true_random", "partial_mimic"):
        kw["actual"] = args.actual
    endpoint = make_endpoint(args.kind, args.model, seed=args.seed, **kw)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(endpoint, args.latency_scale))
    print(f"mock relay on 127.0.0.1:{args.port}  kind={args.kind}  claimed={args.model}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
