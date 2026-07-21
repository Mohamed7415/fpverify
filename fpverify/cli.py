"""命令行入口。

  python -m fpverify.cli enroll --base-url URL --api-key KEY --model NAME --out ref.json
  python -m fpverify.cli audit  --base-url URL --api-key KEY --model NAME --ref ref.json
                                [--alpha 0.01] [--delta 0.02] [--max-queries 600] [--report out.json]

成本提示：enroll 默认 20 样本/cell × 36 cell = 720 次单 token 请求；
audit 有早停，预算上限 --max-queries（默认 600），明显造假通常 <100 次就出结论。
"""

from __future__ import annotations

import argparse
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .betting import BettingConfig
from .endpoints import HTTPEndpoint
from .fingerprint import Fingerprint
from .report import render_text, save_json
from .verifier import Verifier


def _endpoint(args) -> HTTPEndpoint:
    key = args.api_key or os.environ.get("FP_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        print("缺少 API key：--api-key 或环境变量 FP_API_KEY / OPENAI_API_KEY")
        sys.exit(1)
    return HTTPEndpoint(args.base_url, key, args.model, timeout=args.timeout)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fpverify",
                                 description="LLM 行为指纹验证：检测中转站模型注水（研究实现）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("enroll", help="从可信渠道入册参考指纹")
    pe.add_argument("--base-url", required=True)
    pe.add_argument("--api-key")
    pe.add_argument("--model", required=True)
    pe.add_argument("--samples", type=int, default=20, help="每个 cell 的采样数（默认 20）")
    pe.add_argument("--timeout", type=float, default=60)
    pe.add_argument("--out", default="ref.json")

    pa = sub.add_parser("audit", help="序贯审计中转站端点")
    pa.add_argument("--base-url", required=True)
    pa.add_argument("--api-key")
    pa.add_argument("--model", required=True, help="向中转站宣称请求的模型名")
    pa.add_argument("--ref", required=True)
    pa.add_argument("--alpha", type=float, default=0.01, help="FPR 上界（默认 0.01）")
    pa.add_argument("--delta", type=float, default=None,
                    help="良性漂移容差；缺省=用参考指纹自动标定（推荐）")
    pa.add_argument("--max-queries", type=int, default=600)
    pa.add_argument("--timeout", type=float, default=60)
    pa.add_argument("--report", help="保存 JSON 报告")

    args = ap.parse_args(argv)

    if args.cmd == "enroll":
        ep = _endpoint(args)
        v = Verifier()
        n_cells = 36  # len(probes.all_cells())
        print(f"入册参考指纹: {args.model}")
        print(f"预算: 约 {args.samples} 样本/cell × {n_cells} cell ≈ {args.samples * n_cells} 次单 token 请求")
        fp = v.enroll(ep, args.model, samples_per_cell=args.samples,
                      progress=lambda i, n: print(f"  进度 {i}/{n}"))
        fp.save(args.out)
        print(f"完成。总样本 {fp.total_samples()}，有效率 {fp.overall_validity():.1%}")
        print(f"已保存: {args.out}")
        ep.close()
        return 0

    if args.cmd == "audit":
        ep = _endpoint(args)
        ref = Fingerprint.load(args.ref)
        cfg = BettingConfig(alpha=args.alpha, delta=args.delta)
        v = Verifier(cfg)
        print(f"审计端点（宣称 {args.model}，参考 {ref.model}）")
        res = v.audit(ep, ref, max_queries=args.max_queries,
                      progress=lambda n, m, w: print(f"  进度 {n}/{m}  财富={w:.3g}"))
        print()
        print(render_text(res, ref.model))
        if args.report:
            save_json(res, ref.model, args.report)
            print(f"JSON 报告已保存: {args.report}")
        ep.close()
        return {"PASS": 0, "SUSPECT": 3, "FAIL": 4}.get(res.verdict, 2)

    return 2


if __name__ == "__main__":
    sys.exit(main())
