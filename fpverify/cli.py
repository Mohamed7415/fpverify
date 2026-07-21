"""命令行入口。

  python -m fpverify.cli enroll   --base-url URL --api-key KEY --model NAME --out ref.json
  python -m fpverify.cli audit    --base-url URL --api-key KEY --model NAME --ref ref.json
                                  [--alpha 0.01] [--delta 0.02] [--max-queries 600] [--report out.json]
  python -m fpverify.cli identify --base-url URL --api-key KEY --model NAME
                                  [--claimed 库条目] [--channel api] [--samples 8] [--report out.json]
  python -m fpverify.cli library  [--library refs]

enroll/audit 面向**有官方渠道**的审计者；identify 面向**只有中转站 key** 的用户——
对照公共指纹库 refs/ 做降档识别：库内→验真伪（带 FPR 保证），库外→最像谁/未知。

成本提示：enroll 默认 20 样本/cell × 36 cell = 720 次单 token 请求；
audit 有早停，预算上限 --max-queries（默认 600），明显造假通常 <100 次就出结论；
identify 预算 = --samples × 库频道 cell 数（例如 8 × 36 = 288 次）。
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

    pi = sub.add_parser("identify", help="对照公共指纹库做降档识别（无需自己入册）")
    pi.add_argument("--base-url", required=True)
    pi.add_argument("--api-key")
    pi.add_argument("--model", required=True, help="向端点请求时使用的模型名")
    pi.add_argument("--claimed", help="声称的库条目 id/名称（缺省 = 用 --model 在库里解析）")
    pi.add_argument("--library", help="指纹库目录（缺省 = 仓库自带 refs/）")
    pi.add_argument("--channel", default="api", help="库频道：api / cursor-harness（默认 api）")
    pi.add_argument("--samples", type=int, default=8, help="每 cell 采样数（默认 8）")
    pi.add_argument("--alpha", type=float, default=0.01)
    pi.add_argument("--seed", type=int, default=None)
    pi.add_argument("--timeout", type=float, default=60)
    pi.add_argument("--report", help="保存 JSON 报告")

    pl = sub.add_parser("library", help="列出公共指纹库条目")
    pl.add_argument("--library", help="指纹库目录（缺省 = 仓库自带 refs/）")

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

    if args.cmd in ("identify", "library"):
        from .library import Library, default_library_path, identify

        lib = Library.load(args.library or default_library_path())

        if args.cmd == "library":
            print(f"指纹库: {lib.root}（更新于 {lib.meta.get('updated_at', '?')}）")
            for ch in lib.channels():
                print(f"\n[{ch}]")
                for e in lib.by_channel(ch):
                    print(f"  {e.id:<16} {e.model:<28} {e.family:<8} "
                          f"n={e.samples_per_cell}/cell  {e.enrolled_at}")
            if not lib.by_channel("api"):
                print("\napi 频道暂无条目——有官方 key 的话，欢迎按 refs/README.md 贡献。")
            return 0

        ep = _endpoint(args)
        claimed = args.claimed or args.model
        print(f"降档识别: 端点声称『{claimed}』, 频道 {args.channel}, "
              f"预算 {args.samples} 样本/cell")
        res = identify(ep, lib, claimed, channel=args.channel,
                       samples_per_cell=args.samples, alpha=args.alpha, seed=args.seed,
                       progress=lambda n, m, w: print(
                           f"  进度 {n}/{m}" + (f"  财富={w:.3g}" if w is not None else "")))
        print()
        print(f"判定: {res.verdict}")
        print(f"  {res.detail}")
        if res.warning:
            print(f"  警告: {res.warning}")
        if res.ranking:
            print("  与库内模型的距离（聚合 JSD，越小越像）:")
            for mid, d in res.ranking[:8]:
                e = lib.get(mid)
                name = e.model if e else mid
                mark = " ← 声称" if res.claimed_entry == mid else ""
                print(f"    {d:.3f}  {name}{mark}")
            print(f"  解读带: ≤{res.bands['match']} 一致 / ≥{res.bands['unknown']} 未知")
        if args.report:
            import json as _json
            from pathlib import Path as _Path
            _Path(args.report).write_text(
                _json.dumps(res.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"JSON 报告已保存: {args.report}")
        ep.close()
        return {"PASS": 0, "BEST_MATCH": 0, "SUSPECT": 3, "FAIL": 4, "UNKNOWN": 5}.get(res.verdict, 2)

    return 2


if __name__ == "__main__":
    sys.exit(main())
