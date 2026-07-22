# -*- coding: utf-8 -*-
"""对中转站跑铁律双跑：同一个被测模型，分别对照两份参考条目的铁律题。

用法:
  py -3.13 -X utf8 experiments/relay_ironlaw_check.py \
      --base-url https://.../v1 --model gpt-5.6 --refs gpt56-sol fable5 [--n 10]

key 从环境变量 RELAY_KEY 读取，不进命令行历史。
输出:每道铁律题的观测计数 vs 两份参考的众数，肉眼可判。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpverify.library import Library, default_library_path
from fpverify.normalize import normalize
from fpverify.reproduce import top_invariants

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def ask(base_url: str, key: str, model: str, system: str, prompt: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": prompt}],
        "temperature": 1.0, "max_tokens": 16,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="被测端点上的模型名")
    ap.add_argument("--refs", nargs="+", required=True, help="要对照的库条目 id")
    ap.add_argument("--n", type=int, default=10, help="每题请求次数")
    ap.add_argument("--top", type=int, default=5, help="每份参考取几条铁律")
    a = ap.parse_args()

    key = os.environ.get("RELAY_KEY")
    if not key:
        sys.exit("请先设置环境变量 RELAY_KEY")

    lib = Library.load(default_library_path())
    for ref_id in a.refs:
        entry = lib.get(ref_id)
        if entry is None:
            sys.exit(f"库里没有条目 {ref_id}")
        invs = top_invariants(lib.fingerprint(entry), k=a.top)
        print("=" * 72)
        print(f"参考: {entry.model}（{ref_id}，渠道 {entry.channel}）  被测: {a.model} @ 中转站")
        print("=" * 72)
        hits = total = 0
        for inv in invs:
            c: collections.Counter = collections.Counter()
            for _ in range(a.n):
                try:
                    raw = ask(a.base_url, key, a.model, inv.system, inv.prompt)
                    c[normalize(inv.parse, raw)] += 1
                except Exception as e:  # noqa: BLE001 —— 网络失败也计入，透明呈现
                    c[f"<error:{type(e).__name__}>"] += 1
            obs = "  ".join(f"{k}×{v}" for k, v in c.most_common(4))
            match = c.most_common(1)[0][0] == inv.expected if c else False
            hits += match
            total += 1
            mark = "对上" if match else "偏离"
            print(f"[{mark}] {inv.prompt!r}")
            print(f"       参考众数 {inv.expected}（占比 {inv.share:.0%}）   观测 {obs}")
        print(f"小结: 众数吻合 {hits}/{total}")
        print()


if __name__ == "__main__":
    main()
