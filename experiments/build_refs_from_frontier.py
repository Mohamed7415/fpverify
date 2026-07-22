# -*- coding: utf-8 -*-
"""把前沿实测原始数据（experiments/frontier/batch_*.json）构建成公共指纹库 refs/。

产出：
  refs/harness/<model>.json   9 个模型的参考指纹（Fingerprint 序列化格式）
  refs/manifest.json          库清单（含渠道/日期/来源/样本量元数据）

渠道标注为 cursor-harness：采样发生在 Cursor agent harness 内（带系统提示、
温度不受控），只适用于同渠道比对与 identify 演示；裸 API 审计请用 api 频道
的参考（等待社区贡献，见 refs/README.md）。

用法：  python -X utf8 experiments/build_refs_from_frontier.py
幂等：重跑覆盖生成文件。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fpverify import probes
from fpverify.fingerprint import Fingerprint
from fpverify.normalize import classify_validity, normalize

FRONTIER = ROOT / "experiments" / "frontier"
REFS = ROOT / "refs"

MODELS = {
    "fable5":       ("Claude Fable 5", "claude"),
    "fable5-think": ("Claude Fable 5 thinking", "claude"),
    "sonnet5-think": ("Claude Sonnet 5 thinking", "claude"),
    "opus48-think": ("Claude Opus 4.8 thinking", "claude"),
    "gpt56-sol":    ("GPT-5.6 sol", "openai"),
    "gpt56-terra":  ("GPT-5.6 terra", "openai"),
    "glm52":        ("GLM-5.2", "zhipu"),
    "composer25":   ("Composer 2.5", "cursor"),
    "grok45":       ("Grok 4.5", "xai"),
}

SOURCE = ("Cursor subagent 采样，9 模型 × 11 个全新独立实例，2026-07；"
          "原始数据 experiments/frontier/batch_*.json，本脚本一键重建")


def answer_cell(key: str) -> tuple[str, str]:
    """原始数据键 -> 探针 cell。rand_num_100_zh 是唯一的中文题。"""
    if key.endswith("_zh"):
        return (key[:-3], "zh")
    return (key, "en")


def main() -> int:
    records = []
    for f in sorted(FRONTIER.glob("batch_*.json")):
        records.extend(json.loads(f.read_text(encoding="utf-8")))
    if not records:
        print(f"未找到原始数据: {FRONTIER}/batch_*.json")
        return 1

    per_model = defaultdict(list)
    for r in records:
        per_model[r["model"]].append(r["answers"])

    (REFS / "harness").mkdir(parents=True, exist_ok=True)
    entries = []
    for mid, (display, family) in MODELS.items():
        rows = per_model.get(mid, [])
        if not rows:
            print(f"警告：{mid} 无数据，跳过")
            continue
        fp = Fingerprint(model=display,
                         params={"temperature": "uncontrolled (agent harness)",
                                 "max_tokens": None},
                         note=f"channel=cursor-harness; n={len(rows)}/cell; {SOURCE}")
        for answers in rows:
            for key, val in answers.items():
                task, lang = answer_cell(key)
                if task not in probes.TASK_BY_ID:
                    continue
                ptype = probes.parse_type(task)
                text = str(val)
                fp.add((task, lang), normalize(ptype, text),
                       classify_validity(ptype, text))
        out = REFS / "harness" / f"{mid}.json"
        fp.save(out)
        entries.append({
            "id": mid, "model": display, "family": family,
            "channel": "cursor-harness", "protocol": "harness-battery",
            "file": f"harness/{mid}.json",
            "enrolled_at": "2026-07", "samples_per_cell": len(rows),
            "source": SOURCE,
            "note": "样本量小（n=11/cell）；harness 套卷协议采集，与在线单题冷探针跨协议，仅限同协议比对",
        })
        print(f"  {out.name}: {fp.total_samples()} 样本, {len(fp.cells)} cell")

    manifest = {
        "version": 1,
        "updated_at": "2026-07-21",
        "channels": {
            "api": "裸 OpenAI 兼容 API 直连入册，冷单题协议（可硬判定审计中转站）——征集社区贡献中",
            "cursor-harness": "Cursor agent harness 内套卷协议采样；与在线单题探针跨协议，只做相对排名 / identify 演示",
            "simulation": "仿真分布（仅作格式示例/测试）",
        },
        "entries": entries,
    }
    (REFS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"清单: {REFS / 'manifest.json'}（{len(entries)} 个条目）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
