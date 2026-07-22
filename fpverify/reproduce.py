"""自行复核包（reproduce pack）：让用户不经过 fpverify 也能亲手验证结论。

设计动机：identify 的判定再严谨，对用户也是黑箱。复核包把参考指纹变成
"任何人可重跑的实验"——但**指纹是（模型 × 渠道 × 协议 × 档位）的条件分布**，
复核必须复现参考的采集协议，否则对不上不代表造假（实证见
experiments/frontier/PROTOCOL.md：GPT-5.6 sol 套卷第 9 题抛硬币 = tails 11/11，
冷启动单问同一模型 = heads）。

因此复核包按参考条目的渠道生成两种协议：

  cursor-harness 条目（套卷协议）——
    导出采集时的**套卷原文**（一次十题、JSON 作答）。复核 = N 个全新实例各答
    一整卷：A. Cursor 子代理（同渠道同协议，最严谨）；B. Codex CLI 循环
    `codex exec`（近似渠道）；C. official_api.py 把整卷发给 OpenAI 兼容端点
    （跨渠道，仅看方向）；D. 官网手点也必须**整卷粘贴**到全新对话。

  api 条目（冷协议）——
    与 fpverify enroll/audit 同协议：每题独立请求、全新对话。
    官方 API 脚本逐题冷问；官网手点每题新开一个对话。

通用铁则：**同一个对话里连问 N 次不算 N 个样本**——模型看得见自己之前的答案
会刻意换。样本 = 全新对话/全新实例。

这只是人肉复核的经验判读，不替代序贯统计检验；harness 参考与在线端点探针
属跨协议对比，只能看"最像谁"的方向，不做 PASS/FAIL 硬判定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import probes
from .fingerprint import Fingerprint
from .library import (
    HARNESS_PROTOCOL, PROBE_PROTOCOL, Library, LibraryEntry, entry_protocol,
)

DEFAULT_RUNS = 10
MIN_SHARE = 0.75   # 众数占比阈值：低于此不算"铁律"
MIN_N = 8          # 参考样本量下限

# ---------------------------------------------------------------- harness 套卷协议
# 逐字复刻 2026-07-21 首采提示词（experiments/frontier/PROTOCOL.md）。不得改动：
# 措辞、题序、JSON 模板都是分布的条件。
HARNESS_BATTERY = """You are participating in a statistical sampling task. Do NOT use any tools. Do NOT explore the codebase. Do NOT explain anything. Reply with ONLY one JSON object — no markdown fences, no extra text before or after.

Answer each question independently and spontaneously with the first value that comes to mind. Do not try to be consistent, do not avoid repeating values across questions, do not overthink.

1. rand_num_100 — Name a random number between 1 and 100.
2. rand_num_10 — Pick a random number between 1 and 10.
3. fav_num — What is your favorite number?
4. rand_letter — Pick a random letter of the English alphabet.
5. rand_color — Name a random color.
6. fav_color — What is your favorite color?
7. rand_animal — Name a random animal.
8. rand_city — Name a random city.
9. coin_flip — Flip a coin in your head: heads or tails?
10. rand_num_100_zh — 说一个1到100之间的随机数。

Reply with exactly this JSON shape (fill in your values):
{"rand_num_100": <int 1-100>, "rand_num_10": <int 1-10>, "fav_num": <int>, "rand_letter": "<single letter>", "rand_color": "<one word>", "fav_color": "<one word>", "rand_animal": "<one word>", "rand_city": "<one word>", "coin_flip": "<heads or tails>", "rand_num_100_zh": <int 1-100>}"""

# (JSON key, 题面, 指纹 cell, 解析类型)
HARNESS_ITEMS = [
    ("rand_num_100",    "Name a random number between 1 and 100.",   ("rand_num_100", "en"), "int"),
    ("rand_num_10",     "Pick a random number between 1 and 10.",    ("rand_num_10", "en"),  "int"),
    ("fav_num",         "What is your favorite number?",             ("fav_num", "en"),      "int"),
    ("rand_letter",     "Pick a random letter of the English alphabet.", ("rand_letter", "en"), "letter"),
    ("rand_color",      "Name a random color.",                      ("rand_color", "en"),   "word"),
    ("fav_color",       "What is your favorite color?",              ("fav_color", "en"),    "word"),
    ("rand_animal",     "Name a random animal.",                     ("rand_animal", "en"),  "word"),
    ("rand_city",       "Name a random city.",                       ("rand_city", "en"),    "word"),
    ("coin_flip",       "Flip a coin in your head: heads or tails?", ("coin_flip", "en"),    "word"),
    ("rand_num_100_zh", "说一个1到100之间的随机数。",                  ("rand_num_100", "zh"), "int"),
]


@dataclass(frozen=True)
class Invariant:
    cell: str        # "coin_flip::en"
    task: str
    lang: str
    parse: str       # int / letter / word
    system: str      # 单值系统提示（冷协议用；套卷协议为空）
    prompt: str      # 发送原文（套卷协议 = 套卷内该题的题面）
    n_templates: int
    expected: str    # 参考众数（归一化后的 token）
    share: float     # 众数占比
    n: int           # 参考样本量

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def top_invariants(fp: Fingerprint, k: int = 6,
                   min_share: float = MIN_SHARE, min_n: int = MIN_N) -> list[Invariant]:
    """从参考指纹里挑出人肉可验的铁律 cell，按众数占比降序（冷协议 api 包用）。"""
    rows: list[Invariant] = []
    for cell, counter in fp.cells.items():
        task, lang = cell
        t = probes.TASK_BY_ID.get(task)
        if t is None or lang not in t.templates:
            continue
        n = sum(counter.values())
        if n < min_n:
            continue
        tok, cnt = max(counter.items(), key=lambda kv: kv[1])
        share = cnt / n
        if share < min_share:
            continue
        rows.append(Invariant(
            cell=f"{task}::{lang}", task=task, lang=lang, parse=t.parse,
            system=probes.SYSTEM_PROMPTS.get(lang, probes.SYSTEM_PROMPTS["en"]),
            prompt=t.templates[lang][0], n_templates=len(t.templates[lang]),
            expected=str(tok), share=share, n=n))
    rows.sort(key=lambda r: (-r.share, -r.n, r.cell))
    return rows[:k]


def battery_invariants(fp: Fingerprint) -> list[Invariant]:
    """套卷协议：按套卷题序返回全部 10 题的参考众数（不做占比过滤，占比即判读权重）。"""
    rows: list[Invariant] = []
    for key, question, cell, parse in HARNESS_ITEMS:
        counter = fp.cells.get(cell)
        if not counter:
            continue
        n = sum(counter.values())
        tok, cnt = max(counter.items(), key=lambda kv: kv[1])
        rows.append(Invariant(
            cell=f"{cell[0]}::{cell[1]}", task=cell[0], lang=cell[1], parse=parse,
            system="", prompt=question, n_templates=1,
            expected=str(tok), share=cnt / n, n=n))
    return rows


# ---------------------------------------------------------------- 套卷协议文本

def harness_cursor_prompt_text(entry: LibraryEntry, runs: int) -> str:
    indented = "\n".join("  " + ln for ln in HARNESS_BATTERY.splitlines())
    return f"""复核实验：{entry.model}（渠道 cursor-harness，套卷协议）
把这一整段粘贴给 Cursor（或任何支持子代理的 agent IDE）。

---

帮我跑一个模型行为复核实验，严格执行，不要省略步骤：

1. 启动 {runs} 个全新的 subagent，模型选「{entry.model}」（如当前档位名不完全一致，
   选最接近的一档，并在结果里注明实际用了哪个）。各 subagent 彼此独立、
   不共享上下文、不复用。
2. 每个 subagent 只收到下面 <battery> 标签内的原文，一字不改（含 JSON 模板），
   不要附加任何解释或系统说明：

<battery>
{indented}
</battery>

3. 原样收集每个 subagent 回的 JSON。不要纠正、不要补齐、不要替它们改答案。
4. 输出一张表：行 = 题目 key，列 = 各 subagent 的答案；最后一列统计众数及次数
   （例如 tails 9/{runs}）。中英同义答案归并统计（heads=正面，blue=蓝色）。
   表后不要加任何评价。

---

跑完后对照本包 README.md 的参考表。判读：占比 ≥90% 的强题，{runs} 次里 ≥7 次对上
视为相符；个别题随官方档位轮换漂移属正常；**多数强题对上=相符，多题同时大幅偏离
=与参考不是同一模型**（同渠道条件下）。
"""


def harness_official_api_py_text(entry: LibraryEntry, invs: list[Invariant], runs: int) -> str:
    expected = {inv.task if inv.lang == "en" else f"{inv.task}_{inv.lang}":
                {"expected": inv.expected, "share": round(inv.share, 3), "n": inv.n}
                for inv in invs}
    head = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""套卷复核脚本（独立运行，仅标准库，不依赖 fpverify）。

参考条目: {entry.model}（渠道 {entry.channel}，套卷协议，入册 {entry.enrolled_at}，n={entry.samples_per_cell}/卷）
用法:
  python official_api.py --base-url https://api.openai.com/v1 --model <模型名> --key sk-... [--n {runs}]

每次请求 = 一整卷（全新对话），共 n 卷；逐题与参考众数并排打印。
注意: 参考采自 Cursor agent 渠道，直连 API 没有那层系统提示——答案可能整体偏移，
本脚本结果**仅看方向**（多数题像不像），不构成硬判定。
"""
import argparse, collections, json, re, sys, urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BATTERY = __BATTERY__

EXPECTED = __EXPECTED__

ZH_MERGE = {{"正面": "heads", "反面": "tails", "蓝色": "blue", "蓝": "blue"}}


def norm(v):
    s = str(v).strip().lower()
    return ZH_MERGE.get(s, s)


def ask(base, key, model, timeout=90):
    body = {{"model": model, "temperature": 1.0, "max_tokens": 300,
            "messages": [{{"role": "user", "content": BATTERY}}]}}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={{"Authorization": "Bearer " + key, "Content-Type": "application/json"}})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    text = (d["choices"][0]["message"]["content"] or "").strip()
    m = re.search(r"\\{{.*\\}}", text, re.S)
    return json.loads(m.group(0)) if m else {{}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--n", type=int, default={runs})
    a = ap.parse_args()

    print(f"套卷复核 {{a.model}} @ {{a.base_url}}  （{{a.n}} 卷，每卷全新对话）")
    print(f"参考: {entry.model} / 渠道 {entry.channel}（跨渠道对比，仅看方向）")
    print()
    tallies = {{k: collections.Counter() for k in EXPECTED}}
    for i in range(a.n):
        try:
            ans = ask(a.base_url, a.key, a.model)
        except Exception as e:
            print(f"  卷 {{i + 1}}: <error:{{type(e).__name__}}>")
            continue
        for k in EXPECTED:
            if k in ans:
                tallies[k][norm(ans[k])] += 1
    print()
    hits = total = 0
    for k, ref in EXPECTED.items():
        c = tallies[k]
        top, cnt = c.most_common(1)[0] if c else ("<无有效答案>", 0)
        ok = top == ref["expected"]
        strong = ref["share"] >= 0.9
        hits += ok and strong
        total += strong
        mark = "OK" if ok else "XX"
        print(f'[{{mark}}] {{k:<18}} 观测众数: {{top}} {{cnt}}/{{a.n}}   '
              f'参考: {{ref["expected"]}} {{ref["share"]:.0%}} (n={{ref["n"]}})')
    print()
    print(f"强题（参考占比≥90%）对上 {{hits}}/{{total}}。判读: 多数强题对上=方向相符;")
    print("多题同时偏离=大概率不是同一模型。跨渠道结果不构成硬判定。")


if __name__ == "__main__":
    main()
'''
    return (head
            .replace("__BATTERY__", json.dumps(HARNESS_BATTERY, ensure_ascii=False))
            .replace("__EXPECTED__", json.dumps(expected, ensure_ascii=False, indent=2)))


def harness_pack_readme_text(entry: LibraryEntry, invs: list[Invariant], runs: int) -> str:
    rows = "\n".join(
        f"| {i}. {inv.prompt} | {inv.expected} | {inv.share:.0%} (n={inv.n})"
        f"{' ★' if inv.share >= 0.9 else ''} |"
        for i, inv in enumerate(invs, 1))
    return f"""# 复核包：{entry.model}（套卷协议）

参考来源：公开指纹库 `refs/`（渠道 {entry.channel}，入册 {entry.enrolled_at}，
n={entry.samples_per_cell} 卷）。采集协议：**一个全新实例一次性答完整卷十题
（JSON 作答）**，逐字提示词见 `battery.txt`。

指纹是（模型 × 渠道 × 协议 × 档位）的条件分布——复核必须复现同一协议。

## 参考表（套卷条件下的众数；★ = 占比 ≥90% 的强题）

| 题目 | 参考众数 | 占比 |
|---|---|---|
{rows}

## 动手前先读三条

1. **不要单题冷问。** 新开对话只问一句"抛硬币"得到的是另一个条件分布，
   对不上不代表造假（实测：GPT-5.6 sol 套卷=tails，冷问=heads）。手测必须把
   `battery.txt` **整卷**粘贴进全新对话。
2. **一个实例只算一卷。** 跑 {runs} 卷 = {runs} 个互不相干的全新对话/子代理；
   同一对话里重复跑，模型会看着旧答案刻意换。
3. **容忍个别题漂移。** 官方会轮换内部档位（如 sol-medium 下架换 sol-max），
   个别题的众数会跟着变；判读看多数强题，不看单题。

## 方法（四选一）

- **A. Cursor / 支持子代理的 agent**（同渠道同协议，最严谨）：
  把 `cursor_prompt.md` 整段粘贴过去
- **B. Codex CLI**（近似渠道）：`./codex_loop.sh {runs}` 或 PowerShell 跑
  `codex_loop.ps1`；cursor CLI 把命令换成 `cursor-agent -p` 同理
- **C. OpenAI 兼容端点**（跨渠道，仅看方向）：
  `python official_api.py --base-url <端点>/v1 --model <名> --key sk-... `
  ——把它指向官方 API 或你的中转站都行，两边各跑一遍互相对照更有说服力
- **D. 官网手点**（跨渠道，仅看方向）：把 `battery.txt` 整卷粘贴到全新对话，
  重复 {runs} 个新对话

## 判读

占比 ≥90% 的强题（★），{runs} 卷里 ≥7 卷对上视为相符；个别题漂移属正常；
多数强题同时大幅偏离 → 与参考不是同一模型（同渠道条件下）。
跨渠道（C/D）只看方向，不做硬判定。

统计口径：中英同义答案归并（heads=正面，blue=蓝色）。
"""


# ---------------------------------------------------------------- 冷协议（api 条目）文本

def battery_text(invs: list[Invariant]) -> str:
    """冷协议：逐题清单（每题独立发给全新对话）。"""
    lines = [
        "You answer with a single value only: just one word or one number, "
        "no explanation, no punctuation, no reasoning.",
        "Answer each question on its own line, numbered, in order.",
        "",
    ]
    for i, inv in enumerate(invs, 1):
        lines.append(f"{i}. {inv.prompt}")
    return "\n".join(lines) + "\n"


def cursor_prompt_text(entry: LibraryEntry, invs: list[Invariant], runs: int) -> str:
    battery = battery_text(invs).rstrip("\n")
    indented = "\n".join("  " + ln for ln in battery.splitlines())
    return f"""复核实验：{entry.model}（参考渠道 {entry.channel}，冷协议）
把这一整段粘贴给 Cursor（或任何支持子代理的 agent IDE / Codex）。

---

帮我跑一个模型行为复核实验，严格执行，不要省略步骤：

1. 启动 {runs} 个全新的 subagent，模型选「{entry.model}」（如无此名就用与它对应的可用型号，
   并在结果里注明实际用了哪个）。各 subagent 彼此独立、不共享上下文、不复用。
2. 每个 subagent 只收到下面 <battery> 标签内的原文，不要附加任何解释、系统说明或提示：

<battery>
{indented}
</battery>

3. 原样收集每个 subagent 的回答。不要纠正、不要补齐、不要去重、不要替它们改答案。
4. 输出一张表：行 = 题号，列 = 各 subagent 的答案；最后一列统计最常见答案及其次数
   （例如 tails 9/{runs}）。表后不要加任何评价。

---

跑完后对照本包 README.md 的参考表。判读经验：参考占比 ≥90% 的题，{runs} 次里 ≥7 次
对上视为相符；多题同时大幅偏离 → 与参考不是同一模型（同渠道条件下）。
注意：本参考采自 api 渠道，agent 子代理带有自己的系统提示，结果可能整体偏移，仅看方向。
"""


def codex_ps1_text() -> str:
    # 纯 ASCII，避免 PowerShell 5.1 的编码坑
    return """# Reproduce pack runner for Codex CLI (fresh session per run).
# Usage: .\\codex_loop.ps1 [-N 10] [-Model gpt-5.6]
# For cursor CLI, replace 'codex exec' with 'cursor-agent -p'.
param([int]$N = 10, [string]$Model = "")
$battery = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "battery.txt")
for ($i = 1; $i -le $N; $i++) {
  Write-Host "--- run $i / $N ---"
  if ($Model) { codex exec -m $Model $battery } else { codex exec $battery }
}
Write-Host "Done. Tally the answers yourself against README.md."
"""


def codex_sh_text() -> str:
    return """#!/usr/bin/env sh
# Reproduce pack runner for Codex CLI (fresh session per run).
# Usage: ./codex_loop.sh [N] [MODEL]
# For cursor CLI, replace 'codex exec' with 'cursor-agent -p'.
N="${1:-10}"; MODEL="${2:-}"
DIR="$(cd "$(dirname "$0")" && pwd)"
i=1
while [ "$i" -le "$N" ]; do
  echo "--- run $i / $N ---"
  if [ -n "$MODEL" ]; then codex exec -m "$MODEL" "$(cat "$DIR/battery.txt")"
  else codex exec "$(cat "$DIR/battery.txt")"; fi
  i=$((i + 1))
done
echo "Done. Tally the answers yourself against README.md."
"""


def official_api_py_text(entry: LibraryEntry, invs: list[Invariant], runs: int) -> str:
    probes_json = json.dumps([inv.to_dict() for inv in invs], ensure_ascii=False, indent=2)
    return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冷协议复核脚本（独立运行，仅标准库，不依赖 fpverify）。

参考条目: {entry.model}（渠道 {entry.channel}，入册 {entry.enrolled_at}，n={entry.samples_per_cell}/cell）
用法:
  python official_api.py --base-url https://api.openai.com/v1 --model <官方模型名> --key sk-... [--n {runs}]

每题独立请求 n 次（每次都是全新对话），与参考众数并排打印。
与 fpverify enroll/audit 同协议（单题冷问）。
"""
import argparse, collections, json, re, sys, urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROBES = {probes_json}


def norm(parse, text):
    s = (text or "").strip().lower()
    if parse == "int":
        m = re.search(r"-?\\d+", s)
        return m.group(0) if m else s[:20]
    if parse == "letter":
        m = re.search(r"[a-z]", s)
        return m.group(0) if m else s[:20]
    m = re.search(r"[\\w\\u4e00-\\u9fff]+", s)
    return m.group(0) if m else s[:20]


def ask(base, key, model, system, user, timeout=60):
    body = {{"model": model, "temperature": 1.0, "max_tokens": 16,
            "messages": [{{"role": "system", "content": system}},
                         {{"role": "user", "content": user}}]}}
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={{"Authorization": "Bearer " + key, "Content-Type": "application/json"}})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return (d["choices"][0]["message"]["content"] or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--n", type=int, default={runs})
    a = ap.parse_args()

    print(f"复核 {{a.model}} @ {{a.base_url}}  （每题 {{a.n}} 次独立请求）")
    print(f"参考: {entry.model} / 渠道 {entry.channel}")
    print()
    hits = total = 0
    for p in PROBES:
        c = collections.Counter()
        for _ in range(a.n):
            try:
                c[norm(p["parse"], ask(a.base_url, a.key, a.model, p["system"], p["prompt"]))] += 1
            except Exception as e:
                c[f"<error:{{type(e).__name__}}>"] += 1
        top, cnt = c.most_common(1)[0]
        ok = top == p["expected"]
        hits += ok
        total += 1
        ref = f'{{p["expected"]}} {{p["share"]:.0%}} (n={{p["n"]}})'
        print(f'[{{ "OK" if ok else "XX" }}] {{p["cell"]:<22}} 观测众数: {{top}} {{cnt}}/{{a.n}}   参考: {{ref}}')
        print(f'     题目: {{p["prompt"]}}')
    print()
    print(f"众数对上 {{hits}}/{{total}} 题。经验判读: 大多数题对上=相符; 多题同时偏离=不是同一模型。")
    print("(这是人肉复核, 不替代 fpverify identify 的序贯统计检验。)")


if __name__ == "__main__":
    main()
'''


def pack_readme_text(entry: LibraryEntry, invs: list[Invariant], runs: int) -> str:
    rows = "\n".join(
        f"| {i}. {inv.prompt} | {inv.expected} | {inv.share:.0%} (n={inv.n}) |"
        for i, inv in enumerate(invs, 1))
    return f"""# 复核包：{entry.model}（冷协议）

参考来源：公开指纹库 `refs/`（渠道 {entry.channel}，入册 {entry.enrolled_at}，n={entry.samples_per_cell}/cell）。
采集协议与 fpverify enroll 相同：**每题独立请求、全新对话**。
这里的每一条都可以**不经过 fpverify** 亲手验证。

## 参考表（铁律：该模型众数占比最高的题）

| 题目（原文发送） | 参考众数 | 占比 |
|---|---|---|
{rows}

审计时措辞会随机改写（每题另有多条等价模板），人肉复核用上表固定措辞即可。

## 方法（四选一）

**关键规则：同一个对话里连问 {runs} 次不算数**——模型看得见自己之前的答案会刻意换。
每个样本必须来自全新对话 / 全新实例。

- **A. Cursor / 支持子代理的 agent**（跨渠道，仅看方向）：把 `cursor_prompt.md` 整段粘贴过去
- **B. Codex CLI**（跨渠道，仅看方向）：`./codex_loop.sh {runs}` 或 `codex_loop.ps1`
- **C. 官方 API key**（同渠道同协议，最严谨）：
  `python official_api.py --base-url https://api.openai.com/v1 --model <官方名> --key sk-...`
- **D. 官网手点**（近似渠道）：每题**新开一个对话**问一次，重复 {runs} 个新对话

## 判读

经验法则：参考占比 ≥90% 的题，{runs} 次里 ≥7 次对上视为相符；多题同时大幅偏离
→ 与参考不是同一模型（同渠道条件下）。

这是人肉复核，用来让你**不必信任 fpverify**；严格结论仍以序贯统计检验为准。
"""


# ---------------------------------------------------------------- 组装

def build_pack_texts(entry: LibraryEntry, fp: Fingerprint,
                     k: int = 6, runs: int = DEFAULT_RUNS) -> tuple[list[Invariant], dict[str, str]]:
    """返回 (参考条目列表, 文件名->内容)。按条目协议选择包形态；无可用 cell 时抛 ValueError。"""
    protocol = entry_protocol(entry)
    if protocol == HARNESS_PROTOCOL:
        invs = battery_invariants(fp)
        if not invs:
            raise ValueError(f"参考『{entry.model}』没有套卷对应的 cell，无法生成复核包。")
        files = {
            "README.md": harness_pack_readme_text(entry, invs, runs),
            "battery.txt": HARNESS_BATTERY + "\n",
            "cursor_prompt.md": harness_cursor_prompt_text(entry, runs),
            "codex_loop.ps1": codex_ps1_text(),
            "codex_loop.sh": codex_sh_text(),
            "official_api.py": harness_official_api_py_text(entry, invs, runs),
        }
    else:
        invs = top_invariants(fp, k=k)
        if not invs:
            raise ValueError(
                f"参考『{entry.model}』没有众数占比 ≥{MIN_SHARE:.0%} 且 n≥{MIN_N} 的 cell，"
                f"无法生成人肉复核包（样本量太小或该模型行为太散）。")
        files = {
            "README.md": pack_readme_text(entry, invs, runs),
            "battery.txt": battery_text(invs),
            "cursor_prompt.md": cursor_prompt_text(entry, invs, runs),
            "codex_loop.ps1": codex_ps1_text(),
            "codex_loop.sh": codex_sh_text(),
            "official_api.py": official_api_py_text(entry, invs, runs),
        }
    files["expected.json"] = json.dumps({
        "entry": entry.id, "model": entry.model, "channel": entry.channel,
        "protocol": protocol,
        "enrolled_at": entry.enrolled_at, "samples_per_cell": entry.samples_per_cell,
        "source": entry.source,
        "invariants": [inv.to_dict() for inv in invs],
    }, ensure_ascii=False, indent=2) + "\n"
    return invs, files


def write_pack(library: Library, entry: LibraryEntry, out_dir: str | Path,
               k: int = 6, runs: int = DEFAULT_RUNS) -> tuple[list[Invariant], Path]:
    invs, files = build_pack_texts(entry, library.fingerprint(entry), k=k, runs=runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    return invs, out
