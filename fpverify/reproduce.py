"""自行复核包（reproduce pack）：让用户不经过 fpverify 也能亲手验证结论。

设计动机：identify 的判定再严谨，对用户也是黑箱。这里从参考指纹中挑出
**众数占比最高的 cell**（"铁律"——如 GPT-5.6 sol 的 coin_flip::en = tails 11/11），
生成一个复核包，用户用四种方式之一亲手重跑：

  A. Cursor 等支持子代理的 agent IDE —— 粘贴 cursor_prompt.md，扇出 N 个全新
     subagent，每个一次性回答题目。与 cursor-harness 渠道的参考**同渠道**，最严谨。
  B. Codex CLI —— codex_loop 脚本循环 `codex exec`，每次都是全新会话。
  C. 官方 API key —— official_api.py（纯标准库、零依赖、不 import fpverify），
     直连官方端点采样并与参考表并排打印。
  D. 官网手点 —— 每题新开一个对话问一次，重复 N 个新对话。

方法学要点（必须告诉用户）：**同一个对话里连问十次不算独立样本**——模型看得见
自己之前的答案会刻意换。每个样本必须来自全新对话/全新实例。

渠道匹配：cursor-harness 条目用 A/B 复核是同渠道对比；api 条目用 C。跨渠道
（如用官方 API 复核 harness 参考）答案分布可能有系统性偏移，包内 README 会标注。

这只是人肉复核的经验判读，不替代 identify 的序贯统计检验。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import probes
from .fingerprint import Fingerprint
from .library import Library, LibraryEntry

DEFAULT_RUNS = 10
MIN_SHARE = 0.75   # 众数占比阈值：低于此不算"铁律"
MIN_N = 8          # 参考样本量下限


@dataclass(frozen=True)
class Invariant:
    cell: str        # "coin_flip::en"
    task: str
    lang: str
    parse: str       # int / letter / word
    system: str      # 单值系统提示（该语言）
    prompt: str      # 该 cell 的第一条模板（人肉复核用固定措辞即可）
    n_templates: int
    expected: str    # 参考众数（归一化后的 token）
    share: float     # 众数占比
    n: int           # 参考样本量

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def top_invariants(fp: Fingerprint, k: int = 6,
                   min_share: float = MIN_SHARE, min_n: int = MIN_N) -> list[Invariant]:
    """从参考指纹里挑出人肉可验的铁律 cell，按众数占比降序。"""
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


# ---------------------------------------------------------------- 文本生成

def battery_text(invs: list[Invariant]) -> str:
    """一次性发给全新实例的题目清单（含单值指令）。"""
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
    return f"""复核实验：{entry.model}（参考渠道 {entry.channel}）
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
Write-Host "Done. Tally the last-line answers yourself against README.md."
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
"""官方渠道复核脚本（独立运行，仅标准库，不依赖 fpverify）。

参考条目: {entry.model}（渠道 {entry.channel}，入册 {entry.enrolled_at}，n={entry.samples_per_cell}/cell）
用法:
  python official_api.py --base-url https://api.openai.com/v1 --model <官方模型名> --key sk-... [--n {runs}]

每题独立请求 n 次（每次都是全新对话），与参考众数并排打印。
注意: 统计口径为简化版归一化，仅供人肉判读；跨渠道对比可能有系统性偏移。
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
    harness = entry.channel == "cursor-harness"
    ch_note = (
        "本参考在 **cursor-harness** 渠道采集：方法 A/B 是同渠道对比（最严谨）；"
        "方法 C/D 走官方渠道，答案分布可能有系统性偏移，仅看方向。"
        if harness else
        "本参考在 **api** 渠道采集：方法 C 是同渠道对比（最严谨）；"
        "方法 A/B 走 agent harness，答案分布可能有系统性偏移，仅看方向。")
    return f"""# 复核包：{entry.model}

参考来源：公开指纹库 `refs/`（渠道 {entry.channel}，入册 {entry.enrolled_at}，n={entry.samples_per_cell}/cell）。
这里的每一条都可以**不经过 fpverify** 亲手验证。

## 参考表（铁律：该模型众数占比最高的题）

| 题目（原文发送） | 参考众数 | 占比 |
|---|---|---|
{rows}

审计时措辞会随机改写（每题另有多条等价模板），人肉复核用上表固定措辞即可。

## 方法（四选一）

**关键规则：同一个对话里连问 {runs} 次不算数**——模型看得见自己之前的答案会刻意换。
每个样本必须来自全新对话 / 全新实例。

- **A. Cursor / 支持子代理的 agent**（把 `cursor_prompt.md` 整段粘贴过去，
  子代理模型选与声称一致的那个）
- **B. Codex CLI**：`./codex_loop.sh {runs}`（或 PowerShell 跑 `codex_loop.ps1`），
  `codex exec` 每次都是全新会话；cursor CLI 把命令换成 `cursor-agent -p` 同理
- **C. 官方 API key**：`python official_api.py --base-url https://api.openai.com/v1 --model <官方名> --key sk-...`
- **D. 官网手点**：每题**新开一个对话**问一次，重复 {runs} 个新对话

{ch_note}

## 判读

经验法则：参考占比 ≥90% 的题，{runs} 次里 ≥7 次对上视为相符；多题同时大幅偏离
→ 与参考不是同一模型（同渠道条件下）。

这是人肉复核，用来让你**不必信任 fpverify**；严格结论仍以 `identify` 的
序贯统计检验为准（误判率有数学上界）。
"""


# ---------------------------------------------------------------- 组装

def build_pack_texts(entry: LibraryEntry, fp: Fingerprint,
                     k: int = 6, runs: int = DEFAULT_RUNS) -> tuple[list[Invariant], dict[str, str]]:
    """返回 (铁律列表, 文件名->内容)。铁律不足时抛 ValueError。"""
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
        "expected.json": json.dumps({
            "entry": entry.id, "model": entry.model, "channel": entry.channel,
            "enrolled_at": entry.enrolled_at, "samples_per_cell": entry.samples_per_cell,
            "source": entry.source,
            "invariants": [inv.to_dict() for inv in invs],
        }, ensure_ascii=False, indent=2) + "\n",
    }
    return invs, files


def write_pack(library: Library, entry: LibraryEntry, out_dir: str | Path,
               k: int = 6, runs: int = DEFAULT_RUNS) -> tuple[list[Invariant], Path]:
    invs, files = build_pack_texts(entry, library.fingerprint(entry), k=k, runs=runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    return invs, out
