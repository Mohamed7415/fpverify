"""前沿模型行为指纹实测分析（2026-07-21，Cursor subagent 采样器）。

读入 experiments/frontier/batch_*.json（9 模型 × 11 轮 × 10 题，每轮每模型
一个全新上下文的 subagent 实例），产出：

- experiments/out/frontier_results.json      全部统计数字
- experiments/out/fig_frontier_matrix.png    9×9 聚合 JSD 热力图（英文短名标签）
- experiments/out/fig_frontier_entropy.png   模型×题 熵热力图（含均匀理想参照行）

并在控制台打印中文摘要。固定随机种子、固定模型/题目顺序，可复跑、输出确定。

用法（在项目根目录）：py -3.13 -X utf8 experiments/analyze_frontier.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt
import numpy as np

# 让脚本无论从哪里启动都能 import fpverify（项目根目录 = 本文件上级目录）
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpverify.distance import entropy_bits, jsd_bits  # noqa: E402

# ---------------------------------------------------------------- 常量

FRONTIER_DIR = ROOT / "experiments" / "frontier"
OUT_DIR = ROOT / "experiments" / "out"

# 模型顺序按家族分组（Claude 系 / GPT 系 / 其它），矩阵里家族聚类一眼可见
MODELS = [
    "fable5", "fable5-think", "sonnet5-think", "opus48-think",
    "gpt56-sol", "gpt56-terra",
    "glm52", "composer25", "grok45",
]
FAMILY = {
    "fable5": "claude", "fable5-think": "claude",
    "sonnet5-think": "claude", "opus48-think": "claude",
    "gpt56-sol": "gpt", "gpt56-terra": "gpt",
    "glm52": "glm", "composer25": "composer", "grok45": "grok",
}

QUESTIONS = [
    "rand_num_100", "rand_num_10", "fav_num", "rand_letter",
    "rand_color", "fav_color", "rand_animal", "rand_city",
    "coin_flip", "rand_num_100_zh",
]
INT_QUESTIONS = {"rand_num_100", "rand_num_10", "fav_num", "rand_num_100_zh"}
UPPER_QUESTIONS = {"rand_letter"}

# 有限支撑题目的"理想均匀随机"熵（bit）；开放式题（最爱数/颜色/动物/城市）无定义
IDEAL_ENTROPY = {
    "rand_num_100": math.log2(100),
    "rand_num_100_zh": math.log2(100),
    "rand_num_10": math.log2(10),
    "rand_letter": math.log2(26),
    "coin_flip": 1.0,
}

MAGIC_NUMBERS = [73, 47, 37, 42]

# 重点对照组：(模型A, 模型B, 中文说明)
KEY_PAIRS = [
    ("fable5", "fable5-think", "同底座：默认 vs thinking-max"),
    ("gpt56-sol", "gpt56-terra", "GPT-5.6 兄弟变体"),
    ("fable5", "sonnet5-think", "Claude 家族内"),
    ("fable5", "opus48-think", "Claude 家族内"),
    ("sonnet5-think", "opus48-think", "Claude 家族内"),
    ("fable5", "gpt56-sol", "跨家族：Claude vs GPT"),
    ("glm52", "grok45", "跨家族：GLM vs Grok"),
    ("fable5", "glm52", "跨家族：Claude vs GLM"),
    ("gpt56-sol", "grok45", "跨家族：GPT vs Grok"),
    ("fable5", "composer25", "跨家族：Claude vs Composer"),
]

SEED = 20260721
N_SHUFFLES = 200  # 自比噪声带的重复划分次数

# ---------------------------------------------------------------- 数据读入与清洗


def clean_answer(question: str, value):
    """答案归一化：数值转 int；字母大写；文本类小写并压掉多余空白。"""
    if question in INT_QUESTIONS:
        return int(str(value).strip())
    s = str(value).strip()
    if question in UPPER_QUESTIONS:
        return s.upper()
    return " ".join(s.lower().split())


def load_records():
    """读入全部 batch，返回 (清洗后记录列表, 异常清单)。异常记录跳过并如实上报。"""
    records, issues = [], []
    files = sorted(FRONTIER_DIR.glob("batch_*.json"))
    if not files:
        raise SystemExit(f"未找到数据文件：{FRONTIER_DIR}/batch_*.json")
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            issues.append(f"{f.name}: 文件解析失败（{e}）")
            continue
        for i, rec in enumerate(data):
            try:
                model = rec["model"]
                answers = rec["answers"]
                cleaned = {q: clean_answer(q, answers[q]) for q in QUESTIONS}
                records.append({"model": model, "round": rec.get("round"), "answers": cleaned})
            except (KeyError, ValueError, TypeError) as e:
                issues.append(f"{f.name}[{i}]: 记录异常，已跳过（{type(e).__name__}: {e}）")
    return records, issues


def counts_from_records(recs) -> dict[str, Counter]:
    """一组记录 -> {题目: Counter(答案)}。"""
    counts = {q: Counter() for q in QUESTIONS}
    for r in recs:
        for q in QUESTIONS:
            counts[q][r["answers"][q]] += 1
    return counts


# ---------------------------------------------------------------- 统计工具


def aggregate_jsd(ca: dict[str, Counter], cb: dict[str, Counter]) -> float:
    """两模型间的聚合距离：10 个题分别算 jsd_bits 后取平均。"""
    vals = [jsd_bits(ca[q], cb[q]) for q in QUESTIONS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals)


def quantile(sorted_vals: list[float], p: float) -> float:
    """线性插值分位数（输入需已排序）。"""
    n = len(sorted_vals)
    pos = p * (n - 1)
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def self_noise_band(recs_by_model: dict[str, list], rng: random.Random):
    """同源噪声基线：每模型 11 样本随机对半分（5/6），组间聚合 JSD 的分布。"""
    per_model, pooled = {}, []
    for m in MODELS:
        recs = recs_by_model[m]
        vals = []
        for _ in range(N_SHUFFLES):
            idx = list(range(len(recs)))
            rng.shuffle(idx)
            half = len(idx) // 2
            g1 = [recs[i] for i in idx[:half]]
            g2 = [recs[i] for i in idx[half:]]
            vals.append(aggregate_jsd(counts_from_records(g1), counts_from_records(g2)))
        vals.sort()
        per_model[m] = {
            "mean": sum(vals) / len(vals),
            "p95": quantile(vals, 0.95),
            "min": vals[0],
            "max": vals[-1],
        }
        pooled.extend(vals)
    pooled.sort()
    summary = {
        "n_shuffles_per_model": N_SHUFFLES,
        "half_sizes": "5/6",
        "pooled_mean": sum(pooled) / len(pooled),
        "pooled_p95": quantile(pooled, 0.95),
        "per_model": per_model,
    }
    return summary


def round_floats(obj, nd: int = 4):
    """递归四舍五入所有浮点数，保证 JSON 输出稳定可读。"""
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, nd) for v in obj]
    return obj


def counter_to_json(c: Counter) -> dict:
    """Counter -> 按频次降序的 {str: int}，键转字符串以兼容 JSON。"""
    return {str(k): v for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))}


# ---------------------------------------------------------------- 绘图


def plot_matrix(matrix: np.ndarray, path: Path):
    """9×9 聚合 JSD 热力图，格内标数值。"""
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    vmax = float(matrix.max())
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(MODELS)), MODELS, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(MODELS)), MODELS, fontsize=10)
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            v = matrix[i, j]
            color = "white" if v < 0.6 * vmax else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title(
        f"Frontier models: pairwise aggregate JSD over {len(QUESTIONS)} probes\n"
        "(Cursor subagent sampling, n=11/model, 2026-07-21)",
        fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="aggregate JSD (bits)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entropy(entropy: dict[str, dict[str, float]], path: Path):
    """模型×题 熵热力图，底部附一行"均匀理想熵"参照（开放式题留空）。"""
    rows = MODELS + ["UNIFORM-IDEAL"]
    data = np.full((len(rows), len(QUESTIONS)), np.nan)
    for i, m in enumerate(MODELS):
        for j, q in enumerate(QUESTIONS):
            data[i, j] = entropy[m][q]
    for j, q in enumerate(QUESTIONS):
        if q in IDEAL_ENTROPY:
            data[len(MODELS), j] = IDEAL_ENTROPY[q]

    masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(13, 7))
    vmax = math.log2(100)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(QUESTIONS)), QUESTIONS, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=10)
    for i in range(len(rows)):
        for j in range(len(QUESTIONS)):
            if np.isnan(data[i, j]):
                txt, color = "n/a", "#555555"
            else:
                txt = f"{data[i, j]:.2f}"
                color = "white" if data[i, j] < 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=color)
    ax.axhline(len(MODELS) - 0.5, color="white", linewidth=2)
    ax.set_title(
        "Per-model per-probe Shannon entropy (bits) vs uniform-ideal reference\n"
        "(n=11 samples/model; open-ended probes have no finite ideal)",
        fontsize=12,
    )
    fig.colorbar(im, ax=ax, label="entropy (bits)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records, issues = load_records()
    recs_by_model = {m: [r for r in records if r["model"] == m] for m in MODELS}
    unknown = sorted({r["model"] for r in records} - set(MODELS))
    if unknown:
        issues.append(f"出现未登记的模型短名，已忽略：{unknown}")

    # --- (a) 样本数
    sample_sizes = {m: len(recs_by_model[m]) for m in MODELS}

    # --- (b) 每模型每题：分布 / 众数占比 / 熵
    counts_by_model = {m: counts_from_records(recs_by_model[m]) for m in MODELS}
    per_model = {}
    entropy_table = {}
    for m in MODELS:
        qstats, ent = {}, {}
        for q in QUESTIONS:
            c = counts_by_model[m][q]
            n = sum(c.values())
            mode, mode_n = c.most_common(1)[0]
            h = max(0.0, entropy_bits(c))  # 消掉退化分布的 -0.0
            ent[q] = h
            qstats[q] = {
                "n": n,
                "counts": counter_to_json(c),
                "mode": str(mode),
                "mode_share": mode_n / n,
                "entropy_bits": h,
                "ideal_entropy_bits": IDEAL_ENTROPY.get(q),
            }
        entropy_table[m] = ent
        coin = counts_by_model[m]["coin_flip"]
        n_coin = sum(coin.values())
        per_model[m] = {
            "n_samples": sample_sizes[m],
            "questions": qstats,
            "signature": {
                "rand_num_100_mode": qstats["rand_num_100"]["mode"],
                "rand_num_100_mode_share": qstats["rand_num_100"]["mode_share"],
                "rand_color_mode": qstats["rand_color"]["mode"],
                "rand_animal_mode": qstats["rand_animal"]["mode"],
                "rand_city_mode": qstats["rand_city"]["mode"],
                "coin_heads_share": coin.get("heads", 0) / n_coin,
            },
        }

    # --- (c) 9×9 聚合 JSD 矩阵
    matrix = np.zeros((len(MODELS), len(MODELS)))
    for i, a in enumerate(MODELS):
        for j, b in enumerate(MODELS):
            if i < j:
                d = aggregate_jsd(counts_by_model[a], counts_by_model[b])
                matrix[i, j] = matrix[j, i] = d

    # 家族内 / 跨家族汇总
    within_claude, cross_family, all_pairs = [], [], []
    for i, a in enumerate(MODELS):
        for j in range(i + 1, len(MODELS)):
            b = MODELS[j]
            d = matrix[i, j]
            all_pairs.append((a, b, d))
            if FAMILY[a] == FAMILY[b] == "claude":
                within_claude.append(d)
            elif FAMILY[a] != FAMILY[b]:
                cross_family.append(d)
    family_summary = {
        "within_claude_mean": sum(within_claude) / len(within_claude),
        "within_claude_pairs": len(within_claude),
        "within_gpt_sol_terra": float(matrix[MODELS.index("gpt56-sol"), MODELS.index("gpt56-terra")]),
        "cross_family_mean": sum(cross_family) / len(cross_family),
        "cross_family_min": min(cross_family),
        "cross_family_max": max(cross_family),
        "cross_family_pairs": len(cross_family),
    }
    closest_cross = min((p for p in all_pairs if FAMILY[p[0]] != FAMILY[p[1]]), key=lambda p: p[2])

    # --- (d) 同源噪声基线
    rng = random.Random(SEED)
    noise = self_noise_band(recs_by_model, rng)

    # --- (e) 重点对照组
    key_pairs = []
    for a, b, label in KEY_PAIRS:
        d = float(matrix[MODELS.index(a), MODELS.index(b)])
        key_pairs.append({"pair": f"{a} vs {b}", "label": label, "jsd": d})

    # --- (f) "73 传闻"验证
    def share_of(m: str, q: str, v: int) -> float:
        c = counts_by_model[m][q]
        return c.get(v, 0) / sum(c.values())

    pooled_100 = Counter()
    for m in MODELS:
        pooled_100.update(counts_by_model[m]["rand_num_100"])
    n_pooled = sum(pooled_100.values())
    magic = {
        "fable5_73_share": share_of("fable5", "rand_num_100", 73),
        "fable5-think_73_share": share_of("fable5-think", "rand_num_100", 73),
        "pooled_n": n_pooled,
        "pooled_shares": {str(v): pooled_100.get(v, 0) / n_pooled for v in MAGIC_NUMBERS},
        "pooled_magic_total": sum(pooled_100.get(v, 0) for v in MAGIC_NUMBERS) / n_pooled,
        "pooled_top5": counter_to_json(Counter(dict(pooled_100.most_common(5)))),
    }

    # --- (g) 中英差异：同一模型 rand_num_100 vs rand_num_100_zh 的 JSD
    zh_en = {}
    for m in MODELS:
        c_en = counts_by_model[m]["rand_num_100"]
        c_zh = counts_by_model[m]["rand_num_100_zh"]
        zh_en[m] = {
            "jsd_en_vs_zh": jsd_bits(c_en, c_zh),
            "en_mode": str(c_en.most_common(1)[0][0]),
            "en_mode_share": c_en.most_common(1)[0][1] / sum(c_en.values()),
            "zh_mode": str(c_zh.most_common(1)[0][0]),
            "zh_mode_share": c_zh.most_common(1)[0][1] / sum(c_zh.values()),
        }

    # --- (h) 落盘 JSON + 两张图
    results = {
        "meta": {
            "date": "2026-07-21",
            "sampler": "Cursor subagent（每轮每模型全新上下文实例；同轮 10 题共享上下文）",
            "n_records_loaded": len(records),
            "n_rounds": len(sorted(FRONTIER_DIR.glob("batch_*.json"))),
            "sample_sizes": sample_sizes,
            "data_issues": issues if issues else ["无：全部记录字段完整、类型正常"],
            "seed": SEED,
            "caveat": "样本采自 Cursor agent harness（带系统提示、温度不受控、thinking 模型先思考），"
                      "是'模型+环境'指纹，不能与论文裸 API 数字直接对表。",
        },
        "ideal_entropy_bits": {q: IDEAL_ENTROPY.get(q) for q in QUESTIONS},
        "per_model": per_model,
        "jsd_matrix": {
            "models": MODELS,
            "matrix": matrix.tolist(),
        },
        "family_summary": family_summary,
        "closest_cross_family_pair": {
            "pair": f"{closest_cross[0]} vs {closest_cross[1]}",
            "jsd": closest_cross[2],
        },
        "self_noise_baseline": noise,
        "key_pairs": key_pairs,
        "magic_73": magic,
        "zh_vs_en": zh_en,
    }
    out_json = OUT_DIR / "frontier_results.json"
    out_json.write_text(
        json.dumps(round_floats(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fig1 = OUT_DIR / "fig_frontier_matrix.png"
    fig2 = OUT_DIR / "fig_frontier_entropy.png"
    plot_matrix(matrix, fig1)
    plot_entropy(entropy_table, fig2)

    # --- (i) 控制台中文摘要
    print("=" * 78)
    print("前沿模型行为指纹实测分析（2026-07-21，Cursor subagent 采样）")
    print("=" * 78)

    print("\n[样本] 每模型样本数：", ", ".join(f"{m}={sample_sizes[m]}" for m in MODELS))
    if issues:
        print("[数据异常]")
        for it in issues:
            print("  -", it)
    else:
        print("[数据异常] 无：99 条记录字段完整、类型正常。")

    print("\n[招牌组合] 模型 | rand_num_100 众数(占比) | 颜色 | 动物 | 城市 | 硬币 heads 占比")
    for m in MODELS:
        s = per_model[m]["signature"]
        print(f"  {m:<14} {s['rand_num_100_mode']:>3} ({s['rand_num_100_mode_share']:.0%})"
              f"  {s['rand_color_mode']:<7} {s['rand_animal_mode']:<9} {s['rand_city_mode']:<7}"
              f" heads={s['coin_heads_share']:.0%}")

    print("\n[熵/bit] 模型 × 题（理想均匀值：num100=6.64, num10=3.32, letter=4.70, coin=1.00）")
    col_labels = ["num100", "num10", "favnum", "letter", "color", "favcol",
                  "animal", "city", "coin", "num100zh"]
    print("  " + "model".ljust(15) + "".join(lab.ljust(9) for lab in col_labels))
    for m in MODELS:
        row = "  " + m.ljust(15)
        for q in QUESTIONS:
            row += f"{entropy_table[m][q]:.2f}".ljust(9)
        print(row)
    max_ent = max(entropy_table[m]["rand_num_100"] for m in MODELS)
    print(f"  全部模型 rand_num_100 熵 ≤ {max_ent:.2f} bit，远低于理想 6.64 bit → 非随机性成立。")

    print(f"\n[自比噪声带] 每模型 11 样本对半分（5/6）×{N_SHUFFLES} 次："
          f"合并均值 {noise['pooled_mean']:.3f}，95 分位 {noise['pooled_p95']:.3f}"
          f"（半组仅 5~6 样本，噪声偏大，如实报告）")

    print("\n[重点对照组] 聚合 JSD（对照：自比噪声带均值 "
          f"{noise['pooled_mean']:.3f} / p95 {noise['pooled_p95']:.3f}）")
    for kp in key_pairs:
        flag = "≈噪声带内" if kp["jsd"] <= noise["pooled_p95"] else "高于噪声带"
        print(f"  {kp['pair']:<32} {kp['jsd']:.3f}  [{kp['label']}] {flag}")
    print(f"  家族内（Claude 系 {family_summary['within_claude_pairs']} 对）均值 "
          f"{family_summary['within_claude_mean']:.3f}；"
          f"sol vs terra {family_summary['within_gpt_sol_terra']:.3f}；"
          f"跨家族（{family_summary['cross_family_pairs']} 对）均值 "
          f"{family_summary['cross_family_mean']:.3f}"
          f"（最小 {family_summary['cross_family_min']:.3f}，最大 {family_summary['cross_family_max']:.3f}）")
    print(f"  跨家族最接近的一对：{closest_cross[0]} vs {closest_cross[1]} = {closest_cross[2]:.3f}")

    print("\n[73 传闻] rand_num_100 里回答 73 的占比：")
    print(f"  fable5 = {magic['fable5_73_share']:.0%}，fable5-think = {magic['fable5-think_73_share']:.0%}")
    shares = "，".join(f"{k}: {v:.1%}" for k, v in magic["pooled_shares"].items())
    print(f"  9 模型合并（n={magic['pooled_n']}）魔法数占比：{shares}；"
          f"四个数合计 {magic['pooled_magic_total']:.1%}")

    print("\n[中英差异] 同一模型 rand_num_100（英文问）vs rand_num_100_zh（中文问）的 JSD：")
    for m in MODELS:
        z = zh_en[m]
        print(f"  {m:<14} JSD={z['jsd_en_vs_zh']:.3f}  en众数 {z['en_mode']}({z['en_mode_share']:.0%})"
              f" → zh众数 {z['zh_mode']}({z['zh_mode_share']:.0%})")

    print("\n[输出]")
    print(f"  {out_json}")
    print(f"  {fig1}")
    print(f"  {fig2}")
    print("=" * 78)


if __name__ == "__main__":
    main()
