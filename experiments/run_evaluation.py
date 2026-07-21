# -*- coding: utf-8 -*-
"""红队评估：验证检测器的统计性质与成本/精度权衡，产出图表与 JSON 结果。

实验清单
  E1  FPR 验证：诚实端点 + 跨部署漂移端点，多参考多种子重复审计，
      实测误杀率应 <= alpha。
  E2  检出率与停时：九类对手 × 多种子，检出率、平均查询数（早停效果）。
  E3  预算曲线：max_queries ∈ {60,120,240,480,600} 下的检出率/误杀率，
      给出"花多少 token 买多少可靠性"的操作曲线。
  E4  模型距离矩阵：7 个仿真模型两两聚合 JSD，验证同族接近/异族远离
      （对应论文的谱系恢复与 Palmyra≈Qwen 案例）。
  E5  财富轨迹示意：诚实 vs 各对手的 e-process 财富随查询数演化。

运行:  py -3.13 -X utf8 experiments/run_evaluation.py
输出:  experiments/out/*.png, results.json
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpverify.betting import BettingConfig, SequentialBettingTest
from fpverify.distance import aggregate_distance
from fpverify.verifier import Verifier
from sim.adversaries import make_endpoint, ADVERSARY_KINDS
from sim.mock_models import BASE_MODELS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "experiments" / "out"
OUT.mkdir(parents=True, exist_ok=True)

CLAIMED = "gpt-4o"
ALPHA = 0.01
DELTA = None                 # None = 每次审计前用参考指纹自动标定容差
ENROLL_SAMPLES = 25          # 每 cell 参考样本
AUDIT_BUDGET = 600           # 默认审计预算

# 成本模型（单位：token）：探针 ≈ 系统提示+问题 ≈ 45 输入 token，回答 ≈ 4 输出 token
IN_TOK, OUT_TOK = 45, 4
# 旗舰定价示例（$/1M token）：输入 2.5 / 输出 10
PRICE_IN, PRICE_OUT = 2.5, 10.0


def audit_cost_usd(n_queries: int) -> float:
    return (n_queries * IN_TOK * PRICE_IN + n_queries * OUT_TOK * PRICE_OUT) / 1e6


def make_reference(seed: int):
    v = Verifier(seed=seed)
    trusted = make_endpoint("honest", CLAIMED, seed=seed * 7 + 1)
    return v.enroll(trusted, CLAIMED, samples_per_cell=ENROLL_SAMPLES)


def run_audit(reference, kind: str, seed: int, budget: int = AUDIT_BUDGET, **kw):
    v = Verifier(BettingConfig(alpha=ALPHA, delta=DELTA), seed=seed)
    ep = make_endpoint(kind, CLAIMED, seed=seed, **kw)
    return v.audit(ep, reference, max_queries=budget)


# ---------------------------------------------------------------- E1: FPR
def experiment_fpr(n_refs=4, n_audits_per=50):
    print("[E1] FPR 验证（诚实 + 漂移端点，不应被误杀）")
    results = {"honest": [], "drift": []}
    for r in range(n_refs):
        ref = make_reference(seed=100 + r)
        for i in range(n_audits_per):
            for kind in ("honest", "drift"):
                res = run_audit(ref, kind, seed=10_000 + r * 1000 + i,
                                drift_seed=r * 31 + i if kind == "drift" else None)
                results[kind].append(res.verdict)
    out = {}
    for kind, verdicts in results.items():
        n = len(verdicts)
        fails = sum(1 for v in verdicts if v == "FAIL")
        out[kind] = {"n": n, "false_positive": fails, "fpr": fails / n}
        print(f"  {kind:<8} n={n}  误杀={fails}  FPR={fails / n:.4f}  (alpha={ALPHA})")
    return out


# ---------------------------------------------------------------- E2: 检出率/停时
def experiment_power(n_refs=3, n_audits_per=25):
    print("[E2] 各类对手的检出率与停时")
    kinds = [k for k in ADVERSARY_KINDS if k not in ("honest", "drift")]
    agg = {k: {"detected": 0, "n": 0, "stops": [], "jsds": []} for k in kinds}
    for r in range(n_refs):
        ref = make_reference(seed=200 + r)
        for i in range(n_audits_per):
            for kind in kinds:
                res = run_audit(ref, kind, seed=20_000 + r * 1000 + i)
                a = agg[kind]
                a["n"] += 1
                if res.verdict == "FAIL":
                    a["detected"] += 1
                    a["stops"].append(res.n_queries)
                if res.aggregate_jsd is not None:
                    a["jsds"].append(res.aggregate_jsd)
    out = {}
    for kind, a in agg.items():
        rate = a["detected"] / a["n"]
        mean_stop = statistics.mean(a["stops"]) if a["stops"] else None
        med_jsd = statistics.median(a["jsds"]) if a["jsds"] else None
        cost = audit_cost_usd(mean_stop) if mean_stop else None
        out[kind] = {"n": a["n"], "detection_rate": rate, "mean_queries_to_detect": mean_stop,
                     "median_jsd": med_jsd, "mean_cost_usd": cost}
        print(f"  {kind:<14} 检出率={rate:.3f}  平均停时={mean_stop and round(mean_stop, 1)}  "
              f"中位JSD={med_jsd and round(med_jsd, 3)}  平均成本=${cost and round(cost, 5)}")
    return out


# ---------------------------------------------------------------- E3: 预算曲线
def experiment_budget_curve(budgets=(60, 120, 240, 480, 600), n_refs=2, n_audits_per=20):
    print("[E3] 预算曲线")
    kinds = ["honest", "drift", "swap", "quantized", "filter_en", "partial_mimic"]
    curve = {k: {} for k in kinds}
    for b in budgets:
        for r in range(n_refs):
            ref = make_reference(seed=300 + r)
            for i in range(n_audits_per):
                for kind in kinds:
                    res = run_audit(ref, kind, seed=30_000 + r * 1000 + i, budget=b)
                    slot = curve[kind].setdefault(b, {"FAIL": 0, "PASS": 0, "SUSPECT": 0,
                                                      "INCONCLUSIVE": 0, "queries": []})
                    slot[res.verdict] += 1
                    slot["queries"].append(res.n_queries)
    printable = {}
    for kind, per_budget in curve.items():
        printable[kind] = {}
        for b, slot in per_budget.items():
            n = slot["FAIL"] + slot["PASS"] + slot["SUSPECT"] + slot["INCONCLUSIVE"]
            printable[kind][b] = {
                "fail_rate": slot["FAIL"] / n,
                "pass_rate": slot["PASS"] / n,
                "suspect_rate": slot["SUSPECT"] / n,
                "mean_queries": statistics.mean(slot["queries"]),
            }
    for kind in kinds:
        row = "  ".join(f"{b}:{printable[kind][b]['fail_rate']:.2f}" for b in budgets)
        print(f"  {kind:<14} FAIL率@预算  {row}")
    return printable


# ---------------------------------------------------------------- E4: 距离矩阵
def experiment_distance_matrix():
    print("[E4] 模型两两距离矩阵（谱系检查）")
    models = list(BASE_MODELS.keys())
    fps = {}
    for m in models:
        v = Verifier(seed=hash(m) % 1000)
        fps[m] = v.enroll(make_endpoint("honest", m, seed=5), m, samples_per_cell=25)
    mat = {}
    for a in models:
        for b in models:
            d, _ = aggregate_distance(fps[a].counts(), fps[b].counts(), min_samples=10)
            mat[(a, b)] = d if d is not None else 0.0
    # 打印关键对
    same_family = mat[("qwen3-235b", "qwen3-max")]
    cross = mat[("gpt-4o", "cheap-7b")]
    print(f"  同族 qwen3-235b vs qwen3-max: {same_family:.3f}")
    print(f"  异族 gpt-4o vs cheap-7b:     {cross:.3f}")
    return models, mat


# ---------------------------------------------------------------- E5: 财富轨迹
def experiment_wealth_trajectories():
    print("[E5] 财富轨迹示意")
    from fpverify.calibrate import calibrate_delta
    ref = make_reference(seed=400)
    ref_counts = ref.counts()
    base = BettingConfig(alpha=ALPHA, delta=0.0)
    delta = calibrate_delta(ref_counts, base, horizon=400)
    print(f"  自动标定 delta={delta:.4f}")
    cfg = BettingConfig(alpha=ALPHA, delta=delta)
    kinds = ["honest", "drift", "quantized", "swap", "pin", "true_random"]
    trajectories = {}
    from fpverify import probes as probes_mod
    from fpverify.normalize import normalize
    for kind in kinds:
        rng = random.Random(4242)
        ep = make_endpoint(kind, CLAIMED, seed=99)
        cells = [c for c in ref_counts if sum(ref_counts[c].values()) >= 10]
        t = SequentialBettingTest({c: ref_counts[c] for c in cells}, cfg)
        order = list(cells)
        rng.shuffle(order)
        xs, ys = [0], [1.0]
        idx = 0
        while t.n_obs < 400 and not t.decided:
            cell = order[idx % len(order)]
            idx += 1
            task, lang = cell
            system, user = probes_mod.render_prompt(task, lang, rng)
            ans = ep.ask(system, user)
            tok = normalize(probes_mod.parse_type(task), ans.text)
            t.observe(cell, tok)
            xs.append(t.n_obs)
            ys.append(t.wealth)
        trajectories[kind] = (xs, ys, t.verdict())
        print(f"  {kind:<12} 终态={t.verdict():<4} 查询={t.n_obs}")
    return trajectories


# ---------------------------------------------------------------- 图表
def plot_all(fpr, power, curve, dist, trajectories):
    models, mat = dist

    # 图1：预算曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    budgets = sorted(next(iter(curve.values())).keys())
    for kind, style in [("swap", "-o"), ("quantized", "-s"), ("filter_en", "-^"),
                        ("partial_mimic", "-d"), ("honest", "--x"), ("drift", "--+")]:
        ys = [curve[kind][b]["fail_rate"] for b in budgets]
        ax.plot(budgets, ys, style, label=kind)
    ax.axhline(ALPHA, color="gray", lw=0.8, ls=":", label=f"alpha={ALPHA}")
    ax.set_xlabel("审计预算（查询数上限）")
    ax.set_ylabel("FAIL 判定率")
    ax.set_title("检出率/误杀率 vs 预算（注水类应趋近1，诚实类应≈0）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget_curve.png", dpi=150)
    plt.close(fig)

    # 图2：财富轨迹
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, (xs, ys, verdict) in trajectories.items():
        ax.plot(xs, ys, label=f"{kind} ({verdict})")
    ax.axhline(1 / ALPHA, color="red", lw=1, ls="--", label=f"拒绝阈值 1/α={1 / ALPHA:.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("查询数")
    ax.set_ylabel("e-process 财富（log）")
    ax.set_title("序贯下注财富轨迹：越假越快越界，诚实端点始终贴地")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_wealth_trajectories.png", dpi=150)
    plt.close(fig)

    # 图3：距离矩阵热图
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    n = len(models)
    grid = [[mat[(a, b)] for b in models] for a in models]
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(n), models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), models, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{grid[i][j]:.2f}", ha="center", va="center",
                    color="white" if grid[i][j] < 0.45 else "black", fontsize=7)
    ax.set_title("仿真模型两两聚合 JSD（同族近、异族远）")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_distance_matrix.png", dpi=150)
    plt.close(fig)

    # 图4：停时分布（检出成本）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    kinds = [k for k, v in power.items() if v["mean_queries_to_detect"]]
    means = [power[k]["mean_queries_to_detect"] for k in kinds]
    costs = [power[k]["mean_cost_usd"] * 100 for k in kinds]  # 美分
    bars = ax.bar(kinds, means, color="steelblue")
    for bar, c in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"~{c:.2f}美分", ha="center", fontsize=8)
    ax.set_ylabel("平均查询数（检出即停）")
    ax.set_title("各对手的平均检出成本（按旗舰定价折算）")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stop_times.png", dpi=150)
    plt.close(fig)

    print(f"图表已保存到 {OUT}")


def main():
    t0 = time.time()
    fpr = experiment_fpr()
    power = experiment_power()
    curve = experiment_budget_curve()
    dist = experiment_distance_matrix()
    trajectories = experiment_wealth_trajectories()
    plot_all(fpr, power, curve, dist, trajectories)

    results = {
        "config": {"alpha": ALPHA, "delta": DELTA, "enroll_samples_per_cell": ENROLL_SAMPLES,
                   "audit_budget": AUDIT_BUDGET, "claimed_model": CLAIMED,
                   "cost_model": {"input_tokens_per_query": IN_TOK, "output_tokens_per_query": OUT_TOK,
                                  "price_per_1m_in": PRICE_IN, "price_per_1m_out": PRICE_OUT}},
        "E1_fpr": fpr,
        "E2_power": power,
        "E3_budget_curve": {k: {str(b): v for b, v in d.items()} for k, d in curve.items()},
        "E4_distance_matrix": {f"{a}|{b}": v for (a, b), v in dist[1].items()},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"完成，用时 {results['elapsed_sec']}s。结果: {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
