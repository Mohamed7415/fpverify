# -*- coding: utf-8 -*-
"""决策核心的统计性质：e-变量公平性、FPR 控制、检出力。

这是本项目最重要的测试：验证"不会把真模型冤枉成假"是有数学保证的，
以及"真被换了能抓出来"。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import random
from collections import Counter

from fpverify.betting import BettingConfig, SequentialBettingTest, _RefModel


CELL = ("rand_num_100", "en")


def make_ref(dist: dict, n: int = 600, seed: int = 1) -> Counter:
    rng = random.Random(seed)
    keys, weights = list(dist.keys()), list(dist.values())
    return Counter(rng.choices(keys, weights=weights, k=n))


GPT_DIST = {"42": 30, "37": 20, "57": 18, "73": 10, "69": 8, "88": 6, "7": 4, "100": 4}
CHEAP_DIST = {"7": 30, "42": 22, "1": 14, "100": 12, "50": 12, "69": 10}


def test_e_variable_fairness():
    """E_{p0}[f] == 1（数值验证下注公平性 => 上鞅性质的根基）。"""
    cfg = BettingConfig()
    ref = _RefModel(make_ref(GPT_DIST), cfg)
    # 任意固定的 q̂（模拟在线估计的某个瞬间）。
    # 与实现一致：观测在计数前经过 canon() 折叠到参考支撑（未见答案 -> <other>）。
    raw = Counter({"42": 3, "7": 9, "1": 5})
    q_counts = Counter()
    for a, c in raw.items():
        q_counts[ref.canon(a)] += c
    n, k = sum(q_counts.values()), len(ref.support)

    def q_hat(a):
        return (q_counts[a] + cfg.q_smoothing) / (n + cfg.q_smoothing * k)

    total = 0.0
    for a in ref.support:
        f = (1 - cfg.lam) + cfg.lam * (q_hat(a) / ref.p0(a))
        total += ref.p0(a) * f
    # Σ p0(a)·f(a) = (1-λ)·Σp0 + λ·Σ q̂ = 1（两者在同一支撑上归一）
    assert abs(total - 1.0) < 1e-9


def test_fpr_controlled_under_h0():
    """H0 下（端点=参考同分布）跑 400 次序贯审计，拒绝率应 <= alpha 的合理邻域。"""
    alpha = 0.05
    cfg = BettingConfig(alpha=alpha, delta=0.0)   # delta=0 是最严设置，FPR 仍须受控
    ref_counts = {CELL: make_ref(GPT_DIST, n=600)}
    keys, weights = list(GPT_DIST.keys()), list(GPT_DIST.values())

    rejections = 0
    trials = 400
    for i in range(trials):
        rng = random.Random(1000 + i)
        t = SequentialBettingTest(ref_counts, cfg)
        for _ in range(300):
            t.observe(CELL, rng.choices(keys, weights=weights, k=1)[0])
            if t.decided:
                break
        rejections += int(t.rejected)
    fpr = rejections / trials
    # 二项 95% 置信上界约 alpha + 2.5*sqrt(alpha/trials) ≈ 0.078
    assert fpr <= alpha + 2.5 * math.sqrt(alpha * (1 - alpha) / trials), f"FPR={fpr}"


def test_fpr_robust_to_reference_sampling_noise():
    """参考指纹本身是有限样本（n=200 偏小）时，同分布端点也不应被大量误杀。

    这依赖 Good-Turing <other> 下限与 delta 容差。
    """
    alpha = 0.05
    cfg = BettingConfig(alpha=alpha, delta=0.02)
    ref_counts = {CELL: make_ref(GPT_DIST, n=200, seed=42)}
    keys, weights = list(GPT_DIST.keys()), list(GPT_DIST.values())

    rejections = 0
    trials = 300
    for i in range(trials):
        rng = random.Random(5000 + i)
        t = SequentialBettingTest(ref_counts, cfg)
        for _ in range(300):
            t.observe(CELL, rng.choices(keys, weights=weights, k=1)[0])
            if t.decided:
                break
        rejections += int(t.rejected)
    fpr = rejections / trials
    assert fpr <= alpha + 2.5 * math.sqrt(alpha * (1 - alpha) / trials), f"FPR={fpr}"


def test_power_detects_swap():
    """换成便宜模型后，检验应在几十次观测内高概率拒绝。"""
    cfg = BettingConfig(alpha=0.01, delta=0.02)
    ref_counts = {CELL: make_ref(GPT_DIST, n=600)}
    keys, weights = list(CHEAP_DIST.keys()), list(CHEAP_DIST.values())

    detected = 0
    stops = []
    trials = 200
    for i in range(trials):
        rng = random.Random(9000 + i)
        t = SequentialBettingTest(ref_counts, cfg)
        for _ in range(300):
            t.observe(CELL, rng.choices(keys, weights=weights, k=1)[0])
            if t.decided:
                break
        detected += int(t.rejected)
        if t.rejected:
            stops.append(t.n_obs)
    assert detected / trials >= 0.95, f"power={detected / trials}"
    assert sum(stops) / len(stops) <= 150, f"mean stop={sum(stops) / len(stops)}"


def test_power_detects_pinned_answer():
    """钉死回答（评论区'只回答73'方案）应被极快识破。"""
    cfg = BettingConfig(alpha=0.01, delta=0.02)
    ref_counts = {CELL: make_ref(GPT_DIST, n=600)}
    t = SequentialBettingTest(ref_counts, cfg)
    for _ in range(300):
        t.observe(CELL, "73")
        if t.decided:
            break
    assert t.rejected and t.n_obs <= 60, f"n={t.n_obs}"


def test_power_detects_true_random():
    """真随机回答（对手自以为安全）应被识破——真模型恰恰不随机。"""
    cfg = BettingConfig(alpha=0.01, delta=0.02)
    ref_counts = {CELL: make_ref(GPT_DIST, n=600)}
    rng = random.Random(7)
    t = SequentialBettingTest(ref_counts, cfg)
    for _ in range(400):
        t.observe(CELL, str(rng.randint(1, 100)))
        if t.decided:
            break
    assert t.rejected, "true-random adversary not detected"


def test_auto_calibrated_delta_controls_fpr_small_reference():
    """自动标定的 δ：参考只有 150 样本（现实的小额入册）时，同分布端点的
    误杀率仍应压在 alpha 邻域内——这是标定器存在的意义。"""
    from fpverify.calibrate import calibrate_delta

    alpha = 0.05
    ref_counts = {CELL: make_ref(GPT_DIST, n=150, seed=77)}
    base = BettingConfig(alpha=alpha, delta=0.0)
    delta = calibrate_delta(ref_counts, base, horizon=300, n_sims=200)
    assert 0.0 < delta <= 0.5, f"delta={delta} 不在合理范围"

    cfg = BettingConfig(alpha=alpha, delta=delta)
    keys, weights = list(GPT_DIST.keys()), list(GPT_DIST.values())
    rejections = 0
    trials = 300
    for i in range(trials):
        rng = random.Random(31_000 + i)
        t = SequentialBettingTest(ref_counts, cfg)
        for _ in range(300):
            t.observe(CELL, rng.choices(keys, weights=weights, k=1)[0])
            if t.decided:
                break
        rejections += int(t.rejected)
    fpr = rejections / trials
    assert fpr <= alpha + 2.5 * math.sqrt(alpha * (1 - alpha) / trials), f"FPR={fpr}"


def test_wealth_never_negative_and_anytime():
    """财富为正；观测流可在任意点停下，结论只会从未拒绝->拒绝单向变化。"""
    cfg = BettingConfig(delta=0.01)
    ref_counts = {CELL: make_ref(GPT_DIST, n=600)}
    rng = random.Random(3)
    t = SequentialBettingTest(ref_counts, cfg)
    was_rejected = False
    for _ in range(100):
        t.observe(CELL, rng.choice(list(GPT_DIST.keys())))
        assert t.wealth > 0
        if was_rejected:
            assert t.rejected  # 拒绝态不可逆
        was_rejected = t.rejected
