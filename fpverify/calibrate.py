"""标定与评估工具：自助法零分布、EER、预算曲线、δ 选择。

这些函数支撑研究笔记里"统计保证"与"成本/精度权衡"两块，
并被 experiments/run_evaluation.py 用来产出图表。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import replace

from .distance import aggregate_distance
from .normalize import OTHER

# 标定结果缓存：同一参考指纹 + 同一配置的 δ 是确定性的，评估/批量审计时避免重复模拟
_CAL_CACHE: dict = {}


def _dirichlet(alphas: list, rng: random.Random) -> list:
    """标准库实现的 Dirichlet 抽样（Gamma 归一化）。"""
    draws = [rng.gammavariate(max(a, 1e-6), 1.0) for a in alphas]
    s = sum(draws)
    if s <= 0:
        n = len(alphas)
        return [1.0 / n] * n
    return [d / s for d in draws]


def calibrate_delta(ref_counts_by_cell: dict, cfg, horizon: int = 600,
                    n_sims: int = 250, benign_drift: float = 0.08,
                    quantile: float = 1.0, safety: float = 1.3,
                    floor: float = 0.005, cap: float = 0.5,
                    threshold_margin: float = 4.0, seed: int = 20260721) -> float:
    """数据驱动地标定容差 δ，使"真身端点被误判"的概率被压回 ≈α。

    真身端点的观测流与参考指纹之间存在两类良性失配：
      (a) 参考指纹的有限样本误差 —— 参考只是真分布的一次多项抽样估计，
          真分布还可能包含参考没见过的低频答案（缺失质量）；
      (b) 跨部署良性漂移 —— 同一模型在不同服务栈上的小幅分布差异（论文中位 0.227 vs
          单部署自比 0.140）。
    若 δ 不覆盖它们，在线估计器会"学到"这些失配并让财富持续增长，把真身误杀。

    标定方法（后验预测模拟）：对每次模拟，
      1) 对每个 cell 从 Dirichlet(参考计数 + Jeffreys 0.5) 抽一份"可能的真分布"。
         支撑 = 参考答案 ∪ {<other>}，其中 <other> 的先验计数取 Good-Turing 缺失质量
         下限（单例数），因此模拟流会以真实量级产生"参考没见过的答案"——这是
         纯自助重抽（永远不产生新答案）覆盖不到的失配来源；
      2) 叠加 ±benign_drift 的乘性重加权（注入 (b)）；
      3) 用它生成长度 horizon 的审计流，跑 δ=0 的检验，记录
         need = max_t (log W_t − log(1/(α·margin))) / t —— 让这条轨迹连"阈值的
         1/margin 邻域"都不进入所需的最小 δ（log W_t(δ) = log W_t(0) − δ·t 单调于 δ）。
         margin>1 把"贴近阈值但未越界"的临界轨迹也计入，消除标定的悬崖效应。
    取 need 的 quantile 分位（默认取最大值）× safety，夹在 [floor, cap]。

    诚实说明：加入标定 δ 后，总体保证是"后验预测近似下的 FPR ≲ α"，
    而非点零假设下的精确 Ville 界（那需要 δ=0 且端点分布恰为平滑后的参考）。
    这是把复合零假设（含良性噪声）工程化的标准做法；FPR 由测试实证验证。
    """
    from .betting import SequentialBettingTest  # 延迟导入避免环依赖

    cells = [c for c, cnt in ref_counts_by_cell.items() if sum(cnt.values()) >= 10]
    if not cells:
        return floor

    cache_key = (
        tuple(sorted((str(c), tuple(sorted(ref_counts_by_cell[c].items()))) for c in cells)),
        cfg.alpha, cfg.lam, cfg.smoothing, cfg.q_smoothing, cfg.max_answer_classes,
        horizon, n_sims, benign_drift, quantile, safety, floor, cap, threshold_margin, seed,
    )
    if cache_key in _CAL_CACHE:
        return _CAL_CACHE[cache_key]

    rng = random.Random(seed)
    # 提前量：连 W = (1/α)/margin 的轨迹也要求 δ 压回去，避免临界轨迹在真实审计中越界
    log_thr = math.log(1.0 / cfg.alpha) - math.log(threshold_margin)
    cfg0 = replace(cfg, delta=0.0)

    # 每 cell 的 Dirichlet 参数：参考计数 + Jeffreys 0.5；<other> 计数取 Good-Turing 下限
    prior = {}
    for c in cells:
        cnt = ref_counts_by_cell[c]
        singletons = sum(1 for _, v in cnt.items() if v == 1)
        alphas = dict(cnt)
        alphas[OTHER] = max(alphas.get(OTHER, 0), singletons, 0.5)
        keys = list(alphas.keys())
        prior[c] = (keys, [alphas[k] + 0.5 for k in keys])

    required = []
    for _ in range(n_sims):
        streams = {}
        for c in cells:
            keys, alphas = prior[c]
            p = _dirichlet(alphas, rng)                                   # (a) 后验真分布
            w = [x * (1.0 + benign_drift * (rng.random() * 2 - 1)) for x in p]  # (b) 漂移
            streams[c] = (keys, w)
        t = SequentialBettingTest({c: ref_counts_by_cell[c] for c in cells}, cfg0)
        order = cells[:]
        rng.shuffle(order)
        need = 0.0
        for i in range(horizon):
            c = order[i % len(order)]
            keys, w = streams[c]
            a = rng.choices(keys, weights=w, k=1)[0]
            t.observe(c, a)
            excess = (t.log_w - log_thr) / t.n_obs
            if excess > need:
                need = excess
        required.append(need)
    required.sort()
    idx = min(len(required) - 1, int(quantile * len(required)))
    delta = required[idx] * safety
    delta = min(max(delta, floor), cap)
    _CAL_CACHE[cache_key] = delta
    return delta


def split_half_distance(counts_by_cell: dict, rng: random.Random, min_samples: int = 10):
    """把同一模型每个 cell 的样本按奇偶切成两半，算聚合 JSD。

    对应论文的"同源自比"噪声地板（genuine trial）。返回距离或 None。
    """
    half_a, half_b = {}, {}
    for cell, counts in counts_by_cell.items():
        pool = [a for a, c in counts.items() for _ in range(c)]
        if len(pool) < 2 * min_samples:
            continue
        rng.shuffle(pool)
        mid = len(pool) // 2
        half_a[cell] = Counter(pool[:mid])
        half_b[cell] = Counter(pool[mid:])
    d, _ = aggregate_distance(half_a, half_b, min_samples=min_samples)
    return d


def bootstrap_null_distances(counts_by_cell: dict, samples_per_cell: int,
                             iters: int = 400, seed: int = 0, min_samples: int = 5):
    """自助法：从参考分布反复抽样模拟"同源"验证，得到零假设下的聚合距离分布。

    用于把 e-process 的判定阈值转成人类可读的 JSD 参考带，或做固定阈值基线对照。
    """
    rng = random.Random(seed)
    pools = {}
    for cell, counts in counts_by_cell.items():
        pool = [a for a, c in counts.items() for _ in range(c)]
        if pool:
            pools[cell] = pool
    dists = []
    for _ in range(iters):
        sampled = {c: Counter(rng.choices(p, k=samples_per_cell)) for c, p in pools.items()}
        d, _ = aggregate_distance(counts_by_cell, sampled, min_samples=min_samples)
        if d is not None:
            dists.append(d)
    dists.sort()
    return dists


def percentile(sorted_vals: list, q: float):
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(q * len(sorted_vals)) - 1))
    return sorted_vals[idx]


def equal_error_rate(genuine: list, impostor: list):
    """给定 genuine/impostor 的距离分数，扫阈值求 EER。返回 (eer, threshold)。

    距离越小越像。阈值 tau：距离<=tau 判为"同源(接受)"。
      FAR = 冒充者被接受的比例（impostor <= tau）
      FRR = 真身被拒绝的比例（genuine > tau）
    EER = 使 |FAR-FRR| 最小处的 (FAR+FRR)/2。
    """
    if not genuine or not impostor:
        return None, None
    candidates = sorted(set(genuine + impostor))
    best_gap = float("inf")
    best = (1.0, candidates[0])
    for tau in candidates:
        far = sum(1 for d in impostor if d <= tau) / len(impostor)
        frr = sum(1 for d in genuine if d > tau) / len(genuine)
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best = ((far + frr) / 2.0, tau)
    return best


def auc(genuine: list, impostor: list):
    """distance 越小越像 => genuine 分数应更小。AUC = P(impostor > genuine)。"""
    if not genuine or not impostor:
        return None
    wins = ties = 0
    for g in genuine:
        for i in impostor:
            if i > g:
                wins += 1
            elif i == g:
                ties += 1
    return (wins + 0.5 * ties) / (len(genuine) * len(impostor))
