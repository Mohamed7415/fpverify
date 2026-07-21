"""概率工具：平滑、JSD、熵、聚合指纹距离。

JSD 采用 base-2，∈[0,1]，用于人类可读的效应量与阈值解读（对照论文参考带
0.140 同源噪声 / 0.227 跨服务商 / 0.463 冒充者中位）。
决策不直接用固定 JSD 阈值，而由 betting.py 的序贯 e-process 承担（见研究笔记 §4.3）。
"""

from __future__ import annotations

import math
from collections import Counter


def counts_to_probs(counts: dict, smoothing: float = 0.0, support: set | None = None) -> dict:
    """计数 -> 概率分布。可选拉普拉斯平滑与显式支撑集。"""
    support = set(support) if support else set(counts)
    if not support:
        return {}
    total = sum(counts.get(k, 0) for k in support) + smoothing * len(support)
    if total <= 0:
        n = len(support)
        return {k: 1.0 / n for k in support}
    return {k: (counts.get(k, 0) + smoothing) / total for k in support}


def entropy_bits(counts: dict) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in counts.values() if v > 0)


def jsd_bits(c1: dict, c2: dict) -> float | None:
    """两个计数分布之间的 Jensen-Shannon 散度（base 2，∈[0,1]）。"""
    n1, n2 = sum(c1.values()), sum(c2.values())
    if n1 == 0 or n2 == 0:
        return None
    keys = set(c1) | set(c2)
    p = {k: c1.get(k, 0) / n1 for k in keys}
    q = {k: c2.get(k, 0) / n2 for k in keys}
    m = {k: (p[k] + q[k]) / 2 for k in keys}

    def kl(a):
        s = 0.0
        for k in keys:
            if a[k] > 0:
                s += a[k] * math.log2(a[k] / m[k])
        return s

    val = 0.5 * kl(p) + 0.5 * kl(q)
    # 数值兜底
    return max(0.0, min(1.0, val))


def aggregate_distance(ref_counts: dict, test_counts: dict, min_samples: int = 10):
    """在双方都有 >=min_samples 有效样本的 cell 上平均 JSD。

    返回 (mean_distance, per_cell_dict)；无可用 cell 返回 (None, {})。
    """
    per_cell = {}
    for key, rc in ref_counts.items():
        tc = test_counts.get(key)
        if tc is None:
            continue
        if sum(rc.values()) < min_samples or sum(tc.values()) < min_samples:
            continue
        d = jsd_bits(rc, tc)
        if d is not None:
            per_cell[key] = d
    if not per_cell:
        return None, {}
    return sum(per_cell.values()) / len(per_cell), per_cell


def as_counter(d) -> Counter:
    return d if isinstance(d, Counter) else Counter(d)
