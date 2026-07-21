# -*- coding: utf-8 -*-
"""JSD 与聚合距离的数学性质。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

from fpverify.distance import jsd_bits, entropy_bits, aggregate_distance


def test_jsd_identical_is_zero():
    c = Counter({"42": 30, "37": 20})
    assert jsd_bits(c, c) == 0.0


def test_jsd_disjoint_is_one():
    a = Counter({"1": 10})
    b = Counter({"2": 10})
    assert abs(jsd_bits(a, b) - 1.0) < 1e-9


def test_jsd_symmetric_and_bounded():
    a = Counter({"42": 25, "37": 5})
    b = Counter({"42": 5, "53": 25})
    d1, d2 = jsd_bits(a, b), jsd_bits(b, a)
    assert abs(d1 - d2) < 1e-12
    assert 0.0 <= d1 <= 1.0


def test_jsd_empty_returns_none():
    assert jsd_bits(Counter(), Counter({"1": 5})) is None


def test_aggregate_respects_min_samples():
    ref = {("t", "en"): Counter({"a": 3})}          # 只有 3 个样本
    test = {("t", "en"): Counter({"a": 20})}
    d, per = aggregate_distance(ref, test, min_samples=10)
    assert d is None and per == {}


def test_entropy():
    assert entropy_bits(Counter({"a": 10})) == 0.0
    assert abs(entropy_bits(Counter({"a": 5, "b": 5})) - 1.0) < 1e-9
