# -*- coding: utf-8 -*-
"""端到端：入册 -> 审计各类对手，验证判定方向正确。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from fpverify.betting import BettingConfig
from fpverify.verifier import Verifier
from sim.adversaries import make_endpoint


CLAIMED = "gpt-4o"


@pytest.fixture(scope="module")
def reference():
    v = Verifier(seed=11)
    trusted = make_endpoint("honest", CLAIMED, seed=1)
    return v.enroll(trusted, CLAIMED, samples_per_cell=25)


def _audit(reference, kind, seed=2, **kw):
    # delta=None => 审计前自动标定容差（推荐用法）
    v = Verifier(BettingConfig(alpha=0.01, delta=None), seed=seed)
    ep = make_endpoint(kind, CLAIMED, seed=seed, **kw)
    return v.audit(ep, reference, max_queries=600)


def test_honest_passes(reference):
    res = _audit(reference, "honest", seed=21)
    assert res.verdict == "PASS", res.detail


def test_drift_passes(reference):
    """同模型跨部署漂移不应被误杀（δ 容差的意义）。"""
    res = _audit(reference, "drift", seed=22)
    assert res.verdict == "PASS", f"{res.verdict}: {res.detail} JSD={res.aggregate_jsd}"


def test_swap_fails(reference):
    res = _audit(reference, "swap", seed=23)
    assert res.verdict == "FAIL"
    assert res.n_queries < 250, f"早停失效：用了 {res.n_queries} 次查询"


def test_pin_fails(reference):
    res = _audit(reference, "pin", seed=24)
    assert res.verdict == "FAIL"


def test_true_random_fails(reference):
    res = _audit(reference, "true_random", seed=25)
    assert res.verdict == "FAIL"


def test_filter_en_fails(reference):
    """T2 过滤对手：特判公开英文措辞，仍会在其它措辞/语言暴露。"""
    res = _audit(reference, "filter_en", seed=26)
    assert res.verdict == "FAIL"


def test_partial_mimic_fails(reference):
    res = _audit(reference, "partial_mimic", seed=27)
    assert res.verdict == "FAIL"


def test_cache_flagged(reference):
    res = _audit(reference, "cache", seed=28)
    assert res.verdict == "FAIL"
    # 缓存对手要么被指纹拒绝，要么被缓存筛查标记——两者都可接受，但至少其一
    assert res.cache_flags or "缓存" in res.detail or res.peak_wealth >= res.threshold
