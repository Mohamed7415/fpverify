# -*- coding: utf-8 -*-
"""协同进化新增件的契约与性质测试。

覆盖：闭集最近邻的归属正确性与"不误报同族/真身"、分流红队的路由语义、
业务流量特征模型的方向性、蓝队审计器对 honest 的 FPR 守恒、稀释攻击的容差单调性。
这些测试不触碰现有 fpverify 公共 API，只验证新模块，保证 26 个既有测试不受影响。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
from collections import Counter

from fpverify.verifier import Verifier
from fpverify.nearest import nearest_model
from sim.adversaries import (make_endpoint, RedRelay, BenignRelay,
                             ROUTE_POLICIES, ADVERSARY_KINDS, COEVO_ADVERSARY_KINDS)
from sim.blue_team import BlueAuditor, BLUE_ROUNDS
from sim import traffic

CLAIMED = "gpt-4o"


def _enroll(model, seed=5, n=25):
    v = Verifier(seed=seed)
    return v.enroll(make_endpoint("honest", model, seed=seed * 3 + 1), model, samples_per_cell=n)


# ---------------------------------------------------------------- 兼容性
def test_existing_nine_kinds_intact():
    """新增对手不得破坏现有 9 类 ADVERSARY_KINDS 与 make_endpoint。"""
    assert ADVERSARY_KINDS == ["honest", "drift", "quantized", "swap", "pin",
                               "filter_en", "true_random", "cache", "partial_mimic"]
    for k in ADVERSARY_KINDS:
        ep = make_endpoint(k, CLAIMED, seed=1)
        ans = ep.ask("sys", "Name a random number between 1 and 100.")
        assert hasattr(ans, "text") and hasattr(ans, "model_field")
        assert ans.model_field == CLAIMED


def test_coevo_kinds_registered():
    for k in COEVO_ADVERSARY_KINDS:
        assert k in ROUTE_POLICIES
        ep = make_endpoint(k, CLAIMED, seed=2, eps=0.3)
        assert isinstance(ep, RedRelay)


# ---------------------------------------------------------------- 最近邻
def test_nearest_flags_swap_to_cheap():
    """整体替换成 cheap-7b 时，闭集最近邻应指向 cheap-7b 并 flag。"""
    enrolled = {m: _enroll(m).counts() for m in ["gpt-4o", "cheap-7b", "claude-sonnet-5"]}
    swapped = make_endpoint("swap", CLAIMED, seed=9, actual="cheap-7b")
    v = Verifier(seed=3)
    test_fp = v.enroll(swapped, CLAIMED, samples_per_cell=25)
    nr = nearest_model(test_fp.counts(), enrolled, CLAIMED)
    assert nr.nearest == "cheap-7b"
    assert nr.flagged


def test_nearest_does_not_flag_honest():
    """真身端点的最近邻应是自己，不得误报。"""
    enrolled = {m: _enroll(m).counts() for m in ["gpt-4o", "cheap-7b", "claude-sonnet-5"]}
    honest = make_endpoint("honest", CLAIMED, seed=8)
    v = Verifier(seed=4)
    test_fp = v.enroll(honest, CLAIMED, samples_per_cell=25)
    nr = nearest_model(test_fp.counts(), enrolled, CLAIMED)
    assert nr.nearest == CLAIMED
    assert not nr.flagged


def test_nearest_respects_same_family():
    """最近邻若与声称模型同族，不应判违规（自研旗舰的合法情形）。"""
    enrolled = {m: _enroll(m).counts() for m in ["qwen3-235b", "qwen3-max", "gpt-4o"]}
    # 端点其实是 qwen3-max，但声称 qwen3-235b（同族）
    ep = make_endpoint("swap", "qwen3-235b", seed=7, actual="qwen3-max")
    v = Verifier(seed=5)
    test_fp = v.enroll(ep, "qwen3-235b", samples_per_cell=25)
    fam = {"qwen3-235b": "qwen", "qwen3-max": "qwen", "gpt-4o": "openai"}
    nr = nearest_model(test_fp.counts(), enrolled, "qwen3-235b", same_family=fam)
    assert not nr.flagged, nr.reason


# ---------------------------------------------------------------- 路由语义
def test_route_probe_aware_separates_raw_probe_from_business():
    """裸探针(表层高)应被判旗舰，普通业务(表层低)应被判廉价。"""
    fn = ROUTE_POLICIES["probe_aware"]
    assert fn({"x_surface": 3.4, "x_semantic": 2.3}) == "flagship"
    assert fn({"x_surface": -0.2, "x_semantic": -1.2}) == "cheap"


def test_route_account_serves_flagship_to_stranger():
    """账号级分流：老客户供廉价，陌生/无账号(外部审计)供旗舰。"""
    fn = ROUTE_POLICIES["account_routing"]
    assert fn({}, {"established": True}) == "cheap"
    assert fn({}, {"established": False}) == "flagship"
    assert fn({}, None) == "flagship"


def test_dilution_saves_eps_fraction():
    """稀释攻击对全部流量以概率 eps 供廉价，廉价占比应≈eps（经济=省到的钱）。"""
    dil_rng = random.Random(1)     # 单一 rng：每次调用独立抽样（不可在 lambda 内重建）
    frac = traffic.served_cheap_fraction(
        lambda f, a: ROUTE_POLICIES["dilution"](f, a, eps=0.3, rng=dil_rng),
        n=20000, rng=random.Random(2))
    assert 0.27 <= frac <= 0.33


# ---------------------------------------------------------------- 流量特征模型
def test_probe_camouflage_reduces_distinguishability():
    """伪装等级越高，探针可观测特征越接近业务（表层/语义均值下降）。"""
    rng = random.Random(0)
    raw = [traffic.render_probe("rand_num_100", "en", "raw", rng, "s", "Name a number.") for _ in range(400)]
    nat = [traffic.render_probe("rand_num_100", "en", "native", rng, "s", "Name a number.") for _ in range(400)]
    raw_surf = sum(r.features["x_surface"] for r in raw) / len(raw)
    nat_surf = sum(r.features["x_surface"] for r in nat) / len(nat)
    raw_sem = sum(r.features["x_semantic"] for r in raw) / len(raw)
    nat_sem = sum(r.features["x_semantic"] for r in nat) / len(nat)
    assert nat_surf < raw_surf
    assert nat_sem < raw_sem


def test_extract_features_direction():
    """特征提取器方向性：裸探针文本的表层特征应高于长业务请求。"""
    probe = traffic.extract_features(
        "You answer with a single value only: one number, no explanation.",
        "Name a random number between 1 and 100.")
    biz = traffic.extract_features(
        "You are a helpful software engineering assistant.",
        "Refactor this long function and explain the changes in detail across several files.")
    assert probe["x_surface"] > biz["x_surface"]


# ---------------------------------------------------------------- 蓝队 FPR 守恒
def test_blue_auditor_does_not_kill_honest():
    """任一轮蓝队配置都不得误杀真身端点（FPR 守恒铁律的单点检查）。"""
    ref = _enroll(CLAIMED, seed=6)
    ref_counts = ref.counts()
    for rnd in (0, 1, 2, 3):
        blue = BLUE_ROUNDS[rnd]
        auditor = BlueAuditor(blue, alpha=0.01, seed=100 + rnd)
        relay = BenignRelay(CLAIMED, seed=200 + rnd)
        out = auditor.audit_active(relay, ref_counts, {}, CLAIMED, max_queries=600)
        assert out.verdict == "PASS", f"轮{rnd} 误杀真身: {out.detail}"


def test_blue_detects_structural_swap_under_camouflage():
    """结构性整体替换（无分流）在任何伪装等级下都应被检出（伪装不该削弱对笨对手的检出）。"""
    ref = _enroll(CLAIMED, seed=7)
    blue = BLUE_ROUNDS[2]      # native 伪装
    auditor = BlueAuditor(blue, alpha=0.01, seed=321)
    relay = RedRelay(CLAIMED, policy="dilution", seed=42, eps=1.0)   # eps=1 => 全供廉价 = 整体替换
    out = auditor.audit_active(relay, ref.counts(), {}, CLAIMED, max_queries=600)
    assert out.detected


def test_probe_aware_escapes_raw_but_caught_under_camouflage():
    """协同进化核心事实：probe_aware 分流在裸探针下逃检，在表层伪装下被抓。"""
    ref = _enroll(CLAIMED, seed=11)
    ref_counts = ref.counts()
    # 裸探针（轮0 基线配置）
    raw_blue = BLUE_ROUNDS[0]
    caught_raw = 0
    for s in range(8):
        relay = make_endpoint("probe_aware", CLAIMED, seed=1000 + s)
        out = BlueAuditor(raw_blue, seed=1000 + s).audit_active(relay, ref_counts, {}, CLAIMED, max_queries=600)
        caught_raw += int(out.detected)
    # 表层伪装（轮1 蓝队升级）
    camo_blue = BLUE_ROUNDS[1]
    caught_camo = 0
    for s in range(8):
        relay = make_endpoint("probe_aware", CLAIMED, seed=1000 + s)
        out = BlueAuditor(camo_blue, seed=1000 + s).audit_active(relay, ref_counts, {}, CLAIMED, max_queries=600)
        caught_camo += int(out.detected)
    assert caught_raw == 0, "裸探针本应被 probe_aware 全部路由给旗舰而逃检"
    assert caught_camo >= 7, f"表层伪装后应基本全检出，实际 {caught_camo}/8"
