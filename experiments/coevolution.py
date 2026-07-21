# -*- coding: utf-8 -*-
"""红蓝协同进化（co-evolution）：在 sim 受控沙盒里跑多轮中转站作弊 vs 检测的军备竞赛。

核心不是"能不能算分布距离"，而是**经济约束**下的可区分度博弈（见 docs/COEVOLUTION.md）：
红队只有在真实业务流量上供廉价模型才算省钱；任何"疑似测试就切旗舰"的策略，若代价是
把大量真实业务也误判成测试而供了旗舰，就等于没省钱（廉价流量占比 <20% => 规避无效）。

轮次（每轮：红队出新招 vs 当前蓝队 -> 蓝队升级 -> 复测；预算 ≤600）：
  轮0  基线：现有检测器（裸探针 + 序贯下注）对现有 9 类对手。
  轮1  红：probe_aware（按表层可区分度分流）  蓝：surface_camo（表层伪装探针 + 最近邻）
  轮2  红：semantic_aware（按语义可区分度分流）蓝：native_camo（业务原生伪装探针 + 最近邻）
  轮3  红：account_routing（按账号分流，外部一次性审计者恒被供旗舰）蓝：passive_mixin（被动混入式审计）
  轮4  收敛检验：红队在账号分流上做"账内反探针"最优反应，量化经济相变，判定稳定点。

评分：红队"赢"= 预算内检出率 <50% 且经济上划算（廉价占比≥20%）；
      蓝队"赢"= 检出率 ≥95% 且 honest/drift 的 FPR ≤ α。

运行:  py -3.13 -X utf8 experiments/coevolution.py
输出:  experiments/out/coevo_results.json, experiments/out/fig_coevo.png
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpverify.betting import BettingConfig, SequentialBettingTest
from fpverify.verifier import Verifier
from sim.adversaries import make_endpoint, BenignRelay, ADVERSARY_KINDS
from sim.blue_team import BlueAuditor, BlueConfig, BLUE_ROUNDS, calibrated_cfg, _cells_from_ref
from sim import traffic

OUT = ROOT / "experiments" / "out"
OUT.mkdir(parents=True, exist_ok=True)

CLAIMED = "gpt-4o"
CHEAP = "cheap-7b"
ALPHA = 0.01
ENROLL_SAMPLES = 25
BUDGET = 600
CANDIDATES = ["gpt-4o", "cheap-7b", "claude-sonnet-5", "llama-3.3-70b", "qwen3-235b"]
SAME_FAMILY = {"qwen3-235b": "qwen", "qwen3-max": "qwen"}

# 成本模型（与 run_evaluation 一致）：探针 45 输入 + 4 输出 token；旗舰定价 $2.5/M 入、$10/M 出
IN_TOK, OUT_TOK, PRICE_IN, PRICE_OUT = 45, 4, 2.5, 10.0
RED_WIN_DETECT = 0.50      # 红队赢：检出率 < 此值
BLUE_WIN_DETECT = 0.95     # 蓝队赢：检出率 >= 此值
VIABLE_CHEAP = 0.20        # 经济可行：廉价流量占比 >= 此值


def cost_usd(n_queries: float) -> float:
    return (n_queries * IN_TOK * PRICE_IN + n_queries * OUT_TOK * PRICE_OUT) / 1e6


# ---------------------------------------------------------------- 参考入册
def build_references(n_refs: int, base_seed: int = 700):
    """入册若干 honest gpt-4o 参考指纹，以及闭集候选模型指纹（供最近邻）。"""
    refs = []
    for r in range(n_refs):
        v = Verifier(seed=base_seed + r)
        trusted = make_endpoint("honest", CLAIMED, seed=(base_seed + r) * 7 + 1)
        refs.append(v.enroll(trusted, CLAIMED, samples_per_cell=ENROLL_SAMPLES))
    enrolled = {}
    for m in CANDIDATES:
        v = Verifier(seed=hash(m) % 10000)
        fp = v.enroll(make_endpoint("honest", m, seed=13), m, samples_per_cell=ENROLL_SAMPLES)
        enrolled[m] = fp.counts()
    return refs, enrolled


# ---------------------------------------------------------------- 主动检出率
def detection_active(blue: BlueConfig, red_kind: str, refs, enrolled,
                     n_audits: int, budget: int, base_seed: int,
                     audit_account=None, red_params=None):
    """在给定蓝队配置下，对某类分流红队跑主动审计，统计检出率/停时/成本。"""
    red_params = red_params or {}
    detected = 0
    total = 0
    stops = []
    nearest_hits = 0
    for ri, ref in enumerate(refs):
        ref_counts = ref.counts()
        for i in range(n_audits):
            seed = base_seed + ri * 1000 + i
            relay = make_endpoint(red_kind, CLAIMED, seed=seed, actual=CHEAP,
                                  account=audit_account, **red_params)
            auditor = BlueAuditor(blue, alpha=ALPHA, seed=seed + 5)
            out = auditor.audit_active(relay, ref_counts, enrolled, CLAIMED,
                                       max_queries=budget, same_family=SAME_FAMILY,
                                       audit_account=audit_account)
            total += 1
            if out.detected:
                detected += 1
                stops.append(out.n_queries)
            if out.nearest_flagged:
                nearest_hits += 1
    rate = detected / total if total else 0.0
    mean_stop = statistics.mean(stops) if stops else None
    return {
        "detection_rate": rate,
        "mean_queries_to_detect": mean_stop,
        "mean_cost_usd": cost_usd(mean_stop) if mean_stop else None,
        "nearest_flag_rate": nearest_hits / total if total else 0.0,
        "n": total,
    }


def detection_passive(blue: BlueConfig, red_kind: str, refs, enrolled,
                      accounts, n_audits: int, budget: int, base_seed: int,
                      red_params=None):
    """被动混入式审计（探针夹在真实老账号流量里）对账号级分流的检出率。"""
    red_params = red_params or {}
    detected = 0
    total = 0
    stops = []
    for ri, ref in enumerate(refs):
        ref_counts = ref.counts()
        for i in range(n_audits):
            seed = base_seed + ri * 1000 + i
            relay = make_endpoint(red_kind, CLAIMED, seed=seed, actual=CHEAP, **red_params)
            auditor = BlueAuditor(blue, alpha=ALPHA, seed=seed + 5)
            out = auditor.audit_passive(relay, ref_counts, enrolled, CLAIMED,
                                        accounts, max_queries=budget, same_family=SAME_FAMILY)
            total += 1
            if out.detected:
                detected += 1
                stops.append(out.n_queries)
    rate = detected / total if total else 0.0
    mean_stop = statistics.mean(stops) if stops else None
    return {"detection_rate": rate, "mean_queries_to_detect": mean_stop,
            "mean_cost_usd": cost_usd(mean_stop) if mean_stop else None, "n": total}


# ---------------------------------------------------------------- FPR 守恒
def fpr_check(blue: BlueConfig, refs, n_audits: int, base_seed: int):
    """蓝队配置对 honest / drift 两类良性端点的误杀率（必须 ≤ α）。"""
    out = {}
    for kind in ("honest", "drift"):
        fails = 0
        total = 0
        for ri, ref in enumerate(refs):
            ref_counts = ref.counts()
            for i in range(n_audits):
                seed = base_seed + ri * 1000 + i
                if kind == "honest":
                    relay = BenignRelay(CLAIMED, seed=seed)
                else:
                    relay = BenignRelay(CLAIMED, seed=seed, drift_seed=ri * 31 + i + 1)
                auditor = BlueAuditor(blue, alpha=ALPHA, seed=seed + 5)
                res = auditor.audit_active(relay, ref_counts, {}, CLAIMED,
                                           max_queries=BUDGET, same_family=SAME_FAMILY)
                total += 1
                if res.detected:
                    fails += 1
        out[kind] = {"n": total, "false_positive": fails, "fpr": fails / total if total else 0.0}
    return out


# ---------------------------------------------------------------- 经济：廉价流量占比
def cheap_fraction_for_policy(policy: str, seed: int, n: int = 20000,
                              accounts=None, params=None) -> float:
    """红队某策略在真实业务流量上的廉价供给占比（= 实际省到的钱）。"""
    from sim.adversaries import ROUTE_POLICIES
    params = params or {}
    fn = ROUTE_POLICIES[policy]
    rng = random.Random(seed)
    route = (lambda feats, acc: fn(feats, acc, **params))
    return traffic.served_cheap_fraction(route, n, rng, accounts=accounts)


# ---------------------------------------------------------------- 轮2 经济相变扫描
def phase_transition_semantic(refs, enrolled, seed: int = 4100):
    """扫描红队语义阈值：native 伪装探针的逃检率 vs 业务廉价占比。

    结论：要把 native 探针的逃检率抬高，必须把阈值压进业务分布，
    从而廉价占比坍塌到 <20% —— 经济上不划算。
    """
    thresholds = [1.2, 0.6, 0.0, -0.6, -1.1, -1.6, -2.2, -2.8]
    ref = refs[0]
    ref_counts = ref.counts()
    blue = BLUE_ROUNDS[2]      # native 伪装
    rows = []
    for thr in thresholds:
        # 逃检率：native 探针在该阈值下 semantic_aware 供旗舰的概率 -> 越高越逃得掉
        det = detection_active(blue, "semantic_aware", [ref], enrolled,
                               n_audits=15, budget=BUDGET, base_seed=seed + int(thr * 100),
                               red_params={"sem_thr": thr})
        escape = 1.0 - det["detection_rate"]
        cheap = cheap_fraction_for_policy("semantic_aware", seed=seed, params={"sem_thr": thr})
        rows.append({"sem_thr": thr, "probe_escape_rate": escape, "business_cheap_fraction": cheap})
    return rows


# ---------------------------------------------------------------- 轮3/4 账号级
def account_level_analysis(refs, enrolled, accounts, seed: int = 5200):
    """账号级分流：外部一次性审计（陌生账号）vs 被动混入式审计（老账号），及经济性。"""
    ref = refs[0]
    ref_counts = ref.counts()

    fresh = {"account_id": "auditor-fresh", "established": False, "volume": 1, "age_days": 0}
    # 外部一次性审计（native 伪装 + 最近邻，但用陌生账号）-> 被恒供旗舰而逃检
    ext = detection_active(BLUE_ROUNDS[2], "account_routing", refs, enrolled,
                           n_audits=20, budget=BUDGET, base_seed=seed,
                           audit_account=fresh)
    # 被动混入式审计（探针夹进老账号自然流量）-> 检出
    passive = detection_passive(BLUE_ROUNDS[3], "account_routing", refs, enrolled,
                                accounts, n_audits=20, budget=BUDGET, base_seed=seed + 777)
    # 经济性：账号级分流的（按业务量加权）廉价占比
    cheap = cheap_fraction_for_policy("account_routing", seed=seed, accounts=accounts)
    return {"external_oneshot": ext, "passive_mixin": passive,
            "account_weighted_cheap_fraction": cheap}


def _dilution_detection(ref_counts, cells, cfg, eps, budget, trials, seed):
    """稀释攻击的序贯检出率：对每次观测以概率 eps 用廉价源、否则真身源。"""
    from sim.mock_models import MockModel
    from fpverify import probes as probes_mod
    from fpverify.normalize import normalize
    gen, chp = MockModel(CLAIMED), MockModel(CHEAP)
    detected = 0
    stops = []
    for tt in range(trials):
        rng = random.Random(seed + 17 * tt + int(eps * 1000))
        test = SequentialBettingTest({c: ref_counts[c] for c in cells}, cfg)
        order = list(cells)
        rng.shuffle(order)
        idx = 0
        while test.n_obs < budget and not test.decided:
            cell = order[idx % len(order)]
            idx += 1
            task, lang = cell
            src = chp if rng.random() < eps else gen
            raw = src.sample(task, lang, rng)
            test.observe(cell, normalize(probes_mod.parse_type(task), src.format_answer(task, raw, rng)))
        if test.rejected:
            detected += 1
            stops.append(test.n_obs)
    return detected / trials, (statistics.mean(stops) if stops else None)


def _drift_fpr_at_delta(ref_counts, cells, delta, trials, seed):
    """给定固定 δ，同模型跨部署漂移端点的误杀率（用于展示'调小 δ 会破坏 FPR'）。"""
    from sim.mock_models import MockModel
    from fpverify import probes as probes_mod
    from fpverify.normalize import normalize
    cfg = BettingConfig(alpha=ALPHA, delta=delta)
    fails = 0
    for tt in range(trials):
        rng = random.Random(seed + tt)
        dm = MockModel(CLAIMED, drift_seed=tt + 1)
        test = SequentialBettingTest({c: ref_counts[c] for c in cells}, cfg)
        order = list(cells)
        rng.shuffle(order)
        idx = 0
        while test.n_obs < BUDGET and not test.decided:
            cell = order[idx % len(order)]
            idx += 1
            task, lang = cell
            raw = dm.sample(task, lang, rng)
            test.observe(cell, normalize(probes_mod.parse_type(task), dm.format_answer(task, raw, rng)))
        fails += int(test.rejected)
    return fails / trials


def round4_convergence(refs, enrolled, accounts, seed: int = 6300):
    """收敛检验（军备竞赛稳定点）：内容/账号都不看的**随机稀释攻击**。

    红队对全部流量以概率 eps 供廉价、否则供旗舰。这使伪装/最近邻/被动审计全部失效
    （探针与业务被一视同仁），省到的钱恰好 = eps。它唯一的对手是序贯检验的容差 δ：
      1) 单次有限预算审计（δ 由 FPR 守恒标定）只有当 eps 超过"容差带边缘"ε* 才检出；
         扫描 eps，定位"逃检（<50% 检出）且经济可行（eps≥20%）"的红队安全窗口；
      2) 蓝队想调小 δ 来压低 ε*，会立刻在良性 drift 端点上制造误报（破坏 FPR 守恒铁律）；
         扫描 δ，给出 (drift FPR, 稀释检出率) 的权衡，证明"调小 δ"不是合法对策；
      3) 唯一正解是长程累积被动审计：证据随样本量累积，检出小 eps 需 ~O(1/eps^2) 样本
         —— 用检出所需累积预算体现（同一 δ 下加大预算对 eps<ε* 无效，必须靠分布累积检验，
         这里用"更大预算 + δ 归零后的可分性"作为可分性下界的示意）。
    """
    # 每个参考的标定 δ 差异很大（取决于该次入册样本的噪声）——这本身决定了"容差带"宽窄
    deltas = []
    for ref in refs:
        rc = ref.counts()
        cells = _cells_from_ref(rc)
        deltas.append({"delta": round(calibrated_cfg(rc, cells, ALPHA, BUDGET).delta, 4)})

    # (1) eps 扫描：单次审计检出率（在全部参考上平均，与逐轮台账口径一致）+ 经济性
    eps_rows = []
    for eps in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        dets = []
        stops = []
        for ref in refs:
            rc = ref.counts()
            cells = _cells_from_ref(rc)
            cfg = calibrated_cfg(rc, cells, ALPHA, BUDGET)
            d, ms = _dilution_detection(rc, cells, cfg, eps, BUDGET, 30, seed)
            dets.append(d)
            if ms is not None:
                stops.append(ms)
        det = statistics.mean(dets)
        eps_rows.append({
            "eps_cheap_fraction": eps,           # = 省到的钱
            "detection_rate_single_audit": det,
            "mean_queries_to_detect": (statistics.mean(stops) if stops else None),
            "red_win": det < RED_WIN_DETECT and eps >= VIABLE_CHEAP,
        })

    # (2)(3) δ 权衡与累积预算：用第 0 个参考（保守标定 δ）演示机制
    ref0 = refs[0].counts()
    cells0 = _cells_from_ref(ref0)
    cfg0 = calibrated_cfg(ref0, cells0, ALPHA, BUDGET)

    # (2) δ 权衡：调小 δ 能压低 ε*、检出稀释，但 δ→0 会破坏 drift FPR
    delta_rows = []
    for delta in [round(cfg0.delta, 4), 0.08, 0.05, 0.03, 0.0]:
        drift_fpr = _drift_fpr_at_delta(ref0, cells0, delta, 120, seed + 500)
        det_eps30, _ = _dilution_detection(ref0, cells0,
                                           BettingConfig(alpha=ALPHA, delta=delta),
                                           0.3, BUDGET, 40, seed + 900)
        delta_rows.append({
            "delta": delta,
            "drift_fpr": drift_fpr,
            "fpr_ok": drift_fpr <= ALPHA,
            "detection_rate_eps0.3": det_eps30,
        })

    # (3) 累积预算：同一标定 δ 下，加大**单次审计**预算对 eps<ε* 无效
    # （anytime-valid 检验在容差带内财富为负漂移，加长不越界；只能靠长程分布累积检验）
    budget_rows = []
    for budget in [600, 1800, 4800]:
        det, _ = _dilution_detection(ref0, cells0, cfg0, 0.3, budget, 30, seed + 1300)
        budget_rows.append({"budget": budget, "detection_rate_eps0.3": det})

    # (4) 合法的 FPR 安全杠杆：加大入册样本 -> 收紧后验 -> 缩小容差带 -> 压低 ε*（但不清零）
    enroll_rows = []
    for ns in [25, 50, 100]:
        vv = Verifier(seed=700)
        fp = vv.enroll(make_endpoint("honest", CLAIMED, seed=4901), CLAIMED, samples_per_cell=ns)
        rc = fp.counts()
        cc = _cells_from_ref(rc)
        cfg = calibrated_cfg(rc, cc, ALPHA, BUDGET)
        d20, _ = _dilution_detection(rc, cc, cfg, 0.2, BUDGET, 30, seed + 1700)
        d30, _ = _dilution_detection(rc, cc, cfg, 0.3, BUDGET, 30, seed + 1700)
        enroll_rows.append({"enroll_samples_per_cell": ns, "delta": round(cfg.delta, 4),
                            "detection_eps0.2": d20, "detection_eps0.3": d30})

    return {"delta_across_refs": deltas, "eps_scan": eps_rows,
            "delta_tradeoff": delta_rows, "budget_scan": budget_rows,
            "enroll_sensitivity": enroll_rows}


# ---------------------------------------------------------------- 轮0 基线
def baseline_round(refs, n_audits: int = 20, fpr_audits: int = 70, base_seed: int = 800):
    """现有检测器（Verifier.audit，裸探针）对现有 9 类对手 + honest/drift FPR。"""
    kinds = [k for k in ADVERSARY_KINDS if k not in ("honest", "drift")]
    power = {}
    for kind in kinds:
        detected = 0
        total = 0
        stops = []
        jsds = []
        for ri, ref in enumerate(refs):
            for i in range(n_audits):
                seed = base_seed + ri * 1000 + i
                v = Verifier(BettingConfig(alpha=ALPHA, delta=None), seed=seed)
                ep = make_endpoint(kind, CLAIMED, seed=seed)
                res = v.audit(ep, ref, max_queries=BUDGET)
                total += 1
                if res.verdict == "FAIL":
                    detected += 1
                    stops.append(res.n_queries)
                if res.aggregate_jsd is not None:
                    jsds.append(res.aggregate_jsd)
        mean_stop = statistics.mean(stops) if stops else None
        power[kind] = {
            "detection_rate": detected / total,
            "mean_queries_to_detect": mean_stop,
            "mean_cost_usd": cost_usd(mean_stop) if mean_stop else None,
            "median_jsd": statistics.median(jsds) if jsds else None,
            "n": total,
        }
    # FPR：现有检测器
    fpr = {}
    for kind in ("honest", "drift"):
        fails = 0
        total = 0
        for ri, ref in enumerate(refs):
            for i in range(fpr_audits):
                seed = base_seed + 50_000 + ri * 1000 + i
                v = Verifier(BettingConfig(alpha=ALPHA, delta=None), seed=seed)
                ep = make_endpoint(kind, CLAIMED, seed=seed,
                                   **({"drift_seed": ri * 13 + i + 1} if kind == "drift" else {}))
                res = v.audit(ep, ref, max_queries=BUDGET)
                total += 1
                if res.verdict == "FAIL":
                    fails += 1
        fpr[kind] = {"n": total, "false_positive": fails, "fpr": fails / total}
    return {"power": power, "fpr": fpr}


# ---------------------------------------------------------------- 组织轮次
def scored(detection_rate, cheap_fraction, fpr_ok):
    red_win = detection_rate < RED_WIN_DETECT and (cheap_fraction is None or cheap_fraction >= VIABLE_CHEAP)
    blue_win = detection_rate >= BLUE_WIN_DETECT and fpr_ok
    return {"red_win": bool(red_win), "blue_win": bool(blue_win)}


def run_rounds(refs, enrolled, accounts, n_audits=20, n_fpr=40, seed=3000):
    rounds = []

    # ---- 轮1：probe_aware ----
    old_blue, new_blue = BLUE_ROUNDS[0], BLUE_ROUNDS[1]
    det_old = detection_active(old_blue, "probe_aware", refs, enrolled, n_audits, BUDGET, seed + 100)
    det_new = detection_active(new_blue, "probe_aware", refs, enrolled, n_audits, BUDGET, seed + 150)
    cheap = cheap_fraction_for_policy("probe_aware", seed=seed + 1)
    fpr_new = fpr_check(new_blue, refs, n_fpr, seed + 170)
    fpr_ok = all(v["fpr"] <= ALPHA for v in fpr_new.values())
    rounds.append({
        "round": 1,
        "red_move": {"kind": "probe_aware",
                     "idea": "按表层可区分度分流：像审计探针的短问句/单值提示供旗舰，其余供廉价",
                     "cheap_fraction": cheap},
        "blue_old": old_blue.name, "blue_new": new_blue.name,
        "blue_move": {"name": new_blue.name, "note": new_blue.note},
        "detection_vs_old_blue": det_old,
        "detection_vs_new_blue": det_new,
        "fpr_new_blue": fpr_new,
        "score_red_vs_old": scored(det_old["detection_rate"], cheap, True),
        "score_blue_after_upgrade": scored(det_new["detection_rate"], cheap, fpr_ok),
    })

    # ---- 轮2：semantic_aware ----
    old_blue, new_blue = BLUE_ROUNDS[1], BLUE_ROUNDS[2]
    det_old = detection_active(old_blue, "semantic_aware", refs, enrolled, n_audits, BUDGET, seed + 200)
    det_new = detection_active(new_blue, "semantic_aware", refs, enrolled, n_audits, BUDGET, seed + 250)
    cheap = cheap_fraction_for_policy("semantic_aware", seed=seed + 2)
    fpr_new = fpr_check(new_blue, refs, n_fpr, seed + 270)
    fpr_ok = all(v["fpr"] <= ALPHA for v in fpr_new.values())
    rounds.append({
        "round": 2,
        "red_move": {"kind": "semantic_aware",
                     "idea": "改按语义可区分度分流：认出'说个1-100/颜色'这类低熵琐碎问题供旗舰",
                     "cheap_fraction": cheap},
        "blue_old": old_blue.name, "blue_new": new_blue.name,
        "blue_move": {"name": new_blue.name, "note": new_blue.note},
        "detection_vs_old_blue": det_old,
        "detection_vs_new_blue": det_new,
        "fpr_new_blue": fpr_new,
        "score_red_vs_old": scored(det_old["detection_rate"], cheap, True),
        "score_blue_after_upgrade": scored(det_new["detection_rate"], cheap, fpr_ok),
    })

    # ---- 轮3：account_routing ----
    acc = account_level_analysis(refs, enrolled, accounts, seed + 300)
    fpr_new = fpr_check(BLUE_ROUNDS[3], refs, n_fpr, seed + 370)
    fpr_ok = all(v["fpr"] <= ALPHA for v in fpr_new.values())
    det_old = acc["external_oneshot"]           # 老蓝队（内容审计，陌生账号）-> 逃检
    det_new = acc["passive_mixin"]              # 新蓝队（被动混入）-> 检出
    cheap = acc["account_weighted_cheap_fraction"]
    rounds.append({
        "round": 3,
        "red_move": {"kind": "account_routing",
                     "idea": "放弃按内容识别探针，改按账号分流：老客户/高流量账号供廉价，陌生账号供旗舰",
                     "cheap_fraction": cheap},
        "blue_old": BLUE_ROUNDS[2].name, "blue_new": BLUE_ROUNDS[3].name,
        "blue_move": {"name": BLUE_ROUNDS[3].name, "note": BLUE_ROUNDS[3].note},
        "detection_vs_old_blue": det_old,
        "detection_vs_new_blue": det_new,
        "fpr_new_blue": fpr_new,
        "score_red_vs_old": scored(det_old["detection_rate"], cheap, True),
        "score_blue_after_upgrade": scored(det_new["detection_rate"], cheap, fpr_ok),
        "note": "被动混入式审计能否成立，取决于审计方能否把探针注入'老账号'的自然流量；"
                "纯外部一次性审计者只有陌生账号，无法做到 —— 见收敛分析的原理性极限。",
    })

    # ---- 轮4：dilution（随机稀释，军备竞赛稳定点）----
    r4_eps = 0.25
    det_native = detection_active(BLUE_ROUNDS[3], "dilution", refs, enrolled, n_audits, BUDGET,
                                  seed + 400, red_params={"eps": r4_eps})
    cheap = cheap_fraction_for_policy("dilution", seed=seed + 4, params={"eps": r4_eps})
    fpr_new = fpr_check(BLUE_ROUNDS[3], refs, n_fpr, seed + 470)
    fpr_ok = all(v["fpr"] <= ALPHA for v in fpr_new.values())
    rounds.append({
        "round": 4,
        "red_move": {"kind": "dilution", "eps": r4_eps,
                     "idea": "内容/账号都不看，对全部流量以概率 eps=0.25 供廉价、否则供旗舰："
                             "伪装/最近邻/被动审计全部失效，省到的钱=eps=25%",
                     "cheap_fraction": cheap},
        "blue_old": BLUE_ROUNDS[3].name, "blue_new": BLUE_ROUNDS[3].name + "+cumulative",
        "blue_move": {"name": "cumulative_passive",
                      "note": "无法靠单次审计或调小 δ（会破坏 drift FPR）反制；"
                              "只能长程累积被动样本做分布检验，检出 eps 需 ~O(1/eps^2) 样本"},
        "detection_vs_old_blue": det_native,       # = 被动/伪装最强蓝队对稀释仍逃检
        "detection_vs_new_blue": det_native,       # 单次审计下无提升（见 round4 分析的 δ/预算扫描）
        "fpr_new_blue": fpr_new,
        "score_red_vs_old": scored(det_native["detection_rate"], cheap, True),
        "score_blue_after_upgrade": scored(det_native["detection_rate"], cheap, fpr_ok),
        "note": "这是稳定点：红队以 eps∈[0.2,0.38] 稀释，单次有限预算审计无法在守住 FPR 的前提下检出。"
                "详见 round4_convergence 的 eps/δ/预算扫描。",
    })

    return rounds


# ---------------------------------------------------------------- 图
def plot_all(baseline, rounds, phase_rows, round4_rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) 逐轮：升级前 vs 升级后 检出率
    ax = axes[0, 0]
    rlabels = [f"R{r['round']}\n{r['red_move']['kind']}" for r in rounds]
    old = [r["detection_vs_old_blue"]["detection_rate"] for r in rounds]
    new = [r["detection_vs_new_blue"]["detection_rate"] for r in rounds]
    x = range(len(rounds))
    w = 0.38
    ax.bar([i - w / 2 for i in x], old, w, label="vs old blue (before upgrade)", color="indianred")
    ax.bar([i + w / 2 for i in x], new, w, label="vs new blue (after upgrade)", color="steelblue")
    ax.axhline(BLUE_WIN_DETECT, color="green", ls="--", lw=1, label="blue win >=0.95")
    ax.axhline(RED_WIN_DETECT, color="orange", ls=":", lw=1, label="red win <0.50")
    ax.set_xticks(list(x), rlabels, fontsize=8)
    ax.set_ylabel("detection rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Co-evolution: detection before vs after each blue upgrade")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (0,1) 红队经济性：每轮廉价流量占比
    ax = axes[0, 1]
    cheaps = [r["red_move"]["cheap_fraction"] for r in rounds]
    colors = ["indianred" if c >= VIABLE_CHEAP else "gray" for c in cheaps]
    ax.bar(rlabels, cheaps, color=colors)
    ax.axhline(VIABLE_CHEAP, color="black", ls="--", lw=1, label="viability >=0.20")
    ax.set_ylabel("cheap traffic fraction (money saved)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Red economy: cheap fraction per round")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (1,0) 轮2 经济相变：语义阈值 -> 逃检率 vs 廉价占比
    ax = axes[1, 0]
    thr = [row["sem_thr"] for row in phase_rows]
    esc = [row["probe_escape_rate"] for row in phase_rows]
    chp = [row["business_cheap_fraction"] for row in phase_rows]
    ax.plot(thr, esc, "-o", color="indianred", label="native probe escape rate")
    ax.plot(thr, chp, "-s", color="steelblue", label="business cheap fraction")
    ax.axhline(VIABLE_CHEAP, color="black", ls="--", lw=1, label="viability 0.20")
    ax.set_xlabel("red semantic threshold (lower = catch more probes)")
    ax.set_ylabel("rate / fraction")
    ax.set_title("Round 2 economic phase transition")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,1) 轮4 稳定点：随机稀释 eps -> 单次审计检出率 vs 省到的钱(=eps)；红队安全窗口
    ax = axes[1, 1]
    eps_rows = round4_rows["eps_scan"]
    eps = [row["eps_cheap_fraction"] for row in eps_rows]
    det = [row["detection_rate_single_audit"] for row in eps_rows]
    ax.plot(eps, det, "-o", color="indianred", label="single-audit detection")
    ax.plot(eps, eps, "-s", color="steelblue", label="money saved (=eps)")
    ax.axhline(RED_WIN_DETECT, color="orange", ls=":", lw=1, label="red win <0.50")
    ax.axvline(VIABLE_CHEAP, color="black", ls="--", lw=1, label="viability eps=0.20")
    # 红队安全窗口：eps>=0.2 且 检出<0.5
    safe = [e for e, d in zip(eps, det) if e >= VIABLE_CHEAP and d < RED_WIN_DETECT]
    if safe:
        ax.axvspan(min(safe), max(safe), color="orange", alpha=0.15)
    ax.set_xlabel("dilution eps (cheap fraction on ALL traffic)")
    ax.set_ylabel("rate")
    ax.set_title("Round 4 stable point: dilution attack (shaded=red safe window)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Behavioral-fingerprint Red vs Blue Co-evolution (sim sandbox)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "fig_coevo.png", dpi=150)
    plt.close(fig)
    print(f"图已保存: {OUT / 'fig_coevo.png'}")


# ---------------------------------------------------------------- 主
def main():
    t0 = time.time()
    random.seed(20260721)
    print("[coevo] 入册参考指纹 ...")
    refs, enrolled = build_references(n_refs=3)
    accounts = traffic.sample_accounts(200, random.Random(9001))

    print("[轮0] 基线：现有检测器 vs 现有 9 类对手 ...")
    baseline = baseline_round(refs, n_audits=15)
    for k, v in baseline["power"].items():
        print(f"   {k:<14} 检出率={v['detection_rate']:.3f} 停时={v['mean_queries_to_detect'] and round(v['mean_queries_to_detect'],1)}")
    for k, v in baseline["fpr"].items():
        print(f"   FPR {k:<8} = {v['fpr']:.4f}  ({v['false_positive']}/{v['n']})")

    print("[轮1-3] 协同进化 ...")
    rounds = run_rounds(refs, enrolled, accounts, n_audits=20, n_fpr=40)
    for r in rounds:
        print(f"   R{r['round']} {r['red_move']['kind']}: "
              f"vs旧蓝检出={r['detection_vs_old_blue']['detection_rate']:.2f} "
              f"廉价占比={r['red_move']['cheap_fraction']:.2f} -> "
              f"vs新蓝检出={r['detection_vs_new_blue']['detection_rate']:.2f} "
              f"红赢(旧)={r['score_red_vs_old']['red_win']} 蓝赢(升级后)={r['score_blue_after_upgrade']['blue_win']}")

    print("[轮2] 经济相变扫描 ...")
    phase_rows = phase_transition_semantic(refs, enrolled)
    print("[轮4] 收敛检验（账内反探针经济相变）...")
    round4_rows = round4_convergence(refs, enrolled, accounts)

    plot_all(baseline, rounds, phase_rows, round4_rows)

    results = {
        "config": {"alpha": ALPHA, "budget": BUDGET, "enroll_samples_per_cell": ENROLL_SAMPLES,
                   "claimed": CLAIMED, "cheap": CHEAP, "candidates": CANDIDATES,
                   "scoring": {"red_win_detect_below": RED_WIN_DETECT,
                               "blue_win_detect_above": BLUE_WIN_DETECT,
                               "viable_cheap_fraction": VIABLE_CHEAP},
                   "traffic_model": {"biz_niche_frac": traffic.BIZ_NICHE_FRAC,
                                     "probe_features": traffic.PROBE_FEATURES}},
        "round0_baseline": baseline,
        "rounds": rounds,
        "round2_phase_transition": phase_rows,
        "round4_convergence": round4_rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "coevo_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"完成，用时 {results['elapsed_sec']}s。结果: {OUT / 'coevo_results.json'}")


if __name__ == "__main__":
    main()
