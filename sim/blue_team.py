"""蓝队检测编排（协同进化专用）：把每轮的检测能力打包成一个可运行的审计器。

设计目标：在**不改动 fpverify 公共 API、不改坏现有测试**的前提下，把协同进化里
每一轮蓝队的"升级"表达为一组可组合的检测配置：

  camo            探针伪装等级（raw/surface/native）——决定探针有多像业务流量，
                  从而决定分流红队能不能把探针择出来单独供旗舰；
  use_nearest     是否启用闭集最近邻辅助判据（fpverify.nearest）；
  passive         是否启用被动/混入式审计（把探针混进真实账号的自然流量里发，
                  规避账号级分流）——这是对 account_routing 的原理性反制。

审计流程复用 fpverify 的序贯下注 e-process（决策核心不动，FPR 保证不变）：
主动路径自己驱动 SequentialBettingTest（因为要控制探针伪装与账号上下文），
δ 仍由 calibrate.calibrate_delta 标定，honest/drift 的 FPR 守恒由实验实测。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from fpverify import probes as probes_mod
from fpverify.betting import BettingConfig, SequentialBettingTest
from fpverify.calibrate import calibrate_delta
from fpverify.distance import aggregate_distance
from fpverify.fingerprint import Fingerprint
from fpverify.nearest import nearest_model
from fpverify.normalize import normalize
from . import traffic


@dataclass
class BlueConfig:
    """一轮蓝队的检测配置。"""
    name: str
    camo: str = "raw"                 # 探针伪装等级：raw / surface / native
    use_nearest: bool = False         # 闭集最近邻辅助判据
    passive: bool = False             # 被动/混入式审计（对抗账号级分流）
    nearest_margin: float = 0.05
    note: str = ""


@dataclass
class AuditOutcome:
    verdict: str                      # FAIL（检出）/ PASS / SUSPECT
    n_queries: int
    wealth: float
    peak_wealth: float
    rejected_betting: bool            # 序贯检验是否拒绝
    nearest_flagged: bool             # 最近邻是否判异族
    nearest_info: dict = field(default_factory=dict)
    aggregate_jsd: float | None = None
    detail: str = ""

    @property
    def detected(self) -> bool:
        return self.verdict == "FAIL"


def _cells_from_ref(ref_counts: dict, min_samples: int = 10) -> list:
    return [c for c, cnt in ref_counts.items() if sum(cnt.values()) >= min_samples]


def calibrated_cfg(ref_counts: dict, cells: list, alpha: float, horizon: int,
                   seed: int = 20260721) -> BettingConfig:
    """按参考指纹标定 δ，返回可用于 SequentialBettingTest 的配置。"""
    base = BettingConfig(alpha=alpha, delta=0.0)
    delta = calibrate_delta({c: ref_counts[c] for c in cells}, base,
                            horizon=horizon, seed=seed)
    return BettingConfig(alpha=alpha, delta=delta)


class BlueAuditor:
    """按 BlueConfig 组织一次审计。可对任意 Endpoint 或 RedRelay 运行。

    关键点：审计器通过 traffic.render_probe 按 camo 等级生成探针文本并采样，
    因此分流红队看到的探针"可区分度"由 camo 决定。对 RedRelay 我们直接调用其
    serve() 拿到 (答案, 路由决策)，既真实又能精确记录逃检情况。
    """

    def __init__(self, blue: BlueConfig, alpha: float = 0.01, seed: int = 0):
        self.blue = blue
        self.alpha = alpha
        self.rng = random.Random(seed)

    # ----- 主动审计（自建账号，逐条发探针）-----
    def audit_active(self, relay, ref_counts: dict, enrolled: dict, claimed: str,
                     max_queries: int = 600, samples_cap: int = 40,
                     same_family: dict | None = None,
                     audit_account=None) -> AuditOutcome:
        cells = _cells_from_ref(ref_counts)
        cfg = calibrated_cfg(ref_counts, cells, self.alpha, max_queries)
        test = SequentialBettingTest({c: ref_counts[c] for c in cells}, cfg)
        test_fp = Fingerprint(model="<endpoint-under-test>")

        order = list(cells)
        self.rng.shuffle(order)
        per_cell = {c: 0 for c in cells}
        idx = 0
        n = 0
        while n < max_queries:
            tries = 0
            while per_cell[order[idx % len(order)]] >= samples_cap and tries < len(order):
                idx += 1
                tries += 1
            if tries >= len(order):
                break
            cell = order[idx % len(order)]
            idx += 1
            task, lang = cell
            base_sys, base_user = probes_mod.render_prompt(task, lang, self.rng)
            req = traffic.render_probe(task, lang, self.blue.camo, self.rng,
                                       base_sys, base_user, account=audit_account)
            tok = self._serve_and_normalize(relay, req)
            test_fp.add(cell, tok, "valid", None)
            per_cell[cell] += 1
            n += 1
            test.observe(cell, tok)
            if test.decided:
                break

        return self._finalize(test, test_fp, ref_counts, enrolled, claimed, n, same_family)

    # ----- 被动/混入式审计（把探针混进真实账号自然流量）-----
    def audit_passive(self, relay, ref_counts: dict, enrolled: dict, claimed: str,
                      accounts: list, max_queries: int = 600,
                      same_family: dict | None = None) -> AuditOutcome:
        """蓝队轮3 反制账号级分流：不新建账号，而是在**已有真实账号**的自然流量里
        夹带探针（native 伪装）。这样分流红队若按账号供廉价，就会把探针也供廉价而暴露；
        若对该账号供旗舰，则相当于对该账号整体放弃省钱。
        """
        cells = _cells_from_ref(ref_counts)
        cfg = calibrated_cfg(ref_counts, cells, self.alpha, max_queries)
        test = SequentialBettingTest({c: ref_counts[c] for c in cells}, cfg)
        test_fp = Fingerprint(model="<endpoint-under-test>")
        # 选一批"老客户"账号作为混入宿主（真实世界里相当于收买/自有高龄账号）
        hosts = [a for a in accounts if a.get("established")] or accounts
        order = list(cells)
        self.rng.shuffle(order)
        n = 0
        idx = 0
        while n < max_queries:
            cell = order[idx % len(order)]
            idx += 1
            task, lang = cell
            base_sys, base_user = probes_mod.render_prompt(task, lang, self.rng)
            host = self.rng.choice(hosts)
            req = traffic.render_probe(task, lang, "native", self.rng,
                                       base_sys, base_user, account=host)
            tok = self._serve_and_normalize(relay, req)
            test_fp.add(cell, tok, "valid", None)
            n += 1
            test.observe(cell, tok)
            if test.decided:
                break
        return self._finalize(test, test_fp, ref_counts, enrolled, claimed, n, same_family)

    # ----- 内部 -----
    def _serve_and_normalize(self, relay, req: traffic.Request) -> str:
        if hasattr(relay, "serve"):
            text, _ = relay.serve(req.task, req.lang, req.features, account=req.account)
        else:
            ans = relay.ask(req.system, req.user)
            text = ans.text
        return normalize(probes_mod.parse_type(req.task), text)

    def _finalize(self, test, test_fp, ref_counts, enrolled, claimed, n,
                  same_family) -> AuditOutcome:
        agg, _ = aggregate_distance(ref_counts, test_fp.counts(), min_samples=8)
        nearest_flagged = False
        nearest_info = {}
        if self.blue.use_nearest and enrolled:
            nr = nearest_model(test_fp.counts(), enrolled, claimed,
                               margin_threshold=self.blue.nearest_margin,
                               same_family=same_family)
            nearest_flagged = nr.flagged
            nearest_info = nr.to_dict()

        rejected = test.rejected
        detected = rejected or nearest_flagged
        verdict = "FAIL" if detected else "PASS"
        if detected:
            why = []
            if rejected:
                why.append("序贯下注拒绝 H0")
            if nearest_flagged:
                why.append("闭集最近邻判异族")
            detail = "；".join(why)
        else:
            detail = "预算内未越界且最近邻一致，未发现注水证据"
        return AuditOutcome(
            verdict=verdict, n_queries=n, wealth=test.wealth, peak_wealth=test.peak_wealth,
            rejected_betting=rejected, nearest_flagged=nearest_flagged,
            nearest_info=nearest_info, aggregate_jsd=agg, detail=detail)


# 协同进化的蓝队升级路线（逐轮）
BLUE_ROUNDS = {
    0: BlueConfig(name="baseline", camo="raw", use_nearest=False, passive=False,
                  note="现有检测器：裸探针 + 序贯下注"),
    1: BlueConfig(name="surface_camo", camo="surface", use_nearest=True, passive=False,
                  note="表层伪装探针（去单值提示/加长）+ 闭集最近邻"),
    2: BlueConfig(name="native_camo", camo="native", use_nearest=True, passive=False,
                  note="业务原生伪装探针（表层+语义都像业务）+ 最近邻"),
    3: BlueConfig(name="passive_mixin", camo="native", use_nearest=True, passive=True,
                  note="被动混入式审计（探针夹在真实账号流量里）对抗账号级分流"),
}
