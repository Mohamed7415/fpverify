"""编排层：enroll（入册参考指纹）与 audit（序贯审计端点）。

audit 把探针流交给序贯下注 e-process 决策：明显的假会很快让财富越界（早停省 token），
明显的真会在预算内始终不越界（PASS）。同时并行采集一份端点指纹用于人类可读的
JSD 诊断与缓存筛查。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from . import probes
from .betting import BettingConfig, SequentialBettingTest
from .calibrate import calibrate_delta
from .distance import aggregate_distance
from .endpoints import Endpoint
from .fingerprint import Fingerprint
from .normalize import normalize, classify_validity
from .screens import screen_response_cache


@dataclass
class AuditResult:
    verdict: str                       # PASS / FAIL / SUSPECT / INCONCLUSIVE
    n_queries: int
    wealth: float
    peak_wealth: float
    threshold: float
    alpha: float
    delta: float
    aggregate_jsd: float | None
    per_cell_jsd: dict = field(default_factory=dict)
    cache_flags: list = field(default_factory=list)
    model_fields_seen: dict = field(default_factory=dict)
    errors: int = 0
    detail: str = ""
    observed_counts: dict = field(default_factory=dict)  # cell -> {答案: 次数}，供脱离本工具独立重算

    def to_dict(self):
        d = dict(self.__dict__)
        d["per_cell_jsd"] = {f"{k[0]}::{k[1]}": v for k, v in self.per_cell_jsd.items()}
        d["cache_flags"] = [f"{c.cell[0]}::{c.cell[1]}" for c in self.cache_flags]
        d["observed_counts"] = {f"{k[0]}::{k[1]}": dict(v) for k, v in self.observed_counts.items()}
        return d


class Verifier:
    def __init__(self, betting_cfg: BettingConfig | None = None, seed: int | None = None):
        self.cfg = betting_cfg or BettingConfig()
        self.rng = random.Random(seed)

    # ---------------------------------------------------------------- 入册
    def enroll(self, endpoint: Endpoint, model_name: str, cells: list | None = None,
               samples_per_cell: int = 20, progress=None) -> Fingerprint:
        """对可信端点采集参考指纹。"""
        cells = cells or probes.all_cells()
        fp = Fingerprint(model=model_name,
                         params={"temperature": 1.0, "max_tokens": 16})
        jobs = [(t, l) for (t, l) in cells for _ in range(samples_per_cell)]
        self.rng.shuffle(jobs)
        for i, (task, lang) in enumerate(jobs):
            system, user = probes.render_prompt(task, lang, self.rng)
            ans = endpoint.ask(system, user)
            if ans.error:
                continue
            ptype = probes.parse_type(task)
            fp.add((task, lang), normalize(ptype, ans.text),
                   classify_validity(ptype, ans.text), ans.latency)
            if progress and (i + 1) % 50 == 0:
                progress(i + 1, len(jobs))
        return fp

    # ---------------------------------------------------------------- 审计
    def audit(self, endpoint: Endpoint, reference: Fingerprint,
              max_queries: int = 600, samples_per_cell_cap: int = 40,
              cells: list | None = None, progress=None) -> AuditResult:
        """序贯审计端点是否为参考模型。"""
        ref_counts = reference.counts()
        cells = cells or list(ref_counts.keys())
        cells = [c for c in cells if sum(ref_counts.get(c, {}).values()) >= 10]
        if not cells:
            return AuditResult("INCONCLUSIVE", 0, 1.0, 1.0, 1.0 / self.cfg.alpha,
                               self.cfg.alpha, self.cfg.delta or 0.0, None,
                               detail="参考指纹样本不足，无法审计。")

        cfg = self.cfg
        if cfg.delta is None:
            # 数据驱动标定：吸收参考采样误差 + 良性部署漂移，保证不误杀真身
            delta = calibrate_delta({c: ref_counts[c] for c in cells}, cfg,
                                    horizon=max_queries)
            cfg = replace(cfg, delta=delta)

        test = SequentialBettingTest({c: ref_counts[c] for c in cells}, cfg)
        test_fp = Fingerprint(model="<endpoint-under-test>")
        model_fields = {}
        errors = 0

        # 轮转 cell（每个 cell 交替采样，避免打满单个任务），随机措辞
        per_cell_count = {c: 0 for c in cells}
        n = 0
        order = list(cells)
        self.rng.shuffle(order)
        idx = 0
        while n < max_queries:
            # 找到下一个还没到上限的 cell
            tries = 0
            while per_cell_count[order[idx % len(order)]] >= samples_per_cell_cap and tries < len(order):
                idx += 1
                tries += 1
            if tries >= len(order):
                break  # 所有 cell 都到上限
            cell = order[idx % len(order)]
            idx += 1

            task, lang = cell
            system, user = probes.render_prompt(task, lang, self.rng)
            ans = endpoint.ask(system, user)
            n += 1
            if ans.error:
                errors += 1
                if errors > max(20, n * 0.5):
                    return AuditResult("INCONCLUSIVE", n, test.wealth, test.peak_wealth,
                                       1.0 / cfg.alpha, cfg.alpha, cfg.delta,
                                       None, errors=errors, detail=f"错误过多：{ans.error}")
                continue

            ptype = probes.parse_type(task)
            tok = normalize(ptype, ans.text)
            test_fp.add(cell, tok, classify_validity(ptype, ans.text), ans.latency)
            if ans.model_field:
                model_fields[ans.model_field] = model_fields.get(ans.model_field, 0) + 1
            per_cell_count[cell] += 1

            test.observe(cell, tok)
            if progress and n % 25 == 0:
                progress(n, max_queries, test.wealth)
            if test.decided:
                break

        agg, per_cell = aggregate_distance(ref_counts, test_fp.counts(), min_samples=8)
        cache_flags = screen_response_cache(test_fp)

        if test.rejected:
            verdict = "FAIL"
            detail = "序贯检验拒绝了'端点是真身'的零假设：行为指纹显著偏离，判定注水/偷换。"
        else:
            # 未拒绝：区分"证据充分的通过"与"预算不足的存疑"
            if n >= 8 * len(cells) or n >= max_queries:
                verdict = "PASS"
                detail = "预算内始终未越界，未发现注水证据（在给定 α/δ 下与参考一致）。"
            else:
                verdict = "SUSPECT"
                detail = "查询量偏少且未定论，建议加大预算复测。"

        if cache_flags and verdict != "FAIL":
            verdict = "FAIL"
            detail += " 另检出疑似响应级缓存（服务层完整性违规，独立于指纹证据）。"

        return AuditResult(
            verdict=verdict, n_queries=n, wealth=test.wealth, peak_wealth=test.peak_wealth,
            threshold=1.0 / cfg.alpha, alpha=cfg.alpha, delta=cfg.delta,
            aggregate_jsd=agg, per_cell_jsd=per_cell, cache_flags=cache_flags,
            model_fields_seen=model_fields, errors=errors, detail=detail,
            observed_counts=test_fp.counts())
