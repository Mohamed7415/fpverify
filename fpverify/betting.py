"""决策核心：带容差的序贯下注 e-process（anytime-valid，严格控 FPR）。

参考 Richter et al. (ICLR 2025) 的 testing-by-betting，改造到类别（categorical）分布，
并适配本项目"验证真身"的语义：

  H0（零假设）：端点就是真身 X（答案 ~ 参考分布 p0，允许 e^δ 容差邻域内的良性波动）
  H1：端点被注水（分布显著偏离 p0）
  拒绝 H0  ==>  判定"注水/FAIL"

为什么这样设定 H0：把 **Type-I error = 把真模型误判为假** 放在被严格控制的一侧，
这正是用户最不能接受的错误（论文用户痛点）。Ville 不等式保证
  P_{H0}( 曾经 W_t >= 1/alpha ) <= alpha，
且**对任意可预测下注策略都成立**——所以我们在线估计端点分布 q̂ 的好坏只影响
检出力（power/省多少 token），不破坏 FPR 有效性。

下注机制（每观测一个答案 a，属于 cell c）：
  f_t = (1-lambda) + lambda * q̂_c(a) / p0_c(a)         # 混合 Kelly，f_t >= 1-lambda > 0
  W_t = W_{t-1} * f_t * exp(-delta)                     # delta 提供容差
在精确 H0（a ~ p0）下 E[f_t] = 1，故 E[W_t | 过去] <= W_{t-1} * e^{-delta} <= W_{t-1}，
W_t 是非负上鞅、W0 = 1 ==> anytime-valid。

q̂_c 只用"当前观测之前"的历史估计（保证可预测性/previsibility），冷启动用均匀分布。
p0_c 用参考指纹计数做加性平滑，支撑集 = 参考出现过的答案 ∪ {<other>}，
未见答案落到 <other>，其 p0 概率恒为正 —— 这既避免除零，也让"频繁给出参考里没有的答案"
成为强注水信号。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from .normalize import OTHER


@dataclass
class BettingConfig:
    alpha: float = 0.01          # 显著性水平 => FPR 上界；拒绝阈值 1/alpha
    # 容差（每次观测的对数余量），吸收参考采样误差与良性部署漂移。
    # None = 审计前用参考指纹自助法自动标定（推荐，见 calibrate.calibrate_delta）。
    delta: float | None = None
    lam: float = 0.5             # 混合 Kelly 系数 ∈(0,1]，越小越保守（方差越低）
    smoothing: float = 0.5       # 参考分布加性平滑
    q_smoothing: float = 0.5     # 在线估计 q̂ 的平滑
    max_answer_classes: int = 64 # 每 cell 参考支撑最多保留的高频类别，其余并入 <other>


class _RefModel:
    """单个 cell 的参考分布 p0（带平滑、<other> 兜底与 Good-Turing 缺失质量下限）。

    参考指纹是有限样本估计：真身端点偶尔会给出参考里没见过的答案。
    若 p0(<other>) 被低估，会把真模型误判成假（Type-I 膨胀）。
    Good-Turing 估计"没见过的答案"的真实概率 ≈ 单例数/样本数，
    我们据此给 <other> 一个保守下限，把估计误差吃进 p0 而不是吃进用户的误判率。
    """

    def __init__(self, ref_counts: Counter, cfg: BettingConfig):
        n_total = sum(ref_counts.values())
        singletons = sum(1 for _, c in ref_counts.items() if c == 1)
        items = ref_counts.most_common(cfg.max_answer_classes)
        self.support = set(k for k, _ in items if k != OTHER)
        self.support.add(OTHER)
        counts = Counter()
        for k, c in ref_counts.items():
            counts[k if k in self.support else OTHER] += c
        # Good-Turing 缺失质量（换算成计数单位）作为 <other> 的下限
        gt_floor = singletons if n_total > 0 else 0.0
        counts[OTHER] = max(counts.get(OTHER, 0), gt_floor, cfg.smoothing)
        self._counts = counts
        self.cfg = cfg
        self._total = sum(self._counts.values()) + cfg.smoothing * len(self.support)

    def p0(self, answer: str) -> float:
        a = answer if answer in self.support else OTHER
        return (self._counts[a] + self.cfg.smoothing) / self._total

    def canon(self, answer: str) -> str:
        return answer if answer in self.support else OTHER


class _OnlineEstimator:
    """端点分布 q̂ 的在线估计，只用历史（previsible）。冷启动=均匀。"""

    def __init__(self, ref: _RefModel, cfg: BettingConfig):
        self.ref = ref
        self.cfg = cfg
        self.counts = Counter()
        self.n = 0

    def q_hat(self, answer: str) -> float:
        a = self.ref.canon(answer)
        k = len(self.ref.support)
        return (self.counts[a] + self.cfg.q_smoothing) / (self.n + self.cfg.q_smoothing * k)

    def update(self, answer: str):
        self.counts[self.ref.canon(answer)] += 1
        self.n += 1


class SequentialBettingTest:
    """跨多个 cell 的序贯下注检验。

    用法：
      t = SequentialBettingTest(ref_counts_by_cell, cfg)
      for (cell, answer) in stream:
          t.observe(cell, answer)
          if t.decided: break
      verdict = t.verdict()   # "FAIL"（拒绝H0=注水）/ "PASS"（预算内未拒绝）
    """

    def __init__(self, ref_counts_by_cell: dict, cfg: BettingConfig | None = None):
        self.cfg = cfg or BettingConfig()
        if self.cfg.delta is None:
            raise ValueError("BettingConfig.delta 为 None：请先用 calibrate.calibrate_delta 标定，"
                             "或经 Verifier.audit（会自动标定）使用。")
        self.refs = {c: _RefModel(Counter(rc), self.cfg) for c, rc in ref_counts_by_cell.items()}
        self.est = {c: _OnlineEstimator(self.refs[c], self.cfg) for c in self.refs}
        self.log_w = 0.0            # log 财富，W0=1 => log_w=0
        self.log_threshold = math.log(1.0 / self.cfg.alpha)
        self.n_obs = 0
        self.peak_log_w = 0.0
        self.rejected = False
        self.history = []           # (n_obs, cell, answer, log_w)

    def observe(self, cell: CellKeyLike, answer: str):
        if cell not in self.refs:
            return  # 未入册的 cell 忽略
        ref = self.refs[cell]
        est = self.est[cell]
        p0 = ref.p0(answer)
        q = est.q_hat(answer)                       # 用更新前的历史 => previsible
        ratio = q / p0 if p0 > 0 else 1.0
        f = (1.0 - self.cfg.lam) + self.cfg.lam * ratio
        f = max(f, 1e-9)
        self.log_w += math.log(f) - self.cfg.delta
        est.update(answer)                          # 观测后再更新历史
        self.n_obs += 1
        self.peak_log_w = max(self.peak_log_w, self.log_w)
        self.history.append((self.n_obs, cell, answer, self.log_w))
        if self.log_w >= self.log_threshold:
            self.rejected = True

    @property
    def decided(self) -> bool:
        return self.rejected

    @property
    def wealth(self) -> float:
        return math.exp(self.log_w)

    @property
    def peak_wealth(self) -> float:
        return math.exp(self.peak_log_w)

    def verdict(self) -> str:
        return "FAIL" if self.rejected else "PASS"

    def summary(self) -> dict:
        return {
            "verdict": self.verdict(),
            "rejected_H0_is_swap": self.rejected,
            "n_observations": self.n_obs,
            "wealth": self.wealth,
            "peak_wealth": self.peak_wealth,
            "threshold": 1.0 / self.cfg.alpha,
            "alpha_fpr_bound": self.cfg.alpha,
            "delta_tolerance": self.cfg.delta,
        }


# 类型别名（仅用于可读性）
CellKeyLike = tuple
