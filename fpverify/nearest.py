"""闭集最近邻识别（蓝队增强，独立于现有公共 API 的新模块）。

序贯下注检验回答的是"端点是不是 claimed 模型 X"这个二元问题。但当中转站
把探针路由给真身、只在别处供便宜货时，主动审计到的分布可能仍贴近 X。
闭集最近邻换一个问题问：**在一批已入册的候选模型里，被测端点的行为最像谁？**

若最近邻不是 claimed 模型，而是某个廉价模型，这本身就是强归属证据
（对应研究笔记 §4.2 的谱系恢复 / 论文 Palmyra≈Qwen 案例）。它不改变
序贯检验的 FPR 保证，只作为一路**辅助判据**：仅在"最近邻是异族且领先足够多"
时才提示，honest/drift 端点的最近邻仍是自己，不会制造误报（见 tests）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .distance import aggregate_distance


@dataclass
class NearestResult:
    claimed: str                       # 声称的模型名
    nearest: str | None                # 最近邻模型名
    claimed_distance: float | None     # 到 claimed 的聚合 JSD
    nearest_distance: float | None     # 到最近邻的聚合 JSD
    margin: float                      # claimed_distance - nearest_distance（>0 表示更像别人）
    ranking: list = field(default_factory=list)   # [(model, distance), ...] 升序
    flagged: bool = False              # 是否达到"更像异族"的判据
    reason: str = ""

    def to_dict(self):
        return {
            "claimed": self.claimed,
            "nearest": self.nearest,
            "claimed_distance": self.claimed_distance,
            "nearest_distance": self.nearest_distance,
            "margin": self.margin,
            "ranking": [[m, d] for m, d in self.ranking],
            "flagged": self.flagged,
            "reason": self.reason,
        }


def nearest_model(test_counts: dict, enrolled: dict, claimed: str,
                  min_samples: int = 8, margin_threshold: float = 0.05,
                  same_family: dict | None = None) -> NearestResult:
    """在闭集候选里找被测端点的最近邻模型。

    参数
      test_counts : cell -> Counter，被测端点的观测指纹
      enrolled    : model_name -> (cell -> Counter)，一批可信入册指纹
      claimed     : 端点声称的模型名（必须在 enrolled 里）
      margin_threshold : claimed 距离比最近邻大多少才判 flagged（默认 0.05，
                         略大于研究笔记里的同源噪声抖动，避免把噪声当证据）
      same_family : 可选 model -> family 标签；若最近邻与 claimed 同族则不 flag
                    （同族在指纹上本就近似，属于"自研旗舰"合法情形，不应误报）

    返回 NearestResult。若样本不足以对 claimed 计距离，flagged=False。
    """
    ranking = []
    for model, ref in enrolled.items():
        d, _ = aggregate_distance(ref, test_counts, min_samples=min_samples)
        if d is not None:
            ranking.append((model, d))
    ranking.sort(key=lambda x: x[1])

    claimed_distance = next((d for m, d in ranking if m == claimed), None)
    if not ranking:
        return NearestResult(claimed, None, claimed_distance, None, 0.0, ranking,
                             False, "无可比 cell（样本不足）")

    nearest, nearest_distance = ranking[0]
    margin = (claimed_distance - nearest_distance) if claimed_distance is not None else 0.0

    fam = same_family or {}
    same = fam.get(nearest) is not None and fam.get(nearest) == fam.get(claimed)

    flagged = (
        claimed_distance is not None
        and nearest != claimed
        and margin >= margin_threshold
        and not same
    )
    if flagged:
        reason = (f"闭集最近邻为 {nearest}（JSD={nearest_distance:.3f}），"
                  f"比声称的 {claimed}（JSD={claimed_distance:.3f}）更近 {margin:.3f}，"
                  f"疑似被替换为该模型或其近亲。")
    elif same:
        reason = f"最近邻 {nearest} 与声称模型同族，属合法情形，不判违规。"
    else:
        reason = f"最近邻即声称模型 {claimed}，归属一致。"

    return NearestResult(claimed, nearest, claimed_distance, nearest_distance,
                         margin, ranking, flagged, reason)
