"""服务层完整性筛查：响应级缓存 / 延迟异常。

依据 Bruckner §VII-C：真正的 T=1.0 采样应有分布方差；若端点用**响应级缓存**回放
存好的补全，会出现"方差塌缩 + 异常低且低方差延迟"的联合签名。这不是指纹证据，
而是端点完整性违规，单独标记。

注意：前缀缓存（prefix/KV cache）只影响成本与延迟、不改变采样分布，不是问题；
我们盯的是"回放整条补全"这种会压塌分布的行为。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .distance import entropy_bits


@dataclass
class ScreenResult:
    cell: tuple
    flagged: bool
    reason: str
    modal_share: float
    entropy: float
    median_latency: float | None


def screen_response_cache(fingerprint, model_median_latency: float | None = None,
                          min_samples: int = 10) -> list[ScreenResult]:
    """对每个 cell 检查响应缓存签名。返回被标记的 cell 列表。

    判据（需同时满足）：
      1) 在温度=1.0 下分布几乎塌缩到单值（modal_share 高且熵极低）；
      2) 该 cell 的中位延迟异常低（< 全模型中位延迟的一半），且延迟方差很小。
    单独满足 1) 很常见（很多任务本就低熵），所以必须叠加延迟证据才判定。
    """
    results = []
    # 汇总全模型延迟中位数（若未显式给出）
    if model_median_latency is None:
        all_lat = [x for lat in fingerprint.latencies.values() for x in lat]
        model_median_latency = statistics.median(all_lat) if all_lat else None

    for cell, counts in fingerprint.cells.items():
        n = sum(counts.values())
        if n < min_samples:
            continue
        modal_share = max(counts.values()) / n
        ent = entropy_bits(counts)
        lat = fingerprint.latencies.get(cell, [])
        med_lat = statistics.median(lat) if lat else None

        variance_collapse = (modal_share >= 0.98 and ent <= 0.15)
        latency_anomaly = False
        if med_lat is not None and model_median_latency:
            low = med_lat < 0.5 * model_median_latency
            stable = (statistics.pstdev(lat) < 0.25 * med_lat) if len(lat) > 1 and med_lat > 0 else False
            latency_anomaly = low and stable

        flagged = variance_collapse and latency_anomaly
        if flagged:
            results.append(ScreenResult(
                cell=cell, flagged=True,
                reason="疑似响应级缓存：T=1.0 下分布塌缩且延迟异常低而稳定",
                modal_share=modal_share, entropy=ent, median_latency=med_lat))
    return results
