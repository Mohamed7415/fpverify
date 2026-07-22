"""人类可读的审计报告渲染（控制台文本 + JSON）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# 论文参考带（Bruckner §VI）：帮助用户解读聚合 JSD 的量级
REFERENCE_BANDS = (
    ("同源自比噪声地板", 0.140),
    ("同模型跨服务商中位", 0.227),
    ("不同模型（冒充者）中位", 0.463),
)


def interpret_jsd(d: float | None) -> str:
    if d is None:
        return "样本不足，无法计算聚合 JSD"
    if d < 0.18:
        return "与参考同源（在单部署自比噪声范围内）"
    if d < 0.30:
        return "接近同源但有部署级漂移（量化/服务栈差异可能）"
    if d < 0.42:
        return "灰色地带：明显漂移，建议加大采样或多语言复测"
    return "落入'不同模型'区间：与参考模型基本不是同一个东西"


def render_text(result, reference_model: str) -> str:
    lines = []
    w = lines.append
    w("=" * 68)
    w(f"行为指纹审计报告   参考模型: {reference_model}")
    w(f"时间: {datetime.now(timezone.utc).isoformat()}")
    w("=" * 68)
    w(f"判定: {result.verdict}")
    w(f"  {result.detail}")
    if result.verdict == "PASS":
        w("  PASS 的含义是本次预算内未检出偏离，不是对商家的认证背书。")
        w("  怀疑本工具偏袒？用官方 key 对两个不同模型交叉审计，错配那次必须 FAIL；")
        w("  --report 导出的 JSON 含全部原始答案计数，可用任意代码独立重算本判定。")
    w("")
    w(f"查询消耗: {result.n_queries} 次（单 token 回答；早停机制生效时远小于预算）")
    w(f"e-process 财富: {result.wealth:.3g}（峰值 {result.peak_wealth:.3g}，拒绝阈值 {result.threshold:.0f}）")
    w(f"FPR 上界 alpha={result.alpha}，容差 delta={result.delta}")
    w("")
    if result.aggregate_jsd is not None:
        w(f"聚合 JSD 距离: {result.aggregate_jsd:.3f} bit —— {interpret_jsd(result.aggregate_jsd)}")
        w("  参考带: " + "；".join(f"{name} {v:.3f}" for name, v in REFERENCE_BANDS))
    if result.per_cell_jsd:
        worst = sorted(result.per_cell_jsd.items(), key=lambda kv: kv[1], reverse=True)[:6]
        w("  差异最大的探测单元:")
        for cell, d in worst:
            w(f"    {cell[0]}/{cell[1]:<3} JSD={d:.3f}")
    if result.cache_flags:
        w("")
        w("[完整性告警] 疑似响应级缓存的单元：")
        for f in result.cache_flags:
            w(f"    {f.cell[0]}/{f.cell[1]} 模态份额={f.modal_share:.2f} 熵={f.entropy:.2f} 中位延迟={f.median_latency}")
    if result.model_fields_seen:
        w("")
        w(f"端点自报 model 字段: {result.model_fields_seen}（可伪造，仅供参考）")
    if result.errors:
        w(f"失败请求: {result.errors}")
    w("=" * 68)
    return "\n".join(lines)


def save_json(result, reference_model: str, path: str | Path):
    payload = {"reference_model": reference_model,
               "generated_at": datetime.now(timezone.utc).isoformat(),
               **result.to_dict()}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
