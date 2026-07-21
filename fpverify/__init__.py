"""fpverify —— 基于单 token 行为指纹的中转站模型注水检测。

科研依据见 docs/RESEARCH_NOTES.md。主要子模块：
  probes       多语言改写探针银行
  normalize    回答归一化
  distance     JSD / 熵 / 聚合距离
  fingerprint  指纹数据结构与入册
  betting      带容差的序贯下注 e-process（决策核心，anytime-valid，控 FPR）
  screens      响应缓存/延迟异常筛查
  calibrate    零分布自助法、EER、预算曲线
  endpoints    端点抽象（HTTP + 进程内仿真）
  verifier     编排 enroll/audit
"""

from .fingerprint import Fingerprint, CellKey
from .verifier import Verifier, AuditResult
from .betting import SequentialBettingTest, BettingConfig

__version__ = "0.1.0"

__all__ = [
    "Fingerprint",
    "CellKey",
    "Verifier",
    "AuditResult",
    "SequentialBettingTest",
    "BettingConfig",
    "__version__",
]
