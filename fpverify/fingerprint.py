"""指纹数据结构：入册、序列化、读写。

指纹 = 每个 cell (task, lang) 的回答类别计数分布 + 元数据（采样参数、时间、延迟样本）。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

CellKey = tuple  # (task: str, lang: str)


def _key_str(key: CellKey) -> str:
    return f"{key[0]}::{key[1]}"


def _str_key(s: str) -> CellKey:
    task, lang = s.split("::", 1)
    return (task, lang)


@dataclass
class Fingerprint:
    model: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    params: dict = field(default_factory=lambda: {"temperature": 1.0, "max_tokens": 16})
    # cell -> Counter(答案 -> 次数)
    cells: dict = field(default_factory=dict)
    # cell -> {valid,invalid,empty}
    validity: dict = field(default_factory=dict)
    # cell -> 延迟样本列表（秒），供缓存筛查
    latencies: dict = field(default_factory=dict)
    note: str = ""

    def add(self, key: CellKey, answer_token: str, validity: str = "valid", latency: float | None = None):
        self.cells.setdefault(key, Counter())[answer_token] += 1
        v = self.validity.setdefault(key, {"valid": 0, "invalid": 0, "empty": 0})
        v[validity] = v.get(validity, 0) + 1
        if latency is not None:
            self.latencies.setdefault(key, []).append(latency)

    def counts(self) -> dict:
        """返回 cell -> Counter（仅用于距离计算，含全部类别）。"""
        return self.cells

    def n_samples(self, key: CellKey) -> int:
        return sum(self.cells.get(key, {}).values())

    def total_samples(self) -> int:
        return sum(sum(c.values()) for c in self.cells.values())

    def overall_validity(self) -> float:
        tv = sum(v.get("valid", 0) for v in self.validity.values())
        tot = sum(sum(v.values()) for v in self.validity.values())
        return tv / tot if tot else 0.0

    # ---- 序列化 ----
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "created_at": self.created_at,
            "params": self.params,
            "note": self.note,
            "cells": {_key_str(k): dict(v) for k, v in self.cells.items()},
            "validity": {_key_str(k): v for k, v in self.validity.items()},
            "latencies": {_key_str(k): v for k, v in self.latencies.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fingerprint":
        fp = cls(model=d["model"], created_at=d.get("created_at", ""),
                 params=d.get("params", {}), note=d.get("note", ""))
        fp.cells = {_str_key(k): Counter(v) for k, v in d.get("cells", {}).items()}
        fp.validity = {_str_key(k): v for k, v in d.get("validity", {}).items()}
        fp.latencies = {_str_key(k): v for k, v in d.get("latencies", {}).items()}
        return fp

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Fingerprint":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
