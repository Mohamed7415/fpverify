"""逼真的"模型行为分布"仿真。

每个模型是一个 (task, lang) -> 类别概率分布 的生成器。分布形态参照
Bruckner 论文 Fig.1 的观察（GPT-4o 散在 42/37/57；Claude 集中 47；
Llama 偏 53；Qwen 几乎恒为 42）以及各任务的普遍偏斜（低熵、模态份额高）。

这些是**仿真**分布，不是真实厂商数据；目的是构造一个"同族接近、异族远离、
同模型跨部署有小幅漂移"的可控世界，用来验证检测器的统计性质。
"""

from __future__ import annotations

import hashlib
import random


# 每个基础模型：task -> {答案: 权重}。语言维度通过轻微扰动派生。
BASE_MODELS = {
    "gpt-4o": {
        "rand_num_100": {"42": 30, "37": 20, "57": 18, "73": 10, "69": 8, "88": 6, "7": 4, "100": 4},
        "rand_num_10":  {"7": 55, "3": 18, "4": 10, "8": 9, "5": 8},
        "fav_num":      {"7": 60, "42": 25, "3": 15},
        "rand_letter":  {"q": 25, "m": 20, "a": 18, "z": 15, "k": 12, "r": 10},
        "rand_color":   {"blue": 48, "teal": 16, "green": 12, "crimson": 12, "purple": 12},
        "fav_color":    {"blue": 70, "green": 20, "teal": 10},
        "rand_animal":  {"octopus": 35, "fox": 22, "giraffe": 18, "elephant": 15, "cat": 10},
        "rand_city":    {"kyoto": 30, "lisbon": 22, "tokyo": 18, "paris": 16, "reykjavik": 14},
        "coin_flip":    {"heads": 78, "tails": 22},
    },
    "claude-sonnet-5": {
        "rand_num_100": {"47": 62, "42": 12, "73": 8, "27": 7, "63": 6, "88": 5},
        "rand_num_10":  {"7": 62, "4": 14, "3": 12, "8": 7, "2": 5},
        "fav_num":      {"42": 55, "7": 30, "3": 15},
        "rand_letter":  {"m": 30, "q": 22, "a": 16, "k": 16, "r": 16},
        "rand_color":   {"blue": 40, "teal": 24, "indigo": 16, "green": 12, "purple": 8},
        "fav_color":    {"blue": 62, "teal": 22, "purple": 16},
        "rand_animal":  {"octopus": 30, "fox": 26, "owl": 18, "elephant": 16, "cat": 10},
        "rand_city":    {"kyoto": 28, "paris": 24, "lisbon": 20, "prague": 16, "tokyo": 12},
        "coin_flip":    {"heads": 70, "tails": 30},
    },
    "llama-3.3-70b": {
        "rand_num_100": {"53": 40, "42": 20, "7": 14, "77": 10, "23": 8, "99": 8},
        "rand_num_10":  {"7": 48, "5": 20, "3": 16, "9": 10, "1": 6},
        "fav_num":      {"7": 50, "42": 30, "13": 20},
        "rand_letter":  {"a": 30, "s": 20, "m": 18, "t": 16, "r": 16},
        "rand_color":   {"blue": 52, "green": 20, "red": 14, "purple": 8, "teal": 6},
        "fav_color":    {"blue": 68, "green": 22, "red": 10},
        "rand_animal":  {"dog": 32, "lion": 24, "elephant": 20, "fox": 14, "cat": 10},
        "rand_city":    {"paris": 34, "tokyo": 24, "london": 20, "newyork": 14, "rome": 8},
        "coin_flip":    {"heads": 66, "tails": 34},
    },
    "qwen3-235b": {
        "rand_num_100": {"42": 88, "7": 4, "77": 3, "88": 3, "50": 2},
        "rand_num_10":  {"7": 78, "3": 10, "5": 6, "8": 6},
        "fav_num":      {"7": 46, "42": 44, "8": 10},
        "rand_letter":  {"a": 40, "q": 18, "m": 16, "k": 14, "z": 12},
        "rand_color":   {"blue": 60, "red": 16, "green": 12, "purple": 12},
        "fav_color":    {"blue": 74, "red": 16, "green": 10},
        "rand_animal":  {"panda": 40, "dragon": 22, "cat": 16, "fox": 12, "tiger": 10},
        "rand_city":    {"beijing": 42, "shanghai": 24, "hangzhou": 18, "tokyo": 16},
        "coin_flip":    {"heads": 60, "tails": 40},
    },
    "qwen3-max": {  # 与 qwen3-235b 同族、近似（用于测同族区分难度）
        "rand_num_100": {"42": 95, "7": 2, "88": 2, "50": 1},
        "rand_num_10":  {"7": 82, "3": 8, "5": 6, "8": 4},
        "fav_num":      {"42": 52, "7": 40, "8": 8},
        "rand_letter":  {"a": 44, "q": 18, "m": 14, "k": 14, "z": 10},
        "rand_color":   {"blue": 64, "red": 14, "green": 12, "purple": 10},
        "fav_color":    {"blue": 76, "red": 14, "green": 10},
        "rand_animal":  {"panda": 44, "dragon": 20, "cat": 16, "tiger": 12, "fox": 8},
        "rand_city":    {"beijing": 46, "shanghai": 22, "hangzhou": 18, "tokyo": 14},
        "coin_flip":    {"heads": 58, "tails": 42},
    },
    "glm-5.2": {
        "rand_num_100": {"66": 34, "42": 20, "88": 16, "6": 12, "99": 10, "23": 8},
        "rand_num_10":  {"6": 40, "7": 26, "8": 16, "3": 10, "9": 8},
        "fav_num":      {"6": 40, "8": 32, "7": 28},
        "rand_letter":  {"a": 34, "g": 20, "m": 16, "z": 16, "k": 14},
        "rand_color":   {"blue": 46, "green": 22, "red": 16, "purple": 16},
        "fav_color":    {"blue": 60, "green": 24, "red": 16},
        "rand_animal":  {"panda": 36, "cat": 22, "dragon": 18, "rabbit": 14, "fox": 10},
        "rand_city":    {"beijing": 40, "shanghai": 26, "chengdu": 20, "xian": 14},
        "coin_flip":    {"heads": 62, "tails": 38},
    },
}

# 廉价小模型（中转站可能拿来冒充旗舰）
BASE_MODELS["cheap-7b"] = {
    "rand_num_100": {"7": 30, "42": 22, "1": 14, "100": 12, "50": 12, "69": 10},
    "rand_num_10":  {"5": 34, "1": 22, "7": 20, "10": 14, "3": 10},
    "fav_num":      {"7": 44, "1": 30, "5": 26},
    "rand_letter":  {"a": 50, "b": 20, "x": 16, "c": 14},
    "rand_color":   {"red": 40, "blue": 30, "green": 18, "yellow": 12},
    "fav_color":    {"red": 46, "blue": 34, "green": 20},
    "rand_animal":  {"cat": 36, "dog": 30, "lion": 20, "bird": 14},
    "rand_city":    {"newyork": 40, "london": 30, "paris": 20, "tokyo": 10},
    "coin_flip":    {"heads": 52, "tails": 48},
}


LANGS = ["en", "zh", "ru", "ar"]


def _lang_perturb(dist: dict, task: str, lang: str, strength: float = 0.15) -> dict:
    """从基础（英文）分布派生某语言分布：确定性地轻微扰动权重。

    同一模型不同语言应相关但不全同（论文：语言是独立探测维度）。
    """
    if lang == "en":
        return dict(dist)
    out = {}
    for k, v in dist.items():
        h = hashlib.sha256(f"{task}|{lang}|{k}".encode()).digest()
        factor = 1.0 + strength * ((h[0] / 255.0) * 2 - 1)  # [1-s, 1+s]
        out[k] = max(0.01, v * factor)
    return out


class MockModel:
    """一个模型的分布采样器。可注入量化/漂移扰动。"""

    def __init__(self, name: str, quantize: float = 0.0, drift_seed: int | None = None):
        if name not in BASE_MODELS:
            raise KeyError(name)
        self.name = name
        self.base = BASE_MODELS[name]
        self.quantize = quantize        # 0=无；>0 让分布更尖锐（模拟量化偏移模态）
        self.drift_seed = drift_seed

    def _dist(self, task: str, lang: str) -> dict:
        d = _lang_perturb(self.base[task], task, lang)
        if self.quantize > 0:
            # 量化：放大模态、抑制长尾（改变 softmax landscape）
            mx = max(d.values())
            d = {k: (v ** (1.0 + self.quantize) if v < mx else v * (1.0 + 0.3 * self.quantize))
                 for k, v in d.items()}
        if self.drift_seed is not None:
            # 跨部署良性漂移：确定性小扰动
            out = {}
            for k, v in d.items():
                h = hashlib.sha256(f"{self.drift_seed}|{task}|{lang}|{k}".encode()).digest()
                out[k] = max(0.01, v * (1.0 + 0.08 * ((h[0] / 255.0) * 2 - 1)))
            d = out
        return d

    def sample(self, task: str, lang: str, rng: random.Random) -> str:
        d = self._dist(task, lang)
        keys = list(d.keys())
        weights = list(d.values())
        return rng.choices(keys, weights=weights, k=1)[0]

    def format_answer(self, task: str, raw: str, rng: random.Random) -> str:
        """把类别值渲染成"模型输出文本"，含少量格式噪声。"""
        # 数字/字母/词直接输出；偶尔加标点或首字母大写
        text = raw
        if task in ("rand_color", "fav_color", "rand_animal", "rand_city", "coin_flip"):
            if rng.random() < 0.3:
                text = text.capitalize()
        if rng.random() < 0.1:
            text += "."
        return text
