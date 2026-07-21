"""回答归一化：把模型的原始文本回答压成一个类别 token。

遵循 Bruckner §IV-B 的确定性流程：Unicode NFC、去标点/引号、大小写折叠、
阿拉伯-印度数字与中文数字映射到拉丁数字、取首 token、颜色词跨语言规范化。
解析失败/离题/多词/拒答归入特殊类别 <other>（弱信号，但保留以免静默丢弃）。
"""

from __future__ import annotations

import re
import unicodedata

OTHER = "<other>"

_PUNCT = "\"'`.,!?。！，、：:；;()（）[]【】<>《》*#…~ \t\r\n"

# 中文数字 -> 阿拉伯数字（覆盖 0-100 常见写法）
_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
# 阿拉伯-印度数字 -> 拉丁
_ARABIC_INDIC = {ord("٠") + i: str(i) for i in range(10)}
_ARABIC_INDIC.update({ord("۰") + i: str(i) for i in range(10)})  # 波斯变体

# 硬币回答规范化（多语言 -> heads/tails）
_COIN = {
    "heads": "heads", "head": "heads", "正面": "heads", "正": "heads",
    "орёл": "heads", "орел": "heads", "وجه": "heads",
    "tails": "tails", "tail": "tails", "反面": "tails", "反": "tails", "背面": "tails",
    "решка": "tails", "كتابة": "tails",
}

# 颜色跨语言规范化（部分常见色 -> 英文规范码）
_COLOR = {
    "红": "red", "红色": "red", "红的": "red", "red": "red", "красный": "red", "أحمر": "red",
    "蓝": "blue", "蓝色": "blue", "blue": "blue", "синий": "blue", "أزرق": "blue",
    "绿": "green", "绿色": "green", "green": "green", "зелёный": "green", "зеленый": "green", "أخضر": "green",
    "黄": "yellow", "黄色": "yellow", "yellow": "yellow", "жёлтый": "yellow", "желтый": "yellow", "أصفر": "yellow",
    "黑": "black", "黑色": "black", "black": "black", "чёрный": "black", "черный": "black", "أسود": "black",
    "白": "white", "白色": "white", "white": "white", "белый": "white", "أبيض": "white",
    "紫": "purple", "紫色": "purple", "purple": "purple", "фиолетовый": "purple", "بنفسجي": "purple",
    "橙": "orange", "橙色": "orange", "orange": "orange", "оранжевый": "orange", "برتقالي": "orange",
    "青": "cyan", "青色": "cyan", "teal": "teal", "cyan": "cyan",
    "粉": "pink", "粉色": "pink", "pink": "pink", "розовый": "pink",
    "靛": "indigo", "indigo": "indigo", "crimson": "crimson", "绯红": "crimson",
}


def _basic(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate(_ARABIC_INDIC)
    return text.strip()


def _first_token(text: str) -> str:
    t = text.strip(_PUNCT).lower()
    if not t:
        return ""
    return re.split(r"[\s,，。.!！?？:：;；、/\\]+", t)[0]


def _cn_number(text: str):
    """解析简单中文数字（十, 二十三, 七 等）。返回 int 或 None。"""
    text = text.strip()
    if not text or any(ch not in _CN_DIGIT for ch in text):
        return None
    if len(text) == 1:
        return _CN_DIGIT[text]
    if "十" in text:
        parts = text.split("十")
        left = parts[0]
        right = parts[1] if len(parts) > 1 else ""
        tens = _CN_DIGIT[left] if left else 1
        ones = _CN_DIGIT[right] if right else 0
        if tens <= 9 and ones <= 9:
            return tens * 10 + ones
    return None


def normalize(parse: str, text: str) -> str:
    """把回答归一化为类别 token。空/离题/多词返回 OTHER。"""
    if not text or not text.strip():
        return OTHER
    raw = _basic(text)
    first_line = raw.splitlines()[0].strip() if raw.splitlines() else raw

    if parse == "int":
        m = re.search(r"-?\d+", first_line.replace(",", "").replace(" ", ""))
        if m:
            return str(int(m.group(0)))
        cn = _cn_number(_first_token(first_line))
        if cn is not None:
            return str(cn)
        return OTHER

    if parse == "letter":
        m = re.search(r"[A-Za-z]", first_line)
        return m.group(0).lower() if m else OTHER

    if parse == "word":
        tok = _first_token(first_line)
        if not tok:
            return OTHER
        if tok in _COIN:
            return _COIN[tok]
        if tok in _COLOR:
            return _COLOR[tok]
        # 允许多词答案时取首词；过长（>24 字符）视为解释性回答 -> OTHER
        if len(tok) > 24:
            return OTHER
        return tok

    return OTHER


def classify_validity(parse: str, text: str) -> str:
    """把回答分类为 valid / invalid / empty（供有效率统计与缓存筛查）。"""
    if not text or not text.strip():
        return "empty"
    return "valid" if normalize(parse, text) != OTHER else "invalid"
