# -*- coding: utf-8 -*-
"""归一化的确定性行为。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpverify.normalize import normalize, OTHER, classify_validity


def test_int_parsing():
    assert normalize("int", "42") == "42"
    assert normalize("int", "  42. ") == "42"
    assert normalize("int", "The number is 73!") == "73"
    assert normalize("int", "٤٢") == "42"          # 阿拉伯-印度数字
    assert normalize("int", "四十二") == "42"        # 中文数字
    assert normalize("int", "七") == "7"
    assert normalize("int", "十") == "10"
    assert normalize("int", "二十三") == "23"
    assert normalize("int", "I refuse to answer") == OTHER


def test_letter_parsing():
    assert normalize("letter", "Q") == "q"
    assert normalize("letter", "q.") == "q"
    assert normalize("letter", "The letter Z") == "t"  # 取首字母（确定性规则）
    assert normalize("letter", "42") == OTHER


def test_word_parsing_and_canonicalization():
    assert normalize("word", "Blue.") == "blue"
    assert normalize("word", "蓝色") == "blue"          # 颜色跨语言规范化
    assert normalize("word", "красный") == "red"
    assert normalize("word", "正面。") == "heads"        # 硬币规范化
    assert normalize("word", "Tails") == "tails"
    assert normalize("word", "орёл") == "heads"
    assert normalize("word", "octopus is my choice") == "octopus"
    assert normalize("word", "") == OTHER


def test_validity():
    assert classify_validity("int", "42") == "valid"
    assert classify_validity("int", "no idea") == "invalid"
    assert classify_validity("int", "  ") == "empty"
