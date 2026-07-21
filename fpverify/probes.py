"""探针银行：语义任务 × 多语言 × 改写族。

对抗设计（针对 T2 过滤型对手）：绝不发送固定"魔法字符串"。每个语义任务在每种语言下
都有多条等价改写模板，审计时随机抽取。同一任务跨语言交叉验证。因此特判单一措辞的对手
会在其它措辞/语言的探针上暴露。

parse 类型决定回答如何归一化为类别（见 normalize.py）：int / letter / word。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProbeTask:
    task: str            # 语义任务 id，如 "rand_num_100"
    parse: str           # 归一化类型：int / letter / word
    # 语言 -> 改写模板列表。每条模板都要求单个词/数的回答。
    templates: dict[str, list[str]] = field(default_factory=dict)

    def languages(self) -> list[str]:
        return list(self.templates.keys())


# 系统提示（多语言），强制简短单值回答，抑制解释与思考。
SYSTEM_PROMPTS = {
    "en": "You answer with a single value only: just one word or one number, no explanation, no punctuation, no reasoning.",
    "zh": "只回答一个值：一个词或一个数字，不要解释、不要标点、不要思考过程。",
    "ru": "Отвечай только одним значением: одно слово или одно число, без объяснений и знаков препинания.",
    "ar": "أجب بقيمة واحدة فقط: كلمة واحدة أو رقم واحد، دون شرح أو علامات ترقيم.",
}


TASKS: list[ProbeTask] = [
    ProbeTask(
        task="rand_num_100", parse="int",
        templates={
            "en": [
                "Name a random number between 1 and 100.",
                "Pick any number from 1 to 100.",
                "Give me a random integer in the range 1-100.",
                "Think of a number between 1 and 100 and tell me.",
                "Choose a number, 1 through 100.",
            ],
            "zh": [
                "说一个1到100之间的随机数。",
                "从1到100里随便挑一个数。",
                "给我一个1到100之间的整数。",
                "随便想一个1到100的数字告诉我。",
            ],
            "ru": [
                "Назови случайное число от 1 до 100.",
                "Выбери любое число от 1 до 100.",
                "Дай случайное целое число в диапазоне 1-100.",
            ],
            "ar": [
                "اذكر رقمًا عشوائيًا بين 1 و 100.",
                "اختر أي رقم من 1 إلى 100.",
            ],
        },
    ),
    ProbeTask(
        task="rand_num_10", parse="int",
        templates={
            "en": [
                "Name a random number between 1 and 10.",
                "Pick any number from 1 to 10.",
                "Give me a random integer in the range 1-10.",
            ],
            "zh": [
                "说一个1到10之间的随机数。",
                "从1到10里随便挑一个数。",
            ],
            "ru": [
                "Назови случайное число от 1 до 10.",
                "Выбери любое число от 1 до 10.",
            ],
            "ar": [
                "اذكر رقمًا عشوائيًا بين 1 و 10.",
            ],
        },
    ),
    ProbeTask(
        task="fav_num", parse="int",
        templates={
            "en": [
                "What is your favorite number? Answer with just the number.",
                "Tell me your favorite number.",
            ],
            "zh": [
                "你最喜欢的数字是几？只回答数字。",
                "说出你最喜欢的一个数字。",
            ],
            "ru": [
                "Какое твое любимое число? Ответь только числом.",
            ],
            "ar": [
                "ما هو رقمك المفضل؟ أجب بالرقم فقط.",
            ],
        },
    ),
    ProbeTask(
        task="rand_letter", parse="letter",
        templates={
            "en": [
                "Name a random letter of the English alphabet.",
                "Pick any single letter A-Z.",
                "Give me a random letter.",
            ],
            "zh": [
                "随便说一个英文字母。",
                "从A到Z里挑一个字母。",
            ],
            "ru": [
                "Назови случайную букву английского алфавита.",
            ],
            "ar": [
                "اذكر حرفًا إنجليزيًا عشوائيًا.",
            ],
        },
    ),
    ProbeTask(
        task="rand_color", parse="word",
        templates={
            "en": [
                "Name a random color.",
                "Pick any color.",
                "Say a color, any color.",
            ],
            "zh": [
                "随便说一个颜色。",
                "挑一个颜色。",
            ],
            "ru": [
                "Назови случайный цвет.",
                "Выбери любой цвет.",
            ],
            "ar": [
                "اذكر لونًا عشوائيًا.",
            ],
        },
    ),
    ProbeTask(
        task="fav_color", parse="word",
        templates={
            "en": [
                "What is your favorite color? One word.",
                "Tell me your favorite color.",
            ],
            "zh": [
                "你最喜欢的颜色是什么？只说一个词。",
                "说出你最喜欢的颜色。",
            ],
            "ru": [
                "Какой твой любимый цвет? Одно слово.",
            ],
            "ar": [
                "ما لونك المفضل؟ كلمة واحدة.",
            ],
        },
    ),
    ProbeTask(
        task="rand_animal", parse="word",
        templates={
            "en": [
                "Name a random animal.",
                "Pick any animal.",
                "Say an animal, any animal.",
            ],
            "zh": [
                "随便说一个动物。",
                "挑一种动物。",
            ],
            "ru": [
                "Назови случайное животное.",
            ],
            "ar": [
                "اذكر حيوانًا عشوائيًا.",
            ],
        },
    ),
    ProbeTask(
        task="rand_city", parse="word",
        templates={
            "en": [
                "Name a random city.",
                "Pick any city in the world.",
            ],
            "zh": [
                "随便说一个城市。",
                "挑一座城市。",
            ],
            "ru": [
                "Назови случайный город.",
            ],
            "ar": [
                "اذكر مدينة عشوائية.",
            ],
        },
    ),
    ProbeTask(
        task="coin_flip", parse="word",
        templates={
            "en": [
                "Flip a coin: heads or tails? One word.",
                "Call a coin flip, heads or tails.",
            ],
            "zh": [
                "抛一枚硬币，正面还是反面？只回答两个字。",
                "猜硬币：正面还是反面？",
            ],
            "ru": [
                "Подбрось монету: орёл или решка? Одно слово.",
            ],
            "ar": [
                "ارمِ عملة: وجه أم كتابة؟ كلمة واحدة.",
            ],
        },
    ),
]

TASK_BY_ID = {t.task: t for t in TASKS}


def all_cells() -> list[tuple[str, str]]:
    """返回全部 (task, lang) 组合。"""
    return [(t.task, lang) for t in TASKS for lang in t.languages()]


def render_prompt(task: str, lang: str, rng: random.Random) -> tuple[str, str]:
    """审计时调用：随机抽一条改写措辞。返回 (system_prompt, user_prompt)。"""
    t = TASK_BY_ID[task]
    user = rng.choice(t.templates[lang])
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    return system, user


def parse_type(task: str) -> str:
    return TASK_BY_ID[task].parse
