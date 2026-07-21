"""业务流量宇宙 + 探针伪装 + 特征模型（协同进化专用，纯仿真、可复现）。

红蓝对抗的**真正战场**不是"能不能算出分布距离"，而是"探针流量与真实业务流量
有多可区分"。本模块用一个可解释、可复现的生成式特征模型刻画这条轴：

  每个请求可被中转站（红队）观测到两个标量特征：
    x_surface   表层可区分度：短问句 + "只回答一个值"式系统提示 => 高
    x_semantic  语义可区分度：能不能被认出是"低熵琐碎问题（说个1-100/颜色…）" => 高

  真实业务流量：x_surface ~ N(0,1)；x_semantic 是混合分布——92% 是高熵日常请求
  （写代码/问答/总结，x_semantic 低），8% 是天然低熵请求（RNG 种子、占位数据、
  给主题选个颜色，x_semantic 高）。这 8% 是红队"语义分流"能薅到的唯一合法借口。

  探针在三种伪装等级下的特征：
    raw      裸探针（现有 fpverify 探针）：x_surface、x_semantic 都很高 => 极易识别
    surface  表层伪装（蓝队轮1）：加长措辞、去掉"只回答一个值"提示 => x_surface 降到 ≈业务，
             但语义上仍明显是"说个1-100" => x_semantic 仍高
    native   能力分叉/业务原生探针（蓝队轮2）：从业务流形里取问法，问一个看起来像普通
             高熵业务请求、但该模型答案恰好低熵且有个性的问题 => x_surface、x_semantic 都 ≈业务。
             探针族跨很多任务类型（数/字母/颜色/动物/城市…），语义特征比单一业务负载更**宽**
             （x_semantic 与业务同均值但方差更大），红队无法用一条阈值把它从业务主流里择干净。

红队的分流策略就是在这两个特征上设阈值：超过阈值判"疑似测试"供旗舰、否则供廉价。
  - 探针被判旗舰 => 答案来自真身 => 逃过检测；
  - 业务被判廉价 => 真省钱（廉价流量占比 = 省到的钱）。
蓝队伪装把探针特征推向业务分布，逼红队为抓探针把阈值压低、连真实业务一起误判成
"疑似测试"从而供了旗舰 —— 廉价流量占比坍塌 => 经济上不划算（铁律）。

以上高斯参数是**建模假设**（写死在此便于复现），不是真实测量；目的是提供一个
双方都能在其上做最优响应的、数字自洽的受控世界。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 特征分布参数
# 业务流量
BIZ_SURFACE = (0.0, 1.0)                    # x_surface ~ N(mu,sigma)
BIZ_SEM_MAJOR = (-1.2, 0.6)                 # 92% 高熵日常
BIZ_SEM_NICHE = (1.8, 0.5)                  # 8% 天然低熵（红队语义分流的合法借口）
BIZ_NICHE_FRAC = 0.08

# 探针（按伪装等级）：(x_surface 参数, x_semantic 参数)
PROBE_FEATURES = {
    "raw":     {"surface": (3.4, 0.5), "semantic": (2.3, 0.4)},
    "surface": {"surface": (0.2, 0.9), "semantic": (2.2, 0.4)},
    "native":  {"surface": (0.0, 1.0), "semantic": (-1.1, 0.65)},
}
CAMO_LEVELS = ["raw", "surface", "native"]

# ---------------------------------------------------------------- 文本模板（仅用于让请求"看起来像真的"，特征以 features 为准）
_BIZ_SYS = [
    "You are a helpful software engineering assistant.",
    "You are a concise customer-support agent for a SaaS product.",
    "You are an expert data analyst. Explain your reasoning.",
    "You help users draft professional emails.",
]
_BIZ_USER_HIGH = [
    "Refactor this Python function to be more readable and add type hints, then explain the changes.",
    "Summarize the following quarterly report into three bullet points for an executive audience.",
    "Write a polite email declining a meeting invitation while proposing two alternative times.",
    "Debug why my React component re-renders on every keystroke and suggest a fix.",
    "Draft a product description for a noise-cancelling headphone, around 80 words.",
]
# 天然低熵业务请求：带真实 (task, lang)，便于被动审计把它们映射到指纹 cell。
# 这些是"真实用户偶尔会问的低熵琐碎请求"，与探针问的是同一批语义任务——
# 正因如此，红队无法在"请求内容"维度把探针和它们区分开（只能靠账号级）。
NICHE_TASKS = [
    ("rand_num_100", "en", "I need a random integer between 1 and 100 to seed a test fixture. Output just the number."),
    ("fav_color", "en", "Pick a color for our dashboard's primary theme; one word is fine."),
    ("rand_animal", "en", "Give me a placeholder animal name for the demo account."),
]
_BIZ_USER_NICHE = [u for _, _, u in NICHE_TASKS]

_PROBE_SYS_RAW = "You answer with a single value only: just one word or one number, no explanation."
_PROBE_SYS_CAMO = "You are a helpful assistant embedded in our internal tooling."


@dataclass
class Request:
    """一次请求：红队能看到 system/user 文本与（此处直接给出的）可观测特征。

    features 是"红队特征提取器"的输出（见 extract_features）。仿真里直接由生成器给出
    以保证分布可控；extract_features 提供一条与之一致的、从纯文本恢复特征的现实路径。
    """
    system: str
    user: str
    task: str
    lang: str
    features: dict
    account: dict | None = None
    is_probe: bool = False


def _g(rng: random.Random, params) -> float:
    return rng.gauss(params[0], params[1])


def sample_business_features(n: int, rng: random.Random) -> list[dict]:
    """采样 n 条业务流量的可观测特征（用于计算廉价流量占比，快路径）。"""
    out = []
    for _ in range(n):
        if rng.random() < BIZ_NICHE_FRAC:
            sem = _g(rng, BIZ_SEM_NICHE)
        else:
            sem = _g(rng, BIZ_SEM_MAJOR)
        out.append({"x_surface": _g(rng, BIZ_SURFACE), "x_semantic": sem})
    return out


def sample_business_request(rng: random.Random, account: dict | None = None) -> Request:
    """采样一条"看起来真实"的业务请求（含文本与特征）。

    天然低熵请求会带上真实 (task, lang)，使被动审计（蓝队轮3）能把这类
    真实流量映射到指纹 cell 上累积统计——这是账号级分流的原理性反制。
    """
    niche = rng.random() < BIZ_NICHE_FRAC
    feats = {"x_surface": _g(rng, BIZ_SURFACE),
             "x_semantic": _g(rng, BIZ_SEM_NICHE if niche else BIZ_SEM_MAJOR)}
    system = rng.choice(_BIZ_SYS)
    if niche:
        task, lang, user = rng.choice(NICHE_TASKS)
    else:
        task, lang, user = "<business>", "en", rng.choice(_BIZ_USER_HIGH)
    return Request(system, user, task=task, lang=lang, features=feats,
                   account=account, is_probe=(False))


def render_probe(task: str, lang: str, camo: str, rng: random.Random,
                 base_system: str, base_user: str, account: dict | None = None) -> Request:
    """蓝队按伪装等级渲染一条探针请求。

    base_system/base_user 来自 fpverify.probes.render_prompt（裸探针文本）。
    camo 决定伪装：raw 直接用裸文本；surface 换掉"只回答一个值"系统提示并加长；
    native 用业务外壳包裹（看起来像普通业务请求）。特征按 PROBE_FEATURES 采样。
    """
    fp = PROBE_FEATURES[camo]
    feats = {"x_surface": _g(rng, fp["surface"]), "x_semantic": _g(rng, fp["semantic"])}
    if camo == "raw":
        system, user = base_system, base_user
    elif camo == "surface":
        system = _PROBE_SYS_CAMO
        user = (f"For a quick internal sanity check, {base_user.rstrip('.')}. "
                f"Just reply with the value inline, no need to explain.")
    else:  # native：伪装成普通业务请求
        system = rng.choice(_BIZ_SYS)
        user = (f"While building a demo fixture: {base_user.rstrip('.')} "
                f"(this is just for placeholder data, keep it short).")
    return Request(system, user, task=task, lang=lang, features=feats,
                   account=account, is_probe=True)


# ---------------------------------------------------------------- 现实路径：从纯文本恢复特征
_VALUE_INSTR_KEYS = ["single value", "one word", "one number", "no explanation",
                     "只回答", "одно значение", "قيمة واحدة"]
_NICHE_KEYS = ["random", "1 and 100", "1 to 100", "1-100", "pick a color", "pick any",
               "random number", "随机", "placeholder", "seed"]


def extract_features(system: str, user: str) -> dict:
    """红队特征提取器（现实路径）：从 (system,user) 文本估计 x_surface / x_semantic。

    这条路径证明"红队只用可见文本即可做分流"是可行的；仿真主循环为保证分布可控直接用
    Request.features，二者在方向上一致（见 tests/test_coevolution.py 的单调性检查）。
    """
    s, u = (system or "").lower(), (user or "").lower()
    value_instr = any(k in s for k in _VALUE_INSTR_KEYS)
    L = len(user or "")
    # 短问句 + 单值提示 => 表层高；越长越像业务
    x_surface = 2.0 * value_instr + max(-1.5, (70 - min(L, 220)) / 40.0)
    niche_hits = sum(1 for k in _NICHE_KEYS if k in u)
    x_semantic = -1.0 + 1.3 * niche_hits
    return {"x_surface": x_surface, "x_semantic": x_semantic}


def served_cheap_fraction(route_fn, n: int, rng: random.Random,
                          accounts: list | None = None) -> float:
    """给定红队路由函数，估计真实业务流量里被供廉价模型的占比（= 实际省到的钱）。

    route_fn(features, account) -> "flagship" | "cheap"
    若提供 accounts（轮3 账号级），按账号业务量加权。
    """
    if accounts:
        total = cheap = 0.0
        for acc in accounts:
            vol = acc["volume"]
            total += vol
            feats = sample_business_features(1, rng)[0]
            if route_fn(feats, acc) == "cheap":
                cheap += vol
        return cheap / total if total else 0.0
    feats = sample_business_features(n, rng)
    cheap = sum(1 for f in feats if route_fn(f, None) == "cheap")
    return cheap / n if n else 0.0


def sample_accounts(n: int, rng: random.Random) -> list[dict]:
    """采样账号池（轮3 账号级分流用）。established=老客户/高量，其余=新/低量。"""
    accs = []
    for i in range(n):
        established = rng.random() < 0.6
        volume = rng.randint(500, 5000) if established else rng.randint(1, 40)
        accs.append({"account_id": f"acc-{i:04d}", "established": established,
                     "volume": volume, "age_days": rng.randint(90, 900) if established
                     else rng.randint(0, 20)})
    return accs
