"""对手仿真：把 MockModel 包装成各种"中转站行为"，返回进程内 Endpoint。

覆盖研究笔记 §3 威胁模型的各层级 + 服务层完整性违规 + 两种"反直觉"情形：

  honest           诚实端点：真身直出
  drift            诚实但跨部署漂移（不同 provider 的同一模型）——不能误杀（考验 δ 容差）
  quantized        量化降级的同名模型（轻度作弊）
  swap             整体替换成另一个模型（经典掺水）
  pin              T2' 钉死对手：系统提示"随机数只答73"之类，钉死部分任务的答案
  filter_en        T2 过滤对手：识别英文已公开措辞路由给真身，其余给便宜货
  true_random      "聪明"对手：用真随机数发生器回答数字类问题（反而暴露——真模型不随机）
  cache            响应级缓存：第一次真采样，之后回放（分布塌缩+低延迟）
  partial_mimic    T3 弱化版：只在最常见任务(1-100随机数)上复刻了真身分布表

协同进化新增的分流对手（见 docs/COEVOLUTION.md，不进 ADVERSARY_KINDS 以免影响
现有实验/研究笔记的既定数字，另列 COEVO_ADVERSARY_KINDS）：

  probe_aware      按"表层可区分度"分流：像审计探针的短问句/单值提示 -> 供旗舰，其余 -> 廉价
  semantic_aware   按"语义可区分度"分流：认出"说个1-100/颜色"这类低熵琐碎问题 -> 供旗舰
  account_routing  按账号分流：老客户/高流量账号 -> 廉价，陌生/新/低流量账号 -> 旗舰
                   （一次性外部审计者必然用陌生账号，因此被恒供旗舰而逃检——原理性极限）

所有分流对手都由统一的 RedRelay 实现（红蓝协同进化的单一事实源），make_endpoint
通过 traffic.extract_features 从可见文本还原分流特征，保证可经标准 ask() 接口运行。
"""

from __future__ import annotations

import random

from fpverify.endpoints import Answer, CallableEndpoint
from fpverify import probes as probes_mod
from .mock_models import MockModel
from . import traffic


def _find_task(user_prompt: str) -> tuple[str, str] | None:
    """对手视角：从措辞反查 (task, lang)。对手能读到流量，这里如实模拟。"""
    for t in probes_mod.TASKS:
        for lang, tmpls in t.templates.items():
            if user_prompt in tmpls:
                return t.task, lang
    return None


def _semantic_guess(user_prompt: str) -> tuple[str | None, str | None]:
    """对手做语义识别（不靠模板匹配）——用于 pin/true_random 这类语义级特判。"""
    p = user_prompt.lower()
    lang = "en"
    if any("\u4e00" <= ch <= "\u9fff" for ch in user_prompt):
        lang = "zh"
    elif any("\u0400" <= ch <= "\u04ff" for ch in user_prompt):
        lang = "ru"
    elif any("\u0600" <= ch <= "\u06ff" for ch in user_prompt):
        lang = "ar"
    if ("100" in p) and any(w in p for w in ["random", "number", "随机", "数", "число", "رقم"]):
        return "rand_num_100", lang
    if ("10" in p) and any(w in p for w in ["random", "number", "随机", "数", "число", "رقم"]):
        return "rand_num_10", lang
    return None, lang


class _Latency:
    """简单的延迟仿真：正常解码 ~N(0.8, 0.15)，缓存命中 ~N(0.05, 0.01)。"""

    def __init__(self, rng):
        self.rng = rng

    def normal(self):
        return max(0.05, self.rng.gauss(0.8, 0.15))

    def cached(self):
        return max(0.005, self.rng.gauss(0.05, 0.01))


# ==================================================================== 分流红队
# 路由策略：feature dict + account -> "flagship" | "cheap"
# 每个策略都自带一个"廉价流量占比"的语义：阈值越松（越倾向判 flagship），
# 抓探针越准但省钱越少。经济约束（见 docs/COEVOLUTION.md）就作用在这条权衡上。

def route_probe_aware(features: dict, account=None, surface_thr: float = 1.3) -> str:
    """轮1 红队：表层特征超阈值判'疑似测试'供旗舰。对裸探针有效，对表层伪装失效。"""
    return "flagship" if features.get("x_surface", 0.0) >= surface_thr else "cheap"


def route_semantic_aware(features: dict, account=None, sem_thr: float = 1.2) -> str:
    """轮2 红队：改看语义特征（是不是'说个1-100'这类低熵问题）。对表层伪装仍有效，
    但会把业务里 8% 的天然低熵请求也误判成旗舰，且对 native 伪装（语义也像业务）失效。"""
    return "flagship" if features.get("x_semantic", 0.0) >= sem_thr else "cheap"


def route_combined(features: dict, account=None,
                   surface_thr: float = 1.3, sem_thr: float = 1.2) -> str:
    """表层或语义任一超阈值即供旗舰（红队想两头堵）。伪装把两个特征都压向业务后，
    要抓到 native 探针必须把阈值压到业务分布内部，从而误判大量业务 -> 省钱坍塌。"""
    if features.get("x_surface", 0.0) >= surface_thr:
        return "flagship"
    if features.get("x_semantic", 0.0) >= sem_thr:
        return "flagship"
    return "cheap"


def route_account(features: dict, account=None) -> str:
    """轮3 红队：不看单条请求，看账号。老客户/高流量账号供廉价（省大钱），
    陌生/新账号供旗舰（外部一次性审计者必然是陌生账号 => 被恒供旗舰 => 逃检）。"""
    if account is None:
        return "flagship"           # 无账号上下文（典型外部审计）=> 保守供旗舰
    return "cheap" if account.get("established") else "flagship"


def route_dilution(features: dict, account=None, eps: float = 0.3, rng=None) -> str:
    """轮4 红队：内容/账号都不看，对**所有**流量以概率 eps 供廉价、否则供旗舰。

    这是抗审计的"稀释攻击"：省到的钱 = eps（对全部流量生效），且探针与业务被一视同仁，
    因此伪装/最近邻/被动审计都无法把探针择出来。它的克星只有序贯检验的容差 δ——
    只要 eps 落在 δ 支撑的"良性邻域"内，单次有限预算审计就无法在不破坏 FPR 的前提下检出
    （见 docs/COEVOLUTION.md 的收敛分析：这是军备竞赛的稳定点/原理性极限）。
    rng 为 None 时退化到模块级随机（仅用于占比估计）；RedRelay 内部改用自带 rng 保证可复现。
    """
    import random as _r
    r = (rng or _r).random()
    return "cheap" if r < eps else "flagship"


ROUTE_POLICIES = {
    "probe_aware": route_probe_aware,
    "semantic_aware": route_semantic_aware,
    "combined": route_combined,
    "account_routing": route_account,
    "dilution": route_dilution,
}


class RedRelay:
    """分流型中转站：对每个请求先决定供旗舰(真身)还是廉价模型，再采样作答。

    这是协同进化里"红队"的统一实现。检测侧看到的是标准 Endpoint.ask()；
    路由所需的可观测特征由 traffic.extract_features 从 (system,user) 文本还原。
    评估侧另有 route()/serve() 直接接口，供 experiments/coevolution.py 精确统计
    廉价流量占比与逃检率（不必经过文本往返）。
    """

    def __init__(self, claimed_model: str, policy: str, seed: int = 0,
                 actual: str = "cheap-7b", account=None, **params):
        self.claimed = claimed_model
        self.policy_name = policy
        self.route_fn = ROUTE_POLICIES[policy]
        self.params = params
        self.rng = random.Random(seed)
        self.lat = _Latency(self.rng)
        self.genuine = MockModel(claimed_model)
        self.cheap = MockModel(actual)
        self.account = account
        self.n_flagship = 0
        self.n_cheap = 0

    def route(self, features: dict, account=None) -> str:
        acc = account if account is not None else self.account
        params = dict(self.params)
        if self.policy_name == "dilution":
            params["rng"] = self.rng       # 用自带 rng 保证稀释决策可复现
        return self.route_fn(features, acc, **params)

    def serve(self, task: str, lang: str, features: dict, account=None) -> tuple[str, str]:
        """按路由决定源模型并采样。返回 (归一化前的文本答案, 'flagship'|'cheap')。"""
        decision = self.route(features, account)
        src = self.genuine if decision == "flagship" else self.cheap
        if decision == "flagship":
            self.n_flagship += 1
        else:
            self.n_cheap += 1
        raw = src.sample(task, lang, self.rng)
        return src.format_answer(task, raw, self.rng), decision

    def ask(self, system: str, user: str) -> Answer:
        found = _find_task(user)
        if not found:
            # 非探针业务请求：走特征路由（仍如实计入 flagship/cheap 便于占比统计）
            feats = traffic.extract_features(system, user)
            self.route(feats)
            return Answer(text="OK", model_field=self.claimed, latency=self.lat.normal())
        task, lang = found
        feats = traffic.extract_features(system, user)
        text, decision = self.serve(task, lang, feats)
        latency = self.lat.normal()
        return Answer(text=text, model_field=self.claimed, latency=latency)


class BenignRelay:
    """诚实端点的 relay 形态：无论请求怎么伪装，都按真身(或跨部署漂移)分布如实作答。

    用于蓝队 FPR 守恒测试：检测器升级（探针伪装 / 最近邻 / 被动审计）之后，
    仍不得把真身端点误杀。它提供与 RedRelay 相同的 serve() 接口，
    使 BlueAuditor 无需区分被测对象即可统一驱动。
    """

    def __init__(self, claimed_model: str, seed: int = 0,
                 drift_seed: int | None = None, quantize: float = 0.0):
        self.claimed = claimed_model
        self.rng = random.Random(seed)
        self.lat = _Latency(self.rng)
        self.model = MockModel(claimed_model, quantize=quantize, drift_seed=drift_seed)
        self.n_flagship = 0
        self.n_cheap = 0

    def route(self, features: dict, account=None) -> str:
        return "flagship"

    def serve(self, task: str, lang: str, features: dict, account=None) -> tuple[str, str]:
        self.n_flagship += 1
        raw = self.model.sample(task, lang, self.rng)
        return self.model.format_answer(task, raw, self.rng), "flagship"

    def ask(self, system: str, user: str) -> Answer:
        found = _find_task(user)
        if not found:
            return Answer(text="OK", model_field=self.claimed, latency=self.lat.normal())
        task, lang = found
        raw = self.model.sample(task, lang, self.rng)
        return Answer(text=self.model.format_answer(task, raw, self.rng),
                      model_field=self.claimed, latency=self.lat.normal())


def make_endpoint(kind: str, claimed_model: str, seed: int = 0, **kw) -> CallableEndpoint:
    rng = random.Random(seed)
    lat = _Latency(rng)

    genuine = MockModel(claimed_model)

    if kind == "honest":
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            raw = genuine.sample(task, lang, rng)
            return Answer(text=genuine.format_answer(task, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "drift":
        drifted = MockModel(claimed_model, drift_seed=kw.get("drift_seed", 7))
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            raw = drifted.sample(task, lang, rng)
            return Answer(text=drifted.format_answer(task, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "quantized":
        quant = MockModel(claimed_model, quantize=kw.get("level", 1.0))
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            raw = quant.sample(task, lang, rng)
            return Answer(text=quant.format_answer(task, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "swap":
        cheap = MockModel(kw.get("actual", "cheap-7b"))
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            raw = cheap.sample(task, lang, rng)
            # model 字段照样撒谎
            return Answer(text=cheap.format_answer(task, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "pin":
        # 语义级特判：数字类问题一律答钉死值（评论区"只回答73"方案），其余给便宜货
        cheap = MockModel(kw.get("actual", "cheap-7b"))
        pin_value = kw.get("pin_value", "73")
        def fn(system, user):
            task, lang = _semantic_guess(user)
            if task in ("rand_num_100", "rand_num_10"):
                return Answer(text=pin_value, model_field=claimed_model, latency=lat.normal())
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            t, l = found
            raw = cheap.sample(t, l, rng)
            return Answer(text=cheap.format_answer(t, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "filter_en":
        # T2：认得英文第一条模板（假设它被公开过），只把它路由给真身
        cheap = MockModel(kw.get("actual", "cheap-7b"))
        published = set()
        for t in probes_mod.TASKS:
            if "en" in t.templates:
                published.add(t.templates["en"][0])
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            src = genuine if user in published else cheap
            raw = src.sample(task, lang, rng)
            return Answer(text=src.format_answer(task, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "true_random":
        # 对手以为"回答真随机"最安全——恰恰相反，真模型是不随机的
        cheap = MockModel(kw.get("actual", "cheap-7b"))
        def fn(system, user):
            task, lang = _semantic_guess(user)
            if task == "rand_num_100":
                return Answer(text=str(rng.randint(1, 100)), model_field=claimed_model, latency=lat.normal())
            if task == "rand_num_10":
                return Answer(text=str(rng.randint(1, 10)), model_field=claimed_model, latency=lat.normal())
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            t, l = found
            raw = cheap.sample(t, l, rng)
            return Answer(text=cheap.format_answer(t, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "cache":
        # 响应级缓存：同一 (task,lang) 第一次真采样，之后 90% 概率回放
        store: dict = {}
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            key = (task, lang)
            if key in store and rng.random() < 0.9:
                return Answer(text=store[key], model_field=claimed_model, latency=lat.cached())
            raw = genuine.sample(task, lang, rng)
            text = genuine.format_answer(task, raw, rng)
            store[key] = text
            return Answer(text=text, model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind == "partial_mimic":
        # T3 弱化版：对手花功夫测出了真身在 rand_num_100 上的分布表并复刻，其余任务露馅
        cheap = MockModel(kw.get("actual", "cheap-7b"))
        def fn(system, user):
            found = _find_task(user)
            if not found:
                return Answer(text="OK", model_field=claimed_model, latency=lat.normal())
            task, lang = found
            src = genuine if task == "rand_num_100" else cheap
            raw = src.sample(task, lang, rng)
            return Answer(text=src.format_answer(task, raw, rng),
                          model_field=claimed_model, latency=lat.normal())
        return CallableEndpoint(fn)

    if kind in ("probe_aware", "semantic_aware", "combined", "account_routing", "dilution"):
        # 协同进化的分流红队：统一由 RedRelay 实现（本身就是 Endpoint）
        return RedRelay(claimed_model, policy=kind, seed=seed,
                        actual=kw.get("actual", "cheap-7b"),
                        account=kw.get("account"),
                        **{k: v for k, v in kw.items() if k not in ("actual", "account")})

    raise ValueError(f"unknown adversary kind: {kind}")


# 现有 9 类对手（研究笔记 §6 的既定评估集，保持稳定不动）
ADVERSARY_KINDS = ["honest", "drift", "quantized", "swap", "pin",
                   "filter_en", "true_random", "cache", "partial_mimic"]

# 协同进化新增的分流对手（独立列出，不并入 ADVERSARY_KINDS）
COEVO_ADVERSARY_KINDS = ["probe_aware", "semantic_aware", "combined",
                         "account_routing", "dilution"]
