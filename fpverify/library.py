"""公共指纹库 + 降档识别（identify）。

用户画像：买了中转站、**没有官方渠道**的普通用户——正因为买不到官方 API 才买中转站，
让他自己去官方入册参考指纹是个死循环。解法：社区把主流模型的参考指纹入册进公开的
refs/ 库（带渠道、日期、来源出处），用户只需提供中转站地址 + key + 声称的模型名。

降档判定阶梯（能精确就精确，不能精确就如实降档）：
  1. 声称的模型在库里   → 序贯下注检验验真伪（FPR ≤ α 保证）+ 全库最近邻佐证；
  2. 声称的模型不在库里 → 报告"行为上与库内谁一致"（BEST_MATCH）；
  3. 谁都不像           → UNKNOWN（未入册的新模型 / 深度定制），如实说不知道。

安全边界：审计流量从用户本机发出，key 不经过任何第三方后台；库本身是公开数据、
不含任何密钥。参考指纹公开会帮对手"照着库演"——防线不在库的保密性，而在措辞/语言
随机化的新鲜探针与"完美模仿 ≈ 白跑一遍真模型"的成本下限（见研究笔记 §5）。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import probes
from .betting import BettingConfig, SequentialBettingTest
from .calibrate import calibrate_delta
from .fingerprint import Fingerprint
from .nearest import nearest_model
from .normalize import normalize, classify_validity
from .screens import screen_response_cache

# JSD 解读带（对照论文参考带：0.140 同源 / 0.227 跨部署 / 0.463 冒充者）
BAND_MATCH = 0.18     # 最近邻距离 ≤ 此值：行为上可认为一致
BAND_UNKNOWN = 0.32   # 最近邻距离 ≥ 此值：库里没有像它的，判未知

# ---- 采集协议：指纹是（模型 × 渠道 × 协议 × 档位）的条件分布，协议是一等属性 ----
# 在线探针协议：enroll/audit/identify 探测端点时用的问法——每题独立请求、全新对话。
PROBE_PROTOCOL = "cold-single"
# harness 参考的采集协议：一个全新实例一次答完整卷十题（见 experiments/frontier/PROTOCOL.md）。
HARNESS_PROTOCOL = "harness-battery"

# 跨协议时同一个模型答案本就不同，唯一正确处理：只比相对排名、不做硬判定。
# 全局唯一措辞，供 identify / CLI / WebUI 复用，避免各处说法不一致。
CROSS_PROTOCOL_NOTE = (
    "参考按套卷协议采集（一个实例一次答完十题），在线识别用的是单题冷探针——"
    "两种协议下同一模型的答案本就不同（实测抛硬币：套卷=tails、冷问=heads）。"
    "故跨协议只比『最像谁』的相对排名，不做 PASS/FAIL 硬判定；要硬判定请用官方 key 走 "
    "enroll+audit（同协议），或用 reproduce 复核包按套卷协议复跑。"
)


def entry_protocol(entry: "LibraryEntry") -> str:
    """条目的采集协议：显式 protocol 优先，否则按渠道推断（harness=套卷，其余=冷探针）。"""
    if getattr(entry, "protocol", ""):
        return entry.protocol
    return HARNESS_PROTOCOL if "harness" in entry.channel else PROBE_PROTOCOL


def default_library_path() -> Path:
    """仓库内置库目录 refs/（相对本文件定位，安装为包后仍可用 --library 覆盖）。"""
    return Path(__file__).resolve().parents[1] / "refs"


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@dataclass
class LibraryEntry:
    id: str
    model: str                # 展示名，如 "Claude Fable 5"
    family: str = ""
    channel: str = "api"      # api / cursor-harness / simulation
    protocol: str = ""        # 采集协议；缺省由 entry_protocol() 按渠道推断
    file: str = ""            # 相对库根目录的指纹文件路径
    enrolled_at: str = ""
    samples_per_cell: int = 0
    source: str = ""
    note: str = ""


class Library:
    def __init__(self, root: Path, entries: list[LibraryEntry], meta: dict | None = None):
        self.root = Path(root)
        self.entries = entries
        self.meta = meta or {}
        self._fps: dict[str, Fingerprint] = {}

    @classmethod
    def load(cls, root: str | Path) -> "Library":
        root = Path(root)
        manifest = root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"指纹库清单不存在: {manifest}\n"
                f"（仓库自带 refs/；也可用 --library 指向自建库目录）")
        d = json.loads(manifest.read_text(encoding="utf-8"))
        entries = [LibraryEntry(**{k: v for k, v in e.items()
                                   if k in LibraryEntry.__dataclass_fields__})
                   for e in d.get("entries", [])]
        meta = {k: v for k, v in d.items() if k != "entries"}
        return cls(root, entries, meta)

    def by_channel(self, channel: str) -> list[LibraryEntry]:
        return [e for e in self.entries if e.channel == channel]

    def channels(self) -> list[str]:
        seen = []
        for e in self.entries:
            if e.channel not in seen:
                seen.append(e.channel)
        return seen

    def get(self, entry_id: str) -> LibraryEntry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def resolve(self, name: str, channel: str | None = None) -> LibraryEntry | None:
        """把用户口中的模型名解析成库条目：先精确 id / 展示名，再唯一子串匹配。"""
        if not name:
            return None
        pool = self.by_channel(channel) if channel else self.entries
        q = _norm_name(name)
        for e in pool:
            if q == _norm_name(e.id) or q == _norm_name(e.model):
                return e
        subs = [e for e in pool
                if q and (q in _norm_name(e.id) or q in _norm_name(e.model)
                          or _norm_name(e.id) in q or _norm_name(e.model) in q)]
        return subs[0] if len(subs) == 1 else None

    def fingerprint(self, entry: LibraryEntry) -> Fingerprint:
        if entry.id not in self._fps:
            self._fps[entry.id] = Fingerprint.load(self.root / entry.file)
        return self._fps[entry.id]


# ---------------------------------------------------------------- identify

@dataclass
class IdentifyResult:
    claimed: str                      # 用户声称的模型名（原话）
    claimed_entry: str | None         # 解析到的库条目 id；None = 声称不在库里
    channel: str
    verdict: str                      # PASS / FAIL / SUSPECT / BEST_MATCH / UNKNOWN / INCONCLUSIVE
    detail: str
    warning: str = ""
    protocol: str = ""                # 参考渠道的采集协议
    protocol_matched: bool = True     # 探针协议是否与参考一致；False → 只能相对排名，无硬判定
    n_queries: int = 0
    errors: int = 0
    ranking: list = field(default_factory=list)    # [(entry_id, 聚合JSD)] 升序
    nearest: str | None = None
    nearest_distance: float | None = None
    claimed_distance: float | None = None
    betting: dict | None = None       # {wealth, threshold, alpha, delta}（仅库内验证时有）
    cache_flags: list = field(default_factory=list)
    bands: dict = field(default_factory=lambda: {"match": BAND_MATCH, "unknown": BAND_UNKNOWN})
    observed_counts: dict = field(default_factory=dict)  # cell -> {答案: 次数}，供脱离本工具独立重算

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ranking"] = [[m, round(x, 4)] for m, x in self.ranking]
        d["observed_counts"] = {f"{k[0]}::{k[1]}": dict(v) for k, v in self.observed_counts.items()}
        return d


def identify(endpoint, library: Library, claimed: str, channel: str = "api",
             samples_per_cell: int = 8, alpha: float = 0.01,
             seed: int | None = None, progress=None) -> IdentifyResult:
    """对端点做降档识别。见模块 docstring 的三级阶梯。

    采样预算 = samples_per_cell × 库频道内出现过的 cell 数（api 全探针 36 cell、
    默认 8 样本 → 288 次单 token 请求）。若声称模型在库里，同一条采样流同时喂给
    序贯下注检验（不额外花钱）；定案后不提前停，把配额采完以便给出"最像谁"的归属。
    """
    entries = library.by_channel(channel)
    if not entries:
        return IdentifyResult(
            claimed=claimed, claimed_entry=None, channel=channel,
            verdict="INCONCLUSIVE",
            detail=f"指纹库的 {channel} 频道还没有条目，无法识别。"
                   f"可用频道: {library.channels() or '（空库）'}。欢迎按 refs/README.md 贡献参考指纹。")

    refs = {e.id: library.fingerprint(e).counts() for e in entries}
    families = {e.id: e.family for e in entries}
    claimed_entry = library.resolve(claimed, channel)

    # 渠道内条目协议一致；在线探针恒为冷单题。协议一致才允许硬判定。
    chan_protocol = entry_protocol(entries[0])
    protocol_matched = (chan_protocol == PROBE_PROTOCOL)

    # cell 全集 = 该频道所有参考里出现过的 cell（且探针银行认识它）
    known = set(probes.all_cells())
    cells = sorted({c for counts in refs.values() for c in counts} & known)
    if not cells:
        return IdentifyResult(claimed=claimed, claimed_entry=None, channel=channel,
                              verdict="INCONCLUSIVE", detail="库条目与探针银行没有交集 cell。")

    rng = random.Random(seed)

    # 库内验证：同一条流喂序贯检验。
    # 仅在协议一致（参考也是冷单题）时才建检验：跨协议下真身也会显著偏离参考
    # （实测同模型抛硬币套卷=tails、冷问=heads），α 保证不成立，只做相对排名。
    test = None
    cfg = BettingConfig(alpha=alpha)
    betting_cells: set = set()
    if claimed_entry is not None and protocol_matched:
        ref_counts = refs[claimed_entry.id]
        bet_ref = {c: ref_counts[c] for c in cells
                   if sum(ref_counts.get(c, {}).values()) >= 10}
        if bet_ref:
            betting_cells = set(bet_ref)
            total_budget = samples_per_cell * len(cells)
            delta = calibrate_delta(bet_ref, cfg, horizon=total_budget)
            cfg = BettingConfig(alpha=alpha, delta=delta)
            test = SequentialBettingTest(bet_ref, cfg)

    # 轮转采样：samples_per_cell 轮 × 每轮全部 cell（轮内乱序、措辞随机）
    test_fp = Fingerprint(model="<endpoint-under-test>")
    n = 0
    errors = 0
    total = samples_per_cell * len(cells)
    for _ in range(samples_per_cell):
        order = list(cells)
        rng.shuffle(order)
        for cell in order:
            task, lang = cell
            system, user = probes.render_prompt(task, lang, rng)
            ans = endpoint.ask(system, user)
            n += 1
            if ans.error:
                errors += 1
                if errors > max(20, n * 0.5):
                    return IdentifyResult(
                        claimed=claimed,
                        claimed_entry=claimed_entry.id if claimed_entry else None,
                        channel=channel, verdict="INCONCLUSIVE",
                        detail=f"错误过多（{errors}/{n}），最后错误：{ans.error}",
                        n_queries=n, errors=errors)
                continue
            ptype = probes.parse_type(task)
            tok = normalize(ptype, ans.text)
            test_fp.add(cell, tok, classify_validity(ptype, ans.text), ans.latency)
            if test is not None and cell in betting_cells and not test.decided:
                test.observe(cell, tok)
            if progress and n % 10 == 0:
                progress(n, total, test.wealth if test is not None else None)

    # 全库最近邻归属
    min_s = min(8, samples_per_cell)
    near = nearest_model(test_fp.counts(), refs,
                         claimed=claimed_entry.id if claimed_entry else claimed,
                         min_samples=min_s, same_family=families,
                         match_band=BAND_MATCH)
    cache_flags = screen_response_cache(test_fp, min_samples=min_s)
    flags = [f"{c.cell[0]}::{c.cell[1]}" for c in cache_flags]

    res = IdentifyResult(
        claimed=claimed,
        claimed_entry=claimed_entry.id if claimed_entry else None,
        channel=channel, verdict="", detail="", n_queries=n, errors=errors,
        ranking=near.ranking, nearest=near.nearest,
        nearest_distance=near.nearest_distance, claimed_distance=near.claimed_distance,
        cache_flags=flags, observed_counts=test_fp.counts(),
        protocol=chan_protocol, protocol_matched=protocol_matched)

    def _finish(r: IdentifyResult) -> IdentifyResult:
        # 参考与在线单题探针跨协议：绝对距离对真身也会整体偏大，只能看相对排名。
        if not protocol_matched:
            r.warning = f"{r.warning}　{CROSS_PROTOCOL_NOTE}" if r.warning else CROSS_PROTOCOL_NOTE
        return r

    # ---- 判定阶梯 ----
    if claimed_entry is not None and test is not None:
        res.betting = {"wealth": test.wealth, "threshold": 1.0 / cfg.alpha,
                       "alpha": cfg.alpha, "delta": cfg.delta}
        if test.rejected:
            res.verdict = "FAIL"
            res.detail = (f"序贯检验拒绝了『端点是 {claimed_entry.model}』的假设"
                          f"（误判概率 ≤ {cfg.alpha}）：行为指纹显著偏离参考，判定注水/偷换。")
            if near.flagged:
                res.detail += f" 归属线索：{near.reason}"
        else:
            res.verdict = "PASS"
            res.detail = (f"预算 {n} 次查询内未发现偷换证据，行为与库内参考"
                          f"『{claimed_entry.model}』一致（α={cfg.alpha}）。"
                          f"注意：低比例随机掺水需长期复测积累证据。")
            if near.flagged:
                res.warning = f"最近邻信号与声称不一致，建议加大采样复测：{near.reason}"
        if flags and res.verdict != "FAIL":
            res.verdict = "FAIL"
            res.detail += " 另检出疑似响应级缓存（服务完整性违规，独立于指纹证据）。"
        return _finish(res)

    # 纯识别降档：声称不在库里，或 harness 频道（套卷参考 × 冷探针，跨协议）
    if not near.ranking:
        res.verdict = "INCONCLUSIVE"
        res.detail = "有效样本不足，无法与库比对（可加大 --samples 或检查端点报错）。"
        return _finish(res)

    best_id, best_d = near.ranking[0]
    best_entry = library.get(best_id)
    best_name = best_entry.model if best_entry else best_id
    if claimed_entry is not None:
        why = (f"『{claimed_entry.model}』的参考与在线单题探针跨协议，只做相对排名、不做硬判定。"
               if not protocol_matched
               else f"『{claimed_entry.model}』样本不足以序贯定论，退为相对排名。")
        cd = (f"；到声称参考的距离 {res.claimed_distance:.3f}"
              if res.claimed_distance is not None else "")
    else:
        why = f"声称的『{claimed}』不在指纹库中，无法直接验真。"
        cd = ""
    if best_d <= BAND_MATCH:
        res.verdict = "BEST_MATCH"
        note = ("与声称一致。" if (claimed_entry is not None and best_id == claimed_entry.id)
                else "若这与声称的档次不符，就是问题。")
        res.detail = (f"{why}端点行为与库内『{best_name}』一致"
                      f"（聚合 JSD={best_d:.3f} ≤ {BAND_MATCH}{cd}）。{note}")
    elif best_d >= BAND_UNKNOWN:
        res.verdict = "UNKNOWN"
        res.detail = (f"{why}端点行为与库内任何模型都不像"
                      f"（最近的『{best_name}』也有 JSD={best_d:.3f} ≥ {BAND_UNKNOWN}{cd}）。"
                      f"可能是未入册的新模型或深度定制部署——如实报告：未知。")
    else:
        res.verdict = "INCONCLUSIVE"
        res.detail = (f"{why}端点最像库内『{best_name}』"
                      f"（JSD={best_d:.3f}{cd}），但落在灰区 ({BAND_MATCH}, {BAND_UNKNOWN})，"
                      f"证据不足以定论。建议加大 --samples。")
    if (claimed_entry is not None and best_id != claimed_entry.id
            and res.claimed_distance is not None
            and res.claimed_distance - best_d >= 0.05):
        hint = (f"排名线索：最像『{best_name}』而非声称的参考"
                f"（差距 {res.claimed_distance - best_d:.3f}）。跨协议对比仅供参考，"
                f"归属以复核包套卷双跑为准。")
        res.warning = f"{hint}　{res.warning}" if res.warning else hint
    if flags:
        res.detail += " 另检出疑似响应级缓存（服务完整性违规）。"
    return _finish(res)
