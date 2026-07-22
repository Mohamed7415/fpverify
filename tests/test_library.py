# -*- coding: utf-8 -*-
"""公共指纹库 + identify 降档识别的行为测试。

覆盖三级阶梯：库内验真（PASS/FAIL+归属）、库外识别（BEST_MATCH）、未知模型（UNKNOWN），
以及仓库自带 refs/ 的完整性。全部离线、固定种子。
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from fpverify import probes
from fpverify.endpoints import Answer, CallableEndpoint
from fpverify.fingerprint import Fingerprint
from fpverify.library import Library, default_library_path, identify
from fpverify.verifier import Verifier
from sim.adversaries import make_endpoint

# 措辞模板 -> cell 反查（供回放端点定位被问的是哪个 cell）
TEMPLATE2CELL = {tpl: (t.task, lang)
                 for t in probes.TASKS
                 for lang, tpls in t.templates.items()
                 for tpl in tpls}


def replay_endpoint(counts: dict, seed: int) -> CallableEndpoint:
    """按给定指纹计数回放答案的端点（模拟'行为与该指纹一致'的服务）。"""
    rng = random.Random(seed)

    def fn(system, user):
        cell = TEMPLATE2CELL[user]
        c = counts.get(cell)
        if not c:
            return Answer(text="")
        toks, weights = zip(*c.items())
        return Answer(text=rng.choices(toks, weights=weights)[0], latency=0.05)

    return CallableEndpoint(fn)


@pytest.fixture(scope="module")
def api_library(tmp_path_factory):
    """用仿真世界造一个 api 频道的小库：gpt-4o 与 qwen3-235b 两个条目。"""
    root = tmp_path_factory.mktemp("refs")
    (root / "api").mkdir()
    entries = []
    for model in ("gpt-4o", "qwen3-235b"):
        v = Verifier(seed=7)
        fp = v.enroll(make_endpoint("honest", model, seed=3), model, samples_per_cell=15)
        fp.save(root / "api" / f"{model}.json")
        entries.append({"id": model, "model": model, "family": model.split("-")[0],
                        "channel": "api", "file": f"api/{model}.json",
                        "enrolled_at": "2026-07-21", "samples_per_cell": 15,
                        "source": "sim"})
    (root / "manifest.json").write_text(
        json.dumps({"version": 1, "updated_at": "2026-07-21", "entries": entries}),
        encoding="utf-8")
    return Library.load(root)


def test_resolve_fuzzy(api_library):
    assert api_library.resolve("GPT 4o", "api").id == "gpt-4o"
    assert api_library.resolve("qwen3-235b").id == "qwen3-235b"
    assert api_library.resolve("完全不存在的名字", "api") is None


def test_identify_claimed_in_library_passes(api_library):
    """真身声称自己 → PASS（库内验真，带 FPR 保证）。api 参考与探针同为冷协议。"""
    ep = make_endpoint("honest", "gpt-4o", seed=41)
    res = identify(ep, api_library, "gpt-4o", channel="api", samples_per_cell=8, seed=41)
    assert res.claimed_entry == "gpt-4o"
    assert res.protocol == "cold-single" and res.protocol_matched
    assert res.verdict == "PASS", f"{res.verdict}: {res.detail}"
    assert res.betting is not None


def test_identify_substitution_fails_with_attribution(api_library):
    """端点实际是 qwen 却声称 gpt-4o → FAIL，且最近邻指向 qwen。"""
    ep = make_endpoint("honest", "qwen3-235b", seed=42)
    res = identify(ep, api_library, "gpt-4o", channel="api", samples_per_cell=8, seed=42)
    assert res.claimed_entry == "gpt-4o"
    assert res.verdict == "FAIL", f"{res.verdict}: {res.detail}"
    assert res.nearest == "qwen3-235b"


def test_identify_unknown_model_degrades_honestly(api_library):
    """库里没有 llama → 不许瞎猜：UNKNOWN 或明示证据不足，绝不给 PASS/FAIL。"""
    ep = make_endpoint("honest", "llama-3.3-70b", seed=43)
    res = identify(ep, api_library, "llama-3.3-70b", channel="api",
                   samples_per_cell=8, seed=43)
    assert res.claimed_entry is None
    assert res.verdict in ("UNKNOWN", "INCONCLUSIVE"), f"{res.verdict}: {res.detail}"
    assert res.ranking, "库外识别也应给出距离排名"


def test_identify_empty_channel_is_graceful(api_library):
    ep = make_endpoint("honest", "gpt-4o", seed=44)
    res = identify(ep, api_library, "gpt-4o", channel="不存在的频道", samples_per_cell=4)
    assert res.verdict == "INCONCLUSIVE"


def test_nearest_attribution_gated_by_match_band():
    """最近邻在一致带外时不得断言"疑似被替换为该模型"——闭集矮子将军只能报归属未知。

    对应真实用户反馈：端点声称 GPT 却被指认换成了更贵的模型，经济学上荒谬；
    实际多为库外便宜模型（蒸馏像老师）。"""
    from collections import Counter
    from fpverify.nearest import nearest_model

    cell = ("coin_flip", "en")
    enrolled = {"A": {cell: Counter({"a": 30})},
                "B": {cell: Counter({"b": 30})}}

    # 带外最近邻：与 B 半重叠、与 A 完全不相交 → nearest=B 但距离远
    far = nearest_model({cell: Counter({"b": 15, "c": 15})}, enrolled, claimed="A")
    assert far.flagged and far.nearest == "B"
    assert far.nearest_distance > 0.18
    assert "归属未知" in far.reason
    assert "疑似被替换为该模型" not in far.reason

    # 带内最近邻：与 B 同分布 → 保留强归属措辞
    close = nearest_model({cell: Counter({"b": 30})}, enrolled, claimed="A")
    assert close.flagged and close.nearest == "B"
    assert close.nearest_distance <= 0.18
    assert "疑似被替换为该模型或其近亲" in close.reason


def test_identify_out_of_library_impostor_not_misattributed(api_library):
    """库外便宜模型冒充 gpt-4o → FAIL 排除声称；只有最近邻落在一致带内才允许强归属。"""
    ep = make_endpoint("honest", "cheap-7b", seed=45)
    res = identify(ep, api_library, "gpt-4o", channel="api", samples_per_cell=8, seed=45)
    assert res.verdict == "FAIL", f"{res.verdict}: {res.detail}"
    if "疑似被替换为该模型" in res.detail:
        assert res.nearest_distance is not None and res.nearest_distance <= res.bands["match"]


# ---------------------------------------------------------------- 仓库自带 refs/

def test_bundled_refs_load():
    lib = Library.load(default_library_path())
    harness = lib.by_channel("cursor-harness")
    assert len(harness) >= 9
    fp = lib.fingerprint(lib.resolve("Claude Fable 5", "cursor-harness"))
    assert len(fp.cells) == 10 and fp.total_samples() == 110


def test_bundled_refs_identify_harness_channel():
    """harness 参考是套卷协议、探针是冷单题——跨协议只许相对排名，不许 PASS/FAIL 硬判定。

    回放 fable5 实测分布的端点：声称 fable5 → 不应 FAIL 且必须声明跨协议；
    声称 composer25（行为差异大的另一家）→ 不应 PASS。"""
    lib = Library.load(default_library_path())
    counts = lib.fingerprint(lib.get("fable5")).counts()

    res = identify(replay_endpoint(counts, 51), lib, "fable5",
                   channel="cursor-harness", samples_per_cell=8, seed=51)
    assert res.claimed_entry == "fable5"
    assert res.protocol == "harness-battery"
    assert not res.protocol_matched, "harness 参考对冷探针必须标记协议不一致"
    assert res.verdict not in ("PASS", "FAIL"), f"跨协议出了硬判定 {res.verdict}: {res.detail}"
    assert "跨协议" in res.warning, "必须向用户解释跨协议只看相对排名"

    res2 = identify(replay_endpoint(counts, 52), lib, "composer25",
                    channel="cursor-harness", samples_per_cell=8, seed=52)
    assert res2.verdict != "PASS", f"回放 fable5 冒充 composer25 竟然 PASS: {res2.detail}"
    assert not res2.protocol_matched
