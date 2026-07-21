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
    """真身声称自己 → PASS（库内验真，带 FPR 保证）。"""
    ep = make_endpoint("honest", "gpt-4o", seed=41)
    res = identify(ep, api_library, "gpt-4o", channel="api", samples_per_cell=8, seed=41)
    assert res.claimed_entry == "gpt-4o"
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


# ---------------------------------------------------------------- 仓库自带 refs/

def test_bundled_refs_load():
    lib = Library.load(default_library_path())
    harness = lib.by_channel("cursor-harness")
    assert len(harness) >= 9
    fp = lib.fingerprint(lib.resolve("Claude Fable 5", "cursor-harness"))
    assert len(fp.cells) == 10 and fp.total_samples() == 110


def test_bundled_refs_identify_harness_channel():
    """回放 fable5 实测分布的端点，在 harness 频道声称 fable5 → 不应被判 FAIL；
    声称 composer25（行为差异大的另一家）→ 不应 PASS。"""
    lib = Library.load(default_library_path())
    counts = lib.fingerprint(lib.get("fable5")).counts()

    res = identify(replay_endpoint(counts, 51), lib, "fable5",
                   channel="cursor-harness", samples_per_cell=8, seed=51)
    assert res.claimed_entry == "fable5"
    assert res.verdict != "FAIL", f"{res.verdict}: {res.detail}"

    res2 = identify(replay_endpoint(counts, 52), lib, "composer25",
                    channel="cursor-harness", samples_per_cell=8, seed=52)
    assert res2.verdict != "PASS", f"回放 fable5 冒充 composer25 竟然 PASS: {res2.detail}"
