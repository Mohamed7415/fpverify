# -*- coding: utf-8 -*-
"""自行复核包（reproduce）的行为测试。

覆盖：铁律提取（占比/样本量阈值、排序）、复核包文件内容（题目原文、参考答案、
可编译的独立脚本）、CLI 子命令、以及仓库自带 harness 参考真的能出包。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from fpverify.cli import main as cli_main
from fpverify.library import Library, default_library_path
from fpverify.reproduce import (
    MIN_SHARE, build_pack_texts, top_invariants, write_pack,
)


@pytest.fixture(scope="module")
def lib():
    return Library.load(default_library_path())


@pytest.fixture(scope="module")
def gpt56(lib):
    entry = lib.get("gpt56-sol")
    assert entry is not None
    return entry, lib.fingerprint(entry)


def test_top_invariants_pick_iron_laws(gpt56):
    """gpt56-sol 的铁律应含 coin_flip=tails（11/11），且全部达占比阈值。"""
    _, fp = gpt56
    invs = top_invariants(fp, k=6)
    assert invs, "参考里应能挑出铁律"
    by_cell = {inv.cell: inv for inv in invs}
    assert "coin_flip::en" in by_cell
    assert by_cell["coin_flip::en"].expected == "tails"
    assert by_cell["coin_flip::en"].share == 1.0
    for inv in invs:
        assert inv.share >= MIN_SHARE
        assert inv.prompt  # 有可发送的原文
    # 按占比降序
    shares = [inv.share for inv in invs]
    assert shares == sorted(shares, reverse=True)


def test_top_invariants_exclude_scattered_cells(gpt56):
    """rand_letter（8/11=0.73）低于阈值，不该入选。"""
    _, fp = gpt56
    cells = {inv.cell for inv in top_invariants(fp, k=20)}
    assert "rand_letter::en" not in cells


def test_pack_texts_complete_and_consistent(gpt56):
    entry, fp = gpt56
    invs, files = build_pack_texts(entry, fp, runs=10)
    assert set(files) == {"README.md", "battery.txt", "cursor_prompt.md",
                          "codex_loop.ps1", "codex_loop.sh",
                          "official_api.py", "expected.json"}
    # battery 含全部题目原文，编号齐全
    for i, inv in enumerate(invs, 1):
        assert f"{i}. {inv.prompt}" in files["battery.txt"]
    # cursor 提示词内嵌 battery、声称模型名与"不许改答案"的规则
    assert entry.model in files["cursor_prompt.md"]
    assert invs[0].prompt in files["cursor_prompt.md"]
    assert "不要纠正" in files["cursor_prompt.md"]
    # expected.json 可解析且与铁律一致
    exp = json.loads(files["expected.json"])
    assert exp["entry"] == entry.id
    assert exp["invariants"][0]["expected"] == invs[0].expected
    # 官方 API 脚本是合法 Python 且不 import fpverify（独立可运行）
    compile(files["official_api.py"], "official_api.py", "exec")
    assert "import fpverify" not in files["official_api.py"]
    assert "from fpverify" not in files["official_api.py"]
    # ps1 纯 ASCII（PowerShell 5.1 编码兼容）
    files["codex_loop.ps1"].encode("ascii")


def test_write_pack_and_cli(tmp_path, lib):
    out = tmp_path / "pack"
    invs, out_dir = write_pack(lib, lib.get("gpt56-sol"), out)
    assert (out_dir / "README.md").exists()
    assert (out_dir / "official_api.py").exists()

    # CLI：模糊名解析 + 出包
    out2 = tmp_path / "pack_cli"
    rc = cli_main(["reproduce", "--claimed", "GPT-5.6 sol", "--out", str(out2)])
    assert rc == 0
    assert (out2 / "cursor_prompt.md").exists()

    # 库里没有的名字 → 退出码 2
    assert cli_main(["reproduce", "--claimed", "不存在的模型",
                     "--out", str(tmp_path / "nope")]) == 2


def test_all_bundled_refs_can_pack(lib, tmp_path):
    """仓库自带的每个 harness 参考都应能生成复核包（或明确报样本不足）。"""
    for e in lib.entries:
        try:
            invs, files = build_pack_texts(e, lib.fingerprint(e))
        except ValueError:
            continue  # 行为太散的模型如实拒绝，也是合法结果
        assert invs and files["README.md"]
