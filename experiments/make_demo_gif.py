# -*- coding: utf-8 -*-
"""生成 README 顶部的终端演示 GIF（docs/demo.gif）。

跑的是真实流程，不是摆拍：本机起 sim/mock_server.py 的诚实端点与作弊端点，
依次执行 enroll → audit(诚实, 应 PASS) → audit(作弊, 应早停 FAIL)，抓取真实
stdout 后渲染成终端风格动图。进度行做了抽稀（600 行进度没人想看完），其余
输出原样保留。

    py -3.13 -X utf8 experiments/make_demo_gif.py
依赖：Pillow（渲染）、httpx（等待 mock 服务器就绪）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo.gif"

PORT_HONEST, PORT_SWAP = 18801, 18802

# 展示用命令（README 同款 ^ 续行写法，两行显示）与实际执行的 argv
STEPS = [
    (("py -3.13 -X utf8 -m fpverify.cli enroll --base-url http://127.0.0.1:18801/v1 ^",
      "      --api-key mock --model gpt-4o --out ref.json"),
     ["-m", "fpverify.cli", "enroll", "--base-url", f"http://127.0.0.1:{PORT_HONEST}/v1",
      "--api-key", "mock", "--model", "gpt-4o", "--out", "REF_PLACEHOLDER"]),
    (("py -3.13 -X utf8 -m fpverify.cli audit --base-url http://127.0.0.1:18801/v1 ^",
      "      --api-key mock --model gpt-4o --ref ref.json"),
     ["-m", "fpverify.cli", "audit", "--base-url", f"http://127.0.0.1:{PORT_HONEST}/v1",
      "--api-key", "mock", "--model", "gpt-4o", "--ref", "REF_PLACEHOLDER"]),
    (("py -3.13 -X utf8 -m fpverify.cli audit --base-url http://127.0.0.1:18802/v1 ^",
      "      --api-key mock --model gpt-4o --ref ref.json"),
     ["-m", "fpverify.cli", "audit", "--base-url", f"http://127.0.0.1:{PORT_SWAP}/v1",
      "--api-key", "mock", "--model", "gpt-4o", "--ref", "REF_PLACEHOLDER"]),
]

BANNER = [
    "# 本地演示：全程 127.0.0.1，无真实服务参与",
    "#   :18801 = 诚实端点（真 gpt-4o 分布）    :18802 = 作弊中转站（声称 gpt-4o，实际供便宜模型）",
]


# ---------------------------------------------------------------- 抓取

def wait_health(port: int, timeout: float = 15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"mock server :{port} 未就绪")


def capture() -> list[tuple[str, list[str]]]:
    env = {**os.environ, "PYTHONUTF8": "1"}
    servers = []
    tmp = Path(tempfile.mkdtemp(prefix="fpv_demo_"))
    ref = str(tmp / "ref.json")
    try:
        for port, kind in ((PORT_HONEST, "honest"), (PORT_SWAP, "swap")):
            servers.append(subprocess.Popen(
                [sys.executable, "-X", "utf8", str(ROOT / "sim" / "mock_server.py"),
                 "--port", str(port), "--kind", kind, "--model", "gpt-4o"],
                cwd=ROOT, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            wait_health(port)

        result = []
        for shown, argv in STEPS:
            argv = [a.replace("REF_PLACEHOLDER", ref) for a in argv]
            print(f"[capture] {shown[0].split(' -m ')[1][:60]} …", flush=True)
            p = subprocess.run([sys.executable, "-X", "utf8", *argv],
                               cwd=ROOT, env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               timeout=600)
            lines = [ln.rstrip() for ln in (p.stdout or "").splitlines()]
            result.append((shown, lines))
        return result
    finally:
        for s in servers:
            s.kill()


def thin_progress(lines: list[str], keep: int = 6) -> list[str]:
    """进度行抽稀：首尾保留，中间等距取样。非进度行原样。"""
    prog = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("进度")]
    if len(prog) <= keep:
        return lines
    step = max(1, len(prog) // keep)
    kept = set(prog[::step]) | {prog[-1]}
    return [ln for i, ln in enumerate(lines) if i not in set(prog) - kept]


# ---------------------------------------------------------------- 渲染

FONT_ASCII = next(p for p in [r"C:\Windows\Fonts\consola.ttf",
                              r"C:\Windows\Fonts\CascadiaMono.ttf",
                              r"C:\Windows\Fonts\cour.ttf"] if Path(p).exists())
FONT_CJK = next(p for p in [r"C:\Windows\Fonts\msyh.ttc",
                            r"C:\Windows\Fonts\simhei.ttf"] if Path(p).exists())

SIZE = 15
W, H = 960, 600
PAD, TITLE_H = 14, 30
LINE_H = SIZE + 6
ROWS = (H - TITLE_H - PAD * 2) // LINE_H

BG = (13, 17, 23)
TITLE_BG = (22, 27, 34)
FG = (201, 209, 217)
DIM = (110, 118, 129)
GREEN = (63, 185, 80)
RED = (248, 81, 73)
BLUE = (88, 166, 255)
AMBER = (210, 153, 34)

f_ascii = ImageFont.truetype(FONT_ASCII, SIZE)
f_cjk = ImageFont.truetype(FONT_CJK, SIZE)


def _wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def draw_mixed(d: ImageDraw.ImageDraw, x: int, y: int, text: str, fill):
    """ASCII 用等宽字体、CJK 用雅黑，逐段绘制。"""
    i = 0
    while i < len(text):
        wide = _wide(text[i])
        j = i
        while j < len(text) and _wide(text[j]) == wide:
            j += 1
        run, font = text[i:j], (f_cjk if wide else f_ascii)
        d.text((x, y), run, font=font, fill=fill)
        x += int(d.textlength(run, font=font))
        i = j
    return x


def line_color(ln: str):
    s = ln.strip()
    if ln.startswith("$") or s.startswith("--"):   # 命令行及其 ^ 续行
        return FG
    if s.startswith("#"):
        return DIM
    if "判定: PASS" in ln:
        return GREEN
    if "判定: FAIL" in ln or "判定: SUSPECT" in ln:
        return RED
    if s.startswith("进度") or set(s) <= {"="} and s:
        return DIM
    if "聚合 JSD" in ln or "e-process" in ln:
        return BLUE
    return (170, 178, 189)


def render_frame(lines: list[str], cursor: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, TITLE_H), fill=TITLE_BG)
    for k, c in enumerate(((248, 81, 73), (210, 153, 34), (63, 185, 80))):
        d.ellipse((12 + k * 20, 9, 24 + k * 20, 21), fill=c)
    d.text((78, 7), "fpverify demo — mock relay, everything on 127.0.0.1",
           font=f_ascii, fill=DIM)
    shown = lines[-ROWS:]
    y = TITLE_H + PAD
    for ln in shown:
        x = PAD
        color = line_color(ln)
        if ln.startswith("$"):
            x = draw_mixed(d, x, y, "$ ", GREEN)
            x = draw_mixed(d, x, y, ln[2:], FG)
        else:
            x = draw_mixed(d, x, y, ln, color)
        y += LINE_H
    if cursor and shown:
        d.rectangle((x + 2, y - LINE_H + 2, x + 2 + SIZE // 2, y - 4), fill=FG)
    return img


def build_gif(steps: list[tuple[str, list[str]]]):
    frames: list[Image.Image] = []
    durations: list[int] = []
    buf: list[str] = list(BANNER) + [""]

    def emit(dur: int, cursor: bool = False):
        frames.append(render_frame(buf, cursor))
        durations.append(dur)

    emit(1400)
    for shown, lines in steps:
        # 敲命令：每帧 8 个字符，续行另起一行
        for k, cmd_line in enumerate(shown):
            prefix = "$ " if k == 0 else ""
            buf.append(prefix)
            for i in range(0, len(cmd_line), 8):
                buf[-1] = prefix + cmd_line[: i + 8]
                emit(45, cursor=True)
        emit(500)
        # 输出：每帧 1 行，长块加速
        lines = thin_progress(lines)
        for k, ln in enumerate(lines):
            buf.append(ln)
            emit(60 if len(lines) > 25 else 110)
        buf.append("")
        emit(1500)
    emit(3500)  # 结尾定格

    OUT.parent.mkdir(parents=True, exist_ok=True)
    q = [f.quantize(colors=64) for f in frames]
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durations,
              loop=0, optimize=True)
    print(f"[gif] {OUT}  帧数={len(q)}  大小={OUT.stat().st_size / 1e6:.2f} MB")


def main():
    steps = capture()
    for shown, lines in steps:
        head = next((ln for ln in lines if "判定" in ln), lines[-1] if lines else "")
        print(f"[capture] 完成，{len(lines)} 行，{head.strip()[:50]}")
    build_gif(steps)


if __name__ == "__main__":
    main()
