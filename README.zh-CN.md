# fpverify — LLM API 行为指纹验证器

检测 OpenAI 兼容端点背后运行的，是否为它声称的模型。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![tests](https://github.com/Mohamed7415/fpverify/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed7415/fpverify/actions/workflows/ci.yml)

[English →](README.md)

问题：API 中转站可以把付费的旗舰模型换成便宜模型或量化版。API 格式不变，响应里的
`model` 字段照写原名，从协议层面看不出区别。

方法：LLM 生成不了随机输出。要求模型"随便说一个 1 到 100 的数"，答案高度集中，
且每个模型集中的位置不同——2026 年 7 月实测 9 个前沿模型 × 各 11 个全新实例，
99 个回答只有 4 种（见[实测一节](#实测前沿模型无法随机)）。几十道此类问题的回答分布
构成稳定的模型签名。fpverify 向端点发送单 token 探针，将观测分布与参考指纹做
Jensen-Shannon 散度比对，由序贯下注检验（e-process）给出判定。误判率有上界：
诚实端点被判 FAIL 的概率 ≤ α = 0.01，且在任意停止点成立。

![终端演示：本机模拟中转站——入册参考、审诚实端点 PASS、审作弊端点早停 FAIL](docs/demo.gif)

（上图为真实运行录制：两个端点都在 127.0.0.1，作弊端点声称 gpt-4o、实际供应
便宜模型；可用 `experiments/make_demo_gif.py` 重新生成。）

## 使用

下文命令为 Windows 写法；Linux / macOS 将 `py -3.13 -X utf8` 换成 `python`。

```bash
git clone https://github.com/Mohamed7415/fpverify
cd fpverify
pip install -r requirements.txt
```

### 场景一：手里只有中转站的 key

启动本地网页检测台：

```bash
py -3.13 -X utf8 -m webui.server
```

浏览器自动打开 `http://127.0.0.1:8765`。填三项：

| 字段 | 内容 |
|---|---|
| Base URL | 中转站地址，以 `/v1` 结尾 |
| API Key | 中转站发给你的 key |
| 模型名 | 点「拉取模型列表」，从中转站实际提供的列表中选 |

「对照库条目」保持默认（按模型名自动匹配）。点「开始检测」。请求数 = 每题采样数 ×
库内题数，默认设置下为几十到三百次单 token 请求，费用几分到几毛钱，几分钟出结果。

判定含义：

| 判定 | 含义 |
|---|---|
| PASS | 预算内未发现偷换证据 |
| FAIL | 行为显著偏离声称模型的参考（误判概率 ≤ 0.01），或检出响应级缓存 |
| BEST_MATCH | 声称的模型不在库里；报告行为与库内哪个模型一致 |
| UNKNOWN | 与库内任何模型都不像 |
| INCONCLUSIVE | 证据不足；加大采样数重测 |

判定下方附「自行复核」表：该模型参考中最确定的几道题、参考答案、一键复制的提示词
与可下载脚本。复核不依赖本工具，方法见场景二。

隐私边界：探针请求从你的电脑直连中转站；key 只存在本机进程内存，不落盘、不上传。
指纹库是仓库内的公开数据，`git pull` 更新。

命令行等价操作：

```bash
py -3.13 -X utf8 -m fpverify.cli library        # 列出指纹库
py -3.13 -X utf8 -m fpverify.cli identify --base-url https://中转站/v1 --api-key KEY --model gpt-4o --samples 8
```

识别逻辑分三档：声称的模型在库里，用序贯检验验真伪（PASS / FAIL）；不在库里，
报告行为最像库内哪个模型（BEST_MATCH）；都不像，报 UNKNOWN。指纹库自带 2026-07
采集的 9 个前沿模型（`cursor-harness` 渠道，仅限同渠道比对）。`api` 渠道的官方 API
参考指纹征集社区贡献，入册一次成本几毛钱，防投毒规则见 [`refs/README.md`](refs/README.md)。

### 场景二：复核判定（不依赖本工具）

```bash
py -3.13 -X utf8 -m fpverify.cli reproduce --claimed gpt-5.6-sol
```

导出该模型的复核包：参考指纹中众数占比最高的几道题及参考答案（GPT-5.6 sol：
抛硬币 = tails 11/11，随机颜色 = orange 11/11）。四种跑法：

1. 把 `cursor_prompt.md` 粘贴给 Cursor 等支持子代理的 agent IDE，扇出 N 个全新
   subagent（与 `cursor-harness` 参考同渠道）；
2. `codex_loop.sh` / `codex_loop.ps1` 循环调用 `codex exec`，每次都是全新会话；
3. `official_api.py`（纯标准库、零依赖）用官方 API key 采样，与参考表并排打印；
4. 在官网手动测：每题新开一个对话。

规则一条：每个样本必须来自全新对话或全新实例。同一对话内连问十次无效——模型能
看到自己之前的答案，会刻意变换。

### 场景三：有官方 API key

参考指纹从官方渠道现场采集，不依赖公共库，证据强度最高。

```bash
# 1. 从官方 API 入册参考指纹（约 720 次单 token 请求，几美分；模型版本更新后重建）
py -3.13 -X utf8 -m fpverify.cli enroll ^
    --base-url https://api.openai.com/v1 --api-key 官方key ^
    --model gpt-4o --samples 20 --out ref_gpt4o.json

# 2. 审计任何声称提供该模型的 OpenAI 兼容端点
py -3.13 -X utf8 -m fpverify.cli audit ^
    --base-url https://中转站地址/v1 --api-key 中转站key ^
    --model gpt-4o --ref ref_gpt4o.json --report audit.json
```

明显偷换通常在十几次查询内触发早停（约 0.2 美分）。报告含聚合 JSD 及论文参考带
（0.140 同源 / 0.227 跨部署 / 0.463 冒充者）。

两条自检，同时也是对本项目 FPR 声明的检验（一个便宜的官方 key 即可完成）：

```bash
# 从官方 API 入册模型 A 的指纹
py -3.13 -X utf8 -m fpverify.cli enroll --base-url https://api.deepseek.com/v1 --api-key $KEY --model deepseek-chat --samples 20 --out ref_a.json

# 用 A 的参考审同一个官方端点：必须 PASS（真实网络下的误判检查）
py -3.13 -X utf8 -m fpverify.cli audit --base-url https://api.deepseek.com/v1 --api-key $KEY --model deepseek-chat --ref ref_a.json

# 用 A 的参考审另一个模型：必须 FAIL（真实模型上的检出检查）
py -3.13 -X utf8 -m fpverify.cli audit --base-url https://api.deepseek.com/v1 --api-key $KEY --model deepseek-reasoner --ref ref_a.json
```

若官方直连端点用它自己新入册的参考审计却得到 FAIL，请附审计 JSON 开 issue；
这会直接推翻本项目的 FPR 声明。

## 本地演示（不需要任何 key）

本节用于验证工具本身，不涉及真实服务。`sim/mock_server.py` 在本机启动假端点：
`--kind honest` 按 gpt-4o 风格分布诚实作答，`--kind swap` 模拟声称 gpt-4o、实际
供应便宜模型的中转站。预期结果：前者 PASS，后者在十几次查询内 FAIL。

```bash
pip install httpx

py -3.13 -X utf8 sim/mock_server.py --port 18801 --kind honest --model gpt-4o
py -3.13 -X utf8 sim/mock_server.py --port 18802 --kind swap   --model gpt-4o

py -3.13 -X utf8 -m fpverify.cli enroll --base-url http://127.0.0.1:18801/v1 --api-key mock --model gpt-4o --out ref.json
py -3.13 -X utf8 -m fpverify.cli audit  --base-url http://127.0.0.1:18801/v1 --api-key mock --model gpt-4o --ref ref.json   # PASS
py -3.13 -X utf8 -m fpverify.cli audit  --base-url http://127.0.0.1:18802/v1 --api-key mock --model gpt-4o --ref ref.json   # FAIL
```

模拟中转站内置九类对手（`--kind`）：`honest / drift / quantized / swap / pin /
filter_en / true_random / cache / partial_mimic`，实现见 `sim/adversaries.py`。

## 实测：前沿模型无法随机

2026 年 7 月，对 9 个前沿模型采样，每个模型 11 个全新独立实例（Cursor subagent
渠道，模型身份由平台保证）。问"随机说一个 1 到 100 的数"，99 个实例的答案只有
4 种：73、47、37、42，其中 73 占 65.7%。全部模型全部题目的熵中位数为 0.44 bit；
均匀随机应为 6.64 bit。

验证方式：开一个全新会话，向 Claude Fable 5 要一个 1 到 100 的随机数。本次实测中
11 个新实例有 9 个回答 73，thinking 版 11 个全部回答 73。

单题答案会撞车（5 个模型的众数都是 73），组合分布才构成指纹：

| 模型（2026-07） | 1–100 随机数（众数） | 颜色 | 动物 | 城市 | 抛硬币 |
|---|---|---|---|---|---|
| Claude Fable 5 | **73**（82%） | teal | otter | Kyoto | heads（100%） |
| Claude Fable 5 thinking | **73**（100%） | teal | otter | Kyoto | heads（100%） |
| Claude Sonnet 5 thinking | **37**（91%） | blue | elephant | Paris | heads（100%） |
| Claude Opus 4.8 thinking | **73**（100%） | blue | fox | Tokyo | heads（100%） |
| GPT-5.6 sol | **73**（91%） | orange | otter | Lisbon | tails（100%） |
| GPT-5.6 terra | **47**（36%） | teal | otter | Lisbon | tails（100%） |
| GLM-5.2 | **73**（91%） | teal | fox | Kyoto | heads（91%） |
| Composer 2.5 | **47**（100%） | purple | elephant | Tokyo | heads（91%） |
| Grok 4.5 | **73**（100%） | teal | otter | Lisbon | heads（45%） |

![9 个前沿模型的两两聚合 JSD 距离矩阵](experiments/out/fig_frontier_matrix.png)

与审计直接相关的结论（完整分析：[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) §7）：

- 同一权重、不同推理模式，指纹相同。Fable 5 与其 thinking 版距离 0.034，在自比
  噪声带内。指纹绑定权重；关闭 thinking 省钱这类模式切换指纹不可见，需要延迟旁路信号。
- 兄弟变体可分。GPT-5.6 sol 与 terra 距离 0.295，超出噪声带 p95 = 0.217
  （n=11，初步证据）。
- 家族聚类不成立。Claude 家族内部平均距离 0.393，跨家族 0.481，同一量级。指纹
  跟随权重，不跟随厂商。
- 语言是独立的轴。同一模型的中英文探针分布 JSD 0.605~1.0；中文语境下多数模型的
  高频数字变为 42。多语言探针提供额外区分度。

原始数据在 `experiments/frontier/batch_*.json`，全部数字与图固定种子可复现：

```bash
py -3.13 -X utf8 experiments/analyze_frontier.py
```

## 检出性能（受控仿真）

对九类已知真相的对手验证（α=0.01，容差自动标定，预算 600 查询；
`experiments/run_evaluation.py`）：

| 对手 | 检出率 | 平均定案查询数 | 估算成本* |
|---|---|---|---|
| 整体换模型 | 100% | 15 | $0.002 |
| 钉死回答（"只说 73"） | 100% | 14 | $0.002 |
| 真随机烟雾弹 | 100% | 14 | $0.002 |
| 特判已知探针措辞 | 100% | 17 | $0.003 |
| 部分分布模仿 | 100% | 17 | $0.003 |
| 响应缓存 | 100% | 89 | $0.014 |
| 量化降级 | 100% | 116 | $0.018 |
| 诚实端点 / 良性漂移 | 400 次审计 0 误判 | — | — |

\* 按旗舰定价（输入 $2.5/M、输出 $10/M）估算。作弊越接近真实分布需要的证据越多；
上表全部在 $0.02 预算内检出。

## 对抗分析

仿真中运行了 4 回合红蓝协同进化：中转站提出规避手段，检测器提出对策，循环。
经济约束：中转站只有在真实业务流量上供应廉价模型才有收益，"疑似测试就供旗舰"
等于零收益。完整记录：[`docs/COEVOLUTION.md`](docs/COEVOLUTION.md)。

- 按内容分流（疑似探针的请求供旗舰）：被业务原生探针与能力分叉探针破解。中转站
  要么把这些探针也路由给旗舰，要么收紧阈值使廉价流量占比从 0.52 降到 0.23，
  规避与收益冲突。
- 按账号分流（陌生账号供旗舰、老账号供廉价）：可击败一次性外部审计。对策是把
  探针混入真实老账号的自然流量，长期被动审计。
- 随机稀释（不看内容与账号，对全部流量以概率 ε 供廉价模型）：收敛点，属统计
  原理性极限。任何保持 FPR ≤ α 的检验必然留出宽度约 δ 的容差带，足够小的 ε 可
  藏入其中。本仿真中攻击者的安全窗口为 ε ≈ 0.20~0.28；收紧 δ 会使良性漂移端点
  的 FPR 升至 0.092。检出稀释率 ε 需要约 1/ε² 的样本，唯一手段是扩大入册样本并
  长期累积证据。

结论：结构化偷换可低成本检出；残余风险是低比例随机稀释，形式为"省下 ε ↔ 审计方
投入 ~1/ε² 样本"的权衡。anytime-valid 设计允许持续审计不断累积这份证据。

## 原理

1. 探针：语义琐碎、答案可归类的单 token 问题（1-100 随机数、随机颜色、抛硬币等），
   多措辞、多语言，抵抗字符串匹配特判。
2. 归一化：答案规范化；未见过的答案归入 OTHER 桶（Good-Turing 缺失质量处理）。
3. 比对：端点经验分布与参考指纹的 Jensen-Shannon 散度，跨探针 cell 聚合。
4. 判定：序贯下注 e-process 逐次累积证据。anytime-valid：任意时点可停，明显案例
   早停，一类错误始终 ≤ α。良性漂移容差 δ 按参考指纹以 Dirichlet 后验预测模拟
   自动标定。

方法基础是论文 One Token Is Enough（Bruckner, arXiv:2607.10252, 2026），该文以
165 个模型、32.6 万次请求确立了单 token 分布指纹。本项目在其上增加：序贯
e-process 决策层（早停 + anytime-valid FPR 控制）、对抗加固（多语言改写探针、
缓存与延迟筛查）、自动标定，以及上节的前沿模型实测。

## 相关工具

| 工具 | 思路 | 判定类型 |
|---|---|---|
| [api-relay-audit](https://github.com/toby-bridges/api-relay-audit) | 安全扫描：注入 / SSE 完整性 / 身份关键词 | 换模只算"信号，非证据" |
| [veridrop](https://github.com/canarybyte/veridrop) | 协议一致性 + Claude thinking 签名（密码学级）+ usage 字段取证 | Claude 最强；其余协议级 |
| [RelayRadar (AI45Lab)](https://github.com/AI45Lab/RelayRadar) | 自适应判别探针（AB3IT），TVD + 置换检验 p 值 | 固定样本假设检验 |
| [relay-radar (AetherCore)](https://github.com/AetherCore-Dev/relay-radar) | 被动风格监控 + LLMmap 探针 | 准确率式打分 |
| [zing](https://github.com/cenbonew/zing) | 能力/知识画像（上下文窗、tokenizer、知识截止） | 画像一致性检查 |
| [KBF (arXiv:2605.29524)](https://arxiv.org/abs/2605.29524) | 知识边界数值召回 | 固定样本二项检验 |

fpverify 与它们的差异：

1. anytime-valid 序贯判定。e-process 在任意停止点保证 FPR ≤ α，因此支持早停
   （明显造假约 15 次查询），也支持持续低频被动审计——应对账号级智能分流的唯一
   形态（见对抗分析）。固定样本检验反复运行会使实际错误率膨胀。
2. 良性漂移容差自动标定（Dirichlet 后验预测），不依赖手调阈值。
3. 2026-07 前沿模型指纹实测，模型身份由平台保证，原始数据全部提交、一条命令复现。
4. 对抗极限分析：明确给出检测失效区（随机稀释 ε ≈ 0.20~0.28，检出 ε 需 ~1/ε²
   样本），而非宣称检测不可战胜。

veridrop 的 Claude thinking 签名校验是密码学级的，与本工具互补；审计 Claude 端点
建议两者同时使用。行为指纹是对所有模型生效、无需服务端配合的通用层。

## 项目结构

```
fpverify/     可复用库：探针、归一化、JSD、e-process、标定、最近邻、公共库 identify、复核包、CLI
refs/         社区公共指纹库（清单 + 各模型分布文件 + 贡献协议）
webui/        本地网页检测台（标准库实现；key 不出本机）
sim/          红队仿真：模型分布、对手、HTTP 模拟中转站、流量模型、蓝队探针
experiments/  评估、前沿实测、红蓝协同进化（FPR、检出率、预算曲线、距离矩阵）
tests/        统计性质测试（公平性、FPR 上界、检出力、端到端、协同进化、公共库识别、复核包）
docs/         研究笔记（问题、威胁模型、方法、实验、前沿实测、多模态路线图）+ 协同进化台账
```

## 路线图

决策核心（JSD + 序贯 e-process + 标定）与模态无关：图像/视频输出经嵌入并量化为
码本后，同一套判定机制适用。将偷换检测扩展到生图/生视频 API，并利用固定种子
可复现作为额外信号，是规划中的 v2。设计见
[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) §8，尚未实现。

## 局限

- 判定是统计证据，不是密码学证明。FAIL 表示分布显著偏离参考，可能原因包括换模、
  量化、版本回滚、缓存。建议保留 JSON 报告并复测后再下结论。
- 前沿模型指纹采自 Cursor agent harness 内部（含系统提示、温度不受控），可证明
  非随机性与可分性，不能与裸 API 数字直接对表；每模型 n=11，自比噪声带偏宽。
- 同权重的推理模式切换（thinking 开/关）指纹不可见，需要延迟/长度旁路信号。
- 能在账号级识别审计流量的对手可击败一次性认证；对策是持续、低频、混入业务流量
  的审计。
- 对完美复刻目标模型完整条件分布的对手没有密码学保证；达到该水平的成本约等于
  运行原模型。

## 许可

MIT
