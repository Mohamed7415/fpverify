# fpverify — LLM API 行为指纹验证器

**你花的是旗舰模型的钱，中转站给你上的到底是什么？**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![tests](https://github.com/Mohamed7415/fpverify/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed7415/fpverify/actions/workflows/ci.yml)

[English →](README.md)

API 中转站/代理商可以悄悄把你付费的旗舰模型换成便宜货或量化版——API 格式一样、
响应里的 `model` 字段照样写旗舰的名字，成本却是他们的零头。`fpverify` 用
**行为指纹**抓这件事：LLM 在"随便说个数"这类问题上**根本做不到随机**，
每个模型的偏好模式是稳定、可测量的签名。

对端点问 15~120 个琐碎的单 token 问题（成本零点几美分），把回答分布和从官方 API
入册的参考指纹比对，给出 PASS/FAIL 判定，并带**统计保证**：把真模型冤枉成假的
概率被压在 α=0.01 以下，且在任意停止点都成立（序贯下注 e-process）。

## LLM 不会随机——我们在前沿模型上实测了

2026 年 7 月，我们对 **9 个前沿模型，每个开 11 个全新独立实例**采样（借 Cursor
subagent 机制，模型身份由平台保证、是铁的地面真相）。问"随机说一个 1 到 100 的数"，
99 个全新实例给出的答案——

> **只有四个：73、47、37、42。**
> 光 `73` 就占 65.7%。全部模型全部题目的熵中位数只有 **0.44 bit**，
> 而理想随机答案应该有 6.64 bit。

**30 秒亲手验证**：在 Cursor 开一个*全新*会话，问 Claude Fable 5 要一个 1-100 的
随机数。我们的实测里，11 个新实例有 9 个说 73（thinking 版 11 个全说 73）。

单列会撞车（五个模型都爱 73），**组合才是指纹**：

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

对审计有直接意义的发现（完整研究：[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) §7）：

- **同一份权重、不同推理模式 → 同一个指纹**。Fable 5 与它的 thinking 版距离仅
  0.034，深入自比噪声带。指纹认的是*权重*；中转站偷偷关 thinking 省钱这种事，
  指纹看不见，得靠延迟旁路信号。
- **兄弟变体分得开**。GPT-5.6 sol vs terra 距离 0.295，超出噪声带（p95=0.217）——
  版本级识别摸得到（n=11，属初步证据）。
- **家族聚类不成立**。Claude 家族内部距离（均值 0.393）和跨家族（0.481）一个
  量级。指纹跟着权重走，不跟厂牌走。
- **语言是第二根轴**。同一模型的中英文探针分布几乎不相交（JSD 0.605~1.0）；
  中文语境下大多数模型把最爱的数换成 42。多语言探针白送区分度。

原始采样数据（`experiments/frontier/batch_*.json`）已随仓库提交；所有数字和图
一条命令可复现（固定随机种子）：

```bash
py -3.13 -X utf8 experiments/analyze_frontier.py
```

## 快速上手——不需要任何 API key

本地起一个诚实端点 + 一个掺水中转站，然后当场抓住后者：

```bash
pip install httpx

py -3.13 -X utf8 sim/mock_server.py --port 18801 --kind honest --model gpt-4o
py -3.13 -X utf8 sim/mock_server.py --port 18802 --kind swap   --model gpt-4o

py -3.13 -X utf8 -m fpverify.cli enroll --base-url http://127.0.0.1:18801/v1 --api-key mock --model gpt-4o --out ref.json
py -3.13 -X utf8 -m fpverify.cli audit  --base-url http://127.0.0.1:18801/v1 --api-key mock --model gpt-4o --ref ref.json   # → PASS
py -3.13 -X utf8 -m fpverify.cli audit  --base-url http://127.0.0.1:18802/v1 --api-key mock --model gpt-4o --ref ref.json   # → FAIL，通常十几次查询
```

模拟中转站内置九类对手（`--kind`）：`honest / drift / quantized / swap / pin /
filter_en / true_random / cache / partial_mimic`，红队实现见 `sim/adversaries.py`。

## 审计真实端点

```bash
# 1. 从你信任的渠道（官方 API）入册参考指纹
#    约 720 次单 token 请求、几美分；模型版本更新后建议重建
py -3.13 -X utf8 -m fpverify.cli enroll ^
    --base-url https://api.openai.com/v1 --api-key 官方key ^
    --model gpt-4o --samples 20 --out ref_gpt4o.json

# 2. 审计任何声称提供该模型的 OpenAI 兼容端点
py -3.13 -X utf8 -m fpverify.cli audit ^
    --base-url https://中转站地址/v1 --api-key 中转站key ^
    --model gpt-4o --ref ref_gpt4o.json --report audit.json
```

- **PASS**：预算内没有发现注水证据；
- **FAIL**：行为指纹显著偏离（误判概率 ≤ α=0.01），或检出响应级缓存；
- 明显造假通常**十几次查询（约 0.2 美分）**就出结论（早停）；
- 报告含聚合 JSD 与论文参考带（0.140 同源 / 0.227 跨部署 / 0.463 冒充者）辅助解读。

### 花几块钱，自己端到端验证一遍

不用信我们的仿真。判卷需要标准答案——所以直接用官方 API（一个便宜的官方 key 就够）：

```bash
# 从官方 API 入册模型 A 的指纹
py -3.13 -X utf8 -m fpverify.cli enroll --base-url https://api.deepseek.com/v1 --api-key $KEY --model deepseek-chat --samples 20 --out ref_a.json

# 再审同一个官方端点 → 必须 PASS（真实网络下的误杀检查）
py -3.13 -X utf8 -m fpverify.cli audit --base-url https://api.deepseek.com/v1 --api-key $KEY --model deepseek-chat --ref ref_a.json

# 用 A 的参考去审另一个模型 → 必须 FAIL（真实模型上的检出检查）
py -3.13 -X utf8 -m fpverify.cli audit --base-url https://api.deepseek.com/v1 --api-key $KEY --model deepseek-reasoner --ref ref_a.json
```

**可证伪声明**：如果你的官方直连端点、用它自己新入册的参考指纹审计却 FAIL 了，
请带着审计 JSON 开 issue——那将直接推翻我们的 FPR 声明。

## 只有中转站的 key？（买中转站的人恰恰买不到官方渠道）

这是最常见的真实处境：没法自己入册参考指纹。`identify` 命令改用 [`refs/`](refs/)
**社区公共指纹库**做降档识别——能精确就精确，不能就如实降档，绝不硬猜：

1. 声称的模型**在库里** → 序贯检验验真伪（PASS/FAIL，误判率 ≤ α）+ 全库距离排名；
2. 声称的模型**不在库里** → 报告"行为上与库内 X 一致"（`BEST_MATCH`）；
3. 谁都不像 → `UNKNOWN`，诚实说不知道。

```bash
py -3.13 -X utf8 -m fpverify.cli library     # 看库里有什么
py -3.13 -X utf8 -m fpverify.cli identify --base-url https://中转站/v1 --api-key 中转站key --model gpt-4o --samples 8
```

嫌命令行麻烦就用**本地网页检测台**——填中转站地址 / key / 声称模型（可一键拉取
中转站自己的模型列表来选），出判定卡片和距离排名：

```bash
py -3.13 -X utf8 -m webui.server             # 自动打开 http://127.0.0.1:8765
```

探针流量从**你自己的电脑**直连中转站，key 不经过我们任何后台（UI 是本地进程，
指纹库是 git 里的公开数据，`git pull` 即更新）。

库目前自带 2026-07 实测的 9 个前沿模型（`cursor-harness` 渠道，仅限同渠道比对）；
`api` 频道——真正能审中转站的裸官方 API 参考——**公开征集贡献**：入册一次几毛钱，
防投毒规则（来源出处 + 独立二次入册在噪声带内交叉验证）见 [`refs/README.md`](refs/README.md)。

## 检出性能（受控仿真）

对九类已知真相的对手做了全面验证（α=0.01、容差自动标定、预算 600 查询；
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
| 诚实端点 / 良性漂移 | 400 次审计 **0 误杀** | — | — |

\* 按旗舰定价（输入 $2.5/M、输出 $10/M）。粗暴作弊十几次查询就暴露；
作弊越"逼真"需要的证据越多，但以上全部在 $0.02 预算内检出。

## 红队 vs 蓝队：防得住哪、防不住哪

我们在仿真里跑了一轮 4 回合的协同进化——中转站出规避招、检测器出对策、循环——
全程守一条经济铁律：中转站只有在**真实业务流量**上供廉价模型才算省钱，"疑似测试就上
旗舰"等于零收益。完整台账：[`docs/COEVOLUTION.md`](docs/COEVOLUTION.md)。

- **按内容分流**（像探针的问句就供旗舰）→ 被破。业务原生 / 能力分叉探针逼中转站要么
  把它们也路由给旗舰，要么把阈值收到廉价流量占比从 0.52 崩到 0.23——**逃检=放弃省钱**。
- **按账号分流**（陌生账号供旗舰、老客户供廉价）→ 能击败任何一次性外部审计，但把探针
  混进真实老账号的自然流量（被动审计）即可破解。
- **随机稀释**（不看内容不看账号，对全部流量以概率 ε 供廉价）→ 收敛稳定点，也是一个
  **统计原理性极限**而非蓝队没想到的招。只要守 FPR≤α 就必然留一条宽度~δ 的容差带，
  足够小的 ε 藏得进去。攻击者的安全窗口 ε≈0.20~0.28（省 20~28%）；想调小 δ 去抓它，
  良性漂移端点 FPR 会冲到 0.092（超 α 九倍）。蓝队唯一合法杠杆是加大证据——扩大入册
  样本 + 长程累积，而抓住 ε 需要约 1/ε² 的样本。

诚实收口：结构化偷换都能低成本抓住；剩下的残余攻击是低速随机稀释，本质是"省下的钱 ε
↔ 审计方要花的样本约 1/ε²"的权衡。`fpverify` 的 anytime-valid 设计，正是让持续审计
能不断累积这份证据的关键。

## 工作原理

1. **探针**：问语义琐碎、答案可归类的单 token 问题（1-100 随机数、随机颜色、
   抛硬币……），多措辞、多语言，抗字符串匹配特判；
2. **归一化**：答案规范化，没见过的答案归入 `OTHER` 桶（Good-Turing 缺失质量处理）；
3. **比对**：端点经验分布 vs 参考指纹的 Jensen-Shannon 散度，跨探针 cell 聚合；
4. **判定**：序贯下注 e-process 逐次积累证据。anytime-valid——随时可停、
   明显案例早停，一类错误始终 ≤ α；良性漂移容差 δ 按参考指纹用 Dirichlet
   后验预测模拟自动标定。

方法源自论文 **《One Token Is Enough》**（Bruckner, arXiv:2607.10252, 2026）——
它用 165 个模型、32.6 万次请求确立了"单 token 分布指纹"。本项目在其上增加了
序贯 e-process 决策层（早停 + anytime-valid FPR 控制）、对抗加固（多语言改写探针、
缓存/延迟筛查）、自动标定，以及上面的前沿模型实测研究。

## 相关工具,以及我们真正不同的地方

中转站审计是个活跃赛道(恰好证明这个痛点是真的)。最接近的几个工具,如实对比:

| 工具 | 思路 | 判定类型 |
|---|---|---|
| [api-relay-audit](https://github.com/toby-bridges/api-relay-audit) | 安全扫描:注入/SSE 完整性/身份关键词 | 换模只算*"信号,非证据"* |
| [veridrop](https://github.com/canarybyte/veridrop) | 协议一致性 + **Claude thinking 签名**(密码学级)+ usage 字段取证 | Claude 最强;其余协议级 |
| [RelayRadar (AI45Lab)](https://github.com/AI45Lab/RelayRadar) | 自适应判别探针(AB3IT),TVD + 置换检验 p 值 | 固定样本假设检验 |
| [relay-radar (AetherCore)](https://github.com/AetherCore-Dev/relay-radar) | 被动风格监控 + LLMmap 探针 | 准确率式打分 |
| [zing](https://github.com/cenbonew/zing) | 能力/知识画像(上下文窗、tokenizer、知识截止) | 画像一致性检查 |
| [KBF (arXiv:2605.29524)](https://arxiv.org/abs/2605.29524) | 知识边界数值召回 | 固定样本二项检验 |

`fpverify` 有而以上全都没有的四样东西:

1. **anytime-valid 序贯判定**——e-process 在*任意*停止点都保证 FPR ≤ α,所以能早停
   (明显造假 ~15 次查询),更关键的是支持**持续低频被动审计**——这是唯一能对付
   账号级智能分流的形态(见协同进化台账)。固定样本检验反复跑会悄悄膨胀错误率,
   这件事它们做不了。
2. **良性漂移容差自动标定**(Dirichlet 后验预测)——"别冤枉跨服务商的诚实部署"
   靠标定解决,不靠手调阈值。
3. **2026 年 7 月的前沿模型指纹实测,模型身份由平台保证**(9 模型 × 11 全新实例,
   原始数据全提交、一条命令复现)——不是拿去年的模型跑民间偏方。
4. **对抗极限分析**——我们明说*检测在哪里失效*(随机稀释 ε≈0.20–0.28 是安全窗口,
   抓 ε 要 ~1/ε² 样本),而不是暗示检测器无敌。

互补而非互斥:veridrop 的 Claude 签名校验是密码学级的——审 Claude 端点建议两个
一起跑。行为指纹是那层**对所有模型都生效、不需要服务端配合**的通用防线。

## 项目结构

```
fpverify/     可复用库：探针、归一化、JSD、e-process、标定、最近邻、公共库 identify、CLI
refs/         社区公共指纹库（清单 + 各模型分布文件 + 贡献协议）
webui/        本地网页检测台（标准库实现;key 不出本机）
sim/          红队仿真：模型分布、对手、HTTP 模拟中转站、流量模型、蓝队探针
experiments/  评估、前沿实测、红蓝协同进化（FPR、检出率、预算曲线、距离矩阵）
tests/        统计性质测试（公平性、FPR 上界、检出力、端到端、协同进化、公共库识别）
docs/         研究笔记（问题、威胁模型、方法、实验、前沿实测、多模态路线图）+ 协同进化台账
```

## 路线图

决策核心（JSD + 序贯 e-process + 标定）是**模态无关**的：把图像/视频输出嵌入、
量化成码本，同一套 PASS/FAIL 机制原样适用。把换模检测扩展到生图/生视频 API——
并把固定种子可复现当作额外的近确定性信号——是规划中的 v2。设计见
[`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) §8，尚未实现。

## 诚实的局限

- 判定是**统计证据，不是铁证**。FAIL 意味着"分布显著偏离参考"，可能原因包括
  换模型、量化、版本回滚、缓存。请保留 JSON 报告并复测一次再下结论。
- 上面的前沿模型指纹采自 *Cursor agent harness 内部*（带系统提示、温度不受控），
  足以证明非随机性与可分性，但不能和裸 API 数字直接对表；每模型 n=11 样本量小，
  自比噪声带偏宽。
- 同权重的推理模式切换（thinking 开/关）指纹看不见，需要延迟/长度旁路信号。
- 能在**账号级**识别审计流量的适应性对手可以击败任何一次性认证；对策是持续、
  低频、混入业务流量的审计（anytime-valid 的设计天然支持）。
- 对"完美复刻目标模型完整条件分布"的对手没有密码学保证——但做到那一步的成本
  约等于真的跑原模型，作弊的利润动机已被抹平。

## 许可

MIT
