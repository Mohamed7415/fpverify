# cursor-harness 频道采集协议（v1，2026-07-21 首采）

`refs/harness/` 全部条目按本协议采集。**指纹是（模型 × 渠道 × 协议 × 档位）四元组的
条件分布**——复核时四者必须全部对齐，缺一个都会得到不同的分布（见下方实证）。

## 协议

每轮 = 一个**全新上下文**的 Cursor subagent 实例，一次性收到下面的套卷原文，
在**同一个回复**里按 JSON 作答十题。跨轮独立（实例不复用、不共享上下文）。
9 模型 × 11 轮，原始数据 `batch_00~10.json`。

### 套卷原文（逐字，不得改动）

```
You are participating in a statistical sampling task. Do NOT use any tools. Do NOT explore the codebase. Do NOT explain anything. Reply with ONLY one JSON object — no markdown fences, no extra text before or after.

Answer each question independently and spontaneously with the first value that comes to mind. Do not try to be consistent, do not avoid repeating values across questions, do not overthink.

1. rand_num_100 — Name a random number between 1 and 100.
2. rand_num_10 — Pick a random number between 1 and 10.
3. fav_num — What is your favorite number?
4. rand_letter — Pick a random letter of the English alphabet.
5. rand_color — Name a random color.
6. fav_color — What is your favorite color?
7. rand_animal — Name a random animal.
8. rand_city — Name a random city.
9. coin_flip — Flip a coin in your head: heads or tails?
10. rand_num_100_zh — 说一个1到100之间的随机数。

Reply with exactly this JSON shape (fill in your values):
{"rand_num_100": <int 1-100>, "rand_num_10": <int 1-10>, "fav_num": <int>, "rand_letter": "<single letter>", "rand_color": "<one word>", "fav_color": "<one word>", "rand_animal": "<one word>", "rand_city": "<one word>", "coin_flip": "<heads or tails>", "rand_num_100_zh": <int 1-100>}
```

### 首采时的模型档位（2026-07-21）

fable5=claude-fable-5-thinking-max（非 think 条目为对应非思考档），
sonnet5-think=claude-sonnet-5-thinking-high，opus48-think=claude-opus-4-8-thinking-high，
**gpt56-sol=gpt-5.6-sol-medium**，gpt56-terra=gpt-5.6-terra-medium，
glm52=glm-5.2-high，composer25=composer-2.5-fast，grok45=cursor-grok-4.5-high。

## 协议敏感性实证（2026-07-22，gpt56-sol）

同模型、同渠道，**冷启动单题**（每实例只问一道）与**套卷**答案可以完全不同：

| 题 | 套卷参考（sol-medium，n=11） | 套卷复测（sol-max，n=5） | 冷启动单题（sol-max，n=3~6） |
|---|---|---|---|
| coin_flip | tails 11/11 | tails 5/5 | **heads/正面 5/6** |
| rand_num_100 | 73 10/11 | 73 5/5 | **47 3/3** |
| rand_color | orange 11/11 | **teal 5/5** | **蓝色 3/3** |
| rand_animal | otter 11/11 | otter 5/5 | 熊猫/海豚/水獭 各1 |
| rand_city | lisbon 11/11 | Kyoto 3/5, Lisbon 2/5 | — |
| rand_letter | k 8/11 | Q 5/5 | — |

补充实证（2026-07-22 晚）：

- **官网 web 渠道**（chatgpt.com 登录会话，冷启动单题，每题新对话）：
  coin_flip = heads 6/6，rand_num_100 = 47 6/6——与子代理冷启动单题基本一致。
  说明冷协议分布跨渠道（harness 子代理 / 官网 web）相当稳定，**套卷 vs 冷问
  才是主导差异轴**。
- **协议翻转不是 GPT 特例**：claude-fable-5-thinking-max 冷启动单题
  coin_flip = tails 5/6（套卷参考 = heads 11/11，方向相反）；
  而 rand_num_100 = 73 5/6 与套卷一致——同一模型有的题跨协议稳定、有的题翻转，
  哪道题稳定必须实测，不能想当然。

三条结论：

1. **套卷协议下参考表大体复现**（coin/73/4/7/blue/otter 全对上）——数据是真的，
   但它只在自己的协议下成立。冷启动单题是另一个条件分布（官网单独问一句
   抛硬币得到 heads ≠ 数据造假）。
2. **档位会在脚下变动**：sol-medium 已从 Cursor 下架（现为 sol-max），个别题随之漂移
   （rand_color orange→teal、rand_letter k→Q、rand_city 部分 lisbon→Kyoto）。
   复核判读要容忍少数题漂移，以多数题对上为准。
3. **用户配置会渗进 harness 渠道**：子代理继承账号的用户规则（如"始终中文回复"），
   冷启动单题时答案会变成中文（正面/蓝色/熊猫）；套卷的 JSON 模板锚定了英文，
   基本免疫此问题。复核统计时把中英同义答案归并（heads=正面）。

## 使用边界

- harness 参考 ↔ harness 套卷复核：同渠道同协议，判读有效。
- harness 参考 ↔ 在线端点单题探针（identify/audit 的冷协议）：**跨协议**，
  绝对距离整体偏大，只能看相对排名，不发 PASS/FAIL 硬判定。
- 对中转站要硬判定：用官方 key 按 fpverify enroll 冷协议现场入册（api 频道），
  与 audit 探针同协议同渠道。
