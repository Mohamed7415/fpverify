# 公共指纹库 / Community Reference Fingerprint Library

这个目录是 `fpverify identify` 的数据源:**没有官方渠道的用户**(买中转站的人恰恰
买不到官方 API)不必自己入册,直接对着这里的社区参考指纹做降档识别:

1. 声称的模型**在库里** → 序贯检验验真伪(FPR ≤ α 保证);
2. 声称的模型**不在库里** → 报告"行为上最像库里的谁"(BEST_MATCH);
3. 谁都不像 → UNKNOWN,如实说不知道。

This directory backs `fpverify identify`: users who *only* have a relay key (and by
definition cannot reach the official API) audit against community-enrolled references
instead of enrolling their own.

## 渠道与协议 / Channels & protocols

指纹是**(模型 × 渠道 × 协议 × 档位)**的条件分布,每个条目都带 `protocol` 字段。
探针(enroll/audit/identify)恒为 `cold-single`(每题独立请求、全新对话);
参考与探针**同协议才发 PASS/FAIL 硬判定**,跨协议只做相对排名。

| channel | protocol | 含义 | 能干什么 |
|---|---|---|---|
| `api` | `cold-single` | 裸 OpenAI 兼容 API 直连入册 | **审计中转站**(与探针同协议同渠道,可硬判定) |
| `cursor-harness` | `harness-battery` | Cursor agent harness 内套卷采样(一实例连答十题;带系统提示、温度不受控) | 相对排名、identify 演示、套卷复核;**不能**当冷协议参考 |
| `simulation` | `cold-single` | 仿真分布 | 格式示例 / 测试 |

协议全文与跨协议实证见 `experiments/frontier/PROTOCOL.md`。

**当前状态**:`cursor-harness` 频道有 2026-07 实测的 9 个前沿模型(n=11/cell,由
`experiments/build_refs_from_frontier.py` 从原始数据一键重建);`api` 频道**正在
征集贡献**——这是整个库最有价值、也最需要人多力量大的部分。

## 贡献一个 api 频道参考 / Contributing an `api` reference

有任何模型的官方 API key?入册一次只要几毛钱、几分钟:

```bash
python -m fpverify.cli enroll \
    --base-url https://api.官方.com/v1 --api-key $KEY \
    --model 精确版本号 --samples 20 --out refs/api/模型id.json
```

然后在 `manifest.json` 的 `entries` 里加一条(照抄现有条目格式),PR 必须包含:

- `channel: "api"`、`protocol: "cold-single"`(enroll 天然就是这个协议);
- `enrolled_at`(日期)、`source`(官方 base-url,**不含 key**)、精确模型版本号;
- 入册命令原样贴在 PR 描述里;
- 官方端点对着这份参考自审计的 PASS 报告(`audit --report`)。

**防投毒**:同一 model+version 需要**第二个独立贡献者**的入册与首份在自比噪声带内
(聚合 JSD ≤ 0.14)才标记为 `verified`;在那之前条目带 `unverified` 注记。
维护者也会抽查。参考指纹只是分布计数,不含任何密钥或个人信息。

**保鲜**:官方模型升版本会让旧参考漂移,条目按 `enrolled_at` 判断新鲜度;
超过 90 天未复核的条目在 identify 输出里带过期警告(TODO)。更新库 = `git pull`。

## 格式 / Format

每个文件是 `fpverify.fingerprint.Fingerprint` 的 JSON 序列化:
`cells` 是 `"task::lang" -> {答案: 次数}` 的计数,另有 `validity` / `params` /
`created_at` 元数据。`manifest.json` 是条目清单,字段见现有条目。
