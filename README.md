# fpverify — Behavioral Fingerprinting for LLM APIs

**You're paying for a flagship model. Is that what you're actually getting?**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![tests](https://github.com/Mohamed7415/fpverify/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed7415/fpverify/actions/workflows/ci.yml)

[中文文档 →](README.zh-CN.md)

API resellers and relay services can silently swap the expensive model you paid for
with a cheaper or quantized one — same API format, same `model` field in the response,
lower cost for them. `fpverify` catches this with **behavioral fingerprinting**:
LLMs are incapable of answering "pick a random number" randomly, and each model's
bias pattern is a stable, measurable signature.

Ask an endpoint ~15–120 trivial one-token questions (fractions of a cent), compare the
answer distribution against a reference fingerprint enrolled from the official API, and
get a PASS/FAIL verdict with a **statistical guarantee**: the probability of falsely
accusing an honest endpoint is capped at α = 0.01, valid at any stopping point
(sequential betting e-process).

## LLMs cannot be random — we measured it on frontier models

In July 2026 we sampled **9 frontier models × 11 fresh, independent instances each**
(via Cursor subagents, so ground-truth identity is platform-guaranteed). Asked
*"name a random number between 1 and 100"*, the 99 fresh instances produced…

> **…only four distinct answers: 73, 47, 37, 42.**
> `73` alone accounts for 65.7%. The median per-question entropy across all models is
> **0.44 bits** — an ideal random answer would carry 6.64 bits.

**Verify this yourself in 30 seconds**: open a *fresh* chat with Claude Fable 5 and ask
for a random number between 1 and 100. In our runs, 9 of 11 fresh instances said 73
(the thinking variant: 11 of 11).

Single answers collide (five models love 73) — the *combination* is the fingerprint:

| Model (July 2026) | Rand 1–100 (mode) | Color | Animal | City | Coin flip |
|---|---|---|---|---|---|
| Claude Fable 5 | **73** (82%) | teal | otter | Kyoto | heads (100%) |
| Claude Fable 5 thinking | **73** (100%) | teal | otter | Kyoto | heads (100%) |
| Claude Sonnet 5 thinking | **37** (91%) | blue | elephant | Paris | heads (100%) |
| Claude Opus 4.8 thinking | **73** (100%) | blue | fox | Tokyo | heads (100%) |
| GPT-5.6 sol | **73** (91%) | orange | otter | Lisbon | tails (100%) |
| GPT-5.6 terra | **47** (36%) | teal | otter | Lisbon | tails (100%) |
| GLM-5.2 | **73** (91%) | teal | fox | Kyoto | heads (91%) |
| Composer 2.5 | **47** (100%) | purple | elephant | Tokyo | heads (91%) |
| Grok 4.5 | **73** (100%) | teal | otter | Lisbon | heads (45%) |

![Pairwise aggregated JSD between 9 frontier models](experiments/out/fig_frontier_matrix.png)

Findings that matter for auditing (full study: [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) §7):

- **Same weights, different mode → same fingerprint.** Fable 5 vs its thinking variant:
  JSD 0.034, deep inside the self-noise band. Fingerprints identify *weights*, and a
  relay silently disabling thinking mode needs latency side-channels instead.
- **Sibling variants are separable.** GPT-5.6 sol vs terra: JSD 0.295, above the noise
  band (p95 = 0.217) — version-level identification is within reach (n=11, preliminary).
- **Family clustering fails.** Claude-internal distances (mean 0.393) are the same
  magnitude as cross-family (0.481). The fingerprint tracks weights, not vendor.
- **Language is a second axis.** The same model answers EN and ZH probes with almost
  disjoint distributions (JSD 0.605–1.0); in Chinese, most models switch their favorite
  number to 42. Multilingual probes are free extra discriminative power.

Raw data (`experiments/frontier/batch_*.json`) is committed; every number and figure
regenerates with one command (fixed seeds):

```bash
python -X utf8 experiments/analyze_frontier.py
```

## Quick start — zero API keys

Run a local honest endpoint and a local cheating relay, then catch the cheater:

```bash
pip install httpx

python sim/mock_server.py --port 18801 --kind honest --model gpt-4o &
python sim/mock_server.py --port 18802 --kind swap   --model gpt-4o &

python -m fpverify.cli enroll --base-url http://127.0.0.1:18801/v1 --api-key mock \
    --model gpt-4o --out ref.json
python -m fpverify.cli audit  --base-url http://127.0.0.1:18801/v1 --api-key mock \
    --model gpt-4o --ref ref.json     # → PASS
python -m fpverify.cli audit  --base-url http://127.0.0.1:18802/v1 --api-key mock \
    --model gpt-4o --ref ref.json     # → FAIL, usually within ~15 queries
```

The mock relay implements nine adversaries (`--kind`): `honest / drift / quantized /
swap / pin / filter_en / true_random / cache / partial_mimic` — see `sim/adversaries.py`.

> Windows: use `py -3.13 -X utf8` instead of `python`.

## Audit a real endpoint

```bash
# 1. Enroll a reference fingerprint from a source you trust (the official API).
#    ~720 one-token requests, a few cents. Re-enroll after model version bumps.
python -m fpverify.cli enroll \
    --base-url https://api.openai.com/v1 --api-key $OFFICIAL_KEY \
    --model gpt-4o --samples 20 --out ref_gpt4o.json

# 2. Audit any OpenAI-compatible endpoint claiming to serve that model.
python -m fpverify.cli audit \
    --base-url https://some-relay.example/v1 --api-key $RELAY_KEY \
    --model gpt-4o --ref ref_gpt4o.json --report audit.json
```

- **PASS** — no evidence of substitution within budget.
- **FAIL** — the behavioral fingerprint deviates significantly (false-positive
  probability capped at α = 0.01), or response-level caching was detected.
- Reports include the aggregated JSD plus reference bands from the paper
  (0.140 same-source / 0.227 cross-deployment / 0.463 impostor) for interpretation.

### Validate it end-to-end for a few cents

You don't have to trust our simulation. Ground truth requires *knowing* what's behind
the endpoint, so use official APIs directly (one cheap provider key is enough):

```bash
# Enroll model A from its official API
python -m fpverify.cli enroll --base-url https://api.deepseek.com/v1 \
    --api-key $KEY --model deepseek-chat --samples 20 --out ref_a.json

# Audit the SAME official endpoint → must PASS (real-network false-positive check)
python -m fpverify.cli audit --base-url https://api.deepseek.com/v1 \
    --api-key $KEY --model deepseek-chat --ref ref_a.json

# Audit a DIFFERENT model against A's reference → must FAIL (real-model detection check)
python -m fpverify.cli audit --base-url https://api.deepseek.com/v1 \
    --api-key $KEY --model deepseek-reasoner --ref ref_a.json
```

**Falsifiability**: if an official, direct-connection endpoint FAILs an audit against
its own freshly-enrolled reference, open an issue with the audit JSON — that would
directly refute our FPR claim.

## Only have a relay key? (you can't enroll your own reference)

People who buy relays are, by definition, the people who *can't* reach the official
API. For them: `identify` audits against the **community reference library** in
[`refs/`](refs/) instead, with a graceful degradation ladder — never overclaiming:

1. Claimed model **in the library** → sequential test verdict (PASS/FAIL, FPR ≤ α)
   plus a nearest-neighbor ranking over the whole library;
2. Claimed model **not in the library** → "behaviorally consistent with X"
   (`BEST_MATCH`) if something matches;
3. Nothing matches → `UNKNOWN`. An honest "we don't know".

```bash
python -m fpverify.cli library     # see what's enrolled
python -m fpverify.cli identify \
    --base-url https://some-relay.example/v1 --api-key $RELAY_KEY \
    --model gpt-4o --samples 8     # ≈ 288 one-token requests
```

Or use the **local web console** — paste base URL / key / claimed model, pick from the
relay's own model list, get a verdict card with the distance ranking:

```bash
python -m webui.server             # opens http://127.0.0.1:8765
```

Probe traffic goes directly from *your machine* to the relay; the key never touches
any backend of ours (the UI is a local process, the library is public data in git —
update it with `git pull`).

The library currently ships 9 frontier models measured in July 2026 on the
`cursor-harness` channel (same-channel comparisons only). The `api` channel — bare
official-API references usable for relay audits — **is open for contributions**: one
enrollment costs cents; anti-poisoning rules (provenance + independent cross-check
within the noise band) are in [`refs/README.md`](refs/README.md).

## Detection performance (controlled simulation)

Validated against nine adversary types with known ground truth
(α = 0.01, auto-calibrated tolerance, budget 600 queries; `experiments/run_evaluation.py`):

| Adversary | Detection | Mean queries to verdict | Est. cost* |
|---|---|---|---|
| Full model swap | 100% | 15 | $0.002 |
| Pinned answers ("always say 73") | 100% | 14 | $0.002 |
| True-random smokescreen | 100% | 14 | $0.002 |
| Filtering known probe phrasings | 100% | 17 | $0.003 |
| Partial distribution mimicry | 100% | 17 | $0.003 |
| Response cache | 100% | 89 | $0.014 |
| Quantized downgrade | 100% | 116 | $0.018 |
| Honest endpoint / benign drift | **0 false alarms** in 400 audits | — | — |

\* at flagship pricing ($2.5/M input, $10/M output). Blatant cheating is exposed in
~15 queries; the subtler the fraud, the more evidence is needed — but everything above
was caught within a $0.02 budget.

## Red team vs blue team: where it holds and where it breaks

We ran a 4-round co-evolution in the simulator — the relay invents an evasion, the
detector counters, repeat — under one economic rule: the relay only profits if it serves
the cheap model on *real* traffic, so "serve the flagship whenever a test is suspected"
counts as zero savings. Full ledger: [`docs/COEVOLUTION.md`](docs/COEVOLUTION.md).

- **Content-based routing** (flagship for anything that looks like a probe) → defeated.
  Business-native, capability-splitting probes force the relay to either route them to
  the flagship too, or push its threshold so hard that its cheap-traffic share collapses
  from 0.52 to 0.23. Evasion becomes economically self-defeating.
- **Account-based routing** (flagship for unfamiliar accounts, cheap for regulars) →
  defeats any one-shot external auditor, but is recovered by blending probes into a real
  account's organic traffic (passive auditing).
- **Random dilution** (serve the cheap model to a fraction ε of *all* traffic, blind to
  content and account) → the convergence point, and a **statistical hard limit** rather
  than a missing trick. Any test that keeps FPR ≤ α leaves a tolerance band of width ~δ,
  and a small enough ε hides inside it. The attacker's safe window is ε ≈ 0.20–0.28
  (saving 20–28%); shrinking δ to catch it pushes benign-drift FPR to 0.092 (9× over α).
  The only legitimate blue-team lever is more evidence — larger enrollment and long-run
  accumulation, where catching ε costs ~1/ε² samples.

Honest takeaway: structured substitution is caught cheaply; the residual attack is
low-rate random dilution, a trade of "money saved (ε)" against "samples the auditor must
spend (~1/ε²)". `fpverify`'s anytime-valid design is exactly what lets a continuous
auditor keep accumulating that evidence.

## How it works

1. **Probe**: ask semantically trivial questions with categorical one-token answers
   ("random number 1–100", "random color", coin flip…), in multiple phrasings and
   languages to resist string-matching filters.
2. **Normalize**: canonicalize answers, map unseen answers to an `OTHER` bucket
   (Good-Turing missing-mass handling).
3. **Compare**: Jensen-Shannon divergence between the endpoint's empirical distribution
   and the enrolled reference, aggregated across probe cells.
4. **Decide**: a sequential betting e-process accumulates evidence query-by-query.
   Anytime-valid: you may stop at any point, early-stop obvious verdicts, and the
   type-I error stays ≤ α. The benign-drift tolerance δ is auto-calibrated per
   reference via Dirichlet posterior-predictive simulation.

Based on **"One Token Is Enough"** (Bruckner, arXiv:2607.10252, 2026), which
established single-token distribution fingerprints (165 models, 326k requests).
On top of it, this project adds the sequential e-process decision layer (early
stopping + anytime-valid FPR control), adversarial hardening (multilingual paraphrase
probes, cache/latency screening), auto-calibration, and the frontier-model study above.

## Project layout

```
fpverify/     reusable library: probes, normalization, JSD, e-process, calibration, nearest-neighbor, community-library identify, CLI
refs/         community reference fingerprint library (manifest + per-model distributions, contribution protocol)
webui/        local web console for relay users (stdlib server; keys never leave your machine)
sim/          red team: model distributions, adversaries, HTTP mock relay, traffic model, blue-team probes
experiments/  evaluation, frontier study, red/blue co-evolution (FPR, power, budget, distance matrices)
tests/        statistical property tests (fairness, FPR bound, power, end-to-end, co-evolution, library/identify)
docs/         research notes (problem, threat model, method, experiments, frontier study, multimodal roadmap) + co-evolution ledger
```

## Roadmap

The decision core (JSD + sequential e-process + calibration) is **modality-agnostic**:
embed an image/video output, quantize it to a codebook, and the same PASS/FAIL machinery
applies unchanged. Extending swap-detection to image/video generation APIs — with
fixed-seed reproducibility as an extra near-deterministic signal — is the planned v2.
Design in [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) §8; not yet implemented.

## Honest limitations

- A verdict is **statistical evidence, not cryptographic proof**. FAIL means "the
  distribution deviates significantly from the reference" — causes include model
  substitution, quantization, version rollback, or caching. Keep the JSON report and
  re-audit before drawing conclusions.
- Frontier-model fingerprints above were sampled *inside the Cursor agent harness*
  (system prompt present, temperature not controlled); they demonstrate non-randomness
  and separability but are not directly comparable to bare-API numbers. n=11 per model
  is small; the self-noise band is wide.
- Same-weights mode changes (thinking on/off) are invisible to the fingerprint;
  detecting those requires latency/length side-channels.
- An adaptive adversary that detects audit traffic at the *account* level defeats any
  one-shot certification; the countermeasure is continuous, low-rate, blended auditing
  (which the anytime-valid design supports natively).
- No guarantee against an adversary that perfectly reproduces the target model's full
  conditional distribution — but doing so is approximately as expensive as running the
  real model, which erases the profit motive.

## License

MIT
