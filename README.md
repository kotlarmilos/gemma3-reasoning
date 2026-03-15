# Gemma3 Reasoning

Fine-tuning [Gemma 3 1B-IT](https://ai.google.dev/gemma) for improved math reasoning using a three-stage pipeline: **SFT → SFT → GRPO**, built with JAX on TPU.

Built as part of the Google Tunix Hackathon: https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemma3-reasoning

## Pipeline

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Stage 1: SFT      │     │   Stage 2: SFT      │     │   Stage 3: GRPO     │
│                     │     │                     │     │                     │
│ GSM8K (70%)         │────▶│ GSM8K + ScienceQA   │────▶│ GSM8K prompts       │
│ ScienceQA (30%)     │     │ + MBPP + XSum       │     │ Math-verifiable     │
│                     │     │ + WritingPrompts    │     │ reward function     │
│ lr=2e-5             │     │ lr=1e-5             │     │ lr=1e-6             │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
   Build reasoning            Expand capabilities          Reinforce correct
   foundation                 without forgetting           math answers
```

**Stage 1** teaches structured math reasoning. **Stage 2** adds diverse capabilities (code, summarization, creative writing) while retaining math via weighted sampling. **Stage 3** uses Group Relative Policy Optimization with a verifiable reward — the model gets 1.0 for correct numerical answers and partial credit for showing work.

**Model**: Gemma 3 1B-IT (google/gemma-3-1b-it)

**LoRA config**:
- Rank: 32, Alpha: 32.0
- Targets: `q_einsum`, `kv_einsum`, `gate_proj`, `down_proj`, `up_proj`, `attn_vec_einsum`

**Training**:
- Sequence length: 1024
- Batch size: 4
- Optimizer: AdamW with warmup + cosine decay
- Hardware: TPU v5e

**GRPO reward function**:
- 1.0 — correct numerical answer (verified against GSM8K gold)
- 0.2 — shows reasoning steps
- 0.1 — reaches a numerical answer
- 0.1 — ends cleanly
- 0.0 — degenerate output (repetition ratio < 30%)

**Datasets**:

| Dataset | Task | Stage |
|---------|------|-------|
| [GSM8K](https://huggingface.co/datasets/gsm8k) | Math word problems | 1, 2, 3 |
| [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Science multiple choice | 1, 2 |
| [MBPP](https://huggingface.co/datasets/mbpp) | Python programming | 2 |
| [XSum](https://huggingface.co/datasets/xsum) | Summarization | 2 |
| [WritingPrompts](https://huggingface.co/datasets/euclaise/writingprompts) | Creative writing | 2 |

## Stack

- [JAX](https://github.com/jax-ml/jax) — accelerated numerical computing
- [Tunix](https://github.com/google/tunix) — LLM post-training (SFT, GRPO)
- [Qwix](https://github.com/google/qwix) — LoRA adapter application
- [Flax NNX](https://github.com/google/flax) — neural network library
- [Optax](https://github.com/google-deepmind/optax) — gradient optimization

## Repository Structure

```
├── gemma3_reasoning.ipynb    # Full pipeline notebook
├── README.md
└── adapter/                  # Generated after training
    ├── adapter_model.safetensors
    ├── adapter_config.json
    └── README.md             # HuggingFace model card
```
