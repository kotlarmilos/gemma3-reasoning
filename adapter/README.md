---
base_model: google/gemma-3-1b-it
library_name: tunix
license: gemma
tags:
  - gemma3
  - lora
  - reasoning
  - math
  - grpo
  - jax
  - tpu
  - sft
datasets:
  - gsm8k
  - derek-thomas/ScienceQA
  - mbpp
  - xsum
  - euclaise/writingprompts
pipeline_tag: text-generation
---

# Gemma 3 1B-IT — Reasoning LoRA

LoRA adapter that improves math reasoning in [Gemma 3 1B-IT](https://huggingface.co/google/gemma-3-1b-it), trained with a three-stage pipeline (SFT → SFT → GRPO) using [Tunix](https://github.com/google/tunix) on TPU v5e.

Writeup: [gemma3-reasoning](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemma3-reasoning).

## Training Pipeline

### Stage 1 — SFT on Math & Science

Builds the reasoning foundation using structured math and science datasets.

- **Data**: GSM8K (70%) + ScienceQA (30%)
- **Steps**: 1000, batch size 4, lr=2e-5 with warmup + cosine decay
- **Sequence length**: 1024

### Stage 2 — SFT on Diverse Tasks

Expands capabilities to code, summarization, and creative writing while retaining math performance through weighted sampling.

- **Data**: GSM8K (25%) + ScienceQA (15%) + MBPP (25%) + XSum (20%) + WritingPrompts (15%)
- **Steps**: 600, batch size 4, lr=1e-5

### Stage 3 — GRPO with Math Reward

Reinforcement learning using Group Relative Policy Optimization. The reward function verifies answer correctness against GSM8K gold labels — no judge model needed.

- **Data**: GSM8K prompts
- **Steps**: 50, lr=1e-6
- **GRPO config**: 4 generations, 2 iterations, β=0.1, ε=0.2
- **Reward**: 1.0 for correct answer, 0.2 for showing reasoning steps, 0.1 for reaching a number, 0.1 for clean ending, 0.0 for degenerate output

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank | 32 |
| Alpha | 32.0 |
| Target modules | `q_einsum`, `kv_einsum`, `gate_proj`, `down_proj`, `up_proj`, `attn_vec_einsum` |

## Usage

This adapter was trained with Tunix/Qwix (JAX). To load and use:

```python
from tunix.models.gemma3 import params, model
from tunix.generate import sampler as sampler_lib
import qwix

# Load base model
base = params.create_model_from_checkpoint(
    params.GEMMA3_1B_IT,
    model.ModelConfig.gemma3_1b_it()
)

# Apply LoRA structure
lora_model = qwix.apply_lora_to_model(
    base,
    qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum",
        rank=32, alpha=32.0,
    ),
    rngs=nnx.Rngs(0),
    **base.get_model_input(),
)

# Load trained adapter weights
from safetensors.numpy import load_file
adapter = load_file("adapter_model.safetensors")
# Merge adapter weights into lora_model state

# Generate
tokenizer = params.create_tokenizer()
sampler = sampler_lib.Sampler(
    transformer=lora_model, tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=1536,
        num_layers=lora_model.config.num_layers,
        num_kv_heads=lora_model.config.num_kv_heads,
        head_dim=lora_model.config.head_dim,
    ),
)

prompt = "<start_of_turn>user\nWhat is 25 * 13? Think step by step.<end_of_turn>\n<start_of_turn>model\n"
out = sampler(
    input_strings=[prompt],
    max_generation_steps=512,
    temperature=0.7, top_k=50, top_p=0.95,
    echo=False, eos_tokens=[106],
)
print(out.text[0])
```

## Technical Details

- **Framework**: JAX + Tunix + Qwix + Flax NNX
- **Hardware**: TPU v5e (Google Colab)
- **Precision**: bfloat16
- **Optimizer**: AdamW with warmup cosine decay (SFT), clipped AdamW (GRPO)
- **GRPO reference model**: Frozen copy of base Gemma 3 1B-IT (KL anchor)

## Source

- **Code**: [github.com/kotlarmilos/gemma3-reasoning](https://github.com/kotlarmilos/gemma3-reasoning)
- **Writeup**: [gemma3-reasoning](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/gemma3-reasoning)

## Citation

```bibtex
@misc{kotlar2025gemma3reasoning,
  title={Gemma 3 1B-IT Reasoning LoRA},
  author={Kotlar, Milos},
  year={2025},
  url={https://github.com/kotlarmilos/gemma3-reasoning}
}
```
