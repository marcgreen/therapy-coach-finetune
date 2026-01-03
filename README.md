# Therapeutic Coaching Fine-tune

A **privacy-first, locally-runnable therapeutic coaching model** fine-tuned on synthetic multi-turn conversations.

> ⚠️ **Not therapy.** This is a self-care coaching tool for stable adults. No professional therapist was involved in this project. Cannot handle crises. Cannot replace professional mental health care.

## Results

Evaluated against a custom **17-criterion therapeutic quality rubric** (15 weighted + 2 safety gates) developed as part of this project:

| Metric | Fine-tuned | Base | Improvement |
|--------|-----------|------|-------------|
| **Mean Score** | 0.831 | 0.701 | +18.5% |
| **Pass Rate** | 57.1% | 20.0% | +37pp |
| **p-value** | | | 0.0165 |

Biggest gains: **naturalness** (+0.296) and **context use** (+0.260)

## Therapeutic Approach

Training data incorporates **9 evidence-based therapeutic frameworks**:

| Framework | Focus | Best For |
|-----------|-------|----------|
| **CBT** | Thought patterns → feelings → behaviors | Anxiety, depression, negative self-talk |
| **DBT** | Distress tolerance, emotion regulation | Intense emotions, interpersonal conflict |
| **ACT** | Psychological flexibility, values-based action | Avoidance, getting "stuck", meaning-making |
| **Motivational Interviewing** | Exploring ambivalence about change | Resistance, "I know I should but..." |
| **Solution-Focused (SFBT)** | What's working, future-oriented | Feeling stuck, building on strengths |
| **Person-Centered** | Unconditional positive regard, reflection | Needing to be heard, self-exploration |
| **Positive Psychology** | Strengths, gratitude, meaning | Building resilience, flourishing |
| **Compassion-Focused (CFT)** | Self-criticism → self-compassion | Shame, perfectionism, harsh inner critic |
| **Behavioral Activation** | Action before motivation | Low energy, depression, avoidance |

The model adaptively applies whichever approach fits the situation rather than adhering to a single modality. See [`reference/therapeutic-frameworks.md`](reference/therapeutic-frameworks.md) for detailed technique documentation.

## Three Deliverables

1. **[Fine-tuned Model](https://huggingface.co/marcgreen/therapeutic-qwen3-14b)** — Qwen 3 14B (4-bit quantized, ~9GB) optimized for therapeutic coaching, runs locally via Ollama/llama.cpp
2. **[Open Dataset](https://huggingface.co/datasets/marcgreen/therapeutic-coaching-v1)** — ~1,300 synthetic multi-turn therapeutic conversations
3. **Claude Code SKILLs** — Domain-agnostic fine-tuning pipeline (design → generate → train) adaptable to any domain

---

## Quick Start

### Using llama.cpp

```bash
# Download the GGUF from HuggingFace
wget https://huggingface.co/marcgreen/therapeutic-qwen3-14b/resolve/main/therapeutic-qwen3-14b-q4_k_m.gguf

# Run with llama-server
llama-server -m therapeutic-qwen3-14b-q4_k_m.gguf --port 8080 -ngl 99
```

### Hardware Requirements

- **RAM:** ~9GB for 4-bit quantized model
- **GPU:** Apple Silicon (Metal) or CUDA recommended
- `-ngl 99` offloads all layers to GPU

---

## How It Works

### Two-Agent Simulation

Training data was generated via conversation simulation between two LLM agents:

1. **User Simulator** — Generates realistic multi-topic messages with human-like "flaws" (burying the lede, tangential rambling, yes-but resistance, contradicting self, etc.)
2. **Therapist** — Responds using eclectic therapeutic techniques adapted to the situation

Conversations range from 3-100 turns, building genuine context and topic continuity.

### Rubric-Based Filtering

Every generated conversation is assessed by an LLM judge against 17 binary criteria:

| Category | Weight | Criteria |
|----------|--------|----------|
| **Comprehension** | 15% | Accurate understanding, tentative framing |
| **Connection** | 20% | Emotional attunement, user empowerment |
| **Naturalness** | 15% | Length calibration, non-formulaic, varied structure |
| **Multi-Topic** | 30% | Topic coverage, depth calibration, segmentation |
| **Context Use** | 20% | History utilization, thread continuity, coaching loops |
| **Safety Gates** | — | No harmful patterns, appropriate crisis handling |

Pass threshold: 0.80 weighted score + no safety gate failures.

Only passing conversations enter the training set. See [`reference/assessment-rubric.md`](reference/assessment-rubric.md) for full criterion definitions.

### Training

- **Base model:** Qwen 3 14B
- **Method:** QLoRA fine-tuning via HuggingFace Jobs
- **Context:** Truncated to 16K tokens due to memory constraints
- **Export:** Merged adapter → GGUF quantization for local inference

---

## Dataset

**[marcgreen/therapeutic-coaching-v1](https://huggingface.co/datasets/marcgreen/therapeutic-coaching-v1)** — ~1,300 synthetic multi-turn therapeutic coaching conversations.

### Contents

- Multi-turn conversations (3-100 exchanges each)
- Covers anxiety, relationships, life transitions, self-worth, emotional regulation, and edge cases
- Synthetic data only—no real therapy transcripts

### Usage

```python
from datasets import load_dataset

dataset = load_dataset("marcgreen/therapeutic-coaching-v1")
```

### Caveats

- Generated by Claude (Sonnet/Haiku for user sim, Sonnet/Opus for therapist)
- Filtered by single-backend assessment—stricter multi-backend filtering could improve quality
- Some phrase repetition patterns exist in the data (e.g., "that's not nothing" appears frequently)

---

## Claude Code SKILLs

This project includes reusable fine-tuning SKILLs that can be adapted to any domain:

| Skill | Purpose |
|-------|---------|
| `finetune-design` | Design rubrics, taxonomies, and persona generation |
| `finetune-generate` | Two-agent conversation simulation and assessment |
| `finetune-train` | HuggingFace Jobs training, GGUF conversion, evaluation |

### Adapting to Your Domain

To use this pipeline for a different domain (e.g., coding assistants, legal, medical Q&A):

1. **Design phase** (`finetune-design`) — Build your domain's taxonomy, rubric, and prompts
2. **Iterate** (`finetune-generate`) — Generate small batches, assess, refine prompts and rubric based on failures
3. **Scale** (`finetune-generate` → `finetune-train`) — Once pass rates stabilize, generate larger datasets and train

The skills guide you through each phase. Expect 3-5 prompt/rubric revision cycles before quality stabilizes.

See `.claude/skills/` for full skill definitions.

---

## Project Structure

```
├── config/
│   ├── input-taxonomy.yaml      # Topic seeds for generation
│   ├── flaw-taxonomy.yaml       # Human flaw patterns for user simulation
│   ├── system-prompt.md         # System prompt for fine-tuned model
│   └── prompts/
│       ├── assistant.md         # Therapist generation prompt
│       ├── user_sim.md          # User simulator prompt
│       └── assessor.md          # Assessment prompt
├── reference/
│   ├── assessment-rubric.md     # 17-criterion rubric documentation
│   └── therapeutic-frameworks.md # 9 therapeutic approaches
├── data/
│   ├── raw/                     # Generated transcripts
│   ├── assessments/             # Assessment results
│   └── processed/               # Filtered training data
├── models/                      # Local model artifacts
├── .claude/skills/              # Claude Code fine-tuning SKILLs
├── transcript_generator.py      # Two-agent conversation generator
├── assessor.py                  # LLM-as-judge evaluation
└── llm_backend.py              # Backend abstraction (OpenAI API, Claude CLI)
```

---

## Development

### Setup

```bash
# Install dependencies
uv sync

# Run linting and type checks
uv run ruff check .
uv run ty check .
```

### Generate Transcripts

```bash
uv run python transcript_generator.py --count 10 --output data/raw/transcripts
```

### Assess Transcripts

```bash
uv run python scripts/assess_remaining.py
```

Supports three backends:
- **Claude Code CLI** — Free during usage periods, but rates own output higher
- **Gemini 3 Flash** — Fast and cheap, good for bulk assessment
- **GPT-5-mini** — Catches issues other backends miss

Using multiple backends reduces self-bias (Claude rating Claude-generated content).

### Full Pipeline

See [`docs/plans/2025-12-30-e2e-finetune-pipeline.md`](docs/plans/2025-12-30-e2e-finetune-pipeline.md) for the complete workflow:

1. **Assess** — `scripts/assess_remaining.py`
2. **Gather passing** — `scripts/gather_passing.py`
3. **Slice into examples** — `scripts/slice_transcripts.py`
4. **Push to HuggingFace** — `scripts/push_dataset.py`
5. **Train** — `scripts/train_therapeutic_model.py` (via HF Jobs)
6. **Convert to GGUF** — `scripts/convert_to_gguf.py`
7. **Evaluate** — `scripts/run_evaluation.py`
