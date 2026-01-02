# Copy-Paste Ready Code

This directory contains domain-agnostic infrastructure code that can be copied directly into any multi-turn conversation fine-tuning project.

**→ See [SETUP-REFERENCE.md](SETUP-REFERENCE.md) for:**
- Project structure and file templates (for agents setting up projects)
- Manual usage examples (for humans using infrastructure.py directly)
- Script running instructions (for humans running the pipeline)

## Files

### `infrastructure.py`

Complete module (~400 lines) with:

| Section | What It Does | Copy-Paste Ready? |
|---------|--------------|-------------------|
| **LLM Backend Abstraction** | Multi-provider interface (OpenAI, Google, Claude CLI) | Yes |
| **Rate Limit Handling** | Detection + retry for all providers | Yes |
| **Google Retry Strategy** | Extracts `retryDelay` from 429 errors | Yes |
| **Checkpoint Management** | JSONL append pattern for crash resilience | Yes |
| **Conversation Slicing** | SHA256-seeded random slice points | Yes |
| **Token Counting** | tiktoken wrapper with message overhead | Yes |
| **Assessment Infrastructure** | Scoring functions, NOT criteria definitions | Yes |

## Usage

```python
# Copy the entire file into your project
cp infrastructure.py /your/project/

# Or import specific components
from infrastructure import (
    get_backend,
    load_checkpoint,
    append_checkpoint,
    get_slice_points,
    count_tokens,
    compute_weighted_score,
)
```

## What You Still Need to Create

The infrastructure handles the HOW. You need to define the WHAT:

| Component | What to Define |
|-----------|----------------|
| **Criteria** | Your domain-specific evaluation criteria |
| **Categories** | How to group and weight your criteria |
| **Safety Gates** | Which criteria auto-reject on failure |
| **Prompts** | User simulator and assistant generation prompts |
| **Taxonomy** | Input distribution (topics, styles, difficulty) |

## Dependencies

```toml
# pyproject.toml
dependencies = [
    "tenacity==8.2.3",
    "pydantic==2.5.0",
    "tiktoken==0.5.2",
    # Provider-specific (include what you use):
    "openai==1.12.0",
    "google-genai==0.3.0",
]
```

## See Also

- [examples/therapy-domain.md](../examples/therapy-domain.md) - Complete domain-specific example
- [finetune-design/SKILL.md](../finetune-design/SKILL.md) - Design phase workflow
