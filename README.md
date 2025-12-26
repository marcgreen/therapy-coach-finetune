## therapay-finetune

Fine-tuning a **privacy-first, locally-runnable therapeutic coaching model** (project spec in `SPEC.md`).

### What's implemented

**Core pipeline:**
- **Transcript generator**: `transcript_generator.py` — generates multi-topic conversations via user simulation loop
- **Assessor**: `assessor.py` — LLM-as-judge evaluation (15 criteria: 13 weighted + 2 safety gate) using Claude CLI
- **LLM backend**: `llm_backend.py` — abstraction layer supporting OpenAI API and Claude CLI

**Reference materials:**
- `reference/assessment-rubric.md` — 15-criterion rubric for conversation quality
- `reference/therapeutic-frameworks.md` — 9 therapeutic approaches + techniques + failure modes

**Config:**
- `config/input-taxonomy.yaml` — topic seeds for generation
- `config/flaw-taxonomy.yaml` — flaw patterns for realistic user simulation
- `config/system-prompt.md` — system prompt for the fine-tuned model
- `config/prompts/` — prompts for assessor, assistant, and user simulator

**Scripts:**
- `run_base_model_eval.py` — run base model evaluation with llama-server
- `run_gemma_interactive.py` — interactive chat with Gemma
- `assess_base_model.py` — assess base model responses

**Output data (in `output/`):**
- Base model responses and assessments
- Evaluation scenarios and style examples

### What's not implemented yet

- SFT training pipeline
- GGUF export automation

### Quick start

Prereqs:
- Python 3.12+
- `uv`
- Claude CLI installed (for assessment and generation)

Install:

```bash
uv sync
```

Generate transcripts:

```bash
uv run python transcript_generator.py --count 1 --output data/raw/transcripts
```

Assess a conversation:

```python
from assessor import assess_conversation, ConversationInput, ConversationTurn

conversation = ConversationInput(turns=[
    ConversationTurn(role="user", content="..."),
    ConversationTurn(role="assistant", content="..."),
])
result = await assess_conversation(conversation)
print(f"Score: {result.score:.2f}, Passed: {result.passed}")
```

### Local model (Gemma 3 12B)

See `CLAUDE.md` for llama-server setup instructions.
