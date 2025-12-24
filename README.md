## therapay-finetune

Fine-tuning a **privacy-first, locally-runnable therapeutic coaching model** (project spec in `SPEC.md`).

### What’s implemented in this repo today

- **Evaluation rubric**: `reference/evaluation-rubric.md` (turn-level CP*/CN*/US*/FT*/SF* + conversation-level CV1–CV6)
- **Therapeutic framework reference**: `reference/therapeutic-frameworks.md` (9 approaches + techniques + failure modes)
- **Parallel evaluator**: `eval/parallel_evaluator.py` (LLM-as-judge using OpenAI Responses API)

### What’s not implemented yet (planned)

- Data generation scripts / pipeline outputs (`config/`, `data/`, training/export automation)

### Quick start

Prereqs:
- Python 3.12+
- `uv`
- `OPENAI_API_KEY` in your environment (the evaluator calls the OpenAI API)

Install:

```bash
uv sync
```

Run a single-turn evaluation:

```bash
uv run python eval/parallel_evaluator.py "user message here" "assistant response here"
```

For multi-turn conversation evaluation, import and call `assess_full_conversation()` from `eval/parallel_evaluator.py`.

