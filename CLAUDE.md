# CLAUDE.md

## Project Overview

Fine-tuning a **privacy-first, locally-runnable therapeutic coaching model** (7B parameters).

**Goals:**
1. A model that runs offline on consumer hardware (Mac/Ollama/GGUF), applies eclectic therapeutic approaches adaptively, and feels genuinely helpful for self-care
2. Reusable Claude Code SKILLs for the full fine-tuning pipeline (domain extraction → synthetic data → model selection → training → evaluation)

**Approach:**
- Generate synthetic multi-turn conversations using DSPy/GEPA optimization
- Evaluate with conversation-level rubric (12 criteria, safety gate)
- SFT fine-tuning on filtered high-quality data
- Export to GGUF for local inference

**Therapeutic frameworks:** CBT, DBT, ACT, Motivational Interviewing, Solution-Focused, Person-Centered, Positive Psychology, CFT, Behavioral Activation

**Python 3.12** | **uv** for package management

## Philosophy: MVP Always

**Find the smallest effective change that gets the job done.**

- Solve the immediate problem, nothing more
- No "while we're here" improvements
- No abstractions until the third use case
- No configurability until it's needed
- If it works and it's clear, ship it

Signs you're over-engineering:
- Adding parameters "for flexibility"
- Creating helpers for one-time operations
- Building for hypothetical future requirements
- Refactoring adjacent code that wasn't broken

## Commands

```bash
uv run python <script.py>      # Run scripts
uv add <package>               # Add dependency
uv sync                        # Install from lockfile
uv run ruff check .            # Lint
uv run ruff format .           # Format
uv run ty check .              # Type check
```

## Code Quality: Write It Right the First Time

**Pre-commit hooks run ruff and ty on every commit.** Write code that passes these checks without needing fixes.

During development:
- Run `uv run ruff check . && uv run ty check .` before considering code complete
- Fix issues as you write, not in a separate pass
- If ty complains about types, add the annotation - don't ignore it

Common mistakes to avoid upfront:
- Missing return type annotations
- Using `Dict`/`List` instead of `dict`/`list`
- Unused imports
- Unhandled `None` from optional returns

## Python Guidelines

### Typing (Required)

- Use modern typing syntax: `dict[str, int]` not `Dict[str, int]`
- Add type hints to all function signatures (parameters and return types)
- Use `|` for union types: `str | None` not `Optional[str]`
- Prefer `Literal` for constrained string values
- Use `type` statement for aliases (3.12+): `type UserId = int`

```python
# Good
def process(items: list[str], config: dict[str, Any] | None = None) -> bool:

# Bad
def process(items, config=None):
```

### Data Structures

- Use `dataclass` for simple data containers
- Use Pydantic `BaseModel` for validated data with parsing/serialization
- Use `NamedTuple` for immutable records

### Code Style

- Follow existing patterns in `assessor.py`
- Docstrings for public functions and classes

### Async

- Use `asyncio` for concurrent I/O operations
- Prefer `asyncio.gather()` for parallel execution
- Use `AsyncOpenAI` client for openai API calls (and always use responses api, not chat completions)
