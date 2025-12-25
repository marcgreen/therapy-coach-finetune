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

## Dependencies

**Always pin exact versions with `==`, never use `>=`.**

```toml
# Good
"openai==2.14.0"
"pytest==9.0.2"

# Bad
"openai>=2.14.0"
"pytest>=8.0.0"
```

**Why:** Reproducible builds. Everyone gets the same versions. No surprise breakage from upstream updates.

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

### Type Safety

- **`cast()` is a code smell.** If you're casting, ask: "what bug am I potentially hiding?"
- When `cast()` is unavoidable (e.g., Python can't narrow `Literal` types in lists), add runtime validation:

```python
# Bad: cast hides typos
cast(CriterionAnswer, "YESS")  # No error, but wrong

# Good: validate then cast
VALID = {"YES", "NO", "NA", "ERROR"}
if ans not in VALID:
    raise ValueError(f"Invalid: {ans}")
cast(CriterionAnswer, ans)  # Now safe
```

### Async

- Use `asyncio` for concurrent I/O operations
- Prefer `asyncio.gather()` for parallel execution
- Use `AsyncOpenAI` client for openai API calls (and always use responses api, not chat completions)

## Testing: Verify Requirements, Not Implementation

### The Core Question

Before writing a test, ask: **"What bug would this catch?"**

If the answer is "none" or "it verifies the code does what the code does," the test is useless.

### Good Tests vs Bad Tests

**Bad: Hardcoded magic numbers from implementation**
```python
def test_weighted_score(self):
    result = compute_score(results, criteria)
    assert result.score == 0.85  # Where does 0.85 come from? The code.
```
This test breaks if you intentionally change weights. It doesn't know if that's a bug or a feature.

**Good: Test the formula, not the result**
```python
def test_weighted_score_matches_formula(self):
    result = compute_score(results, criteria)
    expected = sum(
        result.category_scores[cat] * CATEGORY_WEIGHTS[cat]
        for cat in CATEGORY_WEIGHTS
    )
    assert result.score == pytest.approx(expected)
```
This test verifies the *invariant* (score = weighted sum) regardless of what the weights are.

**Bad: Tautological test**
```python
def test_all_yes_gives_perfect_score(self):
    # ... all YES answers ...
    assert result.score == 1.0  # Why 1.0? Because the code says so.
```

**Good: Test business invariants**
```python
def test_score_is_bounded(self):
    """Score must always be between 0 and 1."""
    assert 0.0 <= result.score <= 1.0

def test_safety_gate_implies_failure(self):
    """If safety gate failed, passed must be False (regardless of score)."""
    if result.safety_gate_failed:
        assert result.passed is False

def test_more_yes_means_higher_or_equal_score(self):
    """Monotonicity: replacing NO with YES should never decrease score."""
    # ...
```

### Test Hierarchy (Prefer Earlier Types)

1. **Property/invariant tests** — "For all valid inputs, X is always true"
2. **Boundary tests** — Edge cases, empty inputs, off-by-one
3. **Example tests** — Specific inputs → specific outputs (use sparingly)

### Red Flags in Tests

| Red Flag | Problem |
|----------|---------|
| Magic numbers (`0.85`, `42`) | Coupled to implementation, not requirements |
| Testing mocks | Verifies nothing about real behavior |
| Identical structure, different data | Probably redundant |
| No edge cases | Missing empty, null, boundary conditions |
| Test name describes implementation | Should describe requirement |

### What to Test

- **Pure functions** — Easy to test, high value. Test invariants.
- **Business rules** — "Safety failure = rejection" is a real requirement.
- **Edge cases** — Empty inputs, missing data, boundary values.

### What NOT to Test

- **Implementation details** — Internal data structures, private methods.
- **Third-party code** — Don't test that `json.loads` works.
- **Mocks** — If you're asserting on mock behavior, you're testing your test setup.
