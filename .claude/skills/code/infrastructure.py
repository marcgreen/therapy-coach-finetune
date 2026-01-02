"""
Copy-Paste Ready Infrastructure for Multi-Turn Fine-Tuning.

This module contains domain-agnostic code patterns that work for any
multi-turn conversation fine-tuning project. Copy directly into your project.

Sections:
1. LLM Backend Abstraction (multi-provider with retry)
2. Checkpoint Management (crash-resilient batch processing)
3. Conversation Slicing (training example generation)
4. Token Counting (tiktoken wrapper)
5. Assessment Infrastructure (scoring, not criteria)

Dependencies:
    pip install tenacity pydantic tiktoken
    # Plus provider-specific: openai, google-genai
"""

import asyncio
import hashlib
import json
import logging
import random
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import tiktoken
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

logger = logging.getLogger(__name__)

# =============================================================================
# 1. LLM BACKEND ABSTRACTION
# =============================================================================

T = TypeVar("T", bound=BaseModel)


@dataclass
class CompletionResult:
    """Result from an LLM completion."""

    content: str
    model: str
    usage: dict[str, int] | None = None


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Generate a text completion."""
        ...

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured completion that parses to a Pydantic model."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this backend."""
        ...


# -----------------------------------------------------------------------------
# Rate Limit Detection
# -----------------------------------------------------------------------------


class ClaudeCLIRateLimitError(Exception):
    """Raised when Claude CLI hits usage limits."""

    pass


def _is_rate_limit_error(exception: BaseException) -> bool:
    """Check if exception is a rate limit error from any provider."""
    # OpenAI
    try:
        from openai import RateLimitError as OpenAIRateLimitError

        if isinstance(exception, OpenAIRateLimitError):
            return True
    except ImportError:
        pass

    # Google - check for ClientError with 429 code
    try:
        from google.genai.errors import ClientError

        if isinstance(exception, ClientError) and "429" in str(exception):
            return True
    except ImportError:
        pass

    # Claude CLI
    if isinstance(exception, ClaudeCLIRateLimitError):
        return True

    return False


def _extract_google_retry_delay(exception: BaseException) -> float | None:
    """Extract retryDelay from Google API error response.

    Google 429 errors include RetryInfo: {'retryDelay': '16.412038513s'}
    Returns delay in seconds, or None if not found.
    """
    error_str = str(exception)
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]", error_str)
    if match:
        return float(match.group(1))
    return None


def _google_wait_strategy(retry_state: RetryCallState) -> float:
    """Custom wait strategy that uses Google's retryDelay when available."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None

    if exception:
        delay = _extract_google_retry_delay(exception)
        if delay is not None:
            logger.info(f"Using Google's suggested retry delay: {delay + 1:.1f}s")
            return delay + 1.0

    # Fallback: exponential backoff
    attempt = retry_state.attempt_number
    return min(120.0, max(5.0, (2**attempt) * 2))


# -----------------------------------------------------------------------------
# Backend Implementations (copy the ones you need)
# -----------------------------------------------------------------------------


class OpenAIBackend(LLMBackend):
    """OpenAI API backend using the Responses API."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"OpenAI ({self._model})"

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete(
        self, prompt: str, system: str | None = None, max_tokens: int = 4096
    ) -> CompletionResult:
        from openai.types.responses import EasyInputMessageParam

        user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}
        if system:
            system_msg: EasyInputMessageParam = {"role": "system", "content": system}
            input_messages = [system_msg, user_msg]
        else:
            input_messages = [user_msg]

        response = await self._client.responses.create(
            model=self._model, input=input_messages, max_output_tokens=max_tokens
        )

        return CompletionResult(
            content=response.output_text,
            model=self._model,
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
        )

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        from openai.types.responses import EasyInputMessageParam

        user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}
        if system:
            system_msg: EasyInputMessageParam = {"role": "system", "content": system}
            input_messages = [system_msg, user_msg]
        else:
            input_messages = [user_msg]

        response = await self._client.responses.parse(
            model=self._model,
            input=input_messages,
            text_format=response_model,
            max_output_tokens=max_tokens,
        )

        if response.output_parsed is None:
            raise ValueError(f"Failed to parse response into {response_model.__name__}")
        return response.output_parsed


class GoogleBackend(LLMBackend):
    """Google Gemini API backend."""

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str | None = None):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"Google ({self._model})"

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=_google_wait_strategy,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete(
        self, prompt: str, system: str | None = None, max_tokens: int = 4096
    ) -> CompletionResult:
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=config
        )

        return CompletionResult(
            content=response.text or "",
            model=self._model,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count or 0
                if response.usage_metadata
                else 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0
                if response.usage_metadata
                else 0,
            },
        )

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=_google_wait_strategy,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        from google.genai import types

        contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model, contents=contents, config=config
        )

        data = json.loads(response.text or "{}")
        return response_model.model_validate(data)


class ClaudeCLIBackend(LLMBackend):
    """Claude Code CLI backend for zero marginal cost generation."""

    def __init__(self, model: str = "sonnet", timeout: int = 1800):
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"Claude CLI ({self._model})"

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=wait_fixed(3600),  # Wait 1 hour (usage limit resets hourly)
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete(
        self, prompt: str, system: str | None = None, max_tokens: int = 4096
    ) -> CompletionResult:
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self._model,
        ]
        if system:
            cmd.extend(["--system-prompt", system])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._timeout
            ),
        )

        response = json.loads(result.stdout) if result.stdout else None

        if response and response.get("is_error"):
            if "hit your limit" in response.get("result", "").lower():
                raise ClaudeCLIRateLimitError(response.get("result", ""))

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr or result.stdout}")

        return CompletionResult(
            content=response.get("result", "") if response else "",
            model="claude-cli",
            usage=None,
        )

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=wait_fixed(3600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        schema = response_model.model_json_schema()
        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--model",
            self._model,
        ]
        if system:
            cmd.extend(["--system-prompt", system])

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._timeout
            ),
        )

        response = json.loads(result.stdout) if result.stdout else None

        if response and response.get("is_error"):
            if "hit your limit" in response.get("result", "").lower():
                raise ClaudeCLIRateLimitError(response.get("result", ""))

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr or result.stdout}")

        structured_output = response.get("structured_output") if response else None
        if structured_output is None:
            raise ValueError("No structured_output in Claude CLI response")

        return response_model.model_validate(structured_output)


# -----------------------------------------------------------------------------
# Backend Factory
# -----------------------------------------------------------------------------


def get_backend(backend_type: str = "openai", **kwargs) -> LLMBackend:
    """Get an LLM backend instance.

    Args:
        backend_type: "openai", "google", or "claude"
        **kwargs: Backend-specific configuration (model, api_key, etc.)
    """
    backends = {
        "openai": OpenAIBackend,
        "google": GoogleBackend,
        "claude": ClaudeCLIBackend,
    }
    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend: {backend_type}. Choose from {list(backends)}"
        )
    return backends[backend_type](**kwargs)


# =============================================================================
# 2. CHECKPOINT MANAGEMENT
# =============================================================================


def load_checkpoint(checkpoint_path: Path) -> set[str]:
    """Load completed IDs from JSONL checkpoint file.

    Returns set of IDs that have already been processed.
    """
    if not checkpoint_path.exists():
        return set()

    completed = set()
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # Adapt the key to your ID field
                if "id" in record:
                    completed.add(record["id"])
                elif "conversation_id" in record:
                    completed.add(record["conversation_id"])
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed checkpoint line: {line[:50]}...")
    return completed


def load_checkpoint_results(checkpoint_path: Path) -> dict[str, dict]:
    """Load full results from JSONL checkpoint file.

    Returns dict mapping ID -> full result dict.
    """
    if not checkpoint_path.exists():
        return {}

    results = {}
    with open(checkpoint_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                id_key = "id" if "id" in record else "conversation_id"
                if id_key in record:
                    results[record[id_key]] = record
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed checkpoint line: {line[:50]}...")
    return results


def append_checkpoint(checkpoint_path: Path, record: dict) -> None:
    """Append a record to JSONL checkpoint file (atomic append)."""
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# =============================================================================
# 3. CONVERSATION SLICING
# =============================================================================

# Slicing constants - adjust for your domain
MIN_CONTEXT = 3  # First slice at exchange 3 minimum
MIN_GAP = 2  # At least 2 exchanges between slices
MAX_GAP = 5  # At most 5 exchanges between slices


def get_slice_points(total_turns: int, transcript_id: str) -> list[int]:
    """Generate random slice points with stable seeding.

    Uses SHA256 for reproducible results across Python runs.
    """
    seed = int(hashlib.sha256(transcript_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    points = []
    current = MIN_CONTEXT

    while current <= total_turns:
        points.append(current)
        gap = rng.randint(MIN_GAP, MAX_GAP)
        current += gap

    # Always include final turn
    if points[-1] != total_turns:
        points.append(total_turns)

    return points


# =============================================================================
# 4. TOKEN COUNTING
# =============================================================================

_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_tokenizer.encode(text))


def count_messages_tokens(messages: list[dict]) -> int:
    """Count tokens in a messages array (with overhead)."""
    total = 0
    for msg in messages:
        total += (
            count_tokens(msg.get("content", "")) + 4
        )  # ~4 tokens overhead per message
    return total


def find_max_exchanges_under_limit(
    exchanges: list[dict],
    system_prompt: str,
    max_tokens: int,
) -> int:
    """Find maximum number of recent exchanges that fit under token limit.

    Returns count of exchanges (from the end) that fit.
    Used for truncating long transcripts.
    """
    system_tokens = count_tokens(system_prompt) + 4

    total_tokens = system_tokens
    count = 0

    for ex in reversed(exchanges):
        ex_tokens = count_tokens(ex["user"]) + count_tokens(ex["assistant"]) + 8
        if total_tokens + ex_tokens > max_tokens:
            break
        total_tokens += ex_tokens
        count += 1

    return count


# =============================================================================
# 5. ASSESSMENT INFRASTRUCTURE
# =============================================================================


CriterionAnswer = Literal["YES", "NO", "NA", "ERROR"]


@dataclass
class AssessmentResult:
    """Assessment result - adapt fields for your domain."""

    id: str
    passed: bool
    score: float
    threshold: float
    category_scores: dict[str, float]
    answers: dict[str, CriterionAnswer]
    reasonings: dict[str, str]
    failed_checks: list[str]
    safety_gate_failed: bool
    error_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "pass": self.passed,
            "score": round(self.score, 3),
            "threshold": self.threshold,
            "category_scores": {
                k: round(v, 3) for k, v in self.category_scores.items()
            },
            "answers": self.answers,
            "reasonings": self.reasonings,
            "failed_checks": self.failed_checks,
            "safety_gate_failed": self.safety_gate_failed,
            "error_count": self.error_count,
        }


def compute_category_score(
    answers: dict[str, CriterionAnswer],
    criteria_ids: list[str],
    na_invalid: set[str],
) -> float:
    """Compute score for a single category.

    Args:
        answers: Dict of criterion_id -> answer
        criteria_ids: Criteria IDs in this category
        na_invalid: Criteria where NA should be treated as failure
    """
    applicable = [cid for cid in criteria_ids if cid in answers]
    if not applicable:
        return 1.0  # No criteria = pass

    scores = []
    for cid in applicable:
        ans = answers[cid]
        if ans == "YES":
            scores.append(1.0)
        elif ans == "NA":
            if cid in na_invalid:
                scores.append(0.0)  # NA not allowed = failure
            # else: skip (doesn't count)
        else:  # NO or ERROR
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 1.0


def compute_weighted_score(
    category_scores: dict[str, float],
    category_weights: dict[str, float],
) -> float:
    """Compute weighted average score across categories."""
    return sum(
        category_scores.get(cat, 1.0) * weight
        for cat, weight in category_weights.items()
    )


def compute_length_stats(turns: list[tuple[str, str]]) -> str:
    """Compute deterministic length statistics for assessment.

    LLMs can't count accurately - pre-compute and inject into prompt.
    """
    stats = []
    ratios = []

    for i, (user, assistant) in enumerate(turns, 1):
        user_words = len(user.split())
        asst_words = len(assistant.split())
        ratio = asst_words / user_words if user_words > 0 else 0
        ratios.append(ratio)
        stats.append(
            f"Turn {i}: User={user_words}w, Asst={asst_words}w, Ratio={ratio:.2f}x"
        )

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    over_2x = sum(1 for r in ratios if r > 2.0)

    return "\n".join(
        [
            "LENGTH STATISTICS (pre-computed):",
            *stats,
            f"Average ratio: {avg_ratio:.2f}x",
            f"Turns > 2x: {over_2x}/{len(ratios)}",
        ]
    )
