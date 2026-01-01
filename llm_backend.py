"""
LLM Backend Abstraction Layer.

Provides a common interface for LLM calls with swappable implementations.
Currently supports OpenAI API; Claude CLI can be added later.

Usage:
    from llm_backend import get_backend, OpenAIBackend

    # Default: OpenAI
    backend = get_backend()
    response = await backend.complete("Your prompt here")

    # With system prompt
    response = await backend.complete(
        prompt="Your prompt",
        system="You are a helpful assistant"
    )

    # Structured output (OpenAI only for now)
    result = await backend.complete_structured(
        prompt="Generate a conversation",
        response_model=ConversationModel,
        system="Generate realistic dialogue"
    )
"""

from abc import ABC, abstractmethod
import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel
import re

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    retry_if_exception,
    before_sleep_log,
    RetryCallState,
)

logger = logging.getLogger(__name__)


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

    # Claude CLI - typed exception
    if isinstance(exception, ClaudeCLIRateLimitError):
        return True

    return False


def _is_transient_google_error(exception: BaseException) -> bool:
    """Check if exception is a transient Google error worth retrying.

    This includes:
    - Rate limit errors (429)
    - JSON parse errors (Gemini sometimes returns malformed JSON)
    - Server errors (5xx)
    """
    # Rate limits
    if _is_rate_limit_error(exception):
        return True

    # JSON parse errors from complete_structured (transient Gemini issue)
    if isinstance(exception, ValueError) and "Failed to parse Google response" in str(
        exception
    ):
        return True

    # Google server errors (5xx)
    try:
        from google.genai.errors import ClientError, ServerError

        if isinstance(exception, ServerError):
            return True
        if isinstance(exception, ClientError) and "5" in str(exception)[:3]:
            return True
    except ImportError:
        pass

    return False


def _extract_google_retry_delay(exception: BaseException) -> float | None:
    """Extract retryDelay from Google API error response.

    Google 429 errors include RetryInfo with format like:
    {'retryDelay': '16.412038513s'} or {'retryDelay': '16s'}

    Returns delay in seconds, or None if not found.
    """
    error_str = str(exception)

    # Look for retryDelay pattern: '16s' or '16.412038513s'
    match = re.search(r"retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]", error_str)
    if match:
        return float(match.group(1))

    # Also check for "Please retry in X.Xs" message
    match = re.search(r"Please retry in (\d+(?:\.\d+)?)s", error_str)
    if match:
        return float(match.group(1))

    return None


def _google_wait_strategy(retry_state: RetryCallState) -> float:
    """Custom wait strategy that uses Google's retryDelay when available.

    Falls back to exponential backoff if retryDelay not found.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None

    if exception:
        delay = _extract_google_retry_delay(exception)
        if delay is not None:
            # Add small buffer (1s) to avoid hitting limit edge
            logger.info(f"Using Google's suggested retry delay: {delay + 1:.1f}s")
            return delay + 1.0

    # Fallback: exponential backoff (2^attempt * 2, min 5s, max 120s)
    attempt = retry_state.attempt_number
    return min(120.0, max(5.0, (2**attempt) * 2))


T = TypeVar("T", bound=BaseModel)


@dataclass
class CompletionResult:
    """Result from an LLM completion."""

    content: str
    model: str
    usage: dict[str, int] | None = None  # tokens used, if available


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Generate a text completion.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            CompletionResult with the generated text
        """
        ...

    @abstractmethod
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured completion that parses to a Pydantic model.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class to parse response into
            system: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Instance of response_model
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this backend."""
        ...


class OpenAIBackend(LLMBackend):
    """OpenAI API backend using the Responses API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        """Initialize OpenAI backend.

        Args:
            model: Model to use (default: gpt-4o-mini for cost efficiency)
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        """
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
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Generate completion using OpenAI Responses API."""
        from openai.types.responses import EasyInputMessageParam

        # Build input following the pattern from assessor.py
        user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}

        if system:
            system_msg: EasyInputMessageParam = {"role": "system", "content": system}
            input_messages = [system_msg, user_msg]
        else:
            input_messages = [user_msg]

        response = await self._client.responses.create(
            model=self._model,
            input=input_messages,
            max_output_tokens=max_tokens,
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
        """Generate structured completion using OpenAI's parse endpoint."""
        from openai.types.responses import EasyInputMessageParam

        # Build input following the pattern from assessor.py
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
    """Google Gemini API backend using the google-genai SDK.

    Uses the async client for all operations.
    Structured output is handled by prompting for JSON and parsing.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash",
        api_key: str | None = None,
    ):
        """Initialize Google Gemini backend.

        Args:
            model: Model to use (default: gemini-3-flash)
            api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
        """
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"Google ({self._model})"

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=_google_wait_strategy,  # Use Google's suggested retry delay
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> CompletionResult:
        """Generate completion using Google Gemini API."""
        from google.genai import types

        # Build contents - Gemini uses a different format
        contents: list[types.Content] = []

        if system:
            # System instruction is passed separately in the config
            pass

        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        # Extract usage metadata with safe defaults
        input_tokens = 0
        output_tokens = 0
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0

        return CompletionResult(
            content=response.text or "",
            model=self._model,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )

    @retry(
        retry=retry_if_exception(_is_transient_google_error),
        stop=stop_after_attempt(10),
        wait=_google_wait_strategy,  # Use Google's suggested retry delay
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Generate structured completion using Gemini's native JSON schema support."""
        from google.genai import types

        # Build contents
        contents: list[types.Content] = []
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))

        # Use native structured output with response_schema
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system if system else None,
            response_mime_type="application/json",
            response_schema=response_model,  # Pass Pydantic model directly
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        # Parse the JSON response
        try:
            content = response.text or ""
            data = json.loads(content)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse Google response as {response_model.__name__}: {e}\nResponse: {response.text}"
            )


class ClaudeCLIBackend(LLMBackend):
    """Claude Code CLI backend for zero marginal cost generation.

    Uses the `claude` CLI tool to generate responses.
    Structured output is handled by prompting for JSON and parsing.
    """

    def __init__(
        self,
        model: str = "opus",
        timeout: int = 1800,  # 30 minutes for long prompts
        validate: bool = True,
    ):
        """Initialize Claude CLI backend.

        Args:
            model: Model alias to use (e.g., "opus", "sonnet", "claude-opus-4-5-20251101")
            timeout: Timeout in seconds for CLI calls (default: 1800 = 30 min)
            validate: If True, verify CLI is installed on init (default: True)

        Raises:
            RuntimeError: If validate=True and CLI is not available
        """
        self._model = model
        self._timeout = timeout

        if validate:
            self._validate_cli()

    def _validate_cli(self) -> None:
        """Verify Claude CLI is installed and working."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude CLI installed but not working: {result.stderr}"
                )
            logger.debug(f"Claude CLI validated: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            ) from None

    @property
    def name(self) -> str:
        return f"Claude CLI ({self._model})"

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=wait_fixed(
            3600
        ),  # Wait 1 hour between retries (usage limit resets hourly)
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,  # noqa: ARG002 - CLI doesn't support this
    ) -> CompletionResult:
        """Generate completion using Claude CLI."""
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

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            ),
        )

        # Parse JSON response (CLI returns JSON even on errors)
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            response = None

        # Check for rate limit errors via is_error field in structured response
        if response and response.get("is_error"):
            error_result = response.get("result", "")
            if "hit your limit" in error_result.lower():
                logger.warning(f"Claude CLI rate limit: {error_result}")
                raise ClaudeCLIRateLimitError(error_result)

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown error (empty output)"
            )
            logger.error(
                f"Claude CLI failed:\n"
                f"  Command: {' '.join(cmd[:6])}...\n"  # Log first 6 args (avoids huge prompts)
                f"  Exit code: {result.returncode}\n"
                f"  Stderr: {result.stderr[:500] if result.stderr else '(empty)'}\n"
                f"  Stdout: {result.stdout[:500] if result.stdout else '(empty)'}"
            )
            raise RuntimeError(
                f"Claude CLI error (exit {result.returncode}): {error_msg}"
            )

        if response is None:
            raise RuntimeError("Claude CLI returned invalid JSON")

        return CompletionResult(
            content=response.get("result", ""),
            model="claude-cli",
            usage=None,  # CLI doesn't report usage
        )

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(10),
        wait=wait_fixed(
            3600
        ),  # Wait 1 hour between retries (usage limit resets hourly)
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,  # noqa: ARG002 - CLI doesn't support this
    ) -> T:
        """Generate structured completion using Claude CLI's native JSON schema."""
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

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            ),
        )

        # Parse JSON response (CLI returns JSON even on errors)
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            response = None

        # Check for rate limit errors via is_error field in structured response
        if response and response.get("is_error"):
            error_result = response.get("result", "")
            if "hit your limit" in error_result.lower():
                logger.warning(f"Claude CLI rate limit: {error_result}")
                raise ClaudeCLIRateLimitError(error_result)

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip()
                or result.stdout.strip()
                or "Unknown error (empty output)"
            )
            logger.error(
                f"Claude CLI failed:\n"
                f"  Command: {' '.join(cmd[:6])}...\n"  # Log first 6 args (avoids huge prompts)
                f"  Exit code: {result.returncode}\n"
                f"  Stderr: {result.stderr[:500] if result.stderr else '(empty)'}\n"
                f"  Stdout: {result.stdout[:500] if result.stdout else '(empty)'}"
            )
            raise RuntimeError(
                f"Claude CLI error (exit {result.returncode}): {error_msg}"
            )

        if response is None:
            raise RuntimeError("Claude CLI returned invalid JSON")

        # With --json-schema, the parsed output is in structured_output (already a dict)
        structured_output = response.get("structured_output")
        if structured_output is None:
            raise ValueError(
                f"No structured_output in Claude CLI response. Response: {response}"
            )

        return response_model.model_validate(structured_output)


# =============================================================================
# Backend Registry and Factory
# =============================================================================

_default_backend: LLMBackend | None = None


def get_backend(
    backend_type: str = "openai",
    **kwargs,
) -> LLMBackend:
    """Get an LLM backend instance.

    Args:
        backend_type: "openai", "claude", or "google"
        **kwargs: Backend-specific configuration

    Returns:
        LLMBackend instance
    """
    if backend_type == "openai":
        return OpenAIBackend(**kwargs)
    elif backend_type == "claude":
        return ClaudeCLIBackend(**kwargs)
    elif backend_type == "google":
        return GoogleBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def set_default_backend(backend: LLMBackend) -> None:
    """Set the default backend for the module."""
    global _default_backend
    _default_backend = backend


def get_default_backend() -> LLMBackend:
    """Get the default backend, creating OpenAI if not set."""
    global _default_backend
    if _default_backend is None:
        _default_backend = OpenAIBackend()
    return _default_backend
