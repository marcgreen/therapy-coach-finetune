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

logger = logging.getLogger(__name__)

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


class ClaudeCLIBackend(LLMBackend):
    """Claude Code CLI backend for zero marginal cost generation.

    Uses the `claude` CLI tool to generate responses.
    Structured output is handled by prompting for JSON and parsing.
    """

    def __init__(
        self,
        model: str = "opus",
        timeout: int = 300,
        validate: bool = True,
    ):
        """Initialize Claude CLI backend.

        Args:
            model: Model alias to use (e.g., "opus", "sonnet", "claude-opus-4-5-20251101")
            timeout: Timeout in seconds for CLI calls
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

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr}")

        response = json.loads(result.stdout)
        return CompletionResult(
            content=response.get("result", ""),
            model="claude-cli",
            usage=None,  # CLI doesn't report usage
        )

    async def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> T:
        """Generate structured completion by prompting for JSON.

        Note: Less reliable than OpenAI's native structured output.
        Consider using OpenAI backend for structured generation.
        """
        # Build a prompt that asks for JSON matching the schema
        schema = response_model.model_json_schema()
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
```json
{json.dumps(schema, indent=2)}
```

Output only the JSON, no other text."""

        result = await self.complete(
            prompt=structured_prompt,
            system=system,
            max_tokens=max_tokens,
        )

        # Parse the response as JSON
        try:
            # Try to extract JSON from the response
            content = result.content.strip()
            if content.startswith("```"):
                # Extract from code block (handles ```json, ```JSON, ``` etc.)
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        if not in_block:
                            # Starting a code block - skip this line (including ```json tag)
                            in_block = True
                            continue
                        else:
                            # Ending the code block
                            break
                    elif in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)

            data = json.loads(content)
            return response_model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse Claude response as {response_model.__name__}: {e}"
            )


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
        backend_type: "openai" or "claude"
        **kwargs: Backend-specific configuration

    Returns:
        LLMBackend instance
    """
    if backend_type == "openai":
        return OpenAIBackend(**kwargs)
    elif backend_type == "claude":
        return ClaudeCLIBackend(**kwargs)
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
