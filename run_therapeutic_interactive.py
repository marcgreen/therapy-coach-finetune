"""Run interactive model test with therapeutic persona.

Uses the same short system prompt from training (config/system-prompt.md),
not the highly-specified one used for data generation.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from llm_backend import ClaudeCLIBackend
from transcript_generator import (
    GeneratorConfig,
    Persona,
    _extract_prompt_section,
    generate_persona,
    load_config,
)

LLAMA_SERVER_URL = "http://localhost:8080"

# The SHORT system prompt used during training (from config/system-prompt.md)
SYSTEM_PROMPT = """You are a supportive therapeutic coach. You help people explore their thoughts and feelings through conversation.

Core approach:
- Engage with what they share, not with stock phrases
- Ask questions to understand, don't assume
- Match the person's energy, pace, and message length
- Return agency - they decide what's right for them
- Stay warm and natural, not clinical
- When they are stuck or looping, offer a simple "why this might be happening" and one small next step to try before the next message.

Boundaries:
- You're a coaching tool, not a licensed therapist
- Don't diagnose conditions or recommend medications
- If they mention potentially urgent physical symptoms (e.g., chest pain, shortness of breath, fainting, new or worsening severe symptoms), encourage medical evaluation. Do not provide medical reassurance or "rule out" serious causes.
- For crisis signals or self-harm hints, do a brief safety check (intent/plan/safety) and then suggest professional resources if needed.

Adapt your style to each person. Some want to explore feelings, others want practical strategies, some just need to be heard."""


def format_prompt(system: str, history: list[dict], current_user: str) -> str:
    """Format multi-turn conversation for llama-server.

    Uses OpenAI-compatible messages format which llama-server handles.
    """
    messages = [{"role": "system", "content": system}]

    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": current_user})

    # Return as JSON for the chat endpoint
    return json.dumps(messages)


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60))
def _completion_request(
    system: str, history: list[dict], current_user: str, max_tokens: int
) -> str:
    """Make request to model with retries using chat completions API."""
    messages = [{"role": "system", "content": system}]

    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": current_user})

    response = httpx.post(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        json={
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


def get_model_response(
    system: str, history: list[dict], current_user: str, max_tokens: int = 600
) -> str:
    """Get response from model via llama-server."""
    try:
        return _completion_request(system, history, current_user, max_tokens)
    except Exception as e:
        return f"[ERROR: {e}]"


def format_history_for_user_sim(exchanges: list[dict]) -> str:
    """Format conversation history for Claude user simulator."""
    if not exchanges:
        return "(This is the start of the conversation)"

    lines = []
    for ex in exchanges:
        lines.append(f"--- Exchange {ex['exchange_number']} ---")
        lines.append(f"User: {ex['user']}")
        lines.append(f"Assistant: {ex['assistant']}")
        lines.append("")
    return "\n".join(lines)


async def generate_user_message(
    backend: ClaudeCLIBackend,
    config: GeneratorConfig,
    persona: Persona,
    exchanges: list[dict],
    exchange_number: int,
    target_exchanges: int,
) -> str:
    """Generate user message using Claude, reacting to conversation history."""
    history_text = format_history_for_user_sim(exchanges)

    # Build persona description
    persona_json = json.dumps(
        {
            "name": persona.name,
            "age_range": persona.age_range,
            "personality_traits": persona.personality_traits,
            "communication_style": persona.communication_style,
            "writing_style": persona.writing_style,
            "trajectory": persona.trajectory,
            "topic_seeds": [
                {
                    "category": t.category,
                    "subtopic": t.subtopic,
                    "complexity": t.complexity,
                    "description": t.description,
                }
                for t in persona.topic_seeds
            ],
        },
        indent=2,
    )

    flaw_text = "No flaw patterns assigned - you're a relatively clear communicator."
    if persona.flaw_patterns:
        flaw_text = "\n".join(
            f"- {f['name']}: {f['description']}" for f in persona.flaw_patterns
        )

    # Extract proper prompt sections from template
    template = config.user_sim_template
    system_prompt = _extract_prompt_section(template, "System Prompt")
    user_prompt_template = _extract_prompt_section(template, "User Prompt")

    # Format system prompt
    system_prompt = system_prompt.replace("{persona}", persona_json)
    system_prompt = system_prompt.replace("{flaw_patterns}", flaw_text)
    system_prompt = system_prompt.replace("{conversation_history}", history_text)
    system_prompt = system_prompt.replace("{exchange_number}", str(exchange_number))
    system_prompt = system_prompt.replace("{target_exchanges}", str(target_exchanges))

    # Format user prompt
    user_prompt = user_prompt_template.replace("{conversation_history}", history_text)
    user_prompt = user_prompt.replace("{exchange_number}", str(exchange_number))
    user_prompt = user_prompt.replace("{target_exchanges}", str(target_exchanges))

    result = await backend.complete(
        prompt=user_prompt,
        system=system_prompt,
        max_tokens=1000,
    )
    return result.content.strip()


async def run_interactive_session(
    config: GeneratorConfig,
    persona: Persona,
    target_exchanges: int = 15,
) -> dict:
    """Run interactive session with Claude user-sim and local model."""
    backend = ClaudeCLIBackend(model="sonnet")
    exchanges: list[dict] = []

    transcript_id = f"therapeutic_session_{persona.seed:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir = Path("data/raw/transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{transcript_id}.json"

    # Build result structure (exchanges will be updated incrementally)
    result = {
        "id": transcript_id,
        "model": "therapeutic-local",
        "persona": {
            "id": f"persona_{persona.seed:04d}",
            "name": persona.name,
            "age_range": persona.age_range,
            "writing_style": persona.writing_style,
            "personality_traits": persona.personality_traits,
            "communication_style": persona.communication_style,
            "trajectory": persona.trajectory,
            "topic_seeds": [
                {
                    "category": t.category,
                    "subtopic": t.subtopic,
                    "complexity": t.complexity,
                    "description": t.description,
                }
                for t in persona.topic_seeds
            ],
            "flaw_patterns": persona.flaw_patterns,
            "seed": persona.seed,
        },
        "exchanges": exchanges,
    }

    print(f"Running interactive session: {target_exchanges} exchanges")
    print(
        f"Persona: {persona.name}, {persona.age_range}, style={persona.writing_style}"
    )
    print(f"Output: {output_path}")
    print()

    for i in range(target_exchanges):
        exchange_num = i + 1

        # Claude generates user message
        user_msg = await generate_user_message(
            backend, config, persona, exchanges, exchange_num, target_exchanges
        )

        # Model generates assistant response
        history = [{"user": e["user"], "assistant": e["assistant"]} for e in exchanges]
        assistant_msg = get_model_response(SYSTEM_PROMPT, history, user_msg)

        exchange = {
            "exchange_number": exchange_num,
            "user": user_msg,
            "assistant": assistant_msg,
        }
        exchanges.append(exchange)

        # Write JSON after each exchange (overwrites previous)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        # Progress
        user_words = len(user_msg.split())
        assistant_words = len(assistant_msg.split())
        print(
            f"[{exchange_num:2d}/{target_exchanges}] User: {user_words}w -> Model: {assistant_words}w"
        )
        print(f"    User: {user_msg[:80]}...")
        print(f"    Model: {assistant_msg[:80]}...")
        print()

    return result


async def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # Load config
    config = load_config()
    persona = generate_persona(config, seed=seed)

    result = await run_interactive_session(config, persona, target_exchanges=target)

    # File already saved incrementally - just print final message
    output_path = Path("data/raw/transcripts") / f"{result['id']}.json"
    print(f"\nSaved to {output_path}")
    print(f"Run: uv run python -m assessor {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
