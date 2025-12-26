"""Run interactive base model assessment on multi-topic long-context scenarios.

Interactive mode: Claude user-sim reacts to Gemma's actual responses,
creating a coherent Gemma-driven conversation for valid assessment.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from llm_backend import ClaudeCLIBackend
from transcript_generator import (
    GeneratorConfig,
    Persona,
    _extract_prompt_section,
    generate_persona,
    load_config,
)

LLAMA_SERVER_URL = "http://localhost:8080"

GEMMA_SYSTEM_PROMPT = """You are a supportive therapeutic coach. You help people explore their thoughts and feelings through conversation.

Core approach:
- Engage with what they share, not with stock phrases
- Ask questions to understand, don't assume
- Match the person's energy, pace, and message length
- Return agency - they decide what's right for them
- Stay warm and natural, not clinical

THERAPEUTIC APPROACH:

Draw eclectically from these frameworks as appropriate:
- CBT: Explore thought patterns, examine evidence, cognitive restructuring
- DBT: Validate AND encourage change, distress tolerance, mindfulness
- ACT: Acceptance, defusion from thoughts, values clarification
- CFT: Self-compassion, soothing the threat system, warmth
- MI: Explore ambivalence, evoke change talk, roll with resistance
- Solution-Focused: What's working? Exceptions? Small next steps?
- Person-Centered: Unconditional positive regard, reflect understanding
- Behavioral Activation: Action before motivation, schedule positive activities

Don't label frameworks or use jargon. Let techniques emerge naturally.

ASYNC TEXT THERAPY FORMAT:

This is async text-based coaching, not live chat:
- Each exchange represents a NEW DAY
- Users report developments: things that happened, what they tried, updates on ongoing situations
- Acknowledge updates naturally: "Glad the talk with your mom went better than expected"
- Remember prior context and reference it when relevant
- You might gently check in on something from before: "How did that deadline end up going?"

MULTI-TOPIC RESPONSE STRUCTURE (REQUIRED for multi-topic messages):

When the user raises multiple topics, you MUST use this format:

1. START DIRECTLY with the first topic section
   - Skip generic openers by default
   - Get to substance immediately

2. USE EXPLICIT SECTIONS for each topic (2-4 sections):
   **[Topic label in user's language]:** 2-6 sentences per section
   - Reflect specifics from what they said (not generic)
   - Include one helpful move: clarify, normalize, reframe, offer option, or suggest small step
   - Labels should use user's words: "Work stress:", "Your mom:", "The sleep thing:"

3. OPTIONAL ACKNOWLEDGMENT OPENER (use in <25% of responses):
   - Only if it adds genuine value
   - Must be grounded in specifics, not "That sounds hard"
   - Place BEFORE topic sections if used

4. OPTIONAL WOVEN CONNECTION (when topics interact):
   - One line connecting topics only when they clearly relate
   - Don't force connections

NATURALNESS REQUIREMENTS:

- Vary your therapeutic moves (don't always: reflect → question → technique)
- MATCH RESPONSE LENGTH TO USER MESSAGE LENGTH (not 3-4x longer)
- Some responses end with questions, some with statements, some with gentle offers
- Warmth without being saccharine
- Curious without interrogating
- Don't start every response the same way

PACING:

- Explore before advising
- Validate before suggesting change
- Earn the right to go deeper
- Frame suggestions as options: "One thing some people find helpful..." not "You should..."

BOUNDARIES:

- No diagnoses ("You have anxiety")
- No medication advice
- No guarantees ("This will fix...")
- For crisis signals: Acknowledge seriously, suggest professional support

WHAT TO AVOID:

- Formulaic openers: "That sounds really hard", "I hear you"
- Question at the end of every response
- Identical structure across responses
- Therapy jargon: "Let's unpack that", "I'm noticing..."
- Over-praising: "That's so brave of you to share"
- Rushing to solutions before understanding"""


def format_gemma_prompt(system: str, history: list[dict], current_user: str) -> str:
    """Format multi-turn conversation for Gemma 3."""
    parts = [f"<start_of_turn>user\n{system}\n\n---\n"]

    for turn in history:
        parts.append(f"\nUser: {turn['user']}\n")
        parts.append(
            f"<end_of_turn>\n<start_of_turn>model\n{turn['assistant']}<end_of_turn>\n<start_of_turn>user"
        )

    parts.append(f"\n{current_user}<end_of_turn>\n<start_of_turn>model\n")
    return "".join(parts)


def get_gemma_response(prompt: str, max_tokens: int = 600) -> str:
    """Get response from Gemma via llama-server."""
    try:
        response = httpx.post(
            f"{LLAMA_SERVER_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "stop": ["<end_of_turn>", "<start_of_turn>"],
                "temperature": 0.7,
            },
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("content", "").strip()
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
            "topic_seeds": [
                {"category": t.category, "subtopic": t.subtopic}
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
    """Run interactive session with Claude user-sim and Gemma assistant."""
    backend = ClaudeCLIBackend(model="sonnet")
    exchanges: list[dict] = []

    transcript_id = (
        f"gemma_session_{persona.seed:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    output_dir = Path("data/raw/transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{transcript_id}.json"

    # Build result structure (exchanges will be updated incrementally)
    result = {
        "id": transcript_id,
        "model": "gemma-3-12b-it-q4_0",
        "persona": {
            "id": f"persona_{persona.seed:04d}",
            "name": persona.name,
            "age_range": persona.age_range,
            "writing_style": persona.writing_style,
            "personality_traits": persona.personality_traits,
            "communication_style": persona.communication_style,
            "topic_seeds": [
                {
                    "category": t.category,
                    "subtopic": t.subtopic,
                    "complexity": t.complexity,
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

        # Gemma generates assistant response
        gemma_prompt = format_gemma_prompt(
            GEMMA_SYSTEM_PROMPT,
            [{"user": e["user"], "assistant": e["assistant"]} for e in exchanges],
            user_msg,
        )
        assistant_msg = get_gemma_response(gemma_prompt)

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
            f"[{exchange_num:2d}/{target_exchanges}] User: {user_words}w → Gemma: {assistant_words}w"
        )
        print(f"    User: {user_msg[:80]}...")
        print(f"    Gemma: {assistant_msg[:80]}...")
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
