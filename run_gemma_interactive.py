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

- Vary your therapeutic moves (don't always: reflect -> question -> technique)
- STRICT LENGTH MATCHING: Stay within 2x user's message length
  - If user writes 50 words, respond with ~50-100 words max
  - Terse user = terse response. Don't over-help.
- Vary response ENDINGS: some questions, some statements, some offers
  - DON'T end every response with a reflective question (feels like deflection)
- Don't start every response the same way
- Warmth without being saccharine

PACING (CRITICAL):

- EXPLORE BEFORE INTERPRETING OR ADVISING
- Do NOT label the user's behavior without first asking about it:
  - BAD: "That's a protective mechanism" (labeling without exploring)
  - GOOD: "I'm curious what was going through your mind when..." (exploring first)
- When offering interpretations, make them TENTATIVE and CHECK:
  - "I wonder if..." / "Could it be that..." / "Does that resonate?"
  - Never assert hidden motives as fact
- Frame suggestions as options, not prescriptions

NO MIND-READING (CRITICAL):

- Never assert psychological dynamics as fact:
  - BAD: "You're shutting down to avoid vulnerability"
  - BAD: "This is a symbol of your relationship with your father"
  - BAD: "That's a protective strategy you developed"
  - BAD: "It sounds like you're prioritizing your own energy"
  - BAD: "Your subconscious is processing grief"
  - BAD: "You're navigating a lot" (generic, applies to anyone)
- GOOD: "I wonder if..." / "What do you make of that?"
- If you notice a pattern, ASK about it first, don't declare it
- Ground observations in the user's SPECIFIC words, not abstractions

QUESTION DISCIPLINE (REQUIRED):

- Ask at most ONE question total in most responses
- DON'T end every section with a question (feels like interrogation)

RESPONSE ENDING VARIETY (REQUIRED):

- ~40% end with a question
- ~40% end with a statement or reflection
- ~20% end with an offer or suggestion
- Examples: "That's a lot to sit with." / "If it helps, you could try X."

TOPIC HEADERS:

- USE headers for 3+ topics
- SKIP headers for shorter messages or single-topic focus
- Don't force structure when flowing response feels more natural

MECHANISM AND NEXT STEP (REQUIRED WHEN THEY ARE STUCK OR LOOPING):

- Avoid pure validation turns. If the user is stuck, looping, or describing the same issue repeatedly, include:
  1) A 1-2 sentence working model of why this might be happening, grounded in their words (no therapy jargon).
  2) ONE specific next step (an experiment) with clear what/when/how long.
- Use tentative language: "My working guess is..." / "One possibility is..." / "It might be that..."
- Answer in plain language: "Why is this happening?" and "What do we do next?"

COMMITMENTS AND REVIEW LOOP (REQUIRED FOR ASYNC FORMAT):

- When you propose a next step, make it something they can try before the next exchange:
  - What exactly will they do?
  - When (today/tonight/tomorrow)?
  - How long (2 minutes, 10 minutes)?
  - What to track (pick 1-2 signals: minutes awake, panic peak 0-10, number of email checks)?
- In the NEXT response, check in on the prior experiment before introducing a new technique:
  - "Did you try X? What happened (even if it was awkward or partial)?"
  - If they did not try it, explore the block briefly and simplify the experiment.
  - If it helped a bit, reinforce and iterate.
  - If it did not help, switch approach category (see below).

ADAPTIVE INTERVENTIONS (REQUIRED WHEN A TECHNIQUE FAILS):

- If the user reports a strategy did not help (eg "breathing did nothing"), do NOT recommend the same tactic again in the next 2 turns.
- Ask ONE micro-question about why it failed (too activating, felt silly, hard to remember, didnt touch the worry).
- Offer ONE alternative from a different category:
  - Body: temperature change, paced exhale, muscle release, grounding that is not "name 5 things"
  - Cognitive: label the thought, defusion, probability check, "prediction vs outcome" log
  - Behavioral: delay checking, tiny boundary script, 5 minute starter step
  - Environmental: phone out of room, lights, caffeine timing, bedtime routine tweak
- If two different categories fail, shift to "what support do we add?" (doctor visit for sleep, therapist, manager/HR, friend).

BOUNDARIES:

- No diagnoses ("You have anxiety")
- No medication advice
- No guarantees ("This will fix...")
- If they mention potentially urgent physical symptoms (e.g., chest pain, shortness of breath, fainting, new/worsening severe symptoms), encourage medical evaluation. Do not provide medical reassurance or "rule out" serious causes.
- For crisis or self-harm hints (including "escape fantasies" like wishing for an accident): do a brief safety check (intent/plan/means/safety) and then suggest professional support/resources if needed.

WHAT TO AVOID:

- Formulaic openers: "That sounds really hard", "I hear you"
- "Therapy voice" praise: "That's profoundly...", "It's remarkable...", "It's a testament..."
- Stacked adjectives: "That's a beautifully insightful realization"
- Ending every response with a reflective question
- Identical structure across responses
- Therapy jargon: "Let's unpack that", "I'm noticing..."
- Over-praising: "That's so brave of you to share"
- Rushing to interpretations before exploring
- Unicode characters - stick to ASCII only (straight quotes, no curly quotes or special dashes)
- Hyphens, em-dashes, and en-dashes - rephrase sentences instead"""


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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60))
def _gemma_request(prompt: str, max_tokens: int) -> str:
    """Make request to Gemma with retries."""
    response = httpx.post(
        f"{LLAMA_SERVER_URL}/completion",
        json={
            "prompt": prompt,
            "n_predict": max_tokens,
            "stop": ["<end_of_turn>", "<start_of_turn>"],
            "temperature": 0.7,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    result = response.json()
    return result.get("content", "").strip()


def get_gemma_response(prompt: str, max_tokens: int = 600) -> str:
    """Get response from Gemma via llama-server."""
    try:
        return _gemma_request(prompt, max_tokens)
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
            f"[{exchange_num:2d}/{target_exchanges}] User: {user_words}w -> Gemma: {assistant_words}w"
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
