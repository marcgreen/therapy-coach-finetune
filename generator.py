# generator.py
"""
Multi-turn therapeutic conversation generator.

Uses two-agent simulation: a user persona and a therapeutic coach.
Generates conversations according to input taxonomy distribution.
"""

import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam

from assessor import ConversationInput, ConversationTurn


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class TaxonomyConfig:
    """Parsed taxonomy configuration."""

    topic: str
    subtopic: str
    style: str
    difficulty: str
    target_turns: int


def load_taxonomy(path: Path = Path("config/input-taxonomy.yaml")) -> dict:
    """Load taxonomy configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def weighted_choice(items: list[dict], weight_key: str = "weight") -> dict:
    """Select item based on weights."""
    weights = [item.get(weight_key, 1.0) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def sample_config(taxonomy: dict) -> TaxonomyConfig:
    """Sample a conversation configuration from taxonomy."""
    # Sample topic and subtopic
    topic_entry = weighted_choice(taxonomy["taxonomy"]["topics"])
    topic = topic_entry["name"]
    subtopic = random.choice(topic_entry["subtopics"])

    # Sample style
    styles = taxonomy["taxonomy"]["styles"]
    style = random.choices(
        list(styles.keys()),
        weights=list(styles.values()),
        k=1,
    )[0]

    # Sample difficulty
    difficulties = taxonomy["taxonomy"]["difficulty"]
    difficulty = random.choices(
        list(difficulties.keys()),
        weights=list(difficulties.values()),
        k=1,
    )[0]

    # Sample conversation length
    lengths = taxonomy["taxonomy"]["conversation_length"]
    length_category = random.choices(
        list(lengths.keys()),
        weights=list(lengths.values()),
        k=1,
    )[0]

    # Get actual turn count
    turn_range = taxonomy["turn_ranges"][length_category]
    target_turns = random.randint(turn_range["min"], turn_range["max"])

    return TaxonomyConfig(
        topic=topic,
        subtopic=subtopic,
        style=style,
        difficulty=difficulty,
        target_turns=target_turns,
    )


# =============================================================================
# Persona Generation
# =============================================================================

PERSONA_PROMPT = """Generate a realistic therapy client persona for a conversation.

Topic: {topic} ({subtopic})
Communication style: {style}
Difficulty level: {difficulty}

Create a persona with:
1. A brief situation description (2-3 sentences)
2. Their emotional state
3. How they communicate (matches the style above)

Also write their opening message to start the conversation.

Output as JSON:
{{
    "persona": "Description of the person and their situation...",
    "opening_message": "Their first message to the coach..."
}}"""


@dataclass
class Persona:
    """Generated user persona."""

    description: str
    opening_message: str
    config: TaxonomyConfig


async def generate_persona(
    client: AsyncOpenAI,
    config: TaxonomyConfig,
) -> Persona:
    """Generate a user persona based on taxonomy config."""
    prompt = PERSONA_PROMPT.format(
        topic=config.topic,
        subtopic=config.subtopic,
        style=config.style,
        difficulty=config.difficulty,
    )

    user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}
    response = await client.responses.create(
        model="gpt-5-mini",
        input=[user_msg],
        reasoning={"effort": "low"},
        temperature=0.9,
    )

    result = json.loads(response.output_text)

    return Persona(
        description=result["persona"],
        opening_message=result["opening_message"],
        config=config,
    )


# =============================================================================
# Turn-by-Turn Generation
# =============================================================================

TURN_TEMPLATES: dict[str, list[str]] = {
    "early": [
        "Share more context about the situation",
        "Express a specific emotion more directly",
        "Ask the assistant a direct question",
        "Show slight resistance or hesitation",
    ],
    "middle": [
        "Go deeper into underlying feelings",
        "Make a connection to past experience",
        "Express ambivalence about change",
        "Have a small insight or realization",
        "Bring up a related concern",
    ],
    "late": [
        "Reflect on what's been discussed",
        "Express what feels different now",
        "Identify a small concrete next step",
        "Show appreciation naturally",
        "Express remaining uncertainty",
    ],
}


def get_turn_guidance(turn_number: int, total_turns: int) -> str:
    """Get guidance for what should happen in this turn."""
    progress = turn_number / total_turns

    if progress <= 0.3:
        phase = "early"
    elif progress <= 0.7:
        phase = "middle"
    else:
        phase = "late"

    return random.choice(TURN_TEMPLATES[phase])


USER_TURN_PROMPT = """You are simulating a therapy client in a conversation.

PERSONA:
{persona}

CONVERSATION SO FAR:
{history}

GUIDANCE FOR THIS TURN:
{guidance}

Write the client's next message. Stay in character. Be natural and realistic.
Output only the message, nothing else."""


THERAPIST_PROMPT = """You are a supportive therapeutic coach. You help people explore their thoughts and feelings through conversation.

Core approach:
- Validate before advising
- Ask questions to understand, don't assume
- Match the person's energy and pace
- Return agency - they decide what's right for them
- Stay warm and natural, not clinical

Boundaries:
- You're a coaching tool, not a licensed therapist
- Don't diagnose conditions or recommend medications
- For crisis situations, acknowledge seriously and suggest professional resources

Adapt your style to each person. Some want to explore feelings, others want practical strategies, some just need to be heard."""


async def generate_user_turn(
    client: AsyncOpenAI,
    persona: str,
    history: str,
    guidance: str,
) -> str:
    """Generate the next user message."""
    prompt = USER_TURN_PROMPT.format(
        persona=persona,
        history=history if history else "(Conversation just starting)",
        guidance=guidance,
    )

    user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}
    response = await client.responses.create(
        model="gpt-5-mini",
        input=[user_msg],
        reasoning={"effort": "low"},
        temperature=0.8,
    )

    return response.output_text.strip()


async def generate_therapist_turn(
    client: AsyncOpenAI,
    system_prompt: str,
    history: list[EasyInputMessageParam],
) -> str:
    """Generate the therapist's response."""
    system_msg: EasyInputMessageParam = {"role": "system", "content": system_prompt}
    messages: list[EasyInputMessageParam] = [system_msg, *history]

    response = await client.responses.create(
        model="gpt-5-mini",
        input=messages,  # type: ignore[arg-type] - list[EasyInputMessageParam] is valid
        reasoning={"effort": "low"},
        temperature=0.7,
    )

    return response.output_text.strip()


def format_history_for_user_sim(turns: list[ConversationTurn]) -> str:
    """Format conversation history for the user simulator."""
    if not turns:
        return ""

    lines = []
    for i, turn in enumerate(turns, 1):
        lines.append(f"Turn {i}:")
        lines.append(f"You: {turn.user}")
        lines.append(f"Coach: {turn.assistant}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Full Conversation Generation
# =============================================================================


async def generate_conversation(
    client: AsyncOpenAI,
    persona: Persona,
    system_prompt: str,
) -> ConversationInput:
    """Generate a complete multi-turn conversation."""
    turns: list[ConversationTurn] = []
    history_for_therapist: list[EasyInputMessageParam] = []

    # First turn uses opening message
    user_msg = persona.opening_message

    for turn_num in range(1, persona.config.target_turns + 1):
        # Generate therapist response
        user_turn: EasyInputMessageParam = {"role": "user", "content": user_msg}
        history_for_therapist.append(user_turn)
        assistant_msg = await generate_therapist_turn(
            client, system_prompt, history_for_therapist
        )
        assistant_turn: EasyInputMessageParam = {
            "role": "assistant",
            "content": assistant_msg,
        }
        history_for_therapist.append(assistant_turn)

        # Record turn
        turns.append(ConversationTurn(user=user_msg, assistant=assistant_msg))

        # Generate next user message (unless this is the last turn)
        if turn_num < persona.config.target_turns:
            guidance = get_turn_guidance(turn_num + 1, persona.config.target_turns)
            history_for_user = format_history_for_user_sim(turns)
            user_msg = await generate_user_turn(
                client, persona.description, history_for_user, guidance
            )

    return ConversationInput(turns=turns, system_prompt=system_prompt)


# =============================================================================
# Batch Generation
# =============================================================================


def extract_system_prompt(markdown_path: Path) -> str:
    """Extract system prompt from markdown file (content between triple backticks)."""
    content = markdown_path.read_text()
    # Find first code block
    match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
    return match.group(1) if match else content


async def generate_batch(
    count: int,
    taxonomy_path: Path = Path("config/input-taxonomy.yaml"),
    system_prompt_path: Path = Path("config/system-prompt.md"),
    output_path: Path = Path("output/generated_conversations.jsonl"),
    concurrency: int = 5,
) -> list[ConversationInput]:
    """Generate a batch of conversations."""
    taxonomy = load_taxonomy(taxonomy_path)
    system_prompt = extract_system_prompt(system_prompt_path)

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one(
        index: int,
    ) -> tuple[int, ConversationInput, TaxonomyConfig]:
        async with semaphore:
            config = sample_config(taxonomy)
            persona = await generate_persona(client, config)
            conversation = await generate_conversation(client, persona, system_prompt)
            print(
                f"Generated {index + 1}/{count}: "
                f"{config.topic}/{config.subtopic} ({len(conversation.turns)} turns)"
            )
            return index, conversation, config

    # Generate all conversations
    tasks = [generate_one(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    conversations = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            if isinstance(result, BaseException):
                print(f"Error: {result}")
                continue

            # After BaseException check, result is the tuple
            index, conversation, config = result
            conversations.append(conversation)

            # Write to JSONL
            record = {
                "id": f"conv_{index:05d}",
                "messages": conversation.to_messages(),
                "metadata": {
                    "topic": config.topic,
                    "subtopic": config.subtopic,
                    "style": config.style,
                    "difficulty": config.difficulty,
                    "turns": len(conversation.turns),
                },
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nGenerated {len(conversations)} conversations to {output_path}")
    return conversations


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    print(f"Generating {count} conversations...")
    asyncio.run(generate_batch(count))
