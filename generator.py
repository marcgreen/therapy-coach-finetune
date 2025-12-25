# generator.py
"""
Single-call therapeutic conversation generator.

Generates complete multi-turn conversations in one API call.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from assessor import ConversationInput, ConversationTurn

load_dotenv()


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
    topic_entry = weighted_choice(taxonomy["taxonomy"]["topics"])
    topic = topic_entry["name"]
    subtopic = random.choice(topic_entry["subtopics"])

    styles = taxonomy["taxonomy"]["styles"]
    style = random.choices(list(styles.keys()), weights=list(styles.values()), k=1)[0]

    difficulties = taxonomy["taxonomy"]["difficulty"]
    difficulty = random.choices(
        list(difficulties.keys()), weights=list(difficulties.values()), k=1
    )[0]

    lengths = taxonomy["taxonomy"]["conversation_length"]
    length_category = random.choices(
        list(lengths.keys()), weights=list(lengths.values()), k=1
    )[0]

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
# Structured Output Models
# =============================================================================


class Turn(BaseModel):
    """A single conversation turn."""

    user: str = Field(description="What the client says")
    assistant: str = Field(
        description="Coach's response (1-3 sentences, warm, concise)"
    )


class GeneratedConversation(BaseModel):
    """Complete generated conversation."""

    turns: list[Turn] = Field(description="The conversation turns")


# =============================================================================
# Generation
# =============================================================================

GENERATION_PROMPT = """Generate a realistic {turns}-turn therapeutic coaching conversation.

SCENARIO:
- Topic: {topic} ({subtopic})
- Client communication style: {style}
- Complexity: {difficulty}

CLIENT STYLES:
- terse: Short messages, few words
- conversational: Natural, flowing
- detailed: Provides full context
- emotional: Expresses intense feelings
- analytical: Notices patterns, systematic

COACH GUIDELINES:
- 1-3 sentences per response (be concise!)
- Validate before advising
- Ask questions to understand
- Return agency to the client
- Warm and natural, not clinical

CONVERSATION ARC:
- Early: Client shares problem, coach validates and explores
- Middle: Deeper understanding, gentle insights
- Late: Small realizations, concrete next steps

Generate a natural, helpful conversation. Coach responses should be SHORT."""


_api_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
)


@_api_retry
async def generate_conversation(
    client: AsyncOpenAI,
    config: TaxonomyConfig,
    system_prompt: str,
) -> ConversationInput:
    """Generate a complete conversation in one API call."""
    prompt = GENERATION_PROMPT.format(
        turns=config.target_turns,
        topic=config.topic,
        subtopic=config.subtopic,
        style=config.style,
        difficulty=config.difficulty,
    )

    user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}
    response = await client.responses.parse(
        model="gpt-5-nano",
        input=[user_msg],
        text_format=GeneratedConversation,
        reasoning={"effort": "low"},
        max_output_tokens=4000,  # Full conversation needs more tokens
    )

    result = response.output_parsed
    if result is None:
        raise ValueError("Failed to parse conversation")

    # Convert to ConversationInput
    turns = [ConversationTurn(user=t.user, assistant=t.assistant) for t in result.turns]
    return ConversationInput(turns=turns, system_prompt=system_prompt)


# =============================================================================
# Batch Generation
# =============================================================================


def extract_system_prompt(markdown_path: Path) -> str:
    """Extract system prompt from markdown file (first code block)."""
    import re

    content = markdown_path.read_text()
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

    async def generate_one(index: int) -> tuple[int, ConversationInput, TaxonomyConfig]:
        async with semaphore:
            config = sample_config(taxonomy)
            conversation = await generate_conversation(client, config, system_prompt)
            print(
                f"Generated {index + 1}/{count}: "
                f"{config.topic}/{config.subtopic} ({len(conversation.turns)} turns)"
            )
            return index, conversation, config

    tasks = [generate_one(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    conversations = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            if isinstance(result, BaseException):
                print(f"Error: {result}")
                continue

            index, conversation, config = result
            conversations.append(conversation)

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


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Generating {count} conversations...")
    asyncio.run(generate_batch(count))
