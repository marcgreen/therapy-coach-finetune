# generate_eval_scenarios.py
"""Generate 50 diverse evaluation scenarios (opening user messages) for base model eval."""

import asyncio
import json
import random
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, Field

load_dotenv()

SCENARIO_PROMPT = """Generate a realistic opening message from someone seeking therapeutic coaching support.

SCENARIO:
- Topic: {topic} ({subtopic})
- Communication style: {style}
- Difficulty: {difficulty}

STYLE GUIDE:
- terse: Very brief, few words (e.g., "feeling anxious about work")
- conversational: Natural, like texting a friend (2-4 sentences)
- detailed: Full context, multiple paragraphs
- emotional: Intense feelings, may be distressed
- analytical: Notices patterns, systematic thinking

DIFFICULTY GUIDE:
- easy: Clear emotion, common situation, straightforward
- medium: Mixed feelings, some complexity
- hard: Ambiguous, layered issues, edge cases

Generate ONLY the user's opening message. No assistant response. Make it feel real and specific."""


class Scenario(BaseModel):
    """A single evaluation scenario."""

    message: str = Field(description="The user's opening message")


async def generate_scenario(
    client: AsyncOpenAI, topic: str, subtopic: str, style: str, difficulty: str
) -> str:
    """Generate a single opening message scenario."""
    prompt = SCENARIO_PROMPT.format(
        topic=topic, subtopic=subtopic, style=style, difficulty=difficulty
    )
    user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}

    for attempt in range(3):
        try:
            response = await client.responses.parse(
                model="gpt-5-nano",
                input=[user_msg],
                text_format=Scenario,
                reasoning={"effort": "minimal"},
                max_output_tokens=1500,
            )
            if response.output_parsed is not None:
                return response.output_parsed.message
        except Exception:
            pass

    # Fallback
    return f"I'm struggling with {subtopic.replace('_', ' ')}."


def load_taxonomy() -> dict:
    """Load taxonomy from YAML."""
    with open("config/input-taxonomy.yaml") as f:
        return yaml.safe_load(f)["taxonomy"]


def sample_from_weights(items: dict[str, float]) -> str:
    """Sample from weighted distribution."""
    names = list(items.keys())
    weights = list(items.values())
    return random.choices(names, weights=weights, k=1)[0]


async def main() -> None:
    taxonomy = load_taxonomy()
    client = AsyncOpenAI()
    output_path = Path("output/eval_scenarios.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build topic weights
    topic_weights = {t["name"]: t["weight"] for t in taxonomy["topics"]}
    topic_subtopics = {t["name"]: t["subtopics"] for t in taxonomy["topics"]}

    # Style and difficulty weights
    style_weights = taxonomy["styles"]
    difficulty_weights = taxonomy["difficulty"]

    scenarios = []
    n_scenarios = 50

    print(f"Generating {n_scenarios} evaluation scenarios...")

    for i in range(n_scenarios):
        # Sample from taxonomy
        topic = sample_from_weights(topic_weights)
        subtopic = random.choice(topic_subtopics[topic])
        style = sample_from_weights(style_weights)
        difficulty = sample_from_weights(difficulty_weights)

        message = await generate_scenario(client, topic, subtopic, style, difficulty)

        scenario = {
            "id": f"eval_{i:03d}",
            "message": message,
            "metadata": {
                "topic": topic,
                "subtopic": subtopic,
                "style": style,
                "difficulty": difficulty,
            },
        }
        scenarios.append(scenario)

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_scenarios}")

    # Write to file
    with open(output_path, "w") as f:
        for scenario in scenarios:
            f.write(json.dumps(scenario) + "\n")

    print(f"\nSaved {len(scenarios)} scenarios to {output_path}")

    # Show distribution
    print("\nDistribution:")
    topic_counts = {}
    style_counts = {}
    for s in scenarios:
        topic_counts[s["metadata"]["topic"]] = (
            topic_counts.get(s["metadata"]["topic"], 0) + 1
        )
        style_counts[s["metadata"]["style"]] = (
            style_counts.get(s["metadata"]["style"], 0) + 1
        )

    print("  Topics:", dict(sorted(topic_counts.items())))
    print("  Styles:", dict(sorted(style_counts.items())))


if __name__ == "__main__":
    asyncio.run(main())
