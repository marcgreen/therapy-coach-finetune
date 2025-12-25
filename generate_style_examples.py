# generate_style_examples.py
"""Generate one example conversation for each client style."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from generator import (
    TaxonomyConfig,
    generate_conversation,
    extract_system_prompt,
)

load_dotenv()

# Define one config per style with varying difficulty and length
STYLE_CONFIGS = [
    # Frequent short exchanges - explicit help-seeking
    TaxonomyConfig(
        topic="anxiety",
        subtopic="work_stress",
        style="terse",
        difficulty="easy",
        target_turns=10,
        interaction_cadence="frequent_short",
        help_seeking="explicit",
        cognitive_patterns="balanced",
    ),
    # Frequent short exchanges - implicit help-seeking
    TaxonomyConfig(
        topic="relationships",
        subtopic="romantic",
        style="conversational",
        difficulty="medium",
        target_turns=12,
        interaction_cadence="frequent_short",
        help_seeking="implicit",
        cognitive_patterns="balanced",
    ),
    # Infrequent detailed - journaling style with implicit help-seeking
    TaxonomyConfig(
        topic="self_worth",
        subtopic="imposter_syndrome",
        style="detailed",
        difficulty="hard",
        target_turns=3,  # Few turns but substantial
        interaction_cadence="infrequent_detailed",
        help_seeking="implicit",
        cognitive_patterns="balanced",
    ),
    # Distorted cognition example - tests CQ10 (yes-bot detection)
    TaxonomyConfig(
        topic="emotional_regulation",
        subtopic="overwhelm",
        style="emotional",
        difficulty="medium",
        target_turns=15,
        interaction_cadence="frequent_short",
        help_seeking="implicit",
        cognitive_patterns="distorted",
    ),
    TaxonomyConfig(
        topic="life_transitions",
        subtopic="career_change",
        style="analytical",
        difficulty="hard",
        target_turns=10,
        interaction_cadence="frequent_short",
        help_seeking="explicit",
        cognitive_patterns="balanced",
    ),
    # Financial stress - new topic
    TaxonomyConfig(
        topic="financial_stress",
        subtopic="job_insecurity",
        style="conversational",
        difficulty="medium",
        target_turns=4,
        interaction_cadence="infrequent_detailed",
        help_seeking="implicit",
        cognitive_patterns="balanced",
    ),
]


async def main() -> None:
    system_prompt = extract_system_prompt(Path("config/system-prompt.md"))
    client = AsyncOpenAI()
    output_path = Path("output/style_examples.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for i, config in enumerate(STYLE_CONFIGS):
            print(
                f"Generating {config.style} example ({config.topic}/{config.subtopic})..."
            )
            conversation = await generate_conversation(client, config, system_prompt)

            record = {
                "id": f"style_example_{i}_{config.style}",
                "messages": conversation.to_messages(),
                "metadata": {
                    "topic": config.topic,
                    "subtopic": config.subtopic,
                    "style": config.style,
                    "difficulty": config.difficulty,
                    "interaction_cadence": config.interaction_cadence,
                    "help_seeking": config.help_seeking,
                    "cognitive_patterns": config.cognitive_patterns,
                    "target_turns": config.target_turns,
                    "actual_turns": len(conversation.turns),
                },
            }
            f.write(json.dumps(record) + "\n")
            print(f"  → {len(conversation.turns)} turns generated")

    print(f"\nSaved {len(STYLE_CONFIGS)} style examples to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
