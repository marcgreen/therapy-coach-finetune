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

# Define diverse configs showcasing different dimensions
STYLE_CONFIGS = [
    # Neurotypical young adult, explicit help-seeking, clean presentation
    TaxonomyConfig(
        topic="anxiety",
        subtopic="work_stress",
        difficulty="easy",
        style="terse",
        interaction_cadence="frequent_short",
        cognitive_patterns="balanced",
        age_context="young_adult",
        cultural_framing="individualist",
        communication_pattern="neurotypical",
        help_seeking="explicit",
        help_relationship="naive",
        presentation="clean_single",
        temporality="acute",
        target_turns=10,
    ),
    # Collectivist framing, implicit help-seeking, family pressure
    TaxonomyConfig(
        topic="relationships",
        subtopic="family",
        difficulty="medium",
        style="conversational",
        interaction_cadence="frequent_short",
        cognitive_patterns="balanced",
        age_context="middle_adult",
        cultural_framing="collectivist",
        communication_pattern="neurotypical",
        help_seeking="implicit",
        help_relationship="experienced",
        presentation="clean_single",
        temporality="chronic",
        target_turns=12,
    ),
    # Direct/literal communication (autistic pattern), detailed journaling
    TaxonomyConfig(
        topic="self_worth",
        subtopic="imposter_syndrome",
        difficulty="hard",
        style="detailed",
        interaction_cadence="infrequent_detailed",
        cognitive_patterns="balanced",
        age_context="young_adult",
        cultural_framing="individualist",
        communication_pattern="direct_literal",
        help_seeking="implicit",
        help_relationship="skeptical",
        presentation="clean_single",
        temporality="chronic",
        target_turns=3,
    ),
    # Distorted cognition + tangential (ADHD pattern) + comorbid presentation
    TaxonomyConfig(
        topic="emotional_regulation",
        subtopic="overwhelm",
        difficulty="hard",
        style="emotional",
        interaction_cadence="frequent_short",
        cognitive_patterns="distorted",
        age_context="young_adult",
        cultural_framing="mixed",
        communication_pattern="tangential_energetic",
        help_seeking="implicit",
        help_relationship="experienced",
        presentation="comorbid",
        temporality="building",
        target_turns=15,
    ),
    # Older adult, triggered by life event, limited vocabulary
    TaxonomyConfig(
        topic="life_transitions",
        subtopic="loss_grief",
        difficulty="medium",
        style="conversational",
        interaction_cadence="frequent_short",
        cognitive_patterns="balanced",
        age_context="older_adult",
        cultural_framing="individualist",
        communication_pattern="limited_vocabulary",
        help_seeking="implicit",
        help_relationship="naive",
        presentation="clean_single",
        temporality="triggered",
        target_turns=10,
    ),
    # Testing behavior, somatic presentation
    TaxonomyConfig(
        topic="anxiety",
        subtopic="health_anxiety",
        difficulty="hard",
        style="analytical",
        interaction_cadence="frequent_short",
        cognitive_patterns="balanced",
        age_context="middle_adult",
        cultural_framing="individualist",
        communication_pattern="neurotypical",
        help_seeking="explicit",
        help_relationship="testing",
        presentation="somatic",
        temporality="chronic",
        target_turns=12,
    ),
    # Teenager, decoy presentation (starts with school, real issue emerges)
    TaxonomyConfig(
        topic="relationships",
        subtopic="family",
        difficulty="hard",
        style="terse",
        interaction_cadence="frequent_short",
        cognitive_patterns="balanced",
        age_context="teenager",
        cultural_framing="mixed",
        communication_pattern="neurotypical",
        help_seeking="implicit",
        help_relationship="skeptical",
        presentation="decoy",
        temporality="chronic",
        target_turns=14,
    ),
    # Dependent help relationship, spiraling presentation
    TaxonomyConfig(
        topic="financial_stress",
        subtopic="job_insecurity",
        difficulty="medium",
        style="emotional",
        interaction_cadence="infrequent_detailed",
        cognitive_patterns="distorted",
        age_context="middle_adult",
        cultural_framing="collectivist",
        communication_pattern="neurotypical",
        help_seeking="explicit",
        help_relationship="dependent",
        presentation="spiraling",
        temporality="acute",
        target_turns=4,
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
                    # What
                    "topic": config.topic,
                    "subtopic": config.subtopic,
                    "difficulty": config.difficulty,
                    # How
                    "style": config.style,
                    "interaction_cadence": config.interaction_cadence,
                    "cognitive_patterns": config.cognitive_patterns,
                    # Who
                    "age_context": config.age_context,
                    "cultural_framing": config.cultural_framing,
                    "communication_pattern": config.communication_pattern,
                    # Relationship
                    "help_seeking": config.help_seeking,
                    "help_relationship": config.help_relationship,
                    # Presentation
                    "presentation": config.presentation,
                    "temporality": config.temporality,
                    # Stats
                    "target_turns": config.target_turns,
                    "actual_turns": len(conversation.turns),
                },
            }
            f.write(json.dumps(record) + "\n")
            print(f"  → {len(conversation.turns)} turns generated")

    print(f"\nSaved {len(STYLE_CONFIGS)} style examples to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
