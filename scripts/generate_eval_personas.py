"""
Generate NEW evaluation personas that were NOT used in training.

For fair evaluation, we need personas distinct from training data.
Training used seeds 0-4999 (for ~155 transcripts).
Evaluation uses seeds starting from 9000.

Usage:
    # Generate 15 evaluation personas (recommended for statistical power)
    uv run python scripts/generate_eval_personas.py --count 15

    # Generate with specific seed range
    uv run python scripts/generate_eval_personas.py --count 15 --start-seed 9000
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from transcript_generator import (
    GeneratorConfig,
    Persona,
    generate_persona,
    load_config,
)

# Evaluation personas use seeds starting from 9000 to avoid overlap with training
# Training used seeds 0-4999 (for ~155 transcripts with various lengths)
DEFAULT_EVAL_SEED_START = 9000


def generate_eval_personas(
    count: int,
    start_seed: int = DEFAULT_EVAL_SEED_START,
    config: GeneratorConfig | None = None,
) -> list[Persona]:
    """Generate evaluation personas with seeds distinct from training."""
    if config is None:
        config = load_config()

    personas = []
    for i in range(count):
        seed = start_seed + i
        persona = generate_persona(config, seed=seed, target_topics=4)
        personas.append(persona)

    return personas


def save_personas(personas: list[Persona], output_path: Path) -> None:
    """Save personas to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "created_at": datetime.now().isoformat(),
        "purpose": "evaluation",
        "count": len(personas),
        "seed_range": f"{personas[0].seed}-{personas[-1].seed}",
        "personas": [p.to_dict() for p in personas],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(personas)} personas to {output_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate NEW evaluation personas (not used in training)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script generates personas for model evaluation that are DISTINCT from
training data. Training used seeds 0-4999, so evaluation uses 9000+.

Recommended: 15 personas × 3 conversations each = 45 transcripts per model
for statistical significance (paired t-test, p < 0.05).
""",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of personas to generate (default: 15)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_EVAL_SEED_START,
        help=f"Starting seed (default: {DEFAULT_EVAL_SEED_START})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/personas.json"),
        help="Output file (default: data/eval/personas.json)",
    )

    args = parser.parse_args()

    print(f"Generating {args.count} evaluation personas...")
    print(f"Using seeds {args.start_seed} to {args.start_seed + args.count - 1}")

    personas = generate_eval_personas(
        count=args.count,
        start_seed=args.start_seed,
    )

    # Print summary
    flaw_counts = sum(1 for p in personas if p.flaw_patterns is not None)
    print("\nPersona Summary:")
    print(f"  Total: {len(personas)}")
    print(f"  With flaws: {flaw_counts}")
    print(f"  Clear communicators: {len(personas) - flaw_counts}")

    # Show trajectory distribution
    trajectories = {}
    for p in personas:
        trajectories[p.trajectory] = trajectories.get(p.trajectory, 0) + 1
    print(f"  Trajectories: {trajectories}")

    save_personas(personas, args.output)


if __name__ == "__main__":
    main()
