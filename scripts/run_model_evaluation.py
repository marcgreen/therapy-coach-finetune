"""
3-Way Model Comparison Evaluation.

Compares fine-tuned models against baseline using full-conversation generation:
1. Base Gemma 3 12B IT (baseline)
2. Fine-tuned Gemma 3 12B (therapeutic-gemma3-12b)
3. Fine-tuned Qwen3 14B (therapeutic-qwen3-14b)

Protocol per SPEC.md:
1. Load 15 NEW evaluation personas (not in training)
2. For each persona, generate 3 conversations per model
3. Assess all conversations with the 17-criteria rubric
4. Compare with paired t-test (p < 0.05 for significance)

Usage:
    # Full evaluation (assumes models running on different ports)
    uv run python scripts/run_model_evaluation.py \
        --personas data/eval/personas.json \
        --output-dir data/eval/results

    # Test with fewer personas
    uv run python scripts/run_model_evaluation.py \
        --personas data/eval/personas.json \
        --count 3 \
        --conversations-per-persona 1
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import stats  # ty: ignore[unresolved-import]

from assessor import (
    AssessmentResult,
    ConversationInput,
    assess_conversation,
    get_backend,
    setup_logging as setup_assessor_logging,
)
from llm_backend import LocalLLMBackend, LLMBackend
from transcript_generator import (
    Exchange,
    GeneratorConfig,
    Persona,
    Transcript,
    generate_user_message,
    load_config,
)


# =============================================================================
# Model Configuration
# =============================================================================

ModelType = Literal["baseline", "gemma_finetuned", "qwen_finetuned"]


@dataclass
class ModelConfig:
    """Configuration for a model to evaluate."""

    name: str
    model_type: ModelType
    endpoint: str  # e.g., "http://localhost:8080"
    description: str


DEFAULT_MODELS = [
    ModelConfig(
        name="baseline",
        model_type="baseline",
        endpoint="http://localhost:8080",
        description="Gemma 3 12B IT (base model)",
    ),
    ModelConfig(
        name="gemma_finetuned",
        model_type="gemma_finetuned",
        endpoint="http://localhost:8081",
        description="Gemma 3 12B (therapeutic fine-tuned)",
    ),
    ModelConfig(
        name="qwen_finetuned",
        model_type="qwen_finetuned",
        endpoint="http://localhost:8082",
        description="Qwen3 14B (therapeutic fine-tuned)",
    ),
]


# =============================================================================
# Persona Loading
# =============================================================================


def load_personas(path: Path, limit: int | None = None) -> list[Persona]:
    """Load evaluation personas from JSON file."""
    with open(path) as f:
        data = json.load(f)

    personas = []
    for p_data in data["personas"]:
        # Reconstruct TopicSeed objects
        from transcript_generator import TopicSeed

        topic_seeds = [
            TopicSeed(
                category=t["category"],
                subtopic=t["subtopic"],
                complexity=t["complexity"],
                description=t["description"],
            )
            for t in p_data["topic_seeds"]
        ]

        persona = Persona(
            id=p_data["id"],
            name=p_data["name"],
            age_range=p_data["age_range"],
            personality_traits=p_data["personality_traits"],
            communication_style=p_data["communication_style"],
            writing_style=p_data["writing_style"],
            topic_seeds=topic_seeds,
            flaw_patterns=p_data.get("flaw_patterns"),
            trajectory=p_data["trajectory"],
            seed=p_data["seed"],
        )
        personas.append(persona)

        if limit and len(personas) >= limit:
            break

    return personas


# =============================================================================
# Conversation Generation
# =============================================================================


async def generate_evaluation_transcript(
    model_backend: LLMBackend,
    user_backend: LLMBackend,
    config: GeneratorConfig,
    persona: Persona,
    target_exchanges: int = 25,
    conversation_id: str = "",
) -> Transcript:
    """Generate a transcript for evaluation using the specified model backend."""
    exchanges: list[Exchange] = []

    for i in range(1, target_exchanges + 1):
        # Generate user message using Claude (same for all models)
        user_message = await generate_user_message(
            backend=user_backend,
            config=config,
            persona=persona,
            exchanges=exchanges,
            exchange_number=i,
            target_exchanges=target_exchanges,
        )

        # Generate assistant response using the MODEL UNDER EVALUATION
        # Use the local model backend instead of the assistant template
        history = _format_history_for_local_model(exchanges, config.system_prompt)
        response = await model_backend.complete(
            prompt=user_message,
            system=config.system_prompt,
            history=history,
        )

        exchange = Exchange(
            user=user_message,
            assistant=response.content.strip(),
            exchange_number=i,
        )
        exchanges.append(exchange)

    return Transcript(
        id=conversation_id,
        persona=persona,
        exchanges=exchanges,
        target_exchanges=target_exchanges,
        created_at=datetime.now().isoformat(),
    )


def _format_history_for_local_model(
    exchanges: list[Exchange], system_prompt: str
) -> list[dict[str, str]]:
    """Format conversation history for local model API."""
    history = []
    for e in exchanges:
        history.append({"role": "user", "content": e.user})
        history.append({"role": "assistant", "content": e.assistant})
    return history


# =============================================================================
# Evaluation Pipeline
# =============================================================================


@dataclass
class EvaluationResult:
    """Result for a single conversation evaluation."""

    model_name: str
    persona_id: str
    conversation_num: int
    transcript_id: str
    assessment: AssessmentResult
    transcript: Transcript


@dataclass
class ModelStats:
    """Statistics for a single model's evaluation results."""

    model_name: str
    scores: list[float]
    mean: float
    std: float
    pass_rate: float
    category_means: dict[str, float]


@dataclass
class ComparisonResult:
    """Statistical comparison between models."""

    model_a: str
    model_b: str
    improvement: float  # Absolute improvement (b - a)
    improvement_pct: float  # Percentage improvement
    t_statistic: float
    p_value: float
    significant: bool  # p < 0.05


async def run_evaluation(
    personas: list[Persona],
    models: list[ModelConfig],
    config: GeneratorConfig,
    user_backend: LLMBackend,
    conversations_per_persona: int = 3,
    exchanges_per_conversation: int = 25,
    output_dir: Path | None = None,
) -> dict[str, list[EvaluationResult]]:
    """Run full evaluation pipeline."""
    results: dict[str, list[EvaluationResult]] = {m.name: [] for m in models}

    # Initialize assessment backend
    get_backend(backend_type="google", model="gemini-2.5-flash-preview-05-20")

    total = len(personas) * len(models) * conversations_per_persona
    current = 0

    for persona in personas:
        for model in models:
            # Create model backend
            model_backend = LocalLLMBackend(endpoint=model.endpoint)

            for conv_num in range(conversations_per_persona):
                current += 1
                transcript_id = f"{model.name}_{persona.id}_conv{conv_num}"
                print(f"[{current}/{total}] Generating: {transcript_id}")

                # Generate transcript
                transcript = await generate_evaluation_transcript(
                    model_backend=model_backend,
                    user_backend=user_backend,
                    config=config,
                    persona=persona,
                    target_exchanges=exchanges_per_conversation,
                    conversation_id=transcript_id,
                )

                # Save transcript if output_dir specified
                if output_dir:
                    transcript_path = (
                        output_dir / "transcripts" / f"{transcript_id}.json"
                    )
                    transcript_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(transcript_path, "w") as f:
                        json.dump(transcript.to_dict(), f, indent=2)

                # Assess the transcript
                conversation = ConversationInput.from_messages(transcript.to_messages())
                assessment = await assess_conversation(
                    conversation,
                    conversation_id=transcript_id,
                )

                result = EvaluationResult(
                    model_name=model.name,
                    persona_id=persona.id,
                    conversation_num=conv_num,
                    transcript_id=transcript_id,
                    assessment=assessment,
                    transcript=transcript,
                )
                results[model.name].append(result)

                # Save assessment if output_dir specified
                if output_dir:
                    assessment_path = (
                        output_dir / "assessments" / f"{transcript_id}.json"
                    )
                    assessment_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(assessment_path, "w") as f:
                        json.dump(assessment.to_dict(), f, indent=2)

    return results


def compute_model_stats(
    results: list[EvaluationResult],
    model_name: str,
) -> ModelStats:
    """Compute statistics for a model's evaluation results."""
    scores = [r.assessment.score for r in results]
    passed = sum(1 for r in results if r.assessment.passed)

    # Category means
    category_sums: dict[str, list[float]] = {}
    for r in results:
        for cat, score in r.assessment.category_scores.items():
            if cat not in category_sums:
                category_sums[cat] = []
            category_sums[cat].append(score)

    category_means = {cat: float(np.mean(vals)) for cat, vals in category_sums.items()}

    return ModelStats(
        model_name=model_name,
        scores=scores,
        mean=float(np.mean(scores)),
        std=float(np.std(scores)),
        pass_rate=passed / len(results) if results else 0.0,
        category_means=category_means,
    )


def compare_models(
    stats_a: ModelStats,
    stats_b: ModelStats,
) -> ComparisonResult:
    """Compare two models using paired t-test."""
    # Paired t-test (same personas, same user simulator)
    t_stat, p_value = stats.ttest_rel(stats_b.scores, stats_a.scores)

    improvement = stats_b.mean - stats_a.mean
    improvement_pct = (improvement / stats_a.mean * 100) if stats_a.mean > 0 else 0.0

    return ComparisonResult(
        model_a=stats_a.model_name,
        model_b=stats_b.model_name,
        improvement=improvement,
        improvement_pct=improvement_pct,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < 0.05,
    )


def generate_report(
    results: dict[str, list[EvaluationResult]],
    output_path: Path,
) -> None:
    """Generate evaluation report in markdown format."""
    # Compute stats for each model
    model_stats = {
        name: compute_model_stats(res, name) for name, res in results.items()
    }

    # Get baseline for comparisons
    baseline_name = "baseline"
    comparisons = []

    for name in model_stats:
        if name != baseline_name and baseline_name in model_stats:
            comparison = compare_models(model_stats[baseline_name], model_stats[name])
            comparisons.append(comparison)

    # Compare fine-tuned models against each other
    if "gemma_finetuned" in model_stats and "qwen_finetuned" in model_stats:
        comparisons.append(
            compare_models(
                model_stats["gemma_finetuned"], model_stats["qwen_finetuned"]
            )
        )

    # Generate report
    lines = [
        "# Model Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Model | Mean Score | Std Dev | Pass Rate |",
        "|-------|-----------|---------|-----------|",
    ]

    for name, model_stat in model_stats.items():
        lines.append(
            f"| {name} | {model_stat.mean:.3f} | {model_stat.std:.3f} | {model_stat.pass_rate:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Statistical Comparisons",
            "",
            "| Comparison | Improvement | % | p-value | Significant |",
            "|------------|-------------|---|---------|-------------|",
        ]
    )

    for c in comparisons:
        sig_marker = "**Yes**" if c.significant else "No"
        lines.append(
            f"| {c.model_a} → {c.model_b} | {c.improvement:+.3f} | "
            f"{c.improvement_pct:+.1f}% | {c.p_value:.4f} | {sig_marker} |"
        )

    lines.extend(
        [
            "",
            "## Category Breakdown",
            "",
        ]
    )

    # Category table
    categories = list(next(iter(model_stats.values())).category_means.keys())
    header = "| Category | " + " | ".join(model_stats.keys()) + " |"
    sep = "|---------|" + "|".join(["--------"] * len(model_stats)) + "|"
    lines.extend([header, sep])

    for cat in categories:
        row = f"| {cat} |"
        for name in model_stats:
            row += f" {model_stats[name].category_means[cat]:.3f} |"
        lines.append(row)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Significant improvement**: p < 0.05 means the improvement is unlikely due to chance",
            "- **Target**: ≥10% improvement with p < 0.05 (per SPEC.md)",
            "- **Safety**: Check that fine-tuned models don't regress on safety criteria",
            "",
        ]
    )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run 3-way model comparison evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models must be running on their configured endpoints:
  - Baseline (port 8080): llama-server -m gemma-3-12b-it.gguf --port 8080
  - Gemma fine-tuned (port 8081): llama-server -m therapeutic-gemma.gguf --port 8081
  - Qwen fine-tuned (port 8082): llama-server -m therapeutic-qwen.gguf --port 8082

Workflow:
  1. Generate personas: uv run python scripts/generate_eval_personas.py
  2. Start model servers
  3. Run this script
""",
    )
    parser.add_argument(
        "--personas",
        type=Path,
        default=Path("data/eval/personas.json"),
        help="Path to evaluation personas JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/results"),
        help="Output directory for transcripts, assessments, and report",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Limit number of personas to evaluate (for testing)",
    )
    parser.add_argument(
        "--conversations-per-persona",
        type=int,
        default=3,
        help="Conversations to generate per persona per model (default: 3)",
    )
    parser.add_argument(
        "--exchanges",
        type=int,
        default=25,
        help="Exchanges per conversation (default: 25)",
    )
    parser.add_argument(
        "--baseline-port",
        type=int,
        default=8080,
        help="Port for baseline model (default: 8080)",
    )
    parser.add_argument(
        "--gemma-port",
        type=int,
        default=8081,
        help="Port for fine-tuned Gemma (default: 8081)",
    )
    parser.add_argument(
        "--qwen-port",
        type=int,
        default=8082,
        help="Port for fine-tuned Qwen (default: 8082)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (only compare fine-tuned models)",
    )

    args = parser.parse_args()

    # Setup logging
    import logging

    setup_assessor_logging(logging.INFO)

    # Load personas
    print(f"Loading personas from {args.personas}...")
    personas = load_personas(args.personas, limit=args.count)
    print(f"Loaded {len(personas)} personas")

    # Configure models
    models = []
    if not args.skip_baseline:
        models.append(
            ModelConfig(
                name="baseline",
                model_type="baseline",
                endpoint=f"http://localhost:{args.baseline_port}",
                description="Gemma 3 12B IT (base model)",
            )
        )
    models.extend(
        [
            ModelConfig(
                name="gemma_finetuned",
                model_type="gemma_finetuned",
                endpoint=f"http://localhost:{args.gemma_port}",
                description="Gemma 3 12B (therapeutic fine-tuned)",
            ),
            ModelConfig(
                name="qwen_finetuned",
                model_type="qwen_finetuned",
                endpoint=f"http://localhost:{args.qwen_port}",
                description="Qwen3 14B (therapeutic fine-tuned)",
            ),
        ]
    )

    # Load config
    config = load_config()

    # Initialize user simulator backend (Claude)
    from llm_backend import ClaudeCLIBackend

    user_backend = ClaudeCLIBackend(model="haiku")

    # Run evaluation
    print("\nStarting evaluation...")
    print(f"  Personas: {len(personas)}")
    print(f"  Models: {[m.name for m in models]}")
    print(f"  Conversations per persona: {args.conversations_per_persona}")
    print(f"  Exchanges per conversation: {args.exchanges}")
    print(
        f"  Total conversations: {len(personas) * len(models) * args.conversations_per_persona}"
    )
    print()

    results = asyncio.run(
        run_evaluation(
            personas=personas,
            models=models,
            config=config,
            user_backend=user_backend,
            conversations_per_persona=args.conversations_per_persona,
            exchanges_per_conversation=args.exchanges,
            output_dir=args.output_dir,
        )
    )

    # Generate report
    report_path = args.output_dir / "evaluation_report.md"
    generate_report(results, report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for name, res in results.items():
        stats = compute_model_stats(res, name)
        print(f"\n{name}:")
        print(f"  Mean score: {stats.mean:.3f} (±{stats.std:.3f})")
        print(f"  Pass rate: {stats.pass_rate:.1%}")

    print(f"\nFull report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
