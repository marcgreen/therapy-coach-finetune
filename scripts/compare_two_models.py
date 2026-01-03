"""
2-Way Model Comparison: Finetune vs Base.

Compare a fine-tuned model against its base model using controlled testing:
- Same personas (seeds) for both models
- Same user simulator (Claude)
- Multiple runs per seed for statistical power
- Paired t-test for significance testing

WORKFLOW (one model at a time - recommended for single GPU):

    # Step 1: Run finetune model
    llama-server -m ~/models/qwen3-finetune.gguf --port 8080 -ngl 99
    uv run python scripts/compare_two_models.py --model finetune --port 8080 \
        --seeds 9000 9001 9002 9003 9004 --runs-per-seed 3

    # Step 2: Swap to base model
    llama-server -m ~/models/qwen3-base.gguf --port 8080 -ngl 99
    uv run python scripts/compare_two_models.py --model base --port 8080 \
        --seeds 9000 9001 9002 9003 9004 --runs-per-seed 3

    # Step 3: Generate comparison report
    uv run python scripts/compare_two_models.py --compare-only

ALTERNATIVE (both models simultaneously on different ports):

    uv run python scripts/compare_two_models.py --both \
        --finetune-port 8080 --base-port 8081 \
        --seeds 9000 9001 9002 9003 9004 --runs-per-seed 3
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import stats  # ty: ignore[unresolved-import]

from assessor import setup_logging as setup_assessor_logging
from llm_backend import ClaudeCLIBackend, LocalLLMBackend
from transcript_generator import (
    Exchange,
    GeneratorConfig,
    Persona,
    Transcript,
    generate_persona,
    generate_user_message,
    load_config,
)


@dataclass
class EvalResult:
    """Result for a single conversation evaluation."""

    model_name: str
    seed: int
    run_num: int
    transcript_id: str
    score: float
    passed: bool
    category_scores: dict[str, float]
    safety_gate_failed: bool
    transcript_path: Path | None = None


@dataclass
class ModelStats:
    """Statistics for one model."""

    name: str
    results: list[EvalResult]
    mean: float = 0.0
    std: float = 0.0
    pass_rate: float = 0.0
    category_means: dict[str, float] = field(default_factory=dict)

    def compute(self) -> None:
        """Compute statistics from results."""
        scores = [r.score for r in self.results]
        self.mean = float(np.mean(scores))
        self.std = float(np.std(scores))
        self.pass_rate = sum(1 for r in self.results if r.passed) / len(self.results)

        # Category means
        cat_sums: dict[str, list[float]] = {}
        for r in self.results:
            for cat, score in r.category_scores.items():
                if cat not in cat_sums:
                    cat_sums[cat] = []
                cat_sums[cat].append(score)
        self.category_means = {c: float(np.mean(v)) for c, v in cat_sums.items()}


async def generate_transcript(
    model_endpoint: str,
    user_backend: ClaudeCLIBackend,
    config: GeneratorConfig,
    persona: Persona,
    transcript_id: str,
    target_exchanges: int = 15,
    output_path: Path | None = None,
) -> Transcript:
    """Generate a transcript using the specified model endpoint.

    If output_path is provided, writes progress after each exchange.
    """
    model_backend = LocalLLMBackend(endpoint=model_endpoint)
    exchanges: list[Exchange] = []
    created_at = datetime.now().isoformat()

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

        # Build history for model
        history = []
        for e in exchanges:
            history.append({"role": "user", "content": e.user})
            history.append({"role": "assistant", "content": e.assistant})

        # Retry up to 3 times if model returns empty response
        assistant_content = ""
        for attempt in range(3):
            response = await model_backend.complete(
                prompt=user_message,
                system=config.system_prompt,
                history=history,
            )
            assistant_content = response.content.strip()
            if assistant_content:
                break
            print(f"      [Retry {attempt + 1}/3] Empty response, retrying...")

        # If still empty after retries, use placeholder
        if not assistant_content:
            assistant_content = "[Model returned empty response]"
            print("      [Warning] Model returned empty response after 3 retries")

        exchange = Exchange(
            user=user_message,
            assistant=assistant_content,
            exchange_number=i,
        )
        exchanges.append(exchange)

        # Write progress after each exchange
        if output_path:
            transcript = Transcript(
                id=transcript_id,
                persona=persona,
                exchanges=exchanges,
                target_exchanges=target_exchanges,
                created_at=created_at,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(transcript.to_dict(), f, indent=2)

    return Transcript(
        id=transcript_id,
        persona=persona,
        exchanges=exchanges,
        target_exchanges=target_exchanges,
        created_at=created_at,
    )


async def generate_model_transcripts(
    model_name: str,
    model_endpoint: str,
    seeds: list[int],
    runs_per_seed: int,
    config: GeneratorConfig,
    user_backend: ClaudeCLIBackend,
    output_dir: Path,
    target_exchanges: int = 15,
) -> int:
    """Generate transcripts for a model across all seeds and runs.

    Resumes from existing - skips transcripts that already exist.
    Returns number of transcripts generated.
    """
    total = len(seeds) * runs_per_seed
    current = 0
    skipped = 0
    generated = 0

    for seed in seeds:
        persona = generate_persona(config, seed=seed, target_topics=4)

        for run_num in range(1, runs_per_seed + 1):
            current += 1
            transcript_id = f"{model_name}_seed{seed}_run{run_num}"

            # Path for this transcript
            transcript_path = output_dir / "transcripts" / f"{transcript_id}.json"

            # Skip if transcript already exists with enough exchanges
            if transcript_path.exists():
                with open(transcript_path) as f:
                    data = json.load(f)
                if len(data.get("exchanges", [])) >= target_exchanges:
                    skipped += 1
                    print(
                        f"[{current}/{total}] {model_name}: seed={seed}, run={run_num} -> SKIP (exists)"
                    )
                    continue

            print(
                f"[{current}/{total}] {model_name}: seed={seed}, run={run_num} -> {transcript_path}"
            )

            # Generate transcript (saves incrementally after each exchange)
            await generate_transcript(
                model_endpoint=model_endpoint,
                user_backend=user_backend,
                config=config,
                persona=persona,
                transcript_id=transcript_id,
                target_exchanges=target_exchanges,
                output_path=transcript_path,
            )
            generated += 1
            print(f"      Done ({target_exchanges} exchanges)")

    print(f"\nGenerated: {generated}, Skipped: {skipped}")
    return generated


def load_results_from_dir(output_dir: Path, model_name: str) -> list[EvalResult]:
    """Load previously saved assessment results for a model."""
    results = []
    assessment_dir = output_dir / "assessments"

    if not assessment_dir.exists():
        return results

    for path in assessment_dir.glob(f"{model_name}_seed*_run*.json"):
        with open(path) as f:
            data = json.load(f)

        # Parse transcript_id to get seed and run_num
        # Format: {model_name}_seed{seed}_run{run_num}
        parts = path.stem.split("_")
        seed = int(parts[-2].replace("seed", ""))
        run_num = int(parts[-1].replace("run", ""))

        result = EvalResult(
            model_name=model_name,
            seed=seed,
            run_num=run_num,
            transcript_id=path.stem,
            score=data["score"],
            passed=data["passed"],
            category_scores=data["category_scores"],
            safety_gate_failed=data["safety_gate_failed"],
            transcript_path=output_dir / "transcripts" / f"{path.stem}.json",
        )
        results.append(result)

    return results


def compare_models(
    finetune_stats: ModelStats,
    base_stats: ModelStats,
) -> dict[str, float | bool]:
    """Run paired t-test comparing models."""
    # Align scores by (seed, run_num) for paired comparison
    finetune_by_key = {(r.seed, r.run_num): r.score for r in finetune_stats.results}
    base_by_key = {(r.seed, r.run_num): r.score for r in base_stats.results}

    paired_finetune = []
    paired_base = []
    for key in finetune_by_key:
        if key in base_by_key:
            paired_finetune.append(finetune_by_key[key])
            paired_base.append(base_by_key[key])

    if len(paired_finetune) < 2:
        return {
            "improvement": 0.0,
            "improvement_pct": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "n_pairs": len(paired_finetune),
        }

    t_stat, p_value = stats.ttest_rel(paired_finetune, paired_base)

    improvement = finetune_stats.mean - base_stats.mean
    improvement_pct = (
        (improvement / base_stats.mean * 100) if base_stats.mean > 0 else 0.0
    )

    return {
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_pairs": len(paired_finetune),
    }


def generate_report(
    finetune_stats: ModelStats,
    base_stats: ModelStats,
    comparison: dict[str, float | bool],
    output_path: Path,
) -> None:
    """Generate markdown comparison report."""
    lines = [
        "# Model Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        "| Model | Mean Score | Std Dev | Pass Rate | N |",
        "|-------|-----------|---------|-----------|---|",
        f"| {finetune_stats.name} | {finetune_stats.mean:.3f} | "
        f"{finetune_stats.std:.3f} | {finetune_stats.pass_rate:.1%} | "
        f"{len(finetune_stats.results)} |",
        f"| {base_stats.name} | {base_stats.mean:.3f} | "
        f"{base_stats.std:.3f} | {base_stats.pass_rate:.1%} | "
        f"{len(base_stats.results)} |",
        "",
        "## Statistical Comparison",
        "",
        f"- **Improvement**: {comparison['improvement']:+.3f} "
        f"({comparison['improvement_pct']:+.1f}%)",
        f"- **t-statistic**: {comparison['t_statistic']:.3f}",
        f"- **p-value**: {comparison['p_value']:.4f}",
        f"- **Significant (p < 0.05)**: "
        f"{'✅ YES' if comparison['significant'] else '❌ NO'}",
        f"- **Paired samples**: {comparison['n_pairs']}",
        "",
        "## Category Breakdown",
        "",
        f"| Category | {finetune_stats.name} | {base_stats.name} | Diff |",
        "|----------|"
        + "-" * len(finetune_stats.name)
        + "|"
        + "-" * len(base_stats.name)
        + "|------|",
    ]

    for cat in finetune_stats.category_means:
        ft_score = finetune_stats.category_means.get(cat, 0)
        base_score = base_stats.category_means.get(cat, 0)
        diff = ft_score - base_score
        lines.append(f"| {cat} | {ft_score:.3f} | {base_score:.3f} | {diff:+.3f} |")

    # Safety analysis
    ft_safety_fails = sum(1 for r in finetune_stats.results if r.safety_gate_failed)
    base_safety_fails = sum(1 for r in base_stats.results if r.safety_gate_failed)

    lines.extend(
        [
            "",
            "## Safety Analysis",
            "",
            "| Model | Safety Failures | Safety Rate |",
            "|-------|-----------------|-------------|",
            f"| {finetune_stats.name} | {ft_safety_fails} | "
            f"{1 - ft_safety_fails / len(finetune_stats.results):.1%} |",
            f"| {base_stats.name} | {base_safety_fails} | "
            f"{1 - base_safety_fails / len(base_stats.results):.1%} |",
            "",
            "## Per-Seed Breakdown",
            "",
            "| Seed | Finetune (avg) | Base (avg) | Diff |",
            "|------|---------------|------------|------|",
        ]
    )

    # Group by seed
    ft_by_seed: dict[int, list[float]] = {}
    base_by_seed: dict[int, list[float]] = {}

    for r in finetune_stats.results:
        if r.seed not in ft_by_seed:
            ft_by_seed[r.seed] = []
        ft_by_seed[r.seed].append(r.score)

    for r in base_stats.results:
        if r.seed not in base_by_seed:
            base_by_seed[r.seed] = []
        base_by_seed[r.seed].append(r.score)

    for seed in sorted(ft_by_seed.keys()):
        ft_avg = np.mean(ft_by_seed.get(seed, [0]))
        base_avg = np.mean(base_by_seed.get(seed, [0]))
        diff = ft_avg - base_avg
        lines.append(f"| {seed} | {ft_avg:.3f} | {base_avg:.3f} | {diff:+.3f} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Target**: ≥10% improvement with p < 0.05 for meaningful fine-tuning win",
            "- **Safety**: Fine-tuned model should NOT regress on safety criteria",
            "- Use category breakdown to identify specific strengths/weaknesses",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport saved to {output_path}")


async def run_comparison(
    finetune_port: int,
    base_port: int,
    seeds: list[int],
    runs_per_seed: int,
    target_exchanges: int,
    output_dir: Path,
    finetune_name: str = "finetune",
    base_name: str = "base",
) -> None:
    """Generate transcripts for both models."""
    config = load_config()
    user_backend = ClaudeCLIBackend(model="haiku")

    total_conversations = len(seeds) * runs_per_seed * 2
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON EVALUATION")
    print(f"{'=' * 60}")
    print(f"Seeds: {seeds}")
    print(f"Runs per seed: {runs_per_seed}")
    print(f"Exchanges per conversation: {target_exchanges}")
    print(f"Total conversations: {total_conversations}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    # Generate finetune transcripts
    print(f"\n=== Generating {finetune_name} (port {finetune_port}) ===\n")
    await generate_model_transcripts(
        model_name=finetune_name,
        model_endpoint=f"http://localhost:{finetune_port}",
        seeds=seeds,
        runs_per_seed=runs_per_seed,
        config=config,
        user_backend=user_backend,
        output_dir=output_dir,
        target_exchanges=target_exchanges,
    )

    # Generate base transcripts
    print(f"\n=== Generating {base_name} (port {base_port}) ===\n")
    await generate_model_transcripts(
        model_name=base_name,
        model_endpoint=f"http://localhost:{base_port}",
        seeds=seeds,
        runs_per_seed=runs_per_seed,
        config=config,
        user_backend=user_backend,
        output_dir=output_dir,
        target_exchanges=target_exchanges,
    )

    print(f"\n{'=' * 60}")
    print("TRANSCRIPT GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print("\nNext steps:")
    print(
        f"  1. Assess: uv run python scripts/assess_transcripts.py {output_dir}/transcripts"
    )
    print("  2. Compare: uv run python scripts/compare_two_models.py --compare-only")
    print(f"{'=' * 60}")


async def run_single_model(
    model_name: str,
    port: int,
    seeds: list[int],
    runs_per_seed: int,
    target_exchanges: int,
    output_dir: Path,
) -> None:
    """Generate transcripts for a single model (for sequential workflow)."""
    config = load_config()
    user_backend = ClaudeCLIBackend(model="haiku")

    total = len(seeds) * runs_per_seed
    print(f"\n{'=' * 60}")
    print(f"GENERATING: {model_name}")
    print(f"{'=' * 60}")
    print(f"Port: {port}")
    print(f"Seeds: {seeds}")
    print(f"Runs per seed: {runs_per_seed}")
    print(f"Total conversations: {total}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    await generate_model_transcripts(
        model_name=model_name,
        model_endpoint=f"http://localhost:{port}",
        seeds=seeds,
        runs_per_seed=runs_per_seed,
        config=config,
        user_backend=user_backend,
        output_dir=output_dir,
        target_exchanges=target_exchanges,
    )

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {model_name}")
    print(f"{'=' * 60}")
    print(f"Transcripts: {output_dir}/transcripts/{model_name}_*.json")
    print(
        f"\nNext: uv run python scripts/assess_transcripts.py {output_dir}/transcripts --pattern '{model_name}_*.json'"
    )
    print(f"{'=' * 60}")


def run_comparison_only(
    finetune_name: str,
    base_name: str,
    output_dir: Path,
) -> None:
    """Generate comparison report from saved results."""
    finetune_results = load_results_from_dir(output_dir, finetune_name)
    base_results = load_results_from_dir(output_dir, base_name)

    if not finetune_results:
        print(f"ERROR: No results found for '{finetune_name}' in {output_dir}")
        return
    if not base_results:
        print(f"ERROR: No results found for '{base_name}' in {output_dir}")
        return

    print(f"Loaded {len(finetune_results)} results for {finetune_name}")
    print(f"Loaded {len(base_results)} results for {base_name}")

    # Compute stats
    finetune_stats = ModelStats(name=finetune_name, results=finetune_results)
    finetune_stats.compute()

    base_stats = ModelStats(name=base_name, results=base_results)
    base_stats.compute()

    # Compare
    comparison = compare_models(finetune_stats, base_stats)

    # Generate report
    report_path = output_dir / "comparison_report.md"
    generate_report(finetune_stats, base_stats, comparison, report_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{finetune_name}:")
    print(f"  Mean: {finetune_stats.mean:.3f} (±{finetune_stats.std:.3f})")
    print(f"  Pass rate: {finetune_stats.pass_rate:.1%}")
    print(f"  Conversations: {len(finetune_results)}")
    print(f"\n{base_name}:")
    print(f"  Mean: {base_stats.mean:.3f} (±{base_stats.std:.3f})")
    print(f"  Pass rate: {base_stats.pass_rate:.1%}")
    print(f"  Conversations: {len(base_results)}")
    print(
        f"\nImprovement: {comparison['improvement']:+.3f} ({comparison['improvement_pct']:+.1f}%)"
    )
    print(f"p-value: {comparison['p_value']:.4f}")
    if comparison["significant"]:
        print("✅ STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("❌ Not statistically significant")
    print(f"\nFull report: {report_path}")
    print(f"{'=' * 60}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare fine-tuned model against base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW (one model at a time - recommended for single GPU):

    # Step 1: Assess finetune model
    llama-server -m finetune.gguf --port 8080 -ngl 99
    uv run python scripts/compare_two_models.py --model finetune --port 8080 \\
        --seeds 9000 9001 9002 9003 9004 --runs-per-seed 3

    # Step 2: Swap to base model
    llama-server -m base.gguf --port 8080 -ngl 99
    uv run python scripts/compare_two_models.py --model base --port 8080 \\
        --seeds 9000 9001 9002 9003 9004 --runs-per-seed 3

    # Step 3: Generate comparison report
    uv run python scripts/compare_two_models.py --compare-only

ALTERNATIVE (both models simultaneously):

    uv run python scripts/compare_two_models.py --both \\
        --finetune-port 8080 --base-port 8081 \\
        --seeds 9000 9001 9002 9003 9004 --runs-per-seed 3
""",
    )

    # Mode selection (mutually exclusive)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--model",
        type=str,
        choices=["finetune", "base"],
        help="Assess a single model (for sequential workflow)",
    )
    mode.add_argument(
        "--both",
        action="store_true",
        help="Assess both models (requires both running on different ports)",
    )
    mode.add_argument(
        "--compare-only",
        action="store_true",
        help="Generate comparison report from existing results",
    )

    # Common options
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for single model (with --model)",
    )
    parser.add_argument(
        "--finetune-port",
        type=int,
        default=8080,
        help="Port for fine-tuned model (with --both)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8081,
        help="Port for base model (with --both)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[9000, 9001, 9002, 9003, 9004],
        help="Seeds for persona generation (default: 9000-9004)",
    )
    parser.add_argument(
        "--runs-per-seed",
        type=int,
        default=3,
        help="Conversations per seed per model (default: 3)",
    )
    parser.add_argument(
        "--exchanges",
        type=int,
        default=15,
        help="Exchanges per conversation (default: 15)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/comparison"),
        help="Output directory (default: data/eval/comparison)",
    )
    parser.add_argument(
        "--finetune-name",
        type=str,
        default="finetune",
        help="Name for fine-tuned model (default: finetune)",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="base",
        help="Name for base model (default: base)",
    )

    args = parser.parse_args()

    # Setup logging
    import logging

    setup_assessor_logging(logging.INFO)

    if args.compare_only:
        # Just generate report from saved results
        run_comparison_only(
            finetune_name=args.finetune_name,
            base_name=args.base_name,
            output_dir=args.output_dir,
        )
    elif args.both:
        # Run both models
        asyncio.run(
            run_comparison(
                finetune_port=args.finetune_port,
                base_port=args.base_port,
                seeds=args.seeds,
                runs_per_seed=args.runs_per_seed,
                target_exchanges=args.exchanges,
                output_dir=args.output_dir,
                finetune_name=args.finetune_name,
                base_name=args.base_name,
            )
        )
    else:
        # Single model mode
        asyncio.run(
            run_single_model(
                model_name=args.model,
                port=args.port,
                seeds=args.seeds,
                runs_per_seed=args.runs_per_seed,
                target_exchanges=args.exchanges,
                output_dir=args.output_dir,
            )
        )


if __name__ == "__main__":
    main()
