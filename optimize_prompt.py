"""Use DSPy to optimize the system prompt for base model therapeutic coaching.

Extracts conversation scenarios from existing passing transcripts,
uses the base model (via llama-server) to generate responses,
and uses the existing assessor rubric as the optimization metric.

Uses Claude CLI (Haiku 4.5) as the judge — free with your plan.

Prerequisites:
    # Start the base model server
    llama-server -m ~/models/gemma-3-12b-it-q4_0.gguf --port 8080 -ngl 99

Usage:
    # Quick test (5 examples, light optimization)
    uv run python optimize_prompt.py --num-examples 5 --auto light

    # Medium run (20 examples)
    uv run python optimize_prompt.py --num-examples 20 --auto medium

    # Use MIPROv2 instead of GEPA (no checkpointing)
    uv run python optimize_prompt.py --optimizer mipro --auto light

    # Resume interrupted GEPA run (auto if same log-dir)
    uv run python optimize_prompt.py --num-examples 20 --auto medium
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy
from dspy.adapters.types import History
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback
from dspy.utils.syncify import run_async

from assessor import (
    AssessmentResult,
    ConversationInput,
    ConversationTurn,
    assess_conversation,
    get_backend,
    setup_logging,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Data Loading
# =============================================================================

PASSING_TRANSCRIPTS_PATH = Path("data/processed/passing_transcripts.json")


def load_examples(
    num_examples: int,
    min_exchanges: int = 5,
) -> list[dspy.Example]:
    """Load conversation scenarios from passing transcripts.

    For each transcript, extracts a mid-conversation scenario:
    - history: prior exchanges as dspy.History (proper chat turns)
    - user_message: the target user message
    - response: the gold assistant response (for few-shot demos)
    """
    with open(PASSING_TRANSCRIPTS_PATH) as f:
        manifest = json.load(f)

    examples: list[dspy.Example] = []

    for entry in manifest["transcripts"]:
        if len(examples) >= num_examples:
            break

        source = Path(entry["source_file"])
        if not source.exists():
            continue

        with open(source) as f:
            transcript = json.load(f)

        exchanges = transcript.get("exchanges", [])
        if len(exchanges) < min_exchanges:
            continue

        # Pick exchange 5 (0-indexed: 4) — enough context for meaningful assessment
        target_idx = min(4, len(exchanges) - 1)

        # Build history as dspy.History for proper chat formatting
        history_messages = [
            {
                "user_message": ex["user"],
                "response": ex["assistant"],
            }
            for ex in exchanges[:target_idx]
        ]

        example = dspy.Example(
            history=History(messages=history_messages),
            user_message=exchanges[target_idx]["user"],
            response=exchanges[target_idx]["assistant"],
            # Metadata for metric (not DSPy input fields)
            transcript_id=entry["id"],
            prior_turns=[
                {"user": ex["user"], "assistant": ex["assistant"]}
                for ex in exchanges[:target_idx]
            ],
        ).with_inputs("history", "user_message")

        examples.append(example)

    if not examples:
        print(
            "ERROR: No valid transcripts found. Check data/processed/passing_transcripts.json"
        )
        sys.exit(1)

    print(f"Loaded {len(examples)} examples from passing transcripts")
    return examples


# =============================================================================
# DSPy Signature
# =============================================================================


class TherapistRespond(dspy.Signature):
    """You are a supportive therapeutic coach helping someone explore their thoughts and feelings.

    Respond to the user's latest message given the conversation history.
    Match their energy and length. Address all topics they raise.
    Stay warm and natural, not clinical or formulaic."""

    history: History = dspy.InputField(desc="Previous exchanges in the conversation")
    user_message: str = dspy.InputField(desc="The user's current message to respond to")
    response: str = dspy.OutputField(
        desc="Your therapeutic coaching response, warm and natural"
    )


# =============================================================================
# Metric
# =============================================================================


def make_metric(judge_backend: str, judge_model: str | None) -> Callable[..., Any]:
    """Create assessment metric. Returns ScoreWithFeedback (works for both GEPA and MIPROv2)."""
    get_backend(backend_type=judge_backend, model=judge_model)

    def metric(
        example: dspy.Example,
        pred: dspy.Prediction,
        trace: object = None,
        **kwargs: object,
    ) -> ScoreWithFeedback:
        """Assess a predicted response using the full rubric.

        Returns ScoreWithFeedback — GEPA uses the feedback for evolution,
        MIPROv2 just reads the .score float.
        """
        response_text = pred.response
        if not response_text or not response_text.strip():
            return ScoreWithFeedback(score=0.0, feedback="Empty response generated.")

        # Build full conversation: prior turns + new exchange
        turns = [
            ConversationTurn(user=t["user"], assistant=t["assistant"])
            for t in example.prior_turns
        ]
        turns.append(
            ConversationTurn(user=example.user_message, assistant=response_text)
        )

        conversation = ConversationInput(turns=turns)

        try:
            result: AssessmentResult = run_async(
                assess_conversation(
                    conversation,
                    require_min_turns=False,
                    conversation_id=example.transcript_id,
                )
            )

            # Build feedback from failed criteria for GEPA
            feedback_parts = [
                f"{cid}: {result.reasonings.get(cid, '')}"
                for cid in result.failed_checks
                if result.reasonings.get(cid)
            ]
            if result.safety_gate_failed:
                feedback_parts.extend(
                    f"SAFETY {cid}: {result.reasonings.get(cid, '')}"
                    for cid in result.failed_safety
                )
            feedback = (
                "\n".join(feedback_parts) if feedback_parts else "All criteria passed."
            )

            logger.info(
                f"  [{example.transcript_id}] score={result.score:.3f} "
                f"passed={result.passed} failed={result.failed_checks}"
            )
            return ScoreWithFeedback(score=result.score, feedback=feedback)

        except Exception as e:
            logger.warning(f"  Assessment failed for {example.transcript_id}: {e}")
            return ScoreWithFeedback(score=0.0, feedback=f"Assessment error: {e}")

    return metric


# =============================================================================
# Baseline Evaluation
# =============================================================================


def evaluate_baseline(
    module: dspy.Module,
    examples: list[dspy.Example],
    metric_fn: Callable[..., Any],
) -> float:
    """Run module on examples and return mean score."""
    scores: list[float] = []
    for ex in examples:
        pred = module(history=ex.history, user_message=ex.user_message)
        result = metric_fn(ex, pred)
        scores.append(result.score if hasattr(result, "score") else float(result))

    return sum(scores) / len(scores) if scores else 0.0


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize therapeutic coaching prompt with DSPy"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of transcript examples to use (default: 10)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["gepa", "mipro"],
        default="gepa",
        help="Optimizer: gepa (default, has checkpointing) or mipro",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="Optimization intensity (default: light)",
    )
    parser.add_argument(
        "--base-model-url",
        default="http://localhost:8080/v1",
        help="Base model OpenAI-compatible endpoint (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["claude", "openai", "google"],
        default="claude",
        help="Backend for assessment judge (default: claude)",
    )
    parser.add_argument(
        "--judge-model",
        default="haiku",
        help="Model for assessment judge (default: haiku)",
    )
    parser.add_argument(
        "--log-dir",
        default="output/dspy_optimization",
        help="Log dir for checkpointing/resume (default: output/dspy_optimization)",
    )
    parser.add_argument(
        "--output",
        default="output/optimized_prompt.json",
        help="Output path for results (default: output/optimized_prompt.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # 1. Configure DSPy with the base model
    base_lm = dspy.LM(
        "openai/gemma-3-12b-it",
        api_base=args.base_model_url,
        api_key="none",
        temperature=0.7,
        max_tokens=1024,
    )
    dspy.configure(lm=base_lm)

    print(f"Base model: {args.base_model_url}")
    print(f"Judge: {args.judge_backend} ({args.judge_model})")
    print(f"Optimizer: {args.optimizer} (auto={args.auto})")
    print(f"Log dir: {args.log_dir}")
    print()

    # 2. Load examples — all used for both train and eval (n is small)
    examples = load_examples(args.num_examples)

    # 3. Create module and metric
    module = dspy.Predict(TherapistRespond)
    metric_fn = make_metric(args.judge_backend, args.judge_model)

    # 4. Baseline evaluation
    print("\n--- Baseline (unoptimized) ---")
    baseline_score = evaluate_baseline(module, examples, metric_fn)
    print(f"Baseline mean score: {baseline_score:.3f}")

    # 5. Optimize
    print(f"\n--- Optimizing with {args.optimizer} (auto={args.auto}) ---")

    if args.optimizer == "gepa":
        optimizer = dspy.GEPA(
            metric=metric_fn,
            auto=args.auto,
            num_threads=1,
            log_dir=args.log_dir,  # Enables checkpoint/resume
        )
        optimized = optimizer.compile(module, trainset=examples)
    else:
        optimizer = dspy.MIPROv2(
            metric=metric_fn,
            auto=args.auto,
            num_threads=1,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            verbose=args.verbose,
            log_dir=args.log_dir,
        )
        optimized = optimizer.compile(
            module,
            trainset=examples,
            requires_permission_to_run=False,
        )

    # 6. Evaluate optimized
    print("\n--- Optimized ---")
    optimized_score = evaluate_baseline(optimized, examples, metric_fn)
    print(f"Optimized mean score: {optimized_score:.3f}")
    print(f"Improvement: {optimized_score - baseline_score:+.3f}")

    # 7. Save results
    optimized_instruction = optimized.signature.instructions
    print(f"\n--- Optimized Prompt ---\n{optimized_instruction}\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "optimizer": args.optimizer,
        "auto": args.auto,
        "num_examples": args.num_examples,
        "baseline_score": round(baseline_score, 3),
        "optimized_score": round(optimized_score, 3),
        "improvement": round(optimized_score - baseline_score, 3),
        "optimized_instruction": optimized_instruction,
    }

    if hasattr(optimized, "demos") and optimized.demos:
        results["num_demos"] = len(optimized.demos)
        results["demos"] = [
            {
                "user_message": d.get("user_message", "")[:200],
                "response": d.get("response", "")[:200],
            }
            for d in optimized.demos
        ]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
