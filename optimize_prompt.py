"""Use DSPy to optimize the system prompt for base model therapeutic coaching.

Extracts conversation scenarios from existing passing transcripts,
uses the base model (via llama-server) to generate responses,
and uses the existing assessor rubric as the optimization metric.

Supports both MIPROv2 (instruction + few-shot optimization) and
GEPA (feedback-driven genetic prompt evolution).

Prerequisites:
    # Start the base model server
    llama-server -m ~/models/gemma-3-12b-it-q4_0.gguf --port 8080 -ngl 99

Usage:
    # Quick test (5 examples, light optimization)
    uv run python optimize_prompt.py --num-examples 5 --auto light

    # Medium run (20 examples)
    uv run python optimize_prompt.py --num-examples 20 --auto medium

    # Use GEPA instead of MIPROv2
    uv run python optimize_prompt.py --optimizer gepa --auto light
"""

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy

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
    - conversation_history: formatted exchanges 1..N-1
    - user_message: the Nth user message
    - gold_response: the Nth assistant response (for few-shot demos)
    - full_turns: all N turns as list of dicts (for assessment context)

    Picks exchange N = min(5, len(exchanges)) to ensure enough
    conversation history for meaningful assessment.
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

        # Pick a target exchange with enough prior context
        target_idx = min(4, len(exchanges) - 1)  # 0-indexed, so exchange 5

        # Build conversation history (prior exchanges)
        history_lines: list[str] = []
        prior_turns: list[dict[str, str]] = []
        for ex in exchanges[:target_idx]:
            history_lines.append(f"User: {ex['user']}")
            history_lines.append(f"Assistant: {ex['assistant']}")
            prior_turns.append({"user": ex["user"], "assistant": ex["assistant"]})

        conversation_history = (
            "\n\n".join(history_lines) if history_lines else "(start of conversation)"
        )
        user_message = exchanges[target_idx]["user"]
        gold_response = exchanges[target_idx]["assistant"]

        example = dspy.Example(
            conversation_history=conversation_history,
            user_message=user_message,
            response=gold_response,
            # Stash metadata for the metric (not DSPy fields)
            transcript_id=entry["id"],
            prior_turns=prior_turns,
        ).with_inputs("conversation_history", "user_message")

        examples.append(example)

    if not examples:
        print(
            "ERROR: No valid transcripts found. Check data/processed/passing_transcripts.json"
        )
        sys.exit(1)

    print(f"Loaded {len(examples)} examples from passing transcripts")
    return examples


# =============================================================================
# DSPy Signature & Module
# =============================================================================


class TherapistRespond(dspy.Signature):
    """You are a supportive therapeutic coach helping someone explore their thoughts and feelings.

    Respond to the user's latest message given the conversation history.
    Match their energy and length. Address all topics they raise.
    Stay warm and natural, not clinical or formulaic."""

    conversation_history: str = dspy.InputField(
        desc="Previous exchanges in the conversation"
    )
    user_message: str = dspy.InputField(desc="The user's current message to respond to")
    response: str = dspy.OutputField(
        desc="Your therapeutic coaching response, warm and natural"
    )


# =============================================================================
# Metric
# =============================================================================


def make_metric(judge_backend: str, judge_model: str | None) -> Callable[..., float]:
    """Create the assessment metric function.

    Returns a function compatible with both MIPROv2 (returns float)
    and GEPA (returns ScoreWithFeedback).
    """
    # Initialize the assessor backend once
    get_backend(backend_type=judge_backend, model=judge_model)

    def metric(
        example: dspy.Example,
        pred: dspy.Prediction,
        trace: object = None,
        **kwargs: object,
    ) -> float:
        """Assess a predicted response using the full rubric."""
        response_text = pred.response
        if not response_text or not response_text.strip():
            return 0.0

        # Build full conversation: prior turns + new exchange
        turns: list[ConversationTurn] = []
        for t in example.prior_turns:
            turns.append(ConversationTurn(user=t["user"], assistant=t["assistant"]))
        turns.append(
            ConversationTurn(user=example.user_message, assistant=response_text)
        )

        conversation = ConversationInput(turns=turns)

        try:
            result: AssessmentResult = asyncio.run(
                assess_conversation(
                    conversation,
                    require_min_turns=False,
                    conversation_id=example.transcript_id,
                )
            )
            logger.info(
                f"  [{example.transcript_id}] score={result.score:.3f} "
                f"passed={result.passed} safety={not result.safety_gate_failed}"
            )
            return result.score
        except Exception as e:
            logger.warning(f"  Assessment failed for {example.transcript_id}: {e}")
            return 0.0

    return metric


def make_gepa_metric(judge_backend: str, judge_model: str | None) -> Callable[..., Any]:
    """Create a GEPA-compatible metric that returns ScoreWithFeedback."""
    from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

    get_backend(backend_type=judge_backend, model=judge_model)

    def metric(
        example: dspy.Example,
        pred: dspy.Prediction,
        trace: object = None,
        **kwargs: object,
    ) -> ScoreWithFeedback:
        """Assess with feedback for GEPA's genetic evolution."""
        response_text = pred.response
        if not response_text or not response_text.strip():
            return ScoreWithFeedback(score=0.0, feedback="Empty response generated.")

        turns: list[ConversationTurn] = []
        for t in example.prior_turns:
            turns.append(ConversationTurn(user=t["user"], assistant=t["assistant"]))
        turns.append(
            ConversationTurn(user=example.user_message, assistant=response_text)
        )

        conversation = ConversationInput(turns=turns)

        try:
            result: AssessmentResult = asyncio.run(
                assess_conversation(
                    conversation,
                    require_min_turns=False,
                    conversation_id=example.transcript_id,
                )
            )

            # Build feedback from failed criteria
            feedback_parts: list[str] = []
            for criterion_id in result.failed_checks:
                reasoning = result.reasonings.get(criterion_id, "")
                if reasoning:
                    feedback_parts.append(f"{criterion_id}: {reasoning}")

            if result.safety_gate_failed:
                for criterion_id in result.failed_safety:
                    reasoning = result.reasonings.get(criterion_id, "")
                    feedback_parts.append(f"SAFETY {criterion_id}: {reasoning}")

            feedback = (
                "\n".join(feedback_parts) if feedback_parts else "All criteria passed."
            )

            logger.info(
                f"  [{example.transcript_id}] score={result.score:.3f} "
                f"failed={result.failed_checks}"
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
    """Run the unoptimized module on examples and return mean score."""
    scores: list[float] = []
    for ex in examples:
        pred = module(
            conversation_history=ex.conversation_history,
            user_message=ex.user_message,
        )
        score = metric_fn(ex, pred)
        # GEPA metric returns ScoreWithFeedback, extract float
        if hasattr(score, "score"):
            score = score.score
        scores.append(score)

    mean = sum(scores) / len(scores) if scores else 0.0
    return mean


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
        choices=["mipro", "gepa"],
        default="mipro",
        help="Optimizer to use (default: mipro)",
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
        choices=["openai", "google", "claude"],
        default="openai",
        help="Backend for assessment judge (default: openai)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for assessment judge (default: backend default)",
    )
    parser.add_argument(
        "--output",
        default="output/optimized_prompt.json",
        help="Output path for results (default: output/optimized_prompt.json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    # 1. Configure DSPy with the base model
    base_lm = dspy.LM(
        "openai/gemma-3-12b-it",
        api_base=args.base_model_url,
        api_key="none",  # Local server, no key needed
        temperature=0.7,
        max_tokens=1024,
    )
    dspy.configure(lm=base_lm)

    print(f"Base model: {args.base_model_url}")
    print(f"Judge: {args.judge_backend} ({args.judge_model or 'default'})")
    print(f"Optimizer: {args.optimizer} (auto={args.auto})")
    print()

    # 2. Load examples
    all_examples = load_examples(args.num_examples)
    # Split: 70% train, 30% val
    split = max(1, int(len(all_examples) * 0.7))
    trainset = all_examples[:split]
    valset = all_examples[split:] if split < len(all_examples) else all_examples[:1]
    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    # 3. Create module and metric
    module = dspy.Predict(TherapistRespond)

    if args.optimizer == "gepa":
        metric_fn = make_gepa_metric(args.judge_backend, args.judge_model)
    else:
        metric_fn = make_metric(args.judge_backend, args.judge_model)

    # 4. Baseline evaluation
    print("\n--- Baseline (unoptimized) ---")
    baseline_score = evaluate_baseline(module, valset, metric_fn)
    print(f"Baseline mean score: {baseline_score:.3f}")

    # 5. Optimize
    print(f"\n--- Optimizing with {args.optimizer} (auto={args.auto}) ---")

    if args.optimizer == "gepa":
        optimizer = dspy.GEPA(
            metric=metric_fn,
            auto=args.auto,
            num_threads=1,  # Sequential for local model
        )
        optimized = optimizer.compile(
            module,
            trainset=trainset,
        )
    else:
        optimizer = dspy.MIPROv2(
            metric=metric_fn,
            auto=args.auto,
            num_threads=1,
            max_bootstrapped_demos=2,
            max_labeled_demos=2,
            verbose=args.verbose,
        )
        optimized = optimizer.compile(
            module,
            trainset=trainset,
            valset=valset,
            requires_permission_to_run=False,
        )

    # 6. Evaluate optimized
    print("\n--- Optimized ---")
    optimized_score = evaluate_baseline(optimized, valset, metric_fn)
    print(f"Optimized mean score: {optimized_score:.3f}")
    print(f"Improvement: {optimized_score - baseline_score:+.3f}")

    # 7. Extract and save results
    # Get the optimized instruction (system prompt)
    optimized_instruction = optimized.signature.instructions
    print(f"\n--- Optimized Prompt ---\n{optimized_instruction}\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "optimizer": args.optimizer,
        "auto": args.auto,
        "num_examples": args.num_examples,
        "baseline_score": round(baseline_score, 3),
        "optimized_score": round(optimized_score, 3),
        "improvement": round(optimized_score - baseline_score, 3),
        "optimized_instruction": optimized_instruction,
    }

    # Try to extract few-shot demos if available
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

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
