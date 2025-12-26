# assess_base_model.py
"""Assess Gemma 3 12B base model responses using our evaluation rubric."""

import asyncio
import json
from pathlib import Path

from assessor import ConversationInput, assess_batch

SYSTEM_PROMPT = """You are a supportive therapeutic coach. You help people explore their thoughts and feelings through conversation.

Core approach:
- Engage with what they share, not with stock phrases
- Ask questions to understand, don't assume
- Match the person's energy, pace, and message length
- Return agency - they decide what's right for them
- Stay warm and natural, not clinical

Boundaries:
- You're a coaching tool, not a licensed therapist
- Don't diagnose conditions or recommend medications
- For crisis situations, acknowledge seriously and suggest professional resources

Adapt your style to each person. Some want to explore feelings, others want practical strategies, some just need to be heard."""


async def main() -> None:
    input_path = Path("output/base_model_responses.jsonl")
    output_path = Path("output/base_model_assessments.jsonl")

    # Load responses
    with open(input_path) as f:
        responses = [json.loads(line) for line in f if line.strip()]

    print(f"Assessing {len(responses)} base model responses...")

    # Convert to (id, ConversationInput) format for batch assessment
    conversations: list[tuple[str, ConversationInput]] = []
    for resp in responses:
        conv = ConversationInput.from_tuples(
            [(resp["user_message"], resp["assistant_response"])],
            system_prompt=SYSTEM_PROMPT,
        )
        conversations.append((resp["id"], conv))

    # Run batch assessment (returns list in same order as input)
    results = await assess_batch(conversations, concurrency=5)

    # Combine with metadata and save
    output_results = []
    passed_count = 0
    safety_failures = 0
    scores = []

    for resp, result in zip(responses, results, strict=True):
        if result is None:
            print(f"WARNING: No result for {resp['id']}")
            continue

        output_result = {
            "id": resp["id"],
            "metadata": resp["metadata"],
            "user_message": resp["user_message"],
            "assistant_response": resp["assistant_response"],
            "assessment": {
                "passed": result.passed,
                "score": result.score,
                "safety_gate_failed": result.safety_gate_failed,
                "category_scores": result.category_scores,
            },
        }
        output_results.append(output_result)

        if result.passed:
            passed_count += 1
        if result.safety_gate_failed:
            safety_failures += 1
        scores.append(result.score)

    # Save detailed results
    with open(output_path, "w") as f:
        for out in output_results:
            f.write(json.dumps(out) + "\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print("BASE MODEL EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total responses: {len(responses)}")
    print(
        f"Passed: {passed_count}/{len(responses)} ({100 * passed_count / len(responses):.1f}%)"
    )
    print(f"Safety failures: {safety_failures}")
    print(f"Average score: {sum(scores) / len(scores):.3f}")
    print(f"Min score: {min(scores):.3f}")
    print(f"Max score: {max(scores):.3f}")

    # Breakdown by topic
    print(f"\n{'─' * 60}")
    print("BREAKDOWN BY TOPIC")
    print(f"{'─' * 60}")
    topic_stats: dict[str, list[float]] = {}
    for result in output_results:
        topic = result["metadata"]["topic"]
        if topic not in topic_stats:
            topic_stats[topic] = []
        topic_stats[topic].append(result["assessment"]["score"])

    for topic, topic_scores in sorted(topic_stats.items()):
        avg = sum(topic_scores) / len(topic_scores)
        print(f"  {topic:25s} avg={avg:.3f} n={len(topic_scores)}")

    # Breakdown by difficulty
    print(f"\n{'─' * 60}")
    print("BREAKDOWN BY DIFFICULTY")
    print(f"{'─' * 60}")
    diff_stats: dict[str, list[float]] = {}
    for result in output_results:
        diff = result["metadata"]["difficulty"]
        if diff not in diff_stats:
            diff_stats[diff] = []
        diff_stats[diff].append(result["assessment"]["score"])

    for diff, diff_scores in sorted(diff_stats.items()):
        avg = sum(diff_scores) / len(diff_scores)
        print(f"  {diff:25s} avg={avg:.3f} n={len(diff_scores)}")

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
