# run_base_model_eval.py
"""Run Gemma 3 12B on evaluation scenarios and collect responses."""

import json
import sys
from pathlib import Path

import httpx

LLAMA_SERVER_URL = "http://localhost:8080"

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


def format_gemma_prompt(system: str, user_message: str) -> str:
    """Format prompt for Gemma 3 chat template."""
    # Gemma 3 uses <start_of_turn> and <end_of_turn> tokens
    return f"""<start_of_turn>user
{system}

---

{user_message}<end_of_turn>
<start_of_turn>model
"""


def run_completion(prompt: str, max_tokens: int = 300) -> str:
    """Send completion request to llama-server."""
    try:
        response = httpx.post(
            f"{LLAMA_SERVER_URL}/completion",
            json={
                "prompt": prompt,
                "n_predict": max_tokens,
                "stop": ["<end_of_turn>"],
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        return result.get("content", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def main() -> None:
    input_path = Path("output/eval_scenarios.jsonl")
    output_path = Path("output/base_model_responses.jsonl")

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run generate_eval_scenarios.py first.")
        sys.exit(1)

    # Load scenarios
    with open(input_path) as f:
        scenarios = [json.loads(line) for line in f if line.strip()]

    print(f"Running Gemma 3 12B on {len(scenarios)} scenarios...")
    print(f"Server: {LLAMA_SERVER_URL}")
    print()

    results = []
    for i, scenario in enumerate(scenarios):
        prompt = format_gemma_prompt(SYSTEM_PROMPT, scenario["message"])
        response = run_completion(prompt)

        result = {
            "id": scenario["id"],
            "user_message": scenario["message"],
            "assistant_response": response,
            "metadata": scenario["metadata"],
        }
        results.append(result)

        # Progress
        print(
            f"[{i + 1:2d}/{len(scenarios)}] {scenario['metadata']['topic']}/{scenario['metadata']['subtopic']}"
        )
        print(f"    Response: {response[:80]}...")
        print()

    # Save results
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\nSaved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    main()
