# generator.py
"""
Single-call therapeutic conversation generator.

Generates complete multi-turn conversations in one API call.
"""

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError
from openai.types.responses import EasyInputMessageParam
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)

from assessor import ConversationInput, ConversationTurn

load_dotenv()


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class TaxonomyConfig:
    """Parsed taxonomy configuration."""

    # What they're discussing
    topic: str
    subtopic: str
    difficulty: str

    # How they communicate
    style: str
    interaction_cadence: str  # frequent_short or infrequent_detailed
    cognitive_patterns: str  # balanced or distorted

    # Who they are
    age_context: str  # young_adult, middle_adult, older_adult, teenager
    cultural_framing: str  # individualist, collectivist, mixed
    communication_pattern: (
        str  # neurotypical, direct_literal, tangential_energetic, limited_vocabulary
    )

    # Their relationship to help
    help_seeking: str  # explicit or implicit
    help_relationship: str  # naive, experienced, skeptical, dependent, testing

    # How the problem presents
    presentation: str  # clean_single, comorbid, somatic, decoy, spiraling
    temporality: str  # acute, chronic, triggered, building

    target_turns: int


def load_taxonomy(path: Path = Path("config/input-taxonomy.yaml")) -> dict:
    """Load taxonomy configuration from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def weighted_choice(items: list[dict], weight_key: str = "weight") -> dict:
    """Select item based on weights."""
    weights = [item.get(weight_key, 1.0) for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def _sample_weighted(options: dict[str, float]) -> str:
    """Sample a key from a dict of {key: weight}."""
    return random.choices(list(options.keys()), weights=list(options.values()), k=1)[0]


def sample_config(taxonomy: dict) -> TaxonomyConfig:
    """Sample a conversation configuration from taxonomy."""
    tax = taxonomy["taxonomy"]

    # What they're discussing
    topic_entry = weighted_choice(tax["topics"])
    topic = topic_entry["name"]
    subtopic = random.choice(topic_entry["subtopics"])
    difficulty = _sample_weighted(tax["difficulty"])

    # How they communicate
    style = _sample_weighted(tax["styles"])
    cadence = _sample_weighted(tax["interaction_cadence"])
    cognitive_patterns = _sample_weighted(tax["cognitive_patterns"])

    # Who they are
    age_context = _sample_weighted(tax["age_context"])
    cultural_framing = _sample_weighted(tax["cultural_framing"])
    communication_pattern = _sample_weighted(tax["communication_pattern"])

    # Their relationship to help
    help_seeking = _sample_weighted(tax["help_seeking"])
    help_relationship = _sample_weighted(tax["help_relationship"])

    # How the problem presents
    presentation = _sample_weighted(tax["presentation"])
    temporality = _sample_weighted(tax["temporality"])

    # Turn count depends on interaction cadence
    turn_range = taxonomy["turn_ranges"][cadence]
    target_turns = random.randint(turn_range["min"], turn_range["max"])

    return TaxonomyConfig(
        topic=topic,
        subtopic=subtopic,
        difficulty=difficulty,
        style=style,
        interaction_cadence=cadence,
        cognitive_patterns=cognitive_patterns,
        age_context=age_context,
        cultural_framing=cultural_framing,
        communication_pattern=communication_pattern,
        help_seeking=help_seeking,
        help_relationship=help_relationship,
        presentation=presentation,
        temporality=temporality,
        target_turns=target_turns,
    )


# =============================================================================
# Structured Output Models
# =============================================================================


class Turn(BaseModel):
    """A single conversation turn."""

    user: str = Field(description="What the client says")
    assistant: str = Field(
        description="Coach's response - length should match the client's cadence"
    )


class GeneratedConversation(BaseModel):
    """Complete generated conversation."""

    turns: list[Turn] = Field(description="The conversation turns")


# =============================================================================
# Generation
# =============================================================================

GENERATION_PROMPT = """Generate a realistic {turns}-turn therapeutic coaching conversation.

=== SCENARIO ===
Topic: {topic} ({subtopic})
Complexity: {difficulty}

=== WHO THE CLIENT IS ===
Age context: {age_context}
Cultural framing: {cultural_framing}
Communication pattern: {communication_pattern}

AGE CONTEXTS:
- young_adult (18-25): Identity formation, career start, first independence. Language: casual, uncertain about life direction.
- middle_adult (26-50): Career/family responsibilities, midlife questions. Language: practical, time-pressured.
- older_adult (50+): Health concerns, legacy, loss, transitions. Language: reflective, may reference past.
- teenager (13-17): School, identity, family dynamics. Language: informal, may be guarded or dramatic.

CULTURAL FRAMING:
- individualist: Frames issues as personal ("I feel", "I want"). Focus on self-growth and boundaries.
- collectivist: Frames issues relationally ("My family expects", "I can't disappoint them"). Duty, shame, obligation matter.
- mixed: Navigating between cultural expectations. Tension between "what I want" and "what's expected."

COMMUNICATION PATTERNS:
- neurotypical: Standard conversational patterns, reads subtext, emotional inference.
- direct_literal: Autistic-pattern. Direct, literal, prefers explicit communication. May miss implied meaning. Don't assume they're "cold" - they're precise.
- tangential_energetic: ADHD-pattern. Topic-jumping, high energy, may interrupt self mid-thought, circles back. Follow their energy, don't force linear structure.
- limited_vocabulary: Struggles to articulate emotions. Uses "bad", "weird", "off" for many feelings. Coach should help name emotions without putting words in their mouth.

=== HOW THEY COMMUNICATE ===
Style: {style}
Cadence: {cadence}
Help-seeking: {help_seeking}

STYLES: terse (few words), conversational (natural flow), detailed (full context), emotional (intense feelings), analytical (notices patterns)

CADENCE:
- frequent_short: Quick back-and-forth, short messages both sides
- infrequent_detailed: Journal-entry style, substantial messages, few turns

HELP-SEEKING:
- explicit: Directly asks "What should I do?"
- implicit: Expresses experience, expects guidance without asking

=== RELATIONSHIP TO HELP ===
Help relationship: {help_relationship}

- naive: First time seeking help. Doesn't know what to expect. May need more explanation of process.
- experienced: Knows therapeutic language. May have expectations or compare to past therapy.
- skeptical: Doubts it will work. "My partner made me try this." Coach should acknowledge skepticism, not argue.
- dependent: Seeks excessive reassurance. Coach should gently return agency, not feed dependency.
- testing: Pushes boundaries, checks if coach will break guidelines. Coach should hold boundaries warmly.

=== PROBLEM PRESENTATION ===
Presentation: {presentation}
Temporality: {temporality}
Cognitive patterns: {cognitive_patterns}

PRESENTATION:
- clean_single: One clear issue, straightforward
- comorbid: Multiple issues intertwined (e.g., anxiety + relationship + work). Let them bleed together naturally.
- somatic: Physical symptoms masking emotional. "My chest hurts", "I can't sleep." Coach should explore gently without diagnosing.
- decoy: Surface problem hides real issue. Client starts with safe topic, real concern emerges mid-conversation.
- spiraling: Jumps between concerns, hard to focus. Coach should help organize without dismissing.

TEMPORALITY:
- acute: Just happened, high distress. Needs grounding before exploration.
- chronic: Longstanding pattern. "I've always been this way."
- triggered: Recent event activated old wound.
- building: Escalating across conversation, getting more distressed as they talk.

COGNITIVE PATTERNS:
- balanced: Realistic perspective, even if distressed
- distorted: Catastrophizing, all-or-nothing, mind-reading. Coach validates FEELING, not CONCLUSION.

=== COACH GUIDELINES ===
- Match the client's communication pattern. Don't force neurotypical norms on direct_literal clients.
- Respect cultural framing. Don't push individualist values on collectivist clients.
- Skip formulaic validation ("That sounds hard"). Just respond naturally.
- Ask questions sparingly - guidance often helps more than another question.
- Return agency. They decide what's right for them.
- If they're testing, hold boundaries warmly without being defensive.

=== CONVERSATION ARC ===
Early: Client shares what's on their mind
Middle: Deeper understanding, gentle insights
Late: Small realizations, concrete next steps (unless temporality is "building" - then end with grounding)

Generate a natural, human conversation. Avoid AI tells."""


_api_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=30),
    retry=retry_if_exception_type((APIError, APITimeoutError, RateLimitError)),
)


@_api_retry
async def generate_conversation(
    client: AsyncOpenAI,
    config: TaxonomyConfig,
    system_prompt: str,
) -> ConversationInput:
    """Generate a complete conversation in one API call."""
    prompt = GENERATION_PROMPT.format(
        turns=config.target_turns,
        topic=config.topic,
        subtopic=config.subtopic,
        difficulty=config.difficulty,
        style=config.style,
        cadence=config.interaction_cadence,
        cognitive_patterns=config.cognitive_patterns,
        age_context=config.age_context,
        cultural_framing=config.cultural_framing,
        communication_pattern=config.communication_pattern,
        help_seeking=config.help_seeking,
        help_relationship=config.help_relationship,
        presentation=config.presentation,
        temporality=config.temporality,
    )

    user_msg: EasyInputMessageParam = {"role": "user", "content": prompt}
    response = await client.responses.parse(
        model="gpt-5-nano",
        input=[user_msg],
        text_format=GeneratedConversation,
        reasoning={"effort": "low"},
        max_output_tokens=4000,  # Full conversation needs more tokens
    )

    result = response.output_parsed
    if result is None:
        raise ValueError("Failed to parse conversation")

    # Convert to ConversationInput
    turns = [ConversationTurn(user=t.user, assistant=t.assistant) for t in result.turns]
    return ConversationInput(turns=turns, system_prompt=system_prompt)


# =============================================================================
# Batch Generation
# =============================================================================


def extract_system_prompt(markdown_path: Path) -> str:
    """Extract system prompt from markdown file (first code block)."""
    import re

    content = markdown_path.read_text()
    match = re.search(r"```\n(.*?)\n```", content, re.DOTALL)
    return match.group(1) if match else content


async def generate_batch(
    count: int,
    taxonomy_path: Path = Path("config/input-taxonomy.yaml"),
    system_prompt_path: Path = Path("config/system-prompt.md"),
    output_path: Path = Path("output/generated_conversations.jsonl"),
    concurrency: int = 5,
) -> list[ConversationInput]:
    """Generate a batch of conversations."""
    taxonomy = load_taxonomy(taxonomy_path)
    system_prompt = extract_system_prompt(system_prompt_path)

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one(index: int) -> tuple[int, ConversationInput, TaxonomyConfig]:
        async with semaphore:
            config = sample_config(taxonomy)
            conversation = await generate_conversation(client, config, system_prompt)
            print(
                f"Generated {index + 1}/{count}: "
                f"{config.topic}/{config.subtopic} ({len(conversation.turns)} turns)"
            )
            return index, conversation, config

    tasks = [generate_one(i) for i in range(count)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    conversations = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in results:
            if isinstance(result, BaseException):
                print(f"Error: {result}")
                continue

            index, conversation, config = result
            conversations.append(conversation)

            record = {
                "id": f"conv_{index:05d}",
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
                    "turns": len(conversation.turns),
                },
            }
            f.write(json.dumps(record) + "\n")

    print(f"\nGenerated {len(conversations)} conversations to {output_path}")
    return conversations


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Generating {count} conversations...")
    asyncio.run(generate_batch(count))
