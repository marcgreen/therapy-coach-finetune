"""
Transcript Generator for Multi-Topic Therapeutic Coaching Conversations.

Generates synthetic transcripts by orchestrating:
1. Persona generation (with flaw patterns from taxonomy)
2. Topic seed sampling (from input taxonomy)
3. Exchange loop (user simulator → assistant → repeat)

Usage:
    # Generate a single transcript (for testing)
    uv run python transcript_generator.py --count 1 --output data/raw/transcripts

    # Generate pilot batch
    uv run python transcript_generator.py --count 3 --output data/raw/transcripts
"""

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from llm_backend import ClaudeCLIBackend, LLMBackend

# =============================================================================
# Logging Configuration
# =============================================================================

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the generator."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TopicSeed:
    """A topic seed for the persona."""

    category: str  # e.g., "anxiety", "relationships", "edge_cases"
    subtopic: str  # e.g., "work_stress", "crisis_signals"
    complexity: str  # "low", "medium", "high"
    description: str  # Brief description for the user simulator


# Writing styles - how the user composes messages
WRITING_STYLES = [
    "formal",  # Complete sentences, proper punctuation, articulate
    "casual",  # Relaxed but readable, normal punctuation
    "text-speak",  # Heavy shorthand: idk, tbh, ngl, lowercase, minimal punctuation
    "terse",  # Very short sentences, minimal elaboration
    "verbose",  # Long, detailed, sometimes rambling
]


# Trajectory types - emotional arc across the conversation
TRAJECTORIES = ["volatile", "improving", "deteriorating", "stable"]
TRAJECTORY_WEIGHTS = [0.66, 0.11, 0.11, 0.12]  # volatile most common


@dataclass
class Persona:
    """A user persona for transcript generation."""

    id: str
    name: str
    age_range: str
    personality_traits: list[str]
    communication_style: str
    writing_style: str  # How they write: formal, casual, text-speak, etc.
    topic_seeds: list[TopicSeed]
    flaw_patterns: list[dict[str, str]] | None  # None = clear communicator
    trajectory: str  # volatile, improving, deteriorating, stable
    seed: int  # Random seed used to generate this persona

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "age_range": self.age_range,
            "personality_traits": self.personality_traits,
            "communication_style": self.communication_style,
            "writing_style": self.writing_style,
            "topic_seeds": [
                {
                    "category": t.category,
                    "subtopic": t.subtopic,
                    "complexity": t.complexity,
                    "description": t.description,
                }
                for t in self.topic_seeds
            ],
            "flaw_patterns": self.flaw_patterns,
            "trajectory": self.trajectory,
            "seed": self.seed,
        }


@dataclass
class Exchange:
    """A single exchange in a conversation (user message + assistant response)."""

    user: str
    assistant: str
    exchange_number: int


@dataclass
class Transcript:
    """A complete transcript with metadata."""

    id: str
    persona: Persona
    exchanges: list[Exchange]
    target_exchanges: int
    created_at: str
    generator_version: str = "1.0.0"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "persona": self.persona.to_dict(),
            "exchanges": [
                {
                    "exchange_number": e.exchange_number,
                    "user": e.user,
                    "assistant": e.assistant,
                }
                for e in self.exchanges
            ],
            "target_exchanges": self.target_exchanges,
            "created_at": self.created_at,
            "generator_version": self.generator_version,
        }

    def to_messages(self) -> list[dict[str, str]]:
        """Convert to TRL-compatible messages format for training."""
        messages: list[dict[str, str]] = []
        for exchange in self.exchanges:
            messages.append({"role": "user", "content": exchange.user})
            messages.append({"role": "assistant", "content": exchange.assistant})
        return messages


# =============================================================================
# Configuration Loading
# =============================================================================


@dataclass
class GeneratorConfig:
    """Configuration for transcript generation."""

    input_taxonomy: dict
    flaw_taxonomy: dict
    system_prompt: str
    user_sim_template: str
    assistant_template: str

    # Derived settings
    no_flaw_probability: float = field(init=False)
    flaw_assignment: dict = field(init=False)
    manifestation_frequency: dict = field(init=False)

    def __post_init__(self) -> None:
        """Extract settings from taxonomies."""
        persona_config = self.flaw_taxonomy.get("persona_config", {})
        self.no_flaw_probability = persona_config.get("no_flaw_probability", 0.5)
        self.flaw_assignment = persona_config.get(
            "flaw_assignment", {"primary_count": 1, "secondary_count": [1, 2]}
        )
        self.manifestation_frequency = persona_config.get(
            "manifestation_frequency", {"primary": 0.6, "secondary": 0.15}
        )


def load_config(config_dir: Path = Path("config")) -> GeneratorConfig:
    """Load all configuration files."""
    # Load taxonomies
    with open(config_dir / "input-taxonomy.yaml") as f:
        input_taxonomy = yaml.safe_load(f)

    with open(config_dir / "flaw-taxonomy.yaml") as f:
        flaw_taxonomy = yaml.safe_load(f)

    # Load system prompt (extract from markdown)
    system_prompt_path = config_dir / "system-prompt.md"
    system_prompt = _extract_system_prompt(system_prompt_path)

    # Load prompt templates
    with open(config_dir / "prompts" / "user_sim.md") as f:
        user_sim_template = f.read()

    with open(config_dir / "prompts" / "assistant.md") as f:
        assistant_template = f.read()

    return GeneratorConfig(
        input_taxonomy=input_taxonomy,
        flaw_taxonomy=flaw_taxonomy,
        system_prompt=system_prompt,
        user_sim_template=user_sim_template,
        assistant_template=assistant_template,
    )


def _extract_system_prompt(path: Path) -> str:
    """Extract system prompt from markdown file (content between ``` blocks)."""
    content = path.read_text()
    lines = content.split("\n")

    in_code_block = False
    prompt_lines = []

    for line in lines:
        if line.strip() == "```" and not in_code_block:
            in_code_block = True
            continue
        elif line.strip() == "```" and in_code_block:
            break  # End of first code block
        elif in_code_block:
            prompt_lines.append(line)

    return "\n".join(prompt_lines)


# =============================================================================
# Sampling Functions
# =============================================================================


def weighted_choice(items: list[dict], weight_key: str = "weight") -> dict:
    """Select an item based on weights."""
    weights = [item.get(weight_key, 1.0) for item in items]
    total = sum(weights)
    normalized = [w / total for w in weights]
    return random.choices(items, weights=normalized, k=1)[0]


def sample_topic_seeds(
    config: GeneratorConfig, count: int = 4, rng: random.Random | None = None
) -> list[TopicSeed]:
    """Sample topic seeds from the input taxonomy."""
    if rng is None:
        rng = random.Random()

    taxonomy = config.input_taxonomy.get("taxonomy", {})
    topics = taxonomy.get("topics", [])
    difficulties = taxonomy.get("difficulty", {"easy": 0.3, "medium": 0.5, "hard": 0.2})

    seeds = []
    for _ in range(count):
        # Sample topic category
        topic = weighted_choice(topics)
        category = topic["name"]

        # Sample subtopic
        subtopics = topic.get("subtopics", [category])
        subtopic = rng.choice(subtopics) if subtopics else category

        # Sample complexity
        complexity = weighted_choice(
            [{"name": k, "weight": v} for k, v in difficulties.items()]
        )["name"]

        # Generate description based on subtopic
        description = _generate_topic_description(category, subtopic, complexity)

        seeds.append(
            TopicSeed(
                category=category,
                subtopic=subtopic,
                complexity=complexity,
                description=description,
            )
        )

    return seeds


def _generate_topic_description(category: str, subtopic: str, complexity: str) -> str:
    """Generate a brief description for a topic seed."""
    # Simple template-based descriptions
    templates = {
        "anxiety": {
            "work_stress": "Feeling overwhelmed by work demands and deadlines",
            "social_anxiety": "Struggling with social situations and fear of judgment",
            "health_anxiety": "Worrying excessively about health symptoms",
            "general_worry": "Persistent worry about various life aspects",
            "panic": "Experiencing panic attacks or intense anxiety episodes",
        },
        "relationships": {
            "romantic": "Navigating challenges in romantic relationship",
            "family": "Dealing with family dynamics and expectations",
            "friendship": "Struggling with friendships and social connections",
            "coworker": "Managing difficult workplace relationships",
            "loneliness": "Feeling isolated and disconnected from others",
        },
        "life_transitions": {
            "career_change": "Considering or going through a career change",
            "relocation": "Adjusting to a new location or environment",
            "loss_grief": "Processing loss or grief",
            "new_role": "Adapting to a new role (parent, manager, etc.)",
            "major_decision": "Facing a significant life decision",
        },
        "self_worth": {
            "low_confidence": "Struggling with self-confidence",
            "imposter_syndrome": "Feeling like a fraud despite achievements",
            "self_criticism": "Being overly critical of oneself",
            "perfectionism": "Dealing with perfectionist tendencies",
            "identity_confusion": "Questioning identity or sense of self",
        },
        "emotional_regulation": {
            "anger_management": "Struggling to manage anger",
            "persistent_sadness": "Experiencing ongoing low mood",
            "overwhelm": "Feeling emotionally overwhelmed",
            "emotional_numbness": "Feeling disconnected from emotions",
            "mood_swings": "Experiencing unpredictable mood changes",
        },
        "edge_cases": {
            "crisis_signals": "Expressing thoughts of self-harm or hopelessness",
            "medical_advice": "Seeking diagnosis or medication guidance",
            "out_of_scope": "Bringing up legal or non-therapeutic issues",
            "vague_input": "Providing minimal context about their situation",
            "hostile_user": "Testing boundaries or being confrontational",
        },
    }

    base = templates.get(category, {}).get(subtopic, f"Dealing with {subtopic}")

    if complexity == "hard":
        base += " (complex, layered situation)"
    elif complexity == "easy":
        base += " (relatively straightforward)"

    return base


def sample_flaw_patterns(
    config: GeneratorConfig, rng: random.Random | None = None
) -> list[dict[str, str]] | None:
    """Sample flaw patterns for a persona. Returns None for clear communicators."""
    if rng is None:
        rng = random.Random()

    # Check if this should be a no-flaw persona
    if rng.random() < config.no_flaw_probability:
        return None

    flaw_taxonomy = config.flaw_taxonomy.get("flaw_taxonomy", {})
    assignment = config.flaw_assignment

    patterns = []

    # Sample primary pattern (1)
    # Choose from communication OR emotional category
    primary_categories = ["communication", "emotional"]
    primary_cat = rng.choice(primary_categories)
    primary_options = flaw_taxonomy.get(primary_cat, [])
    if primary_options:
        primary = weighted_choice(primary_options)
        patterns.append(
            {
                "name": primary["name"],
                "category": primary_cat,
                "level": "primary",
                "description": primary.get("description", ""),
            }
        )

    # Sample secondary patterns (1-2)
    secondary_count = rng.choice(assignment.get("secondary_count", [1, 2]))
    secondary_categories = ["resistance", "memory"]

    for _ in range(secondary_count):
        sec_cat = rng.choice(secondary_categories)
        sec_options = flaw_taxonomy.get(sec_cat, [])
        if sec_options:
            secondary = weighted_choice(sec_options)
            # Avoid duplicates
            if not any(p["name"] == secondary["name"] for p in patterns):
                patterns.append(
                    {
                        "name": secondary["name"],
                        "category": sec_cat,
                        "level": "secondary",
                        "description": secondary.get("description", ""),
                    }
                )

    return patterns if patterns else None


# =============================================================================
# Persona Generation
# =============================================================================

# Name pools for variety
FIRST_NAMES = [
    # Gender-neutral
    "Alex",
    "Jordan",
    "Taylor",
    "Casey",
    "Riley",
    "Jamie",
    "Sam",
    "Charlie",
    # Latino/Hispanic
    "Sofia",
    "Miguel",
    "Carmen",
    "Luis",
    "Elena",
    "Diego",
    "Rosa",
    "Javier",
    # East Asian
    "Wei",
    "Mei",
    "Kenji",
    "Yuki",
    "Jin",
    "Hana",
    "Min",
    "Soo",
    # South Asian
    "Priya",
    "Raj",
    "Ananya",
    "Vikram",
    "Neha",
    "Arjun",
    "Deepa",
    "Amit",
    # African/African-American
    "Amara",
    "Kwame",
    "Zara",
    "Malik",
    "Nia",
    "Kofi",
    "Imani",
    "Darius",
    # Middle Eastern
    "Layla",
    "Omar",
    "Fatima",
    "Karim",
    "Yasmin",
    "Hassan",
    "Noor",
    "Amir",
    # European
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Sophie",
    "Marcus",
    "Anna",
    "David",
    # Mixed/Other
    "Maya",
    "Kai",
    "Zoe",
    "Leo",
    "Mia",
    "Ethan",
    "Ava",
    "Lucas",
]

PERSONALITY_TRAITS = [
    "anxious",
    "thoughtful",
    "skeptical",
    "warm",
    "guarded",
    "analytical",
    "emotional",
    "practical",
    "idealistic",
    "pessimistic",
    "optimistic",
    "introverted",
    "expressive",
    "reserved",
    "caring",
    "independent",
]

COMMUNICATION_STYLES = [
    "direct",
    "indirect",
    "verbose",
    "terse",
    "emotional",
    "analytical",
]


def generate_persona(
    config: GeneratorConfig,
    seed: int,
    target_topics: int = 4,
) -> Persona:
    """Generate a persona with topic seeds and optional flaw patterns."""
    rng = random.Random(seed)

    # Basic info
    name = rng.choice(FIRST_NAMES)
    age_ranges = ["18-25", "26-35", "36-45", "46-55", "55+"]
    age_range = rng.choice(age_ranges)

    # Personality
    num_traits = rng.randint(2, 4)
    traits = rng.sample(PERSONALITY_TRAITS, num_traits)

    # Communication style
    comm_style = rng.choice(COMMUNICATION_STYLES)

    # Writing style (weighted: casual most common, text-speak for younger)
    if age_range == "18-25":
        writing_weights = [0.1, 0.3, 0.4, 0.1, 0.1]  # More text-speak
    elif age_range in ("26-35", "36-45"):
        writing_weights = [0.2, 0.4, 0.1, 0.15, 0.15]  # More casual/formal
    else:
        writing_weights = [0.3, 0.4, 0.05, 0.15, 0.1]  # More formal, less text-speak
    writing_style = rng.choices(WRITING_STYLES, weights=writing_weights, k=1)[0]

    # Topic seeds
    topic_seeds = sample_topic_seeds(config, count=target_topics, rng=rng)

    # Flaw patterns (or None for clear communicator)
    flaw_patterns = sample_flaw_patterns(config, rng=rng)

    # Trajectory (emotional arc across conversation)
    trajectory = rng.choices(TRAJECTORIES, weights=TRAJECTORY_WEIGHTS, k=1)[0]

    persona_id = f"persona_{seed:04d}"

    return Persona(
        id=persona_id,
        name=name,
        age_range=age_range,
        personality_traits=traits,
        communication_style=comm_style,
        writing_style=writing_style,
        topic_seeds=topic_seeds,
        flaw_patterns=flaw_patterns,
        trajectory=trajectory,
        seed=seed,
    )


# =============================================================================
# Exchange Generation
# =============================================================================


def format_conversation_history(exchanges: list[Exchange]) -> str:
    """Format conversation history for prompts."""
    if not exchanges:
        return "(This is the start of the conversation)"

    lines = []
    for e in exchanges:
        lines.append(f"--- Exchange {e.exchange_number} ---")
        lines.append(f"User: {e.user}")
        lines.append(f"Assistant: {e.assistant}")
        lines.append("")

    return "\n".join(lines)


def format_persona_for_prompt(persona: Persona) -> str:
    """Format persona information for the user simulator prompt."""
    lines = [
        f"Name: {persona.name}",
        f"Age range: {persona.age_range}",
        f"Personality: {', '.join(persona.personality_traits)}",
        f"Communication style: {persona.communication_style}",
        f"Trajectory: {persona.trajectory}",
        "",
        "Topic seeds (things on your mind):",
    ]

    for i, topic in enumerate(persona.topic_seeds, 1):
        lines.append(f"  {i}. {topic.description} [{topic.category}/{topic.subtopic}]")

    return "\n".join(lines)


def format_flaw_patterns_for_prompt(
    flaw_patterns: list[dict[str, str]] | None,
) -> str:
    """Format flaw patterns for the user simulator prompt."""
    if not flaw_patterns:
        return "(No flaw patterns assigned - you are a clear communicator)"

    lines = []
    for pattern in flaw_patterns:
        level = pattern.get("level", "secondary")
        name = pattern.get("name", "unknown")
        desc = pattern.get("description", "")
        lines.append(f"- [{level.upper()}] {name}: {desc}")

    return "\n".join(lines)


async def generate_user_message(
    backend: LLMBackend,
    config: GeneratorConfig,
    persona: Persona,
    exchanges: list[Exchange],
    exchange_number: int,
    target_exchanges: int,
) -> str:
    """Generate a user message using the user simulator."""
    # Build the prompt from template
    # Extract the system prompt section from the template
    template = config.user_sim_template

    # Find the system prompt section (between first ``` pair)
    system_prompt = _extract_prompt_section(template, "System Prompt")
    user_prompt = _extract_prompt_section(template, "User Prompt")

    # Format variables
    persona_str = format_persona_for_prompt(persona)
    flaw_str = format_flaw_patterns_for_prompt(persona.flaw_patterns)
    history = format_conversation_history(exchanges)

    # Substitute variables in system prompt
    system_prompt = system_prompt.replace("{persona}", persona_str)
    system_prompt = system_prompt.replace("{flaw_patterns}", flaw_str)

    # Substitute variables in user prompt
    user_prompt = user_prompt.replace("{conversation_history}", history)
    user_prompt = user_prompt.replace("{exchange_number}", str(exchange_number))
    user_prompt = user_prompt.replace("{target_exchanges}", str(target_exchanges))

    # Generate
    result = await backend.complete(prompt=user_prompt, system=system_prompt)
    return result.content.strip()


async def generate_assistant_response(
    backend: LLMBackend,
    config: GeneratorConfig,
    exchanges: list[Exchange],
    user_message: str,
) -> str:
    """Generate an assistant response."""
    template = config.assistant_template

    # Extract sections
    system_prompt = _extract_prompt_section(template, "System Prompt")
    user_prompt = _extract_prompt_section(template, "User Prompt")

    # Format variables
    history = format_conversation_history(exchanges)

    system_prompt = system_prompt.replace("{system_prompt}", config.system_prompt)
    user_prompt = user_prompt.replace("{conversation_history}", history)
    user_prompt = user_prompt.replace("{user_message}", user_message)

    # Generate
    result = await backend.complete(prompt=user_prompt, system=system_prompt)
    return result.content.strip()


def _extract_prompt_section(template: str, section_name: str) -> str:
    """Extract a section from a markdown template (content between ``` blocks after ## Section Name)."""
    lines = template.split("\n")
    in_section = False
    in_code_block = False
    section_lines = []

    for line in lines:
        if line.startswith("## ") and section_name in line:
            in_section = True
            continue

        if in_section:
            if line.strip().startswith("```") and not in_code_block:
                in_code_block = True
                continue
            elif line.strip() == "```" and in_code_block:
                break  # End of code block
            elif in_code_block:
                section_lines.append(line)

    return "\n".join(section_lines)


# =============================================================================
# Transcript Generation
# =============================================================================


async def generate_transcript(
    backend: LLMBackend,
    config: GeneratorConfig,
    persona: Persona,
    target_exchanges: int = 15,
) -> Transcript:
    """Generate a complete transcript."""
    exchanges: list[Exchange] = []
    transcript_id = (
        f"transcript_{persona.seed:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger.info(
        f"Generating transcript {transcript_id} "
        f"({target_exchanges} exchanges, "
        f"{'no flaws' if persona.flaw_patterns is None else f'{len(persona.flaw_patterns)} flaws'})"
    )

    for i in range(1, target_exchanges + 1):
        logger.debug(f"  Exchange {i}/{target_exchanges}")

        # Generate user message
        user_message = await generate_user_message(
            backend=backend,
            config=config,
            persona=persona,
            exchanges=exchanges,
            exchange_number=i,
            target_exchanges=target_exchanges,
        )

        # Generate assistant response
        assistant_response = await generate_assistant_response(
            backend=backend,
            config=config,
            exchanges=exchanges,
            user_message=user_message,
        )

        # Create exchange
        exchange = Exchange(
            user=user_message,
            assistant=assistant_response,
            exchange_number=i,
        )
        exchanges.append(exchange)

    return Transcript(
        id=transcript_id,
        persona=persona,
        exchanges=exchanges,
        target_exchanges=target_exchanges,
        created_at=datetime.now().isoformat(),
    )


# =============================================================================
# Checkpointing
# =============================================================================


def save_transcript(transcript: Transcript, output_dir: Path) -> Path:
    """Save a transcript to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{transcript.id}.json"

    with open(output_path, "w") as f:
        json.dump(transcript.to_dict(), f, indent=2)

    logger.info(f"Saved transcript to {output_path}")
    return output_path


# =============================================================================
# Main Entry Point
# =============================================================================


async def generate_batch(
    count: int,
    output_dir: Path,
    target_exchanges: int = 15,
    start_seed: int = 0,
) -> list[Path]:
    """Generate a batch of transcripts."""
    config = load_config()

    # Initialize Claude CLI backend
    try:
        backend = ClaudeCLIBackend(validate=True)
    except RuntimeError as e:
        logger.error(f"Backend initialization failed: {e}")
        sys.exit(1)

    logger.info(
        f"Generating {count} transcripts with {target_exchanges} exchanges each"
    )

    saved_paths = []
    for i in range(count):
        seed = start_seed + i

        # Generate persona
        persona = generate_persona(config, seed=seed)

        # Generate transcript
        transcript = await generate_transcript(
            backend=backend,
            config=config,
            persona=persona,
            target_exchanges=target_exchanges,
        )

        # Save immediately (checkpoint)
        path = save_transcript(transcript, output_dir)
        saved_paths.append(path)

    return saved_paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic therapeutic coaching transcripts"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of transcripts to generate (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/transcripts"),
        help="Output directory (default: data/raw/transcripts)",
    )
    parser.add_argument(
        "--exchanges",
        type=int,
        default=15,
        help="Target exchanges per transcript (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Starting seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    paths = asyncio.run(
        generate_batch(
            count=args.count,
            output_dir=args.output,
            target_exchanges=args.exchanges,
            start_seed=args.seed,
        )
    )

    print(f"\nGenerated {len(paths)} transcript(s):")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
