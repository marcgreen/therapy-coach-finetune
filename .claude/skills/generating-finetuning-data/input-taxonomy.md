# Input Taxonomy

An input taxonomy defines the distribution of user messages to generate. Without systematic variation, synthetic data clusters around common patterns and the model overfits.

## Structure

```yaml
taxonomy:
  topics:
    - name: "topic_name"
      weight: 0.30           # 30% of examples
      subtopics: [...]       # Optional refinement

  styles:
    - terse                  # "help me"
    - conversational         # "Hey, I've been thinking about..."
    - detailed               # Multi-paragraph context
    - emotional              # Strong feelings expressed
    - analytical             # Logical, structured

  difficulty:
    easy: 0.30               # Clear intent, common scenario
    medium: 0.50             # Some ambiguity, requires judgment
    hard: 0.20               # Edge cases, complex scenarios

  edge_cases:
    - out_of_scope           # Should trigger refusal
    - ambiguous              # Needs clarification
    - multi_intent           # User wants several things
```

## Example: Therapeutic Coaching

```yaml
taxonomy:
  topics:
    - name: anxiety
      weight: 0.20
      subtopics: [work_stress, social, health, general_worry, panic]
    - name: relationships
      weight: 0.20
      subtopics: [romantic, family, friendship, coworker, loneliness]
    - name: life_transitions
      weight: 0.15
      subtopics: [career_change, relocation, loss_grief, new_role, decision]
    - name: self_worth
      weight: 0.15
      subtopics: [confidence, imposter, self_criticism, perfectionism, identity]
    - name: emotional_regulation
      weight: 0.15
      subtopics: [anger, sadness, overwhelm, numbness, mood_swings]
    - name: edge_cases
      weight: 0.15
      subtopics:
        - crisis_signals    # Suicidal ideation, self-harm mentions
        - medical_advice    # Requests for diagnoses, medication info
        - out_of_scope      # Legal, financial, non-therapeutic
        - vague             # Minimal context, unclear intent
        - hostile           # Aggressive, testing boundaries

  styles:
    terse: 0.15            # "feeling anxious"
    conversational: 0.40   # Natural, flowing
    detailed: 0.25         # Full context provided
    emotional: 0.15        # Intense feelings expressed
    analytical: 0.05       # "I notice a pattern..."

  difficulty:
    easy: 0.30             # Clear emotion, common situation
    medium: 0.50           # Mixed feelings, some complexity
    hard: 0.20             # Ambiguous, layered, edge cases
```

## Example: Customer Support

```yaml
taxonomy:
  topics:
    - name: billing
      weight: 0.25
      subtopics: [charges, refunds, subscriptions, payment_methods]
    - name: technical
      weight: 0.30
      subtopics: [bugs, setup, integration, performance]
    - name: account
      weight: 0.20
      subtopics: [access, settings, security, deletion]
    - name: product
      weight: 0.15
      subtopics: [features, limitations, roadmap, comparison]
    - name: edge_cases
      weight: 0.10
      subtopics: [escalation, legal, abuse, out_of_scope]

  styles:
    - frustrated: 0.25       # "This is broken AGAIN"
    - confused: 0.30         # "I don't understand..."
    - professional: 0.25     # Clear, business-like
    - urgent: 0.15           # Time pressure
    - appreciative: 0.05     # Positive context

  difficulty:
    easy: 0.40               # FAQ-level, clear solution
    medium: 0.40             # Requires investigation
    hard: 0.20               # Edge cases, escalation needed
```

## Generating from Taxonomy

```python
import random

def sample_from_taxonomy(taxonomy: dict) -> dict:
    """Sample one configuration from taxonomy."""
    topic = weighted_choice(taxonomy["topics"])
    subtopic = random.choice(topic["subtopics"])
    style = weighted_choice(taxonomy["styles"])
    difficulty = weighted_choice(taxonomy["difficulty"])

    return {
        "topic": topic["name"],
        "subtopic": subtopic,
        "style": style,
        "difficulty": difficulty,
    }

def generate_input_prompt(config: dict, domain: str) -> str:
    """Create prompt for generating a user message."""
    return f"""
Generate a realistic user message for a {domain} assistant.

Topic: {config["topic"]} ({config["subtopic"]})
Communication style: {config["style"]}
Difficulty level: {config["difficulty"]}

The message should feel natural and authentic.
Output only the user message, nothing else.
"""
```

## Diversity Validation

After generation, verify diversity:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def check_diversity(messages: list[str], threshold: float = 0.8) -> dict:
    """Check that messages are sufficiently diverse."""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(messages)
    sim_matrix = cosine_similarity(tfidf)

    # Check for near-duplicates
    np.fill_diagonal(sim_matrix, 0)
    max_similarities = sim_matrix.max(axis=1)
    duplicates = (max_similarities > threshold).sum()

    return {
        "total": len(messages),
        "near_duplicates": duplicates,
        "mean_similarity": sim_matrix.mean(),
        "max_similarity": max_similarities.max(),
        "pass": duplicates < len(messages) * 0.05,  # <5% duplicates
    }
```

## Anti-patterns

| Pattern | Problem | Fix |
|---------|---------|-----|
| All same topic | Model only handles that topic | Use weighted distribution |
| Only "conversational" style | Fails on terse inputs | Include all styles |
| No edge cases | Model hallucinates on boundaries | 10-15% edge cases |
| Generated in batches by topic | Subtle clustering | Shuffle before training |
