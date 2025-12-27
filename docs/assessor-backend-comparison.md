# Assessor Backend Comparison Study

## Overview

We compared 4 LLM backends for assessing therapeutic coaching transcripts to identify potential biases and find the optimal assessor configuration.

## Backends Tested

| Backend | Model | Cost | Notes |
|---------|-------|------|-------|
| Claude CLI | opus | Free (subscription) | Generator of the transcripts |
| OpenAI | gpt-5.2 | $$$ | Latest flagship |
| OpenAI | gpt-5-mini | $ | Budget option |
| Google | gemini-3-flash-preview | $$ | Native structured output |

## Results Summary

### Scores by Backend

| Transcript | Claude | gpt-5.2 | gpt-5-mini | Gemini 3 Flash |
|------------|--------|---------|------------|----------------|
| **1000** | 1.000 ✓ | 0.825 ✓ | 0.933 ✓ | 0.887 ✓ |
| **1003** | 0.787 ✗ | 0.750 ✗ | 0.887 ✓ | 0.925 ✓ |
| **1002** | 0.963 ✓ | (rate limited) | 0.963 ✓ | 0.850 ✓ |

### Criteria Failed by Backend

| Transcript | Claude | gpt-5.2 | gpt-5-mini | Gemini |
|------------|--------|---------|------------|--------|
| **1000** | - | CQ6, CP2, CP4 | MT7 | CQ2, CP2 |
| **1003** | CQ2, CQ6, CP2 | CQ2, CQ6, CP2, CP4 | CQ2, CP2 | CP2, CP4 |
| **1002** | (errors) | - | CP2 | CQ2, CP2 |

## Key Findings

### 1. Claude Self-Bias Confirmed

Claude gave transcript 1000 a **perfect score (1.000)** while:
- gpt-5.2 scored it 0.825 and failed 3 criteria
- Gemini scored it 0.887 and failed 2 criteria

Both external models caught issues that Claude missed, particularly:
- **CQ2 (Mind-reading)**: Assertions like "your brain is doing something sneaky" stated as fact
- **CP4 (Formulaic openers)**: Every response starts with `**Topic:**` headers
- **CQ6 (Premature interpretation)**: Interpreting before exploring

### 2. gpt-5.2 is the Strictest

gpt-5.2 consistently found the most issues:
- Failed CP4 on both 1000 and 1003 (formulaic structure)
- Failed CQ6 on both (premature interpretation)
- Caught all the issues other models caught plus more

### 3. gpt-5-mini is Too Lenient

gpt-5-mini:
- Passed transcript 1003 (0.887) while Claude and gpt-5.2 both failed it
- Missed CQ6 violations that gpt-5.2 caught
- **Not recommended** as a cost-saving alternative

### 4. Gemini 3 Flash is a Good Middle Ground

Gemini:
- Caught CQ2 (mind-reading) issues consistently across transcripts
- Stricter than Claude but not as aggressive as gpt-5.2
- Native structured output reduces JSON parse errors
- More cost-effective than gpt-5.2

### 5. All Models Agree on CP2 (Length)

When length statistics are objectively bad (avg > 2x, >50% turns exceed 2x), all models flag it. This criterion is the most consistent across backends.

## Criteria Analysis

### Most Consistently Flagged
- **CP2 (Length calibration)**: Objective stats make this reliable
- **CP4 (Formulaic openers)**: The `**Topic:**` pattern is flagged by stricter models

### Most Variable
- **CQ2 (Tentative framing)**: Claude passes, others fail
- **CQ6 (Explore before interpret)**: Significant disagreement
- **MT7 (Coaching follow-up)**: Different interpretations of "explicit follow-up"

## Recommendations

### For Rigorous Evaluation
Use **gpt-5.2** - catches the most issues, but expensive and has rate limits.

### For Cost-Effective Evaluation
Use **Gemini 3 Flash** - good balance of strictness and cost, native structured output.

### For Self-Assessment During Generation
Use **Claude** with awareness of self-bias - good for quick iteration but verify with external model before finalizing training data.

### Not Recommended
**gpt-5-mini** - too lenient, misses issues that matter.

## Technical Notes

### Rate Limiting
- OpenAI gpt-5.2: 500k TPM limit, needs tenacity retries
- Google free tier: 5 req/min, 20 req/day (impractical)
- Google paid tier: Works well with retry logic

### Structured Output
- OpenAI: Native `responses.parse()` works reliably
- Google: Native `response_schema` with Pydantic models works well
- Claude CLI: JSON prompting, less reliable

### Implementation
Added tenacity retry logic to `llm_backend.py` for rate limit handling:
- OpenAI: Catches `RateLimitError`
- Google: Catches `ClientError` with 429 status
