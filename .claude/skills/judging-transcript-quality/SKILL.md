---
name: judging-transcript-quality
description: Use when evaluating synthetic conversation transcripts for fine-tuning suitability. Covers structural analysis, repetition detection, therapeutic quality assessment, and fine-tuning risk identification.
---

# Judging Transcript Quality for Fine-Tuning

## Core Principle

**Quantitative pattern analysis BEFORE qualitative judgment.**

Read samples to understand the data, then run automated checks to find systemic issues invisible to spot-checking. Structural rigidity and phrase repetition are the silent killers of fine-tuning data.

## When to Use

- After generating a batch of synthetic transcripts
- Before committing data to fine-tuning pipeline
- When pass rates seem good but you suspect hidden issues
- Periodic quality audits of accumulated data

## The Analysis Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │ 1. Sample    │───▶│ 2. Quantify  │───▶│ 3. Check  │ │
│  │    Reading   │    │    Patterns  │    │   Red     │ │
│  └──────────────┘    └──────────────┘    │   Flags   │ │
│                                          └─────┬─────┘ │
│  ┌──────────────┐    ┌──────────────┐          │       │
│  │ 5. Verdict   │◀───│ 4. Domain    │◀─────────┘       │
│  │    & Fixes   │    │    Quality   │                  │
│  └──────────────┘    └──────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Sample Reading

**Goal**: Understand the data structure and get qualitative feel.

Read 2-3 complete transcripts:
- One from early in the batch
- One from middle
- One from late

**What to notice**:
- Response structure (headers, paragraphs, formatting)
- Opening and closing patterns
- How topics evolve across exchanges
- User message variation

```bash
# Quick structure check
head -c 10000 data/raw/transcripts/short/transcript_XXXX.json | python3 -m json.tool
```

---

## Phase 2: Quantitative Pattern Analysis

Run these checks on the FULL dataset, not samples.

### 2.1 Structural Rigidity

```python
import json
from pathlib import Path
from collections import Counter

files = sorted(Path('data/raw/transcripts/short').glob('transcript_*.json'))

bold_headers = []
for f in files:
    d = json.load(open(f))
    for ex in d['exchanges']:
        resp = ex['assistant']
        bold_count = resp.count('**') // 2
        bold_headers.append(bold_count)

print(f"Responses with 0 headers: {bold_headers.count(0)}")
print(f"Responses with 4+ headers: {len([x for x in bold_headers if x >= 4])}")
print(f"Avg headers per response: {sum(bold_headers)/len(bold_headers):.1f}")
```

**Red flags**:
| Metric | Threshold | Problem |
|--------|-----------|---------|
| 100% same structure | Any | Model learns rigid template |
| Avg headers > 3 | Per response | Over-structured |
| 0% structural variation | — | Will produce robotic outputs |

### 2.2 Phrase Repetition (The Model "Tells")

```python
SIGNATURE_PHRASES = [
    "that's not nothing", "i want to", "here's what i",
    "i'm curious", "that makes sense", "that tracks",
    "that's actually", "that's real", "that's growth"
]

phrase_counts = {p: 0 for p in SIGNATURE_PHRASES}
total = 0

for f in files:
    d = json.load(open(f))
    for ex in d['exchanges']:
        resp = ex['assistant'].lower()
        total += 1
        for phrase in SIGNATURE_PHRASES:
            phrase_counts[phrase] += resp.count(phrase)

for phrase, count in sorted(phrase_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100
    print(f'"{phrase}": {count} ({pct:.0f}% of responses)')
```

**Red flags**:
| Frequency | Severity | Action |
|-----------|----------|--------|
| >50% of responses | Critical | Must de-duplicate |
| 20-50% | High | Should vary |
| 10-20% | Moderate | Monitor |
| <10% | OK | Natural variation |

### 2.3 Response Length Analysis

```python
ratios = []
for f in files:
    d = json.load(open(f))
    for ex in d['exchanges']:
        ratio = len(ex['assistant']) / max(len(ex['user']), 1)
        ratios.append(ratio)

print(f"Avg assistant/user ratio: {sum(ratios)/len(ratios):.1f}x")
print(f"Max ratio: {max(ratios):.1f}x")
```

**Red flags**:
| Ratio | Issue |
|-------|-------|
| >5x consistently | Over-explaining |
| <1.5x | Under-developed responses |
| High variance (>3 std) | Inconsistent quality |

### 2.4 Style Adaptation Check

```python
# Group by user communication style
styles = {}
for f in files:
    d = json.load(open(f))
    style = d['persona']['writing_style']
    if style not in styles:
        styles[style] = []
    for ex in d['exchanges']:
        styles[style].append(len(ex['assistant']))

for style, lengths in sorted(styles.items()):
    avg = sum(lengths) / len(lengths)
    print(f"{style}: avg response {avg:.0f} chars")
```

**Red flag**: If terse users get 3000+ char responses, the model isn't adapting.

---

## Phase 3: Domain-Specific Red Flags

For therapeutic coaching transcripts, check these specific issues:

### 3.1 Therapeutic Quality

```python
ISSUES = {
    'premature_advice': 0,
    'dismissive': 0,
    'missed_crisis': 0
}

for f in files:
    d = json.load(open(f))
    for i, ex in enumerate(d['exchanges']):
        resp = ex['assistant']
        user = ex['user']

        # Premature advice: "you should" without questions first
        if 'you should' in resp.lower() and '?' not in resp[:500]:
            ISSUES['premature_advice'] += 1

        # Dismissive language
        if "don't worry" in resp.lower():
            ISSUES['dismissive'] += 1

        # Missed crisis signals
        crisis_words = ['kill myself', 'end it', 'suicide', 'no point']
        for cw in crisis_words:
            if cw in user.lower():
                if '988' not in resp and 'crisis' not in resp.lower():
                    ISSUES['missed_crisis'] += 1

print(ISSUES)
```

### 3.2 Conversation Arc Analysis

```python
# Check if all conversations end positively (unrealistic)
positive_endings = 0
for f in files:
    d = json.load(open(f))
    last = d['exchanges'][-1]['assistant'].lower()
    if any(p in last for p in ['you did', "you've", 'proud', 'earned']):
        positive_endings += 1

pct = positive_endings / len(files) * 100
print(f"Positive endings: {pct:.0f}%")
```

**Red flag**: >90% positive endings = unrealistic success trajectory

### 3.3 Praise Distribution

```python
PRAISE = ['that\'s growth', 'that\'s huge', 'that counts', 'you did the thing']

early_praise, late_praise = 0, 0
early_total, late_total = 0, 0

for f in files:
    d = json.load(open(f))
    n = len(d['exchanges'])
    for i, ex in enumerate(d['exchanges']):
        resp = ex['assistant'].lower()
        praise_count = sum(resp.count(p) for p in PRAISE)

        if i < n // 3:
            early_praise += praise_count
            early_total += 1
        elif i > 2 * n // 3:
            late_praise += praise_count
            late_total += 1

early_avg = early_praise / early_total
late_avg = late_praise / late_total
print(f"Early: {early_avg:.2f} praise/response")
print(f"Late: {late_avg:.2f} praise/response")
print(f"Ratio: {late_avg/early_avg:.1f}x")
```

**Red flag**: >2x praise increase = model learning to artificially escalate positivity

---

## Phase 4: Aggregate Metrics

Compile findings into a summary table:

| Category | Metric | Value | Status |
|----------|--------|-------|--------|
| Structure | Header uniformity | X% | |
| Repetition | Top phrase frequency | X% | |
| Length | Avg response chars | X | |
| Adaptation | Style variance | X | |
| Domain | Premature advice | X instances | |
| Domain | Missed crisis | X instances | |
| Arc | Positive endings | X% | |

**Status legend**:
- OK: Within acceptable range
- WARN: Needs attention before fine-tuning
- FAIL: Must fix before proceeding

---

## Phase 5: Verdict & Recommendations

### Scoring Rubric

| Score | Meaning | Action |
|-------|---------|--------|
| 8-10 | High quality, minor issues | Proceed with fine-tuning |
| 6-7 | Usable with preprocessing | Filter/augment before training |
| 4-5 | Significant issues | Major revision needed |
| <4 | Unsuitable | Regenerate with different approach |

### Common Fixes

| Issue | Fix |
|-------|-----|
| Structural rigidity | Remove formatting from 30% of responses |
| Phrase repetition | Search-replace with varied alternatives |
| Over-long responses | Filter or truncate outliers |
| No style adaptation | Post-process to match user brevity |
| All positive endings | Add 10-20% stagnation/neutral endings |
| Missing crisis handling | Supplement with explicit crisis examples |

---

## Quick Reference: The Full Check Script

```python
#!/usr/bin/env python3
"""Transcript quality analysis for fine-tuning suitability."""

import json
from pathlib import Path
from collections import Counter

def analyze_transcripts(glob_pattern: str):
    files = sorted(Path('.').glob(glob_pattern))
    print(f"Analyzing {len(files)} transcripts...")

    # Collect metrics
    bold_headers = []
    phrase_counts = Counter()
    response_lengths = []
    user_lengths = []

    PHRASES = ["that's not nothing", "i want to", "here's what i",
               "i'm curious", "that makes sense"]

    for f in files:
        d = json.load(open(f))
        for ex in d['exchanges']:
            resp = ex['assistant']
            user = ex['user']

            bold_headers.append(resp.count('**') // 2)
            response_lengths.append(len(resp))
            user_lengths.append(len(user))

            for phrase in PHRASES:
                phrase_counts[phrase] += resp.lower().count(phrase)

    total = len(response_lengths)

    print(f"\n=== STRUCTURE ===")
    print(f"Avg bold sections: {sum(bold_headers)/total:.1f}")
    print(f"100% have headers: {bold_headers.count(0) == 0}")

    print(f"\n=== REPETITION ===")
    for phrase, count in phrase_counts.most_common(5):
        print(f'  "{phrase}": {count} ({count/total*100:.0f}%)')

    print(f"\n=== LENGTH ===")
    print(f"Avg response: {sum(response_lengths)//total} chars")
    print(f"Avg ratio: {sum(response_lengths)/sum(user_lengths):.1f}x")

if __name__ == "__main__":
    import sys
    pattern = sys.argv[1] if len(sys.argv) > 1 else "data/raw/transcripts/**/*.json"
    analyze_transcripts(pattern)
```

---

## Common Mistakes

| Mistake | Consequence | Prevention |
|---------|-------------|------------|
| Only reading samples | Missing systemic issues | Run quantitative checks on full dataset |
| Ignoring phrase frequency | Model develops "tells" | Check top 10 phrases before training |
| Skipping style adaptation check | Model over-explains to everyone | Verify response length varies by user style |
| Not checking conversation arcs | Unrealistic progress patterns | Analyze praise distribution, endings |
| Trusting rubric pass rate alone | Rubric doesn't catch repetition | Quality judgment is separate from rubric |
