# General-Purpose SKILL Set for E2E Fine-tuning

## The 4-SKILL System (Domain-Agnostic)

```
Knowledge Extraction → Data Generation → Training → Evaluation
         ↓                   ↓               ↓           ↓
     SKILL 1             SKILL 2         SKILL 3     SKILL 4
                                        (existing)
```

---

## SKILL 1: Domain Knowledge Extractor

**Purpose**: Extract structured knowledge from conversation for ANY domain task

**High-Level Description**:
Conducts efficient domain extraction through meta-prompting and eigenquestions. Identifies optimal expert personas, asks 15-25 high-signal questions, outputs structured knowledge with task definition, examples, and evaluation criteria. Supports both constrained domains (with frameworks) and open-ended tasks.

**Universal Components**:

### 1. Meta-Expert Identification
```yaml
For ANY domain, identify:
- Who would be the expert? (role, experience level)
- What makes them effective communicators?
- Rank by: domain knowledge × conversational efficiency

Examples:
- Medical chatbot → Emergency physician (15 yrs), medical educator
- Customer support → Senior support agent, product manager
- Legal assistant → Practicing attorney, legal researcher
- Writing coach → Published author, writing instructor
- Code reviewer → Senior engineer, tech lead
```

### 2. Eigenquestion Framework (Domain-Agnostic)
```yaml
Round 1: Boundary Questions (3-5)
- What IS in scope for this task?
- What is explicitly OUT of scope?
- What are hard constraints? (legal, technical, ethical)

Round 2: Competency Questions (5-7)
- What should the model ALWAYS do correctly?
- What should it NEVER do?
- What are gray areas requiring judgment?
- What are common errors to avoid?

Round 3: Variation Questions (5-7)
- What are the main input categories?
- What makes an input "easy" vs "hard"?
- What are edge cases?
- How does output vary by context?

Round 4: Validation Questions (3-5)
- How do you know if output is correct?
- What metrics matter?
- What's the minimum acceptable performance?

Total: 15-25 questions MAX
```

### 3. Automatic Scope Boundary Generation
```yaml
Based on conversation, AI generates:

WILL do:
[List of capabilities based on task definition]

WILL NOT do:
[Automatic inference of out-of-scope scenarios]

Examples:
- Medical: "Will provide health info, will NOT diagnose or prescribe"
- Legal: "Will explain concepts, will NOT provide legal advice"
- Customer support: "Will troubleshoot issues, will NOT process refunds without authorization"

User approves boundaries (2 minutes).
```

### 4. Eval Set Auto-Generation
```yaml
Generate 50-100 examples covering:
- 60-70%: Common scenarios
- 20-30%: Edge cases
- 10%: Out-of-scope (should trigger refusal)

Stratified by:
- Input category (from Round 3)
- Difficulty level (easy/medium/hard)
- Edge case representation

User spot-checks 10 examples (5 minutes).
If approved → full set ready.
```

### 5. Framework Integration (Optional)
```yaml
If domain has established frameworks:
- Prompt: "Are there established approaches/frameworks for this?"
- If yes: Load relevant frameworks (CBT for therapy, MECE for consulting, etc.)
- Map task to framework components
- Ground responses in framework terminology

If no frameworks:
- Proceed with general task definition
- Build custom taxonomy from conversation
```

**Outputs**:
- `knowledge_base.json` - Task definition, constraints, competencies
- `scope_boundaries.json` - Clear will/won't do statements
- `eval_set.json` - 50-100 diverse examples (70/30 split)
- `input_taxonomy.json` - Categories, difficulty levels, edge cases
- `quality_criteria.json` - What "good" looks like for this task

**Configuration Options**:
- `framework_mode`: "enabled" (for domains with frameworks) or "disabled"
- `eval_set_size`: 50, 75, 100, 200 (default: 100)
- `out_of_scope_ratio`: 0.05-0.20 (default: 0.10)
- `spot_check_size`: 5, 10, 20 (default: 10)

**Time**: 20-30 minutes  
**Manual effort**: ~10 minutes (boundary approval + spot-check)

---

## SKILL 2: Synthetic Data Generator

**Purpose**: Generate high-quality training data for ANY task with automated quality control

**High-Level Description**:
Creates diverse training datasets using configurable generation strategies (few-shot, template-based, contrastive) with automated quality filtering. Supports all TRL formats (SFT, DPO, GRPO, etc.) and scales from pilot to production with minimal manual review.

**Universal Components**:

### 1. Strategy Selection (Auto-Configured)
```yaml
Based on task_type in knowledge_base.json:

Classification/Labeling:
→ Few-shot prompting + structured output
→ Generator: Fast models (Haiku, GPT-4o-mini)

Instruction-Following:
→ Seed diversification + self-critique
→ Generator: Frontier models (Sonnet, GPT-4)

Transformation (code, rewriting):
→ Input-output variation + verification
→ Generator: Specialized models (Claude for code)

Preference Optimization:
→ Contrastive generation (good + multiple bad)
→ Two-model approach (strong + weak)

Reasoning Tasks:
→ Chain-of-thought + verification
→ Generator: Strong reasoning models

If framework_mode = enabled:
→ Use framework templates for structure
```

### 2. Universal Quality Filters
```yaml
Every generated example passes through:

Format Compliance:
- Matches target TRL format (SFT/DPO/GRPO)
- Required fields present
- Valid JSON/structure

Content Quality:
- Length: min_tokens to max_tokens (configurable)
- Completeness: Has all required components
- Coherence: Passes basic NLP quality check

Constraint Compliance:
- Respects scope boundaries from knowledge_base
- No out-of-scope content in in-scope examples
- Refusal examples actually refuse

Diversity:
- Similarity to existing < threshold (default: 0.8)
- Covers different input categories
- Varies in complexity/length

Safety (if enabled):
- No harmful content (toxicity < 0.01)
- No prohibited topics
- Appropriate for target audience

Pass rate target: >70%
If <70% → Adjust generation prompts → Retry
```

### 3. Template-Based Generation (Optional)
```yaml
If framework_mode = enabled:

Use structured templates from frameworks:

Example for CBT (therapy):
1. Empathetic reflection
2. Identify pattern
3. Teach technique
4. Practice prompt
5. Encouragement

Example for customer support:
1. Acknowledge issue
2. Show understanding
3. Provide solution steps
4. Confirm understanding
5. Offer additional help

Templates ensure:
- Consistent structure
- Framework fidelity
- Reduced need for review
```

### 4. Pilot → Scale Pipeline
```yaml
Phase 1: Pilot Generation
- Generate 50-100 examples
- Run automated filters
- Report pass rate

Phase 2: User Spot-Check
- Present 10 passing examples
- User rates: ✓ (good) / ✗ (bad)
- If >80% good → approve scale-up
- If <80% → gather feedback → adjust → regenerate

Phase 3: Scale Generation
- Generate target_size examples (1K-10K)
- Same filters applied
- Optional: 50-example random sample for review

Phase 4: TRL Format Validation
- Apply chat template to verify compatibility
- Check column names match trainer requirements
- Must pass before training
```

### 5. Generation Strategies (Configurable)

**For SFT (Instruction-Following)**:
```python
system_prompt = """
Generate diverse examples for: {task_description}

Vary across:
- Complexity: simple, moderate, challenging
- Style: {style_variations}
- Length: {length_range}
- Context: {context_variations}

Avoid repetitive patterns. Each example should be unique.
"""
```

**For DPO (Preference Pairs)**:
```python
# Chosen: Use strong model
chosen = strong_model.generate(prompt, temperature=0.7)

# Rejected: Multiple strategies
rejected_strategies = [
    "weak_model",        # Use weaker model
    "truncated",         # Cut off good response early
    "off_topic",         # Steer slightly off-topic
    "verbose",           # Add unnecessary verbosity
    "tone_mismatch"      # Wrong tone for context
]

rejected = generate_rejected(prompt, strategy=random.choice(rejected_strategies))
```

**For GRPO (Prompt-Only)**:
```python
# Focus on diverse, challenging prompts
# Model will generate + learn from rewards
prompts = generate_diverse_inputs(
    categories=input_taxonomy,
    complexity_distribution=[0.3, 0.5, 0.2],  # easy, medium, hard
    edge_case_ratio=0.15
)
```

### 6. Configurable Quality Thresholds
```yaml
quality_config:
  min_length: 10  # tokens
  max_length: 4096
  diversity_threshold: 0.7  # semantic similarity
  toxicity_threshold: 0.01
  pass_rate_target: 0.70
  
  # Domain-specific thresholds
  custom_filters:
    - name: "empathy_markers"  # for therapy/support
      enabled: true
      threshold: 0.8
    
    - name: "code_executability"  # for code generation
      enabled: false
      
    - name: "factual_accuracy"  # for knowledge tasks
      enabled: false
      verification_source: "optional_knowledge_base"
```

**Outputs**:
- `training_dataset/` - TRL-compliant dataset (1K-10K examples)
- `generation_report.json` - Pass rates, diversity metrics, costs
- `quality_log.json` - Which filters caught what issues
- `dataset_card.md` - Auto-generated documentation

**Configuration Options**:
- `generation_strategy`: "few_shot" | "template" | "contrastive" | "cot"
- `target_size`: 1000, 2000, 5000, 10000 (default: 2000)
- `training_method`: "sft" | "dpo" | "grpo" | "kto"
- `framework_mode`: true | false
- `pilot_size`: 50, 100, 200 (default: 100)
- `quality_thresholds`: custom threshold object

**Time**: 1-3 hours (mostly automated generation)  
**Manual effort**: 10-15 minutes (pilot spot-check)

---

## SKILL 3: HF LLM Trainer

**Purpose**: Fine-tune models on HF infrastructure

**High-Level Description**:
Uses existing hf-llm-trainer SKILL. Validates dataset format, selects GPU based on model size, submits training job with monitoring, and publishes to Hub. No changes needed - works as-is.

**Use as documented in**:
- https://huggingface.co/blog/hf-skills-training
- https://github.com/huggingface/skills/blob/main/hf-llm-trainer/skills/model-trainer/SKILL.md

**Time**: 2-4 hours (automated)  
**Manual effort**: 0 minutes

---

## SKILL 4: Efficient Model Evaluator

**Purpose**: Evaluate models on key metrics with strategic human spot-checking

**High-Level Description**:
Runs automated evaluation on multiple dimensions (accuracy, refusal, significance) then guides user through strategic spot-checks on high-signal examples. Generates deployment decision based on combined automated + human assessment.

**Universal Components**:

### 1. Evaluation Method Hierarchy
```yaml
Auto-select best evaluation method:

1. Functional Correctness (if applicable):
   - Code: Execute with test cases
   - Structured output: Parse and validate
   - Format requirements: Check with regex/parsers
   → Deterministic, zero bias

2. Exact Match (if outputs constrained):
   - Classification: Check against label
   - Multiple choice: Compare to answer
   - Closed-ended: String matching
   → Simple, interpretable

3. LLM-as-Judge (if needed):
   - Open-ended generation
   - Quality assessment
   - Appropriateness checks
   → Must validate on sample first

4. Human Evaluation:
   - Final validation
   - Ambiguous cases
   - Quality judgment
   → Strategic spot-checking

Priority: Use highest-ranked feasible method.
```

### 2. Automated Metrics Suite
```yaml
Run on held-out eval set:

Core Metrics (all tasks):
- Overall accuracy/quality score
- Performance by category (from input_taxonomy)
- Performance by difficulty (easy/medium/hard)
- Improvement vs. base model (%)
- Statistical significance (t-test, p-value)

Refusal Testing (all tasks):
- Accuracy on out-of-scope examples
- Target: 100% appropriate refusal
- If <100% → flag for attention

Task-Specific Metrics:
- Classification: Precision, Recall, F1
- Generation: Length distribution, diversity
- Code: Pass rate, execution success
- Preference: Win rate, consistency

All automated - no manual review.
```

### 3. Strategic Spot-Check Framework
```yaml
Human reviews carefully selected examples:

Selection Strategy:
- 5 examples: Common scenarios (representative)
- 5 examples: Edge cases (where failures likely)
- 5 examples: Out-of-scope refusals (safety check)
- Optional: 5 base vs. finetuned comparisons

Total: 15-20 examples (15-20 minutes)

Review Questions:
1. Is the response appropriate? (✓/⚠/✗)
2. Would you trust this output? (yes/no)
3. Any concerning patterns? (free text)

Decision Rule:
- 0-2 concerning (✗): Deploy
- 3-4 concerning: Review specific issues, possible quick fixes
- 5+ concerning: Significant revision needed
```

### 4. Multi-Dimensional Scoring (Configurable)
```yaml
Evaluate across dimensions based on task:

Dimension: Accuracy/Quality
- Metric: % correct or quality score
- Target: >80% (configurable)
- Weight: 40%

Dimension: Safety/Boundaries
- Metric: Refusal accuracy on out-of-scope
- Target: 100%
- Weight: 30%

Dimension: Consistency
- Metric: Variance across similar inputs
- Target: <20% variance
- Weight: 15%

Dimension: User Preference (if applicable)
- Metric: Human spot-check approval rate
- Target: >80% approval
- Weight: 15%

Weighted score determines deployment readiness.
```

### 5. LLM-as-Judge Validation (When Needed)
```yaml
If using LLM-as-judge:

Phase 1: Judge Validation (one-time)
- Take 50 examples with known labels/quality
- Have judge evaluate
- Compare to ground truth
- Accuracy threshold: >80%
- If <80%: Try different judge or prompt

Phase 2: Bias Mitigation
- Position bias: Swap order, aggregate
- Self-consistency: Multiple samples, majority vote
- Clear rubric: Specific criteria in prompt

Phase 3: Deployment
- Only use validated judge
- Log confidence scores
- Flag low-confidence for human review
```

### 6. Comparative Analysis
```yaml
Base Model vs. Fine-tuned:

Quantitative:
- Accuracy delta: +X%
- Category-level improvements: [breakdown]
- Edge case improvement: +Y%
- Statistical significance: p = 0.0X

Qualitative (from spot-check):
- Better responses: Z/15 examples
- Worse responses: N/15 examples
- Neutral/similar: M/15 examples

Regression Detection:
- Any categories worse than base?
- Any systematic failures introduced?
- Unexpected behaviors?

Report includes both to inform deployment.
```

### 7. Deployment Decision Logic
```yaml
Auto-generate recommendation:

✅ DEPLOY if:
- Overall improvement >10% (configurable)
- Statistical significance p < 0.05
- Refusal accuracy = 100%
- Human spot-check <3 concerning responses
- No critical regressions

⚠️ REVIEW if:
- Improvement >5% but <10%
- p-value 0.05-0.10 (marginally significant)
- Human spot-check 3-4 concerning
- Minor regressions in edge cases

❌ ITERATE if:
- Improvement <5%
- Not statistically significant
- Refusal accuracy <100%
- Human spot-check >4 concerning
- Critical regressions

Each includes specific recommendations for next steps.
```

**Outputs**:
- `evaluation_report.md` - Automated metrics + spot-check results
- `deployment_decision.json` - Deploy/review/iterate with reasoning
- `detailed_results.json` - Per-example scores and outputs
- `comparison_analysis.json` - Base vs. finetuned breakdown
- `spot_check_worksheet.md` - Formatted for human review

**Configuration Options**:
- `eval_method`: "functional" | "exact_match" | "llm_judge" | "human"
- `spot_check_size`: 10, 15, 20, 30 (default: 15)
- `improvement_threshold`: 0.05, 0.10, 0.15 (default: 0.10)
- `significance_level`: 0.05, 0.01 (default: 0.05)
- `dimensions`: List of dimensions to evaluate (configurable weights)

**Time**: 20-40 minutes  
**Manual effort**: 15-20 minutes (strategic spot-check)

---

## Complete Workflow (Any Domain)

```bash
# Phase 1: Extract Knowledge (25 min, 10 min manual)
claude code: "Use domain-knowledge-extractor for [task description]"
→ Eigenquestion interview
→ Automatic scope boundary generation
→ USER: Approve boundaries (2 min)
→ Auto-generate 100 eval examples
→ USER: Spot-check 10 examples (5 min)
→ Optional: Load frameworks if domain has them
→ Outputs: knowledge_base.json, scope_boundaries.json, eval_set.json

# Phase 2: Generate Data (2-3 hours, 15 min manual)
claude code: "Use synthetic-data-generator with {config}"
→ Auto-select generation strategy based on task type
→ Pilot: 100 examples with automated filtering
→ Report pass rate
→ USER: Spot-check 10 passing examples (10 min)
→ If approved: Scale to target size (2K-5K)
→ TRL format validation
→ Outputs: training_dataset/, generation_report.json

# Phase 3: Train (2-4 hours, 0 min manual)
claude code: "Use hf-llm-trainer with {model} on {dataset}"
→ Automated format validation
→ GPU selection
→ Submit job with monitoring
→ Outputs: finetuned_model on Hub

# Phase 4: Evaluate (30 min, 15 min manual)
claude code: "Use efficient-evaluator with spot-check"
→ Auto-select evaluation method
→ Run automated metrics on held-out set
→ Statistical testing
→ USER: Strategic spot-check (15 examples, 15 min)
→ Auto-generate deployment decision
→ Outputs: evaluation_report.md, deployment_decision.json
```

**Total time**: ~4-8 hours  
**Total manual effort**: ~40 minutes  
**Applicable to**: Any fine-tuning task

---

## Domain Examples Using These SKILLs

### Example 1: Customer Support Chatbot
```yaml
SKILL 1 (Knowledge Extraction):
- Expert: Senior support agent
- Scope: Troubleshoot common issues, escalate complex ones
- Frameworks: None (no established framework)
- Boundaries: Will help with FAQs, won't process refunds

SKILL 2 (Data Generation):
- Strategy: Few-shot + variation
- Format: SFT (conversational)
- Quality filters: Tone (helpful), accuracy, boundaries
- Size: 3K examples

SKILL 4 (Evaluation):
- Method: LLM-as-judge + human spot-check
- Dimensions: Helpfulness, accuracy, tone
- Spot-check: 15 examples
```

### Example 2: Code Review Assistant
```yaml
SKILL 1 (Knowledge Extraction):
- Expert: Senior engineer
- Scope: Review Python code, suggest improvements
- Frameworks: PEP-8, clean code principles
- Boundaries: Will review, won't write full features

SKILL 2 (Data Generation):
- Strategy: Template (before/after code)
- Format: SFT (prompt-completion)
- Quality filters: Code executability, style adherence
- Size: 2K examples

SKILL 4 (Evaluation):
- Method: Functional (code execution + static analysis)
- Dimensions: Correctness, helpfulness, style
- Spot-check: 15 examples
```

### Example 3: Writing Coach
```yaml
SKILL 1 (Knowledge Extraction):
- Expert: Professional editor
- Scope: Improve clarity, style, structure
- Frameworks: Classic writing principles (show don't tell, etc.)
- Boundaries: Will coach, won't ghostwrite

SKILL 2 (Data Generation):
- Strategy: Template (feedback patterns)
- Format: SFT (conversational)
- Quality filters: Constructive tone, specific suggestions
- Size: 4K examples

SKILL 4 (Evaluation):
- Method: LLM-as-judge (validated) + human spot-check
- Dimensions: Helpfulness, specificity, tone
- Spot-check: 20 examples (writing is subjective)
```

### Example 4: Self-Therapy Coach (Our Use Case)
```yaml
SKILL 1 (Knowledge Extraction):
- Expert: Therapist/counselor
- Scope: Teach CBT/DBT skills for everyday stress
- Frameworks: CBT, DBT, ACT (built-in)
- Boundaries: Will teach skills, won't handle crises

SKILL 2 (Data Generation):
- Strategy: Template (framework-based)
- Format: SFT (conversational)
- Quality filters: Empathy, framework adherence, boundaries
- Size: 3K examples

SKILL 4 (Evaluation):
- Method: Automated quality + human spot-check
- Dimensions: Appropriateness, empathy, framework accuracy
- Spot-check: 15 examples
```

---

## Configuration Templates (Quick Start)

### Minimal Config (Fast, Simple Tasks)
```yaml
domain_extractor:
  eval_set_size: 50
  spot_check_size: 5
  framework_mode: disabled

data_generator:
  target_size: 1000
  pilot_size: 50
  generation_strategy: "few_shot"

evaluator:
  eval_method: "exact_match"
  spot_check_size: 10
  improvement_threshold: 0.10
```

### Standard Config (Most Tasks)
```yaml
domain_extractor:
  eval_set_size: 100
  spot_check_size: 10
  framework_mode: disabled

data_generator:
  target_size: 3000
  pilot_size: 100
  generation_strategy: "auto"  # Based on task type

evaluator:
  eval_method: "auto"  # Hierarchy
  spot_check_size: 15
  improvement_threshold: 0.10
```

### Framework-Heavy Config (Therapy, Consulting, etc.)
```yaml
domain_extractor:
  eval_set_size: 100
  spot_check_size: 10
  framework_mode: enabled
  framework_sources: ["CBT", "DBT", "ACT"]  # Load these

data_generator:
  target_size: 4000
  pilot_size: 100
  generation_strategy: "template"  # Use framework templates
  quality_thresholds:
    empathy_markers: 0.8
    framework_adherence: 0.9

evaluator:
  eval_method: "llm_judge"  # Validate first
  spot_check_size: 20
  dimensions: ["appropriateness", "empathy", "framework_accuracy"]
  improvement_threshold: 0.15  # Higher bar for quality
```

---

## Key Advantages of This System

### 1. **Domain-Agnostic**
- Same 4 SKILLs work for any task
- Configuration-driven adaptation
- No task-specific coding needed

### 2. **Minimal Manual Effort**
- 3 spot-check touchpoints (~40 min total)
- Automation handles 95% of work
- Strategic human review on high-signal examples

### 3. **Quality Assured**
- Automated filters catch most issues
- Spot-checks validate generation quality
- Statistical testing ensures real improvement

### 4. **Scalable**
- Pilot → Scale pattern prevents waste
- Same approach works 1K → 10K examples
- Automation scales, manual effort doesn't

### 5. **Transparent**
- Clear reports at each stage
- Deployment decisions explained
- User maintains control with minimal effort

### 6. **Flexible**
- Support frameworks when available
- Multiple evaluation methods
- Configurable thresholds and strategies

---

## Summary: The 4-SKILL System

| SKILL | Purpose | Input | Output | Manual Effort |
|-------|---------|-------|--------|---------------|
| **1. Domain Knowledge Extractor** | Extract task definition via conversation | User conversation | knowledge_base, eval_set, boundaries | 10 min (approve + spot-check) |
| **2. Synthetic Data Generator** | Generate quality training data | knowledge_base | training_dataset (TRL-compliant) | 15 min (pilot spot-check) |
| **3. HF LLM Trainer** | Fine-tune on HF infrastructure | training_dataset | finetuned_model | 0 min (automated) |
| **4. Efficient Model Evaluator** | Evaluate with strategic spot-checks | base_model, finetuned_model, eval_set | evaluation_report, deploy decision | 15 min (strategic spot-check) |

**Total manual effort**: ~40 minutes across entire pipeline  
**Total time**: ~4-8 hours (mostly automated)  
**Works for**: Any fine-tuning task from customer support to therapy coaching to code review