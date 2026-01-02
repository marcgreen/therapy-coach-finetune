---
name: finetune-train
description: Use when training a fine-tuned model and evaluating improvement over base model. Triggers - have filtered training data, ready to submit training job, need to convert to GGUF. Requires finetune-generate first.
---

# Fine-tune Train

Train the model and verify improvement over base model.

## Prerequisites

Complete [finetune-generate](../finetune-generate/SKILL.md) first. You need:

- [ ] `training_data.jsonl` — Filtered, validated training examples
- [ ] Model choice from finetune-design
- [ ] Evaluation rubric (for comparing base vs fine-tuned)

## Scope: SFT Only

This skill covers **Supervised Fine-Tuning (SFT)** only. Other training methods (DPO, GRPO, GEPA) will hopefully be added in the future.

## Outputs

By the end of this phase, you will have:

- [ ] Fine-tuned model (adapter on HuggingFace Hub or local)
- [ ] Merged GGUF file for local deployment
- [ ] `evaluation_report.md` — Statistical comparison of base vs fine-tuned

---

## Workflow

### Step 1: Dataset Preparation

Format and upload your training data:

1. **Verify format** matches training framework expectations:
   ```json
   {"messages": [
     {"role": "system", "content": "..."},
     {"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."}
   ]}
   ```

2. **Push to HuggingFace Hub** (for cloud training):
   ```bash
   huggingface-cli repo create username/dataset-name --type dataset --private
   # Then push via datasets library
   ```

3. **Verify access** — test loading the dataset before submitting training job

**Optional:** Use `hugging-face-dataset-creator` skill for streamlined HF Hub dataset management.

**Gate:** Dataset uploaded and accessible

---

### Step 2: Choose Training Approach

| Approach | Best For | Cost |
|----------|----------|------|
| **HuggingFace Jobs** | Fast, serverless GPU | ~$5-10 for 1K examples |
| **MLX Local** | Apple Silicon, free | 4-6 hours on M3 Max |
| **Cloud GPU** | Full control, large jobs | Varies |

**HuggingFace Jobs** is recommended for most projects — fast iteration, minimal setup.

**Reference:** [training-guide.md](training-guide.md)

---

### Step 3: Configure Training

**QLoRA parameters (typical):**
```python
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**Training hyperparameters:**
```python
config = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=2048,  # Based on token economics from design
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_8bit",
)
```

**Critical for large-vocabulary models (Gemma 3):**
- Vocabulary size affects memory: 262K vocab = 8x larger logits than Llama
- Reduce `max_length` on smaller GPUs (2048 for A10G with Gemma 3 12B)

**Reference:** [training-guide.md#configuration](training-guide.md#configuration)

---

### Step 4: Submit Training

**For HuggingFace Jobs:**

1. **Pre-create output repo** (Jobs can't create repos):
   ```bash
   huggingface-cli repo create username/model-name --type model --private
   ```

2. **Use your actual token** (not `$HF_TOKEN` placeholder):
   ```python
   secrets={"HF_TOKEN": "hf_xxxxx"}  # Your actual token
   ```

3. **Submit and monitor:**
   - Watch logs for training loss progression
   - Expected: Loss decreases from ~2-3 to ~0.8-1.2 over 3 epochs
   - Training time: ~2-3 hours for 1K examples on A10G

**Reference:** [training-guide.md#hf-jobs](training-guide.md#hf-jobs)

---

### Step 5: GGUF Conversion

Convert the fine-tuned adapter to GGUF for local inference:

1. **Download adapter** from HuggingFace Hub
2. **Merge with base model:**
   ```python
   from peft import PeftModel
   model = PeftModel.from_pretrained(base_model, adapter_path)
   merged = model.merge_and_unload()
   merged.save_pretrained("./merged")
   ```
3. **Convert to GGUF** using llama.cpp:
   ```bash
   python convert_hf_to_gguf.py ./merged --outtype q4_k_m
   ```
4. **Download and test locally:**
   ```bash
   llama-server -m model-q4_k_m.gguf --port 8080 -ngl 99
   ```

**Reference:** [training-guide.md#gguf-conversion](training-guide.md#gguf-conversion)

---

### Step 6: Evaluation

Compare fine-tuned model against base model using full-conversation generation.

**Why full-conversation evaluation:**
- Tests the actual use case (multi-turn, not single-turn)
- Captures consistency, context use, relationship building
- More rigorous than perplexity on held-out set

**Protocol:**
1. Generate 10-15 NEW personas (not used in training)
2. For each persona, generate 3 conversations with BOTH models
3. Use the SAME user simulator for both (controlled comparison)
4. Assess all conversations with your rubric
5. Compare scores statistically

**Statistical comparison:**
```python
from scipy import stats

base_scores = [...]
finetuned_scores = [...]

t_stat, p_value = stats.ttest_rel(finetuned_scores, base_scores)
improvement = np.mean(finetuned_scores) - np.mean(base_scores)

# Success criteria:
# - improvement >= 0.10 (10% absolute improvement)
# - p_value < 0.05 (statistically significant)
```

**Reference:** [training-guide.md#evaluation](training-guide.md#evaluation)

**Optional:** Use `hugging-face-evaluation-manager` skill to add evaluation results to your model card on HF Hub.

---

### Step 7: Sanity Checks

Before declaring success, verify:

| Check | Purpose |
|-------|---------|
| Perplexity on held-out set | Did training actually work? |
| Small human eval (5-10 convos) | Does LLM-as-judge agree with humans? |
| Capability regression test | Didn't break general abilities |
| Safety regression | No new harmful patterns |

**Warning signs:**
- Fine-tuned worse than base → Training issue, check data quality
- Huge improvement (>30%) → Suspiciously high, verify evaluation
- Safety regressions → Do not deploy

---

## Decision: Ship or Iterate?

| Result | Action |
|--------|--------|
| ≥10% improvement, p<0.05, no regressions | Ship it |
| <10% improvement | Consider more/better training data |
| Not significant (p>0.05) | More evaluation data or training data |
| Safety regressions | Do not deploy, investigate |

### Red Flags: Rationalizations to Resist

| Rationalization | Reality |
|-----------------|---------|
| "Perplexity improved, we're done" | Low perplexity ≠ good conversations. Full-conversation eval required. |
| "It feels better, ship it" | Feelings aren't evidence. Run statistical comparison (p<0.05). |
| "Default hyperparameters are fine" | Large-vocab models (Gemma 3) OOM with defaults. Check max_length. |
| "Skip GGUF, we'll deploy later" | GGUF conversion is the deployment. Test locally before declaring success. |
| "Safety check is paranoid" | Fine-tuning can introduce regressions. Safety audit is mandatory. |
| "$HF_TOKEN will work" | The placeholder resolves to limited OAuth token. Use your actual token. |

---

## Done When

- [ ] Training completed successfully
- [ ] GGUF converted and tested locally
- [ ] Evaluation shows significant improvement (≥10%, p<0.05)
- [ ] No safety regressions
- [ ] `evaluation_report.md` documents results

---

## Resources

| Resource | What It Contains |
|----------|------------------|
| [code/SETUP-REFERENCE.md](../code/SETUP-REFERENCE.md) | Project structure, script templates |
| [code/infrastructure.py](../code/infrastructure.py) | Copy-paste ready: token counting, slice generation |
| [examples/therapy-domain.md](../examples/therapy-domain.md) | Complete therapy example: evaluation results, model choice |

**HuggingFace Hub integration (optional skills):**
| Skill | Use For |
|-------|---------|
| `hugging-face-dataset-creator` | Push/manage datasets on HF Hub |
| `hugging-face-evaluation-manager` | Add eval results to model cards |

---

## What's Next?

After successful fine-tuning:
- Deploy model locally (Ollama, llama.cpp)
- Monitor real-world usage for issues
- Collect feedback for future iterations
- Consider DPO/RLHF if further refinement needed
