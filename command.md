## 1. First test - NLP tasks with Ollama

Use the task names currently used by the framework:
### Text Classification — ag_news

```bash
python3 main.py --run-campaign \
  --provider ollama \
  --model llama3.1:latest \
  --task text_classification \
  --sample-size 3
```

### Question Answering — SQuAD v2

```bash
python3 main.py --run-campaign \
  --provider ollama \
  --model llama3.1:latest \
  --task question_answering \
  --sample-size 3
```

### Text Generation — CNN/DailyMail

```bash
python3 main.py --run-campaign \
  --provider ollama \
  --model llama3.1:latest \
  --task text_generation \
  --sample-size 3
```

### Full Ollama test across all the tasks and models

```bash
python3 main.py --run-campaign --provider ollama --sample-size 10
```

I recommend keeping `--sample-size 3` for this first validation. **Don't run large experiments yet.**

---

# 2. Then test another provider

The framework already has Google, Groq and Hugging Face integrations. For current provider/model availability,start with **Groq**, because its current model catalogue includes `llama-3.1-8b-instant`; Groq documents a developer-plan limit of 1,000 RPM for that model.

### Groq — Llama 3.1 8B

Classification:

```bash
python3 main.py --run-campaign \
  --provider groq \
  --model llama-3.1-8b-instant \
  --task text_classification \
  --sample-size 3
```

Question Answering:

```bash
python3 main.py --run-campaign \
  --provider groq \
  --model llama-3.1-8b-instant \
  --task question_answering \
  --sample-size 3
```

Text Generation:

```bash
python3 main.py --run-campaign \
  --provider groq \
  --model llama-3.1-8b-instant \
  --task text_generation \
  --sample-size 3
```

**Important:** Groq's current documentation lists `llama-3.1-8b-instant`, but its deprecation page says it is scheduled for shutdown on **August 16, 2026**. 

---

# 3. Hugging Face

The integrated model was:

```text
Qwen/Qwen2.5-7B-Instruct
```

The model still exists on Hugging Face.

If the framework's Hugging Face wrapper accepts that model ID:

### Classification

```bash
python3 main.py --run-campaign \
  --provider huggingface \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task text_classification \
  --sample-size 3
```

### QA

```bash
python3 main.py --run-campaign \
  --provider huggingface \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task question_answering \
  --sample-size 3
```

### Generation

```bash
python3 main.py --run-campaign \
  --provider huggingface \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task text_generation \
  --sample-size 3
```

One caveat: Hugging Face Inference Providers currently provide only a small amount of free monthly credit for free users, so don't launch the large campaign through HF.
