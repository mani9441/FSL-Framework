# LLM Few-Shot Evaluation & Dissertation Benchmarking Framework

An empirical, production-grade Python benchmarking framework designed for systematic research on Large Language Models (LLMs). The framework enables reproducible, multi-dimensional evaluation of open-weight and API-hosted LLMs across diverse Natural Language Processing (NLP) tasks (**Text Classification**, **Question Answering**, **Text Summarization**) under varying few-shot demonstration regimes (**0-shot**, **1-shot**, **3-shot**, **5-shot**).

---

## 🌟 Core Architecture & Capabilities

- 🎯 **Task-Aware Multi-Dimensional Evaluation**:
  - **Text Classification (`ag_news`)**: Accuracy, Macro F1, Precision, Recall, Parse Success Rate.
  - **Question Answering (`squad_v2`)**: Response Correctness, Completeness, Relevance, Exact Match, QA F1.
  - **Text Summarization (`cnn_dailymail`)**: BLEU (BLEU-1/2/4), ROUGE (ROUGE-1/2/L), Parse Success Rate.
  - **Efficiency & Resource Tracking**: Latency (seconds), Prompt Tokens, Completion Tokens, Total Tokens, Estimated Cost.
- 🔬 **Systematic Experimental Matrix**:
  - Evaluates models across 0, 1, 3, and 5 few-shot demonstration counts.
  - Controls demonstration selection (`random`), ordering (`random`), and diversity (`medium`).
  - Maintains strict prompt composition consistency across all experimental conditions.
- 🤖 **Multi-Provider Model Engine**:
  - **Local Models**: Ollama (`ollama`) integration for local, open-weight execution (e.g., Llama 3.1 8B, Mistral 7B, Phi-4 Mini, Qwen 3 8B, Gemma 3 4B).
  - **Cloud API Wrappers**: Groq (`groq`), Hugging Face (`huggingface`), Google Gemini (`google`), and OpenAI (`openai`).
  - Retry handling, rate limit mitigation, and response validation.
- 📊 **7-Stage Reproducible Execution Pipeline**:
  - **Stage 1 — Inference**: Runs model across multi-trial seeds (`[42, 43, 44]`).
  - **Stage 2 — Response Consolidation**: Parses JSONL/CSV responses and validates structural compliance.
  - **Stage 3 — Task Evaluation**: Computes task-specific metrics per trial and aggregate.
  - **Stage 4 — Statistical Analysis**: Computes Welch t-tests, Cohen's d effect sizes, SD, and 95% CIs.
  - **Stage 5 — Visualization**: Plots latency scaling, accuracy-latency trade-offs, and parameter heatmaps.
  - **Stage 6 — Report Generation**: Exports LaTeX booktabs tables, Markdown, HTML, and JSON reports.
  - **Stage 7 — Dissertation Synthesis**: Compiles master campaign datasets and cross-experiment scaling matrices.
- 📈 **Statistical Rigor & Multi-Trial Sampling**:
  - **Multi-Trial Execution**: 3 repeated trials per configuration using fixed seeds (`[42, 43, 44]`).
  - **Hypothesis Testing**: Welch's t-test, Cohen's $d$ effect sizes, 95% Confidence Intervals, and Trial Standard Deviation.

---

## 📁 Git Tracked Repository Structure

```text
├── main.py                     # Primary CLI entrypoint (Campaigns, Resource Validation)
├── pipeline.py                 # Self-contained 7-stage reproducible experiment pipeline
├── command.md                  # Quick CLI command reference for campaign execution
├── .gitignore                  # Git ignore rules for build artifacts, cache, and logs
├── .env.example                # Environment variable configuration template
│
├── config/                     # Configuration management, YAML/JSON schemas, seed & path managers
├── datasets/                   # Subprocess HF dataset loader, preprocessor, formatter & sampler
├── prompt_engine/              # Few-shot example selector, ordering/diversity manager & prompt builder
├── model_interfaces/           # Provider wrappers (Ollama, Groq, HuggingFace, Gemini, OpenAI)
├── evaluation/                 # Metric evaluators for Classification, QA, Generation, Efficiency & Reliability
├── experiment_engine/          # Controller, multi-trial manager, matrix campaign generator & progress tracker
├── response_repository/        # Thread-safe raw response persistence (JSONL & CSV)
├── analysis/                   # Cross-experiment statistical analysis engine & dissertation matrix exporter
├── visualization/              # Publication chart plotting engine (PNG @ 300DPI & vector SVG)
├── reports/                    # Exporter for LaTeX booktabs, Markdown, HTML, JSON & CSV reports
├── resources/                  # Resource validation engine for datasets, models, and environments
├── utilities/                  # Config loaders, logging, constants, and global helper functions
└── tests/                      # Comprehensive 93-test unit & integration testing suite
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (Python 3.13 supported)
- **Ollama** (for local open-weight LLM execution)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd experiment
   ```

2. **Install core dependencies**:
   ```bash
   pip install pandas numpy scipy matplotlib requests pyyaml scikit-learn
   ```

3. **Pull Local Ollama Models** (Optional, for local runs):
   ```bash
   ollama pull llama3.1:latest
   ollama pull mistral:7b
   ollama pull phi4-mini
   ollama pull qwen3:8b
   ollama pull gemma3:4b
   ```

---

## 💻 Usage & Execution Commands

### 1. Resource & Dataset Validation

Validate dataset availability (automated isolated subprocess download from Hugging Face for `ag_news`, `squad_v2`, and `cnn_dailymail`) and model availability:

```bash
python3 main.py --validate-resources
```

### 2. Single Task / Validation Run

Execute a small sample validation run on a single task and model:

```bash
python3 main.py --provider ollama --model llama3.1:latest --task text_classification --sample-size 3
```

### 3. Campaign Matrix Execution

Launch a multi-model, multi-task, multi-shot campaign sweep:

```bash
# Run full campaign across all tasks using Ollama
python3 main.py --run-campaign --provider ollama --sample-size 10

# Filter campaign execution by specific model
python3 main.py --run-campaign --provider ollama --model phi4-mini --sample-size 10

# Filter campaign execution by specific task
python3 main.py --run-campaign --provider ollama --task question_answering --sample-size 10
```

---

## 🧪 Testing

The repository includes a 93-test unit and integration test suite:

```bash
python3 -m unittest discover -s tests
```

---

## 🛡️ Reproducibility Notice

All experiments are fully deterministic when run with fixed random seeds (`seed=42`, trial seeds `[42, 43, 44]`) and `temperature=0.0`. Experiment configurations and output checksums are automatically saved in `reproducibility_manifest.json` for every run.
