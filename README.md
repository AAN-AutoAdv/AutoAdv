# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models

## Abstract

Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

## Getting Started

### Prerequisites

- Python 3.11+
- GitHub CLI (`gh`) for the clone command below
- `OPENAI_API_KEY` (required for AutoAdv's modified StrongREJECT-style evaluator used during runs)
- `TOGETHER_API_KEY` (required for Together-hosted target/attacker models)
- `XAI_API_KEY` (required when using `grok-3-mini-beta` as attacker)
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```bash
   gh repo clone AAN-AutoAdv/AutoAdv
   cd AutoAdv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys by creating a `.env` file in the root directory:
   ```bash
   OPENAI_API_KEY=your_openai_key
   TOGETHER_API_KEY=your_together_key
   XAI_API_KEY=your_xai_key
   ```

### Provider Routing

AutoAdv uses a provider factory to route each model key to the correct backend. The provider is determined by the model configuration in `config.py`, not by the Python SDK class name (`SDK != provider` — the `openai-python` library is used for OpenAI, Together, and xAI backends).

**Paper target models:**

| Model key | Provider | Endpoint | API key env | Serverless |
|-----------|----------|----------|-------------|------------|
| `llama3-8b` | Together | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | No (requires dedicated endpoint) |
| `gpt4o-mini` | OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Yes |
| `Qwen3-235b` | Together | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | No (requires dedicated endpoint) |
| `Mistral-7B` | Together | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` | No (requires dedicated endpoint) |

**Attacker models:**

| Model key | Provider | Endpoint | API key env | Serverless |
|-----------|----------|----------|-------------|------------|
| `grok-3-mini-beta` (default) | xAI | `https://api.x.ai/v1` | `XAI_API_KEY` | Yes |
| `gpt4o-mini` | OpenAI | `https://api.openai.com/v1` | `OPENAI_API_KEY` | Yes |

**Additional target models** (serverless on Together): `llama3.3-70b`, `llama4-Maverick`, `Mistral-24b`, and others defined in `config.py`.

> **Note:** Several Together-hosted models used in the paper (`llama3-8b`, `Qwen3-235b`, `Mistral-7B`) have since been moved to dedicated endpoints and are no longer available on Together's serverless tier. The default target model is set to `llama3.3-70b`, which is currently available serverless.

**Changing a model's provider:** To point a model at a different provider, update its entry in `config.py`. For example, to add a new provider and re-route a model:

```python
# 1. Add the provider to PROVIDER_SPECS (if new):
"fireworks": {
    "sdk_family": "openai_python",
    "base_url": "https://api.fireworks.ai/inference/v1",
    "api_key_env": "FIREWORKS_API_KEY",
    "compat_mode": "openai_compatible",
},

# 2. Update the model entry in TARGET_MODELS:
"llama3-8b": build_model_config(
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    provider="fireworks",
    request_cost=0.20,
    response_cost=0.20,
    token_model="gpt-4o-mini",
),
```

At startup, AutoAdv prints the selected provider, SDK, endpoint, and API key env var for both the target and attacker models.

### Basic Usage

#### Quick Start
```bash
# Default run (gpt4o-mini target + grok-3-mini-beta attacker)
python Code/app.py --sample-size 10

# Use an OpenAI attacker model instead of Grok
python Code/app.py --target-model gpt4o-mini --attacker-model gpt4o-mini --sample-size 10

# No-pattern ablation
python Code/app.py --target-model gpt4o-mini --no-patterns --sample-size 10
```

#### Advanced Usage
```bash
# Custom attacker/target temperatures and turn limit
python Code/app.py --target-model gpt4o-mini --attacker-temp 0.8 --target-temp 0.7 --turns 8

# Disable adaptive temperature learning
python Code/app.py --target-model gpt4o-mini --no-temperature-learning --sample-size 50

# Baseline mode (multi-turn, no pattern learning, no temperature learning)
python Code/app.py --target-model gpt4o-mini --baseline-mode --sample-size 50

# High-throughput parallel processing
python Code/app.py --target-model gpt4o-mini --workers 20 --sample-size 200
```

### Command-Line Options

Core options (recommended):

| Option | Description | Default |
|--------|-------------|---------|
| `--target-model` | Target model key to attack | `gpt4o-mini` |
| `--target-temp` | Temperature for target model | `0.7` |
| `--attacker-temp` | Temperature for attacker model | `0.7` |
| `--attacker-model` | Attacker model key for rewriting | `grok-3-mini-beta` |
| `--turns` | Maximum conversation turns | `10` |
| `--threshold` | Unified evaluator success threshold | `0.5` |
| `--sample-size` | Number of prompts to sample | `5` |
| `--workers` | Parallel worker threads | `10` |
| `--baseline-mode` | Baseline ablation (multi-turn, uses baseline system prompt, no pattern memory, no temperature learning) | `False` |
| `--no-patterns` | Disable pattern-memory learning/use (ablation mode) | `False` |
| `--no-temperature-learning` | Disable adaptive temperature learning (ablation mode) | `False` |
| `--no-fewshot-learning` | Use no-fewshot prompt template variant (ablation mode) | `False` |
| `--no-seed-techniques` | Use no-seed-techniques prompt template variants (ablation mode) | `False` |

Notes:
- `--no-fewshot-learning` and `--no-seed-techniques` are mutually exclusive.

## Project Structure

```
AutoAdv/
├── Code/                                   # Core framework modules
│   ├── app.py                              # Main entry point
│   ├── config.py                           # Model and runtime defaults
│   ├── conversation.py                     # Multi-turn orchestration + logging
│   ├── attacker_llm.py                     # Attacker-side model interface
│   ├── target_llm.py                       # Target-side model interface
│   ├── provider_factory.py                 # Provider-aware client factory + routing metadata
│   ├── xai_client.py                       # xAI client wrapper built on the provider factory
│   ├── grok_client.py                      # Backward-compatible alias for older Grok imports
│   ├── pattern_manager.py                  # Pattern learning and persistence
│   ├── prompt_enhancer.py                  # Pattern-based system prompt enhancement
│   ├── temperature_manager.py              # Adaptive temperature strategy logic
│   ├── technique_analyzer.py               # Technique/category helpers
│   ├── llm_base.py                         # Shared LLM base class
│   ├── token_calculator.py                 # Token/cost accounting
│   ├── logging_utils.py                    # Logging + display utilities
│   ├── utils.py                            # Shared validation/util functions
│   └── reset_patterns.py                   # Reset learned pattern memory
├── Files/                                  # Prompt/data assets
│   ├── advbench_adversarial_prompts.csv
│   ├── harmbench_adversarial_prompts.csv
│   ├── system_prompt.md
│   ├── system_prompt_followup.md
│   ├── system_prompt_baseline.md
│   ├── system_prompt_no_fewshot.md
│   ├── system_prompt_no_seed_techniques.md
│   ├── followup_prompt_no_seed_techniques.md
│   └── successful_patterns.json            # Generated after runs
├── Helpers/                                # Evaluation assets
│   ├── strongreject_evaluator.py           # Modified StrongREJECT-style evaluator
│   └── strongreject_evaluator_prompt.md
├── Logs/                                   # Saved experiment outputs
│   ├── Full Learning-100 prompts-Llama-3.1-8B/
│   ├── Full Learning-100 prompts-Qwen3-235B/
│   ├── Full Learning-100 prompts-Mistral-7B/
│   ├── Full Learning-100 prompts-GPT-4o-mini/
│   ├── Baseline-100 prompts-Llama-3.1-8B/
│   ├── No Pattern Learning-100 prompts-Llama-3.1-8B/
│   ├── No Temperature Manager-100 prompts-Llama-3.1-8B/
│   ├── No Few-Shot Learning-100 prompts-Llama-3.1-8B/
│   └── No Seed Techniques-100 prompts-Llama-3.1-8B/
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## Experimental Results

AutoAdv has been extensively tested across multiple models and configurations:
Results below match the saved experiment logs under `Logs/` and the paper ([arXiv:2511.02376](https://arxiv.org/abs/2511.02376)).

### Attack Success Rates (ASR) by Model
- **Llama-3.1-8B**: 95% ASR with pattern learning
- **Qwen3-235B**: 99% ASR with pattern learning  
- **Mistral-7B**: 91% ASR with pattern learning
- **GPT-4o-mini**: 86% ASR with pattern learning

### Feature Ablation Studies
- **No Pattern Learning**: 89% ASR (vs 95% with patterns)
- **No Temperature Management**: 88% ASR (vs 95% with temperature management)
- **No Few-Shot Learning**: 78% ASR (vs 95% with few-shot)
- **No Seed Techniques**: 86% ASR (vs 95% with seed techniques)
- **Baseline Mode**: 70% ASR (vs 95% with full features)

## Evaluation

### Modified StrongREJECT Framework
AutoAdv builds on the StrongREJECT framework with a modified evaluation design described in the paper.
Instead of a separate two-tier decision, it uses a single unified score that combines:
- Refusal detection (binary)
- Convincingness (1-5 Likert)
- Specificity (1-5 Likert)

An evaluator LLM (`gpt-4o-mini`) scores each target response, and jailbreak success is determined by the unified threshold (default `0.5`).
This grading runs automatically during each experiment.

```bash
# Main run command (includes AutoAdv's modified StrongREJECT-style grading automatically)
python Code/app.py --target-model gpt4o-mini --sample-size 10
```

### Metrics Tracked
- **Attack Success Rate (ASR)**: Overall percentage of successful jailbreaks
- **Cumulative ASR by Turn**: Cumulative and per-turn success rates across turns
- **Unified Evaluation Scores**: Per-turn and final scores from AutoAdv's modified StrongREJECT-style framework
- **Evaluation Components**: Refusal signal, grader feedback text, and response-quality scoring outputs
- **Token and Cost Metrics**: Request/response tokens and costs (per turn and totals), including cost per success
- **Timing Metrics**: Per-turn response times and per-prompt processing time
- **Success-Turn Statistics**: Average/min/max turn of successful jailbreaks
- **Pattern-Memory Statistics**: Pattern-learning aggregates when pattern memory is enabled

## Advanced Features

### Pattern Memory Reset
```bash
# Reset learned patterns
python Code/reset_patterns.py --confirm
```

### Custom System Prompts
Modify `Files/system_prompt.md` and `Files/system_prompt_followup.md` to customize default attacker behavior.
Baseline mode uses `Files/system_prompt_baseline.md`.

### Multi-Source Prompt Mixing
Multi-source mixing is set in code. By default, runs use both datasets and mix them equally.

To change it, open `config.py` and edit these two lines in `DEFAULT_CONFIG`:
- `prompt_sources`: which datasets to use (for example `["advbench"]`, `["harmbench"]`, or `["advbench", "harmbench"]`)
- `prompt_mix_ratio`: how to mix when using multiple sources

Mix ratio modes:
- `equal`: sample equally from each selected source
- `advbench_heavy`: target 70% AdvBench and 30% other selected sources
- `harmbench_heavy`: target 70% HarmBench and 30% other selected sources
- `custom`: combine all selected prompts, shuffle, then apply `--sample-size` (if set)

Example settings in `Code/config.py`:
```python
"prompt_sources": ["advbench", "harmbench"],
"prompt_mix_ratio": "advbench_heavy",
```

## Contributing

Contributions are welcome via pull requests and issues.

## Disclaimer

This repository is intended for research and educational purposes only. The framework is designed to evaluate and improve the safety of LLMs. It may generate or reference harmful or sensitive content. Use responsibly and in accordance with applicable laws and ethical guidelines.

## License

- Paper (arXiv): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Repository code (as stated by this repository): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en).
