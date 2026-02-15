# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models

## Abstract

Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

## Getting Started

### Prerequisites

- Python 3.11+
- GitHub CLI (`gh`) for the clone command below
- `OPENAI_API_KEY` (required for StrongREJECT grading used during runs)
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

### Basic Usage

#### Quick Start
```bash
# Default run (llama3-8b target + grok-3-mini-beta attacker)
python Code/app.py --sample-size 10

# Use an OpenAI attacker model instead of Grok
python Code/app.py --target-model llama3-8b --attacker-model gpt4o-mini --sample-size 10

# Multi-source prompts with custom mixing
python Code/app.py --target-model llama3-8b --prompt-sources advbench harmbench --prompt-mix equal
```

#### Advanced Usage
```bash
# Custom attacker/target temperatures and turn limit
python Code/app.py --target-model llama3-8b --attacker-temp 0.8 --target-temp 0.7 --turns 8

# Disable adaptive temperature learning
python Code/app.py --target-model llama3-8b --no-temperature-learning --sample-size 50

# Baseline mode (single turn, no pattern learning, no temperature learning)
python Code/app.py --target-model llama3-8b --baseline-mode --sample-size 50

# High-throughput parallel processing
python Code/app.py --target-model llama3-8b --workers 20 --sample-size 200
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target-model` | Target model key to attack | `llama3-8b` |
| `--target-temp` | Temperature for target model | `0.7` |
| `--attacker-temp` | Temperature for attacker model | `0.7` |
| `--attacker-model` | Attacker model key for rewriting | `grok-3-mini-beta` |
| `--turns` | Maximum conversation turns | `10` |
| `--threshold` | StrongREJECT threshold for success | `0.5` |
| `--memory` | Enable target conversation memory | `True` |
| `--sample-size` | Number of prompts to sample | `5` |
| `--use_pattern_memory` | Enable pattern-enhanced system prompts at startup | `False` |
| `--workers` | Parallel worker threads | `10` |
| `--verbose` | Verbosity level (`0`, `1`, `2`) | `2` |
| `--prompts` | Prompt CSV path (single-source mode) | `Files/advbench_adversarial_prompts.csv` |
| `--harmbench-prompts` | HarmBench prompt CSV path | `Files/harmbench_adversarial_prompts.csv` |
| `--prompt-sources` | Prompt sources (`advbench`, `harmbench`) | `advbench harmbench` |
| `--prompt-mix` | Prompt mix strategy (`equal`, `advbench_heavy`, `harmbench_heavy`, `custom`) | `equal` |
| `--system-prompt` | Initial system prompt file path | `Files/system_prompt.md` |
| `--followup-prompt` | Follow-up system prompt file path | `Files/system_prompt_followup.md` |
| `--logs-dir` | Directory for output logs | `Logs/` |
| `--save-temp` | Save intermediate files/results | `False` |
| `--no-patterns` | Disable pattern memory | `False` |
| `--no-temperature-learning` | Disable temperature adjustments/learning | `False` |
| `--baseline-mode` | Baseline mode (single-turn, no pattern memory, no temperature learning) | `False` |

Notes:
- `--memory` is currently enabled by default (`True`) and there is no corresponding `--no-memory` flag.
- Pattern behavior is primarily controlled by `--no-patterns`; `--use_pattern_memory` controls early initialization used for prompt enhancement.

## Project Structure

```
AutoAdv/
├── Code/                                   # Core framework modules
│   ├── app.py                              # Main entry point and orchestration
│   ├── attacker_llm.py                     # Attacker model implementation
│   ├── target_llm.py                       # Target model implementation
│   ├── conversation.py                     # Multi-turn conversation management
│   ├── pattern_manager.py                  # Pattern learning and storage
│   ├── temperature_manager.py              # Dynamic temperature adjustment
│   ├── config.py                           # Configuration and model definitions
│   └── ...                                 # Additional utility modules
├── Files/                                  # Data and configuration files
│   ├── advbench_adversarial_prompts.csv    # AdvBench dataset
│   ├── harmbench_adversarial_prompts.csv   # HarmBench dataset
│   ├── system_prompt.md                    # Base attacker system prompt
│   ├── system_prompt_followup.md           # Follow-up system prompt
│   └── successful_patterns.json            # Learned attack patterns
├── Helpers/                                # Evaluation and utility scripts
│   ├── strongreject_evaluator.py           # StrongREJECT grading module (imported by Code/conversation.py)
│   └── strongreject_evaluator_prompt.txt
├── Logs/                                   # Experiment results
│   ├── Full Learning-100 prompts-Llama-3.1-8B/
│   ├── Baseline-100 prompts-Llama-3.1-8B/
│   └── ...                                 # Additional experiment results
└── requirements.txt                        # Python dependencies
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
- **Baseline Mode**: 70% ASR (vs 95% with full features)

### Pattern Learning Effectiveness
The system automatically learns and applies 20+ attack techniques including:
- Educational framing and research context
- Hypothetical scenarios and roleplaying
- Multi-stage breakdowns and confusion techniques
- Domain shifting and cognitive bias exploitation
- And many more sophisticated techniques

## Evaluation

### StrongREJECT Integration
AutoAdv uses the StrongREJECT evaluator for objective assessment of jailbreak success.  
Grading is called automatically during runs from `Code/conversation.py` via `evaluate_with_strongreject(...)`, which imports `Helpers/strongreject_evaluator.py`.

```bash
# Main run command (includes StrongREJECT grading automatically)
python Code/app.py --target-model llama3-8b --sample-size 10
```

### Metrics Tracked
- **Attack Success Rate (ASR)**: Percentage of successful jailbreaks
- **Cumulative ASR**: Success rate by conversation turn
- **Per-Turn Evaluation Score**: StrongREJECT score for each target response
- **Token/Cost Accounting**: Request and response tokens/costs per turn and totals
- **Timing Metrics**: Per-turn response times and per-prompt processing time
- **Pattern Stats**: Learned pattern statistics when pattern memory is enabled

## Advanced Features

### Pattern Memory Reset
```bash
# Reset learned patterns
python Code/reset_patterns.py --confirm
```

### Custom System Prompts
Modify `Files/system_prompt.md` and `Files/system_prompt_followup.md` to customize attacker behavior.

### Multi-Source Prompt Mixing
- **`equal`**: Equal sampling from all selected sources
- **`advbench_heavy`**: 70% AdvBench, 30% others
- **`harmbench_heavy`**: 70% HarmBench, 30% others
- **`custom`**: All prompts combined, then sampled (if `--sample-size` is set)

## Contributing

Contributions are welcome via pull requests and issues.  
There is currently no separate `CONTRIBUTING.md` in this repository.

## Disclaimer

This repository is intended for research and educational purposes only. The framework is designed to evaluate and improve the safety of LLMs. It may generate or reference harmful or sensitive content. Use responsibly and in accordance with applicable laws and ethical guidelines.

## License

The project is stated as [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en).  
Note: this repository currently does not include a top-level `LICENSE` file.