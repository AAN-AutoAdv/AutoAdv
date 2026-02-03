# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models

## Abstract

Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for your chosen models (OpenAI, Together, Anthropic, xAI)
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/nicksaban20/AutoAdv.git
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
   ANTHROPIC_API_KEY=your_anthropic_key
   XAI_API_KEY=your_xai_key
   ```

### Basic Usage

#### Quick Start
```bash
# Basic attack with explicit attacker model
python Code/app.py --target-model llama3-8b --attacker-model gpt4o-mini --sample-size 10

# Full learning mode with pattern memory
python Code/app.py --target-model llama3-8b --use_pattern_memory --sample-size 100 --turns 6

# Multi-source prompts with custom mixing
python Code/app.py --target-model llama3-8b --prompt-sources advbench harmbench --prompt-mix equal
```

#### Advanced Usage
```bash
# Custom temperature and strategy
python Code/app.py --target-model llama3-8b --attacker-temp 0.8 --target-temp 0.7 --turns 8

# Baseline mode (no pattern learning)
python Code/app.py --target-model llama3-8b --baseline-mode --sample-size 50

# High-throughput parallel processing
python Code/app.py --target-model llama3-8b --workers 20 --sample-size 200
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--target-model` | Target model to attack | `llama3-8b` |
| `--attacker-model` | Attacker model for rewriting | `grok-3-mini-beta` |
| `--sample-size` | Number of prompts to test | `5` |
| `--turns` | Maximum conversation turns | `10` |
| `--use_pattern_memory` | Enable pattern learning | `False` |
| `--prompt-sources` | Prompt datasets to use | `advbench harmbench` |
| `--prompt-mix` | How to mix prompt sources | `equal` |
| `--workers` | Parallel processing threads | `10` |
| `--verbose` | Logging verbosity (0-2) | `2` |
| `--baseline-mode` | Disable advanced features | `False` |

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
│   ├── strongreject_evaluator.py           # StrongREJECT evaluation
│   └── strongreject_evaluator_prompt.txt
├── Logs/                                   # Experiment results
│   ├── Full Learning-100 prompts-Llama-3.1-8B/
│   ├── Baseline-100 prompts-Llama-3.1-8B/
│   └── ...                                 # Additional experiment results
└── requirements.txt                        # Python dependencies
```

## Experimental Results

AutoAdv has been extensively tested across multiple models and configurations:

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

## Configuration

### Model Configuration
Edit `Code/config.py` to add new models or modify existing ones:

```python
TARGET_MODELS = {
    "your-model": {
        "name": "model/name",
        "api": "provider",
        "request_cost": 0.15,
        "response_cost": 0.60,
        "token_model": "gpt-4o-mini",
    }
}
```

### Pattern Learning Configuration
Customize pattern categories and analysis in `PATTERN_CONFIG`:

```python
PATTERN_CONFIG = {
    "enabled": True,
    "categories": [
        "educational_framing", "research_context",
        "hypothetical_scenario", "technical_analysis",
        # ... add your own categories
    ]
}
```

## Evaluation

### StrongREJECT Integration
AutoAdv uses the StrongREJECT evaluator for objective assessment of jailbreak success:

```bash
# Run StrongREJECT evaluation
python Helpers/strongreject_evaluator.py --input logs.csv --output results.json
```

### Metrics Tracked
- **Attack Success Rate (ASR)**: Percentage of successful jailbreaks
- **Cumulative ASR**: Success rate by conversation turn
- **Pattern Effectiveness**: Success rate by attack technique
- **Temperature Optimization**: Success rate by temperature settings
- **Model-Specific Performance**: ASR by target model

## Advanced Features

### Pattern Memory Reset
```bash
# Reset learned patterns
python Code/reset_patterns.py --confirm
```

### Custom System Prompts
Modify `Files/system_prompt.md` and `Files/system_prompt_followup.md` to customize attacker behavior.

### Multi-Source Prompt Mixing
- **Equal**: Equal sampling from all sources
- **AdvBench-heavy**: 70% AdvBench, 30% others
- **HarmBench-heavy**: 70% HarmBench, 30% others
- **Custom**: All prompts combined, then sampled

## Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## Disclaimer

This repository is intended for research and educational purposes only. The framework is designed to evaluate and improve the safety of LLMs. It may generate or reference harmful or sensitive content. Use responsibly and in accordance with applicable laws and ethical guidelines.

## License

Our source code is under [CC-BY-NC 4.0 license.](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
