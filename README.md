# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models

## Overview

AutoAdv is a research framework for systematically evaluating and exposing vulnerabilities in the safety mechanisms of Large Language Models (LLMs) through automated, multi-turn adversarial prompting. By leveraging a parametric attacker LLM, AutoAdv generates semantically disguised malicious prompts and iteratively refines its attack strategy based on the target model's responses. This enables comprehensive, realistic assessment of LLM safety in conversational contexts.

**Key Features:**

- **Automated Multi-Turn Jailbreaking:** Dynamically generates and adapts adversarial prompts over multiple conversational turns to bypass LLM safety guardrails.
- **Adaptive Learning:** Analyzes failed jailbreak attempts and iteratively improves attack strategies using techniques such as roleplaying, misdirection, and contextual manipulation.
- **Prompt Rewriting Techniques:** Employs framing, contextualization, obfuscation, format specification, and subtle reframing to maximize attack success.
- **Few-Shot Learning:** Utilizes a curated set of human-authored jailbreak examples to guide and enhance adversarial prompt generation.
- **Dynamic Hyperparameter Tuning:** Adjusts temperature and system prompts in real time based on observed attack performance.
- **Objective Evaluation:** Uses the StrongREJECT framework to quantitatively assess attack success rates (ASR) and model safety.

## Research Impact

AutoAdv reveals that state-of-the-art LLMs—including ChatGPT, Llama, and DeepSeek—remain highly vulnerable to sophisticated, automated multi-turn attacks. Our experiments demonstrate jailbreak success rates of up to 86% for harmful content generation, with multi-turn attacks increasing success rates by up to 51% compared to single-turn approaches. These findings highlight the urgent need for more robust, context-aware LLM safety mechanisms.

## Getting Started

### Prerequisites

- Python 3.11+
- Required packages listed in `requirements.txt`

### Installation

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### 1. Prepare Adversarial Prompts

- Place your initial adversarial prompts in `Files/advbench_adversarial_prompts.csv`.
- (Optional) Add successful human-authored jailbreak examples to seed the attacker.

#### 2. Configure the Attack Pipeline

- Edit configuration parameters in `Code/config.py` (e.g., model selection, temperature, number of turns).
- System prompts and rewriting strategies can be customized in `Files/system_prompt.md` and `Files/system_prompt_followup.md`.

#### 3. Run the Attack Framework

- Execute the main attack script (e.g., `app.py` or your custom pipeline script) to start automated multi-turn adversarial prompting and evaluation.
- Results, including attack logs and success metrics, will be saved in the `Logs/` directory.

#### 4. Evaluate Results

- Use the StrongREJECT evaluator (`Helpers/strongreject_evaluator.py`) to objectively assess the safety of target LLM responses.
- Analyze logs and CSV files in `Logs/` for detailed attack outcomes and ASR statistics.

## Project Structure

- `Code/` — Core framework modules (attacker, target LLMs, prompt rewriting, temperature management, etc.)
- `Files/` — Prompt datasets, system prompts, and pattern storage
- `Helpers/` — StrongREJECT evaluation scripts and prompts
- `Logs/` — Experiment results and attack logs

## Citation

If you use AutoAdv in your research, please cite our paper:

```
@article{AutoAdv2025,
  title={AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models},
  author={Aashray Reddy and Andrew Zagula and Nicholas Saban},
  year={2025},
  note={https://github.com/nicksaban20/AutoAdv}
}
```

## Disclaimer

This repository is intended for research and educational purposes only. The framework is designed to evaluate and improve the safety of LLMs. It may generate or reference harmful or sensitive content. Use responsibly and in accordance with applicable laws and ethical guidelines.

## License

Our source code is under [CC-BY-NC 4.0 license.](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
