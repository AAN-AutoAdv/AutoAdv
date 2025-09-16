# AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models

## Abstract

Large Language Models (LLMs) are susceptible to jailbreaking attacks, in which carefully crafted inputs bypass safety guardrails and elicit harmful outputs. We present AutoAdv, an automated framework for dynamic, multi-turn jailbreaking that both generates adversarial prompts and systematically evaluates vulnerabilities in LLM safety mechanisms. Our method leverages an attacker LLM that rewrites malicious prompts through strategic paraphrasing, hyperparameter tuning, and retrieval from a database of historically effective attack patterns. We evaluate the attack success rate (ASR) with an optimal scoring framework across multiple interaction turns. Extensive empirical testing on state-of-the-art commercial and open-source models reveals significant weaknesses, with AutoAdv achieving an ASR of 95% on Llama-3.1-8B. These findings indicate that current safety mechanisms remain susceptible to sophisticated, multi-turn attacks.

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
