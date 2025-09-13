# AutoAdv Codebase Overview

## Core Architecture

AutoAdv is a self-improving adversarial prompt generation system that learns from successful jailbreaking attempts to enhance future attacks.

### Main Components

#### 1. **app.py** - Main Entry Point
- **Purpose**: Orchestrates the entire jailbreaking process
- **Key Functions**:
  - `load_multi_source_prompts()`: Loads malicious prompts from datasets
  - `load_system_prompts()`: Loads and enhances system prompts with learned patterns
  - `process_prompt()`: Processes individual prompts through the attack pipeline
  - `main()`: Entry point with command-line argument parsing

#### 2. **pattern_manager.py** - Pattern Learning System
- **Purpose**: Learns, stores, and applies successful jailbreaking techniques
- **Key Functions**:
  - `start_tracking()`: Begins tracking a new attack attempt
  - `record_success()`: Records successful techniques when jailbreak occurs
  - `generate_system_prompt_hints()`: Enhances system prompts with learned patterns
  - `rank_and_select_attack()`: Ranks historical attacks by effectiveness
  - `analyze_conversation()`: Extracts patterns from conversation logs

#### 3. **conversation.py** - Attack Orchestration
- **Purpose**: Manages multi-turn conversations between attacker and target
- **Key Functions**:
  - `multi_turn_conversation()`: Main attack loop
  - `evaluate_with_strongreject()`: Evaluates if jailbreak was successful

#### 4. **attacker_llm.py** - Attacker Model
- **Purpose**: Rewrites malicious prompts using learned techniques
- **Key Functions**:
  - `rewrite()`: Rewrites initial malicious prompts
  - `converse()`: Generates follow-up prompts for multi-turn attacks
  - `adjust_temperature_smart()`: Dynamically adjusts generation temperature

#### 5. **target_llm.py** - Target Model
- **Purpose**: The model being attacked (e.g., Llama, GPT)
- **Key Functions**:
  - `respond()`: Generates responses to rewritten prompts

#### 6. **temperature_manager.py** - Temperature Control
- **Purpose**: Sophisticated temperature adjustment based on success patterns
- **Key Functions**:
  - `adjust_temperature()`: Adjusts temperature based on success scores
  - `recommend_strategy()`: Recommends temperature strategies

## Data Flow

```
1. Load malicious prompts from datasets (AdvBench, HarmBench)
   ↓
2. Initialize PatternManager with learned techniques from successful_patterns.json
   ↓
3. For each prompt:
   a. AttackerLLM rewrites using system prompt enhanced with learned patterns
   b. TargetLLM responds to rewritten prompt
   c. StrongREJECT evaluates if jailbreak was successful
   d. If successful: PatternManager learns new techniques
   ↓
4. Save updated patterns for future runs
```

## Key Files

### Core System Files
- **app.py**: Main entry point and orchestration
- **conversation.py**: Attack conversation management
- **pattern_manager.py**: Pattern learning and storage
- **attacker_llm.py**: Attacker model implementation
- **target_llm.py**: Target model implementation

### Configuration Files
- **config.py**: System configuration and model definitions
- **logging_utils.py**: Logging system with color coding

### Utility Files
- **token_calculator.py**: Token counting and cost calculation
- **technique_analyzer.py**: Prompt categorization
- **utils.py**: General utility functions

### Data Files
- **successful_patterns.json**: Learned patterns and successful prompts
- **system_prompt.md**: Base system prompt for attacker
- **system_prompt_followup.md**: Follow-up system prompt

### Standalone Utilities
- **reset_patterns.py**: Manual pattern memory reset tool
- **token_utils.py**: Alternative token calculation (unused)

## Pattern Learning System

The core innovation is the pattern learning system that:

1. **Detects Techniques**: Uses keyword matching to identify 20+ attack techniques
2. **Stores Success Data**: Records successful prompts with metadata
3. **Ranks Effectiveness**: Uses composite scoring to rank historical attacks
4. **Enhances Prompts**: Injects learned patterns into system prompts
5. **Adapts Dynamically**: Learns from each successful attack

## Unused Functions

Several functions are defined but not used in the main codebase:
- PatternManager: `get_successful_temperatures()`, `get_successful_strategies()`, `enhance_followup_prompts()`
- TokenCalculator: `estimate_prompt_cost()`, `format_cost()`
- Utils: `show_stats()`
- TokenUtils: Entire module (alternative implementation)

These are kept for potential future features or alternative implementations.

## Usage

```bash
# Basic usage
python app.py --target_model llama3-8b --attacker_model gpt4o-mini --turns 5

# With custom settings
python app.py --target_model llama3-8b --sample_size 10 --turns 3 --verbosity detailed

# Reset learned patterns
python reset_patterns.py --confirm
```

## Key Features

- **Self-Improving**: Learns from successful attacks to enhance future attempts
- **Multi-Turn**: Supports multi-turn conversations for complex attacks
- **Pattern Learning**: Automatically detects and learns effective techniques
- **Temperature Management**: Dynamic temperature adjustment based on success
- **Model Agnostic**: Supports various attacker and target models
- **Comprehensive Logging**: Detailed logging with color coding and verbosity levels
