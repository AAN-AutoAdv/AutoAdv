import os
import sys
import random
import time
import concurrent.futures
import argparse
import pandas as pd
from tqdm import tqdm

# Add Helpers directory to path
sys.path.insert(
    1,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Helpers"
    ),
)

# Import modules
from config import (
    TARGET_MODELS,
    DEFAULT_PATHS,
    DEFAULT_CONFIG,
    VERBOSE_LEVEL,
    VERBOSE_NORMAL,
    VERBOSE_DETAILED,
    VERBOSE_NONE,
    VERBOSE_LEVEL_NAMES,
)
from logging_utils import log, display_config, ensure_directory_exists
from utils import is_model_available, get_next_filename
from attacker_llm import AttackerLLM
from target_llm import TargetLLM
from conversation import multi_turn_conversation, save_conversation_log
from pattern_manager import PatternManager


def load_prompts(filepath, sample_size=None):
    """
    Load prompts from CSV file with optional sampling.

    Args:
        filepath (str): Path to the CSV file
        sample_size (int, optional): Number of prompts to randomly sample

    Returns:
        list: List of prompts
    """
    log(f"Loading prompts from {filepath}", "info")

    try:
        # Read CSV file
        df_prompts = pd.read_csv(filepath)
        all_prompts = df_prompts["prompt"].tolist()

        # Handle sampling
        if sample_size and sample_size < len(all_prompts):
            prompts = random.sample(all_prompts, sample_size)
            log(
                f"Randomly selected {sample_size} prompts out of {len(all_prompts)}",
                "info",
            )
        else:
            prompts = all_prompts
            if sample_size:
                log(
                    f"Sample size {sample_size} >= total prompts {len(all_prompts)}. Using all prompts.",
                    "info",
                )

        return prompts
    except Exception as e:
        log(f"Error loading prompts: {e}", "error")
        return []


def load_multi_source_prompts(config):
    """
    Load prompts from multiple sources (AdvBench, HarmBench) based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing prompt sources and mix ratio
        
    Returns:
        list: Combined list of prompts from all specified sources
    """
    all_prompts = []
    prompt_sources = config.get("prompt_sources", ["advbench"])
    mix_ratio = config.get("prompt_mix_ratio", "equal")
    sample_size = config.get("sample_size")
    
    # Define source mappings
    source_files = {
        "advbench": config.get("adversarial_prompts", DEFAULT_PATHS["adversarial_prompts"]),
        "harmbench": config.get("harmbench_prompts", DEFAULT_PATHS["harmbench_prompts"])
    }
    
    # Load prompts from each source
    source_prompts = {}
    for source in prompt_sources:
        if source in source_files:
            filepath = source_files[source]
            if os.path.exists(filepath):
                prompts = load_prompts(filepath, sample_size=None)  # Load all first
                if prompts:
                    source_prompts[source] = prompts
                    log(f"Loaded {len(prompts)} prompts from {source}", "info")
            else:
                log(f"Warning: {source} file not found at {filepath}", "warning")
    
    if not source_prompts:
        log("No prompt sources could be loaded", "error")
        return []
    
    # Combine prompts based on mix ratio
    if mix_ratio == "equal":
        # Equal sampling from each source - use smaller of (min_size, sample_size/num_sources)
        num_sources = len(source_prompts)
        if sample_size:
            # If sample_size is specified, divide equally among sources
            prompts_per_source = sample_size // num_sources
            min_size = min(prompts_per_source, min(len(prompts) for prompts in source_prompts.values()))
        else:
            # If no sample_size, use the smallest available dataset size
            min_size = min(len(prompts) for prompts in source_prompts.values())
        
        for source, prompts in source_prompts.items():
            # Randomly sample equal number from each source
            selected = random.sample(prompts, min_size)
            all_prompts.extend([(prompt, source) for prompt in selected])
            log(f"Randomly selected {len(selected)} prompts from {source} (equal mix)", "info")
    
    elif mix_ratio == "advbench_heavy":
        # 70% AdvBench, 30% others
        total_needed = sample_size if sample_size else 100  # Default to 100 if no sample_size
        advbench_count = int(total_needed * 0.7)
        others_count = total_needed - advbench_count
        
        if "advbench" in source_prompts:
            # Randomly sample from AdvBench
            selected_count = min(advbench_count, len(source_prompts["advbench"]))
            selected = random.sample(source_prompts["advbench"], selected_count)
            all_prompts.extend([(prompt, "advbench") for prompt in selected])
            log(f"Randomly selected {selected_count} prompts from advbench (70%)", "info")
            
            # Remaining from other sources
            other_sources = {k: v for k, v in source_prompts.items() if k != "advbench"}
            if other_sources:
                prompts_per_other = others_count // len(other_sources)
                for source, prompts in other_sources.items():
                    count = min(prompts_per_other, len(prompts))
                    selected = random.sample(prompts, count)
                    all_prompts.extend([(prompt, source) for prompt in selected])
                    log(f"Randomly selected {count} prompts from {source} (30%)", "info")
    
    elif mix_ratio == "harmbench_heavy":
        # 70% HarmBench, 30% others  
        total_needed = sample_size if sample_size else 100  # Default to 100 if no sample_size
        harmbench_count = int(total_needed * 0.7)
        others_count = total_needed - harmbench_count
        
        if "harmbench" in source_prompts:
            # Randomly sample from HarmBench
            selected_count = min(harmbench_count, len(source_prompts["harmbench"]))
            selected = random.sample(source_prompts["harmbench"], selected_count)
            all_prompts.extend([(prompt, "harmbench") for prompt in selected])
            log(f"Randomly selected {selected_count} prompts from harmbench (70%)", "info")
            
            # Remaining from other sources
            other_sources = {k: v for k, v in source_prompts.items() if k != "harmbench"}
            if other_sources:
                prompts_per_other = others_count // len(other_sources)
                for source, prompts in other_sources.items():
                    count = min(prompts_per_other, len(prompts))
                    selected = random.sample(prompts, count)
                    all_prompts.extend([(prompt, source) for prompt in selected])
                    log(f"Randomly selected {count} prompts from {source} (30%)", "info")
    
    else:  # "custom" or fallback
        # Combine all prompts, then randomly sample if needed
        for source, prompts in source_prompts.items():
            all_prompts.extend([(prompt, source) for prompt in prompts])
            log(f"Added all {len(prompts)} prompts from {source} (custom mix)", "info")
    
    # Shuffle the combined prompts to ensure random order
    random.shuffle(all_prompts)
    
    # Apply final sampling only for custom mix ratio or if we have more than requested
    if mix_ratio == "custom" and sample_size and sample_size < len(all_prompts):
        all_prompts = random.sample(all_prompts, sample_size)
        log(f"Final sampling: randomly selected {sample_size} prompts from combined sources", "info")
    
    # Extract just the prompts (remove source tags for now, but could be useful for logging)
    final_prompts = [prompt for prompt, source in all_prompts]
    
    log(f"Total prompts loaded: {len(final_prompts)}", "success")
    return final_prompts


def load_system_prompts(initial_prompt_path, followup_prompt_path=None, pattern_manager=None, target_model=None):
    """
    Load system prompts from files and enhance them with learned patterns.

    Args:
        initial_prompt_path (str): Path to the initial system prompt
        followup_prompt_path (str, optional): Path to the followup system prompt
        pattern_manager (PatternManager, optional): Pattern manager for enhancing prompts
        target_model (str, optional): Target model for model-specific enhancements

    Returns:
        tuple: (initial_prompt, followup_prompt)
    """
    # Load initial prompt
    try:
        with open(initial_prompt_path, "r") as f:
            initial_prompt = f.read()
    except Exception as e:
        log(f"Error loading initial system prompt: {e}", "error")
        initial_prompt = None

    # Load followup prompt if available
    followup_prompt = None
    if followup_prompt_path and os.path.exists(followup_prompt_path):
        try:
            with open(followup_prompt_path, "r") as f:
                followup_prompt = f.read()
        except Exception as e:
            log(f"Error loading followup system prompt: {e}", "error")

    # Enhance prompts with learned patterns if available and enabled
    if pattern_manager and initial_prompt:
        # Check if pattern enhancement is enabled (default: True for backward compatibility)
        enhance_enabled = getattr(pattern_manager, '_enhance_enabled', True)
        if enhance_enabled:
            initial_prompt = enhance_prompt_with_patterns(initial_prompt, pattern_manager, target_model, "initial")
            if followup_prompt:
                followup_prompt = enhance_prompt_with_patterns(followup_prompt, pattern_manager, target_model, "followup")

    return initial_prompt, followup_prompt


def enhance_prompt_with_patterns(base_prompt, pattern_manager, target_model=None, prompt_type="initial"):
    """
    Enhance a system prompt with learned successful patterns.
    
    Args:
        base_prompt (str): The base system prompt
        pattern_manager (PatternManager): Pattern manager with learned patterns
        target_model (str, optional): Target model for model-specific patterns
        prompt_type (str): Type of prompt ("initial" or "followup")
    
    Returns:
        str: Enhanced prompt with pattern insights
    """
    try:
        # Generate pattern-based hints
        pattern_hints = pattern_manager.generate_system_prompt_hints()
        
        # Get model-specific insights if target model provided
        model_insights = ""
        if target_model:
            model_insights = generate_model_specific_insights(pattern_manager, target_model)
        
        # Get successful temperature recommendations
        temp_insights = generate_temperature_insights(pattern_manager)
        
        # Build enhancement section
        enhancement = ""
        
        if pattern_hints:
            enhancement += f"\n\n## 🧠 Learned Successful Techniques:\n{pattern_hints}"
        
        if model_insights:
            enhancement += f"\n\n## 🎯 Model-Specific Insights for {target_model}:\n{model_insights}"
        
        if temp_insights:
            enhancement += f"\n\n## 🌡️ Temperature Optimization:\n{temp_insights}"
        
        # Add recent successful examples for followup prompts
        if prompt_type == "followup":
            recent_examples = get_recent_successful_examples(pattern_manager)
            if recent_examples:
                enhancement += f"\n\n## 📈 Recent Successful Patterns:\n{recent_examples}"
        
        return base_prompt + enhancement
        
    except Exception as e:
        log(f"Error enhancing prompt with patterns: {e}", "warning")
        return base_prompt  # Return original prompt if enhancement fails


def generate_model_specific_insights(pattern_manager, target_model):
    """Generate insights specific to the target model."""
    try:
        # Get model-specific success patterns
        model_patterns = pattern_manager.patterns.get("success_by_model", {})
        if target_model not in model_patterns:
            return ""
        
        model_data = model_patterns[target_model]
        insights = []
        
        # Success rate insight
        if "success_rate" in model_data:
            rate = model_data["success_rate"]
            insights.append(f"- Success rate against {target_model}: {rate:.1%}")
        
        # Effective techniques for this model
        if "effective_techniques" in model_data:
            techniques = model_data["effective_techniques"]
            top_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_techniques:
                insights.append("- Most effective techniques:")
                for technique, count in top_techniques:
                    insights.append(f"  • {technique.replace('_', ' ').title()}: {count} successes")
        
        return "\n".join(insights) if insights else ""
        
    except Exception as e:
        log(f"Error generating model insights: {e}", "debug")
        return ""


def generate_temperature_insights(pattern_manager):
    """Generate temperature optimization insights."""
    try:
        # Analyze temperature effectiveness from successful patterns
        effective_prompts = pattern_manager.patterns.get("effective_prompts", [])
        if not effective_prompts:
            return ""
        
        # Extract temperatures from successful attempts
        temps = []
        for prompt_data in effective_prompts:
            if isinstance(prompt_data, dict) and "temperature" in prompt_data:
                temps.append(prompt_data["temperature"])
        
        if not temps:
            return ""
        
        # Calculate temperature statistics
        avg_temp = sum(temps) / len(temps)
        min_temp = min(temps)
        max_temp = max(temps)
        
        insights = [
            f"- Average successful temperature: {avg_temp:.2f}",
            f"- Effective range: {min_temp:.2f} - {max_temp:.2f}",
        ]
        
        # Temperature recommendations
        if avg_temp < 0.5:
            insights.append("- Recommendation: Lower temperatures (more focused) tend to work better")
        elif avg_temp > 1.0:
            insights.append("- Recommendation: Higher temperatures (more creative) tend to work better")
        else:
            insights.append("- Recommendation: Moderate temperatures work well")
        
        return "\n".join(insights)
        
    except Exception as e:
        log(f"Error generating temperature insights: {e}", "debug")
        return ""


def get_recent_successful_examples(pattern_manager, limit=2):
    """Get recent successful prompt examples."""
    try:
        effective_prompts = pattern_manager.patterns.get("effective_prompts", [])
        if not effective_prompts:
            return ""
        
        # Get most recent successful prompts
        recent_prompts = effective_prompts[-limit:] if len(effective_prompts) >= limit else effective_prompts
        
        examples = []
        for i, prompt_data in enumerate(recent_prompts, 1):
            if isinstance(prompt_data, dict):
                prompt_text = prompt_data.get("prompt", "")
                score = prompt_data.get("evaluation_score", 0)
                techniques = prompt_data.get("techniques", [])
                
                # Truncate long prompts
                if len(prompt_text) > 200:
                    prompt_text = prompt_text[:200] + "..."
                
                example = f"{i}. **Score: {score:.2f}** | Techniques: {', '.join(techniques[:3])}\n   \"{prompt_text}\""
                examples.append(example)
        
        return "\n".join(examples) if examples else ""
        
    except Exception as e:
        log(f"Error getting recent examples: {e}", "debug")
        return ""


def process_prompt(prompt, config, pattern_manager=None):
    """
    Process a single prompt through the multi-turn conversation framework.

    Args:
        prompt (str): The prompt to process
        config (dict): Configuration dictionary
        pattern_manager (PatternManager, optional): Pattern manager for tracking successful patterns

    Returns:
        tuple: (conversation_log, is_success)
    """
    # Create attacker and target models
    attacker = AttackerLLM(
        temperature=config["attacker_temp"],
        instructions=config["initial_prompt"],
        followup_instructions=config["followup_prompt"],
        attacker_model_key=config["attacker_model"],
    )

    target = TargetLLM(
        temperature=config["target_temp"],
        target_model_key=config["target_model"],
        memory_enabled=config["target_memory_enabled"],
    )

    # Process prompt
    prompt_summary = prompt[:30] + "..." if len(prompt) > 30 else prompt
    log(f"Processing: {prompt_summary}", "info", VERBOSE_DETAILED)

    try:
        # Ensure the malicious prompt is not reframed before attacker turn 1
        conversation_log = multi_turn_conversation(
            attacker,
            target,
            prompt,  # Pass the original malicious prompt directly
            config["turns"],
            config["strongreject_threshold"],
            pattern_manager,
        )

        # Check if successful
        is_success = conversation_log.get("status") == "success"

        # Store metadata in the log for pattern analysis
        conversation_log["targetModel"] = target.model_key
        conversation_log["attackerTemp"] = attacker.temperature

        # Log success or failure
        if is_success:
            log(
                f"✅ Success! [{conversation_log.get('processing_time', 0):.2f}s] Prompt: {prompt_summary}",
                "success",
            )
        else:
            log(
                f"❌ Failed. [{conversation_log.get('processing_time', 0):.2f}s] Prompt: {prompt_summary}",
                "info",
                VERBOSE_DETAILED,
            )

        return conversation_log, is_success
    except Exception as e:
        import traceback

        log(f"Error processing '{prompt_summary}': {e}", "error")
        log(traceback.format_exc(), "error", VERBOSE_DETAILED)
        return {
            "maliciousPrompt": prompt,
            "turns": [],
            "status": "error",
            "processing_time": 0,
            "error": str(e),
        }, False


def run_experiment(config, pattern_memory=None):
    """
    Run the experiment with the specified configuration.

    Args:
        config (dict): Configuration dictionary
        pattern_memory (PatternManager, optional): Pre-initialized pattern manager

    Returns:
        tuple: (conversation_logs, success_rate)
    """
    # Start timing
    start_time = time.time()

    # Use provided pattern memory or initialize new one
    if pattern_memory is None:
        pattern_memory = (
            PatternManager() if config.get("use_pattern_memory", True) else None
        )

    # Load prompts from multiple sources
    if config.get("prompt_sources") and len(config["prompt_sources"]) > 1:
        # Multi-source loading
        prompts = load_multi_source_prompts(config)
        log(f"Using multi-source prompt loading: {config['prompt_sources']}", "info")
    else:
        # Single source loading (backward compatibility)
        prompts = load_prompts(config["adversarial_prompts"], config["sample_size"])
        log("Using single-source prompt loading", "info")
    
    if not prompts:
        log("No prompts to process.", "error")
        return [], 0

    # Process prompts
    conversation_logs = []
    successes = 0
    total = len(prompts)

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["max_workers"]
    ) as executor:
        # Submit tasks
        future_to_prompt = {
            executor.submit(process_prompt, prompt, config, pattern_memory): prompt
            for prompt in prompts
        }

        # Process results as they complete
        with tqdm(total=total, desc="Processing prompts") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_prompt):
                try:
                    conv_log, is_success = future.result()
                    conversation_logs.append(conv_log)

                    if is_success:
                        successes += 1

                    # Update progress bar
                    progress_bar.update(1)

                    # Save intermediate results if enabled
                    if (
                        config.get("save_temp_files", False)
                        and len(conversation_logs) % 10 == 0
                    ):
                        save_intermediate_results(
                            config, conversation_logs, successes, len(conversation_logs)
                        )
                except Exception as e:
                    prompt = future_to_prompt[future]
                    log(f"Error processing prompt '{prompt}': {e}", "error")

    # Calculate success rate
    success_rate = successes / total if total > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

    # Display results
    log("\nEXECUTION SUMMARY", "result")
    log(f"Success rate: {successes}/{total} ({success_rate:.2f}%)", "result")
    log(f"Total execution time: {total_time:.2f} seconds", "result")

    # Save patterns if using pattern memory
    if pattern_memory:
        if pattern_memory.analyze_logs(conversation_logs):
            log("Updated pattern memory with successful patterns", "success")

    return conversation_logs, success_rate


def save_intermediate_results(config, logs, successes, count):
    """
    Save intermediate results during long runs.

    Args:
        config (dict): Configuration dictionary
        logs (list): Conversation logs so far
        successes (int): Number of successes so far
        count (int): Number of prompts processed so far
    """
    # Create logs directory if it doesn't exist
    logs_dir = ensure_directory_exists(
        config.get("logs_directory", DEFAULT_PATHS["logs_directory"])
    )

    # Create temp file name
    temp_file = os.path.join(logs_dir, f"temp_results_{count}.csv")

    # Create run info
    run_info = {
        "Attacker Temperature": config["attacker_temp"],
        "Target Temperature": config["target_temp"],
        "Number of Turns": config["turns"],
        "Initial Attacker Rewrite Prompt": config["initial_prompt"][:50] + "...",
        "Followup Attacker Rewrite Prompt": (
            config["followup_prompt"][:50] + "..."
            if config["followup_prompt"]
            else "None"
        ),
        "StrongReject Threshold": config["strongreject_threshold"],
        "Target Model Key": config["target_model"],
        "Target Model": TARGET_MODELS[config["target_model"]]["name"],
        "Attacker Model": config["attacker_model"],
    }

    # Save logs
    success_rate_str = f"{(successes / count * 100):.2f}%" if count > 0 else "0.00%"
    save_conversation_log(run_info, logs, success_rate_str, temp_file)
    log(f"Saved intermediate results to {temp_file}", "info")


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Turn Prompting Framework")

    # Model parameters
    parser.add_argument(
        "--target-model",
        type=str,
        default=DEFAULT_CONFIG["target_model"],
        help=f"Target model key. Options: {', '.join(TARGET_MODELS.keys())}",
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=DEFAULT_CONFIG["target_temp"],
        help="Temperature for target model",
    )
    parser.add_argument(
        "--attacker-temp",
        type=float,
        default=DEFAULT_CONFIG["attacker_temp"],
        help="Temperature for attacker model",
    )
    parser.add_argument(
        "--attacker-model",
        type=str,
        default=DEFAULT_CONFIG["attacker_model"],
        help="Attacker model to use",
    )

    # Conversation parameters
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_CONFIG["turns"],
        help="Maximum number of conversation turns",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIG["strongreject_threshold"],
        help="StrongReject threshold for success",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        default=DEFAULT_CONFIG["target_memory_enabled"],
        help="Enable conversation memory for target model",
    )

    # Execution parameters
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_CONFIG["sample_size"],
        help="Number of prompts to sample (None for all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["max_workers"],
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=DEFAULT_CONFIG["verbosity_level"],
        choices=[VERBOSE_NONE, VERBOSE_NORMAL, VERBOSE_DETAILED],
        help="Verbosity level",
    )

    # File paths
    parser.add_argument(
        "--prompts",
        type=str,
        default=DEFAULT_PATHS["adversarial_prompts"],
        help="Path to adversarial prompts CSV (single source mode)",
    )
    parser.add_argument(
        "--harmbench-prompts",
        type=str,
        default=DEFAULT_PATHS["harmbench_prompts"],
        help="Path to HarmBench prompts CSV",
    )
    parser.add_argument(
        "--prompt-sources",
        nargs="+",
        choices=["advbench", "harmbench"],
        default=DEFAULT_CONFIG["prompt_sources"],
        help="Which prompt sources to use (can specify multiple)",
    )
    parser.add_argument(
        "--prompt-mix",
        type=str,
        choices=["equal", "advbench_heavy", "harmbench_heavy", "custom"],
        default=DEFAULT_CONFIG["prompt_mix_ratio"],
        help="How to mix prompts from multiple sources",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../Files/system_prompt.md"),
        help="Path to system prompt file",
    )
    parser.add_argument(
        "--followup-prompt",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "../Files/system_prompt_followup.md"
        ),
        help="Path to followup system prompt file",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=DEFAULT_PATHS["logs_directory"],
        help="Directory for logs",
    )

    # Feature flags
    parser.add_argument(
        "--save-temp", action="store_true", help="Save intermediate results"
    )
    parser.add_argument(
        "--no-patterns", action="store_true", help="Disable pattern memory"
    )

    args = parser.parse_args()

    # Set global verbosity level
    global VERBOSE_LEVEL
    VERBOSE_LEVEL = args.verbose

    # Check target model availability
    log(f"Checking if model '{args.target_model}' is available...", "info")
    if not is_model_available(args.target_model):
        log(
            f"Model '{args.target_model}' is not available. Please choose a different model.",
            "error",
        )
        return False

    # Initialize pattern manager for enhanced prompts
    pattern_memory = PatternManager() if args.use_pattern_memory else None
    if pattern_memory:
        # Set enhancement flag based on configuration
        from config import DEFAULT_CONFIG
        pattern_memory._enhance_enabled = DEFAULT_CONFIG.get("pattern_enhanced_prompts", True)
    
    # Load system prompts with pattern enhancement
    initial_prompt, followup_prompt = load_system_prompts(
        args.system_prompt, args.followup_prompt, pattern_memory, args.target_model
    )
    if not initial_prompt:
        log("Failed to load system prompt.", "error")
        return False
    
    # Log pattern enhancement status
    if pattern_memory:
        pattern_count = len(pattern_memory.patterns.get("effective_prompts", []))
        if pattern_count > 0:
            log(f"Enhanced system prompts with {pattern_count} learned patterns", "success")
        else:
            log("No learned patterns available - using base system prompts", "info")

    # Create configuration
    config = {
        "target_model": args.target_model,
        "target_model_name": TARGET_MODELS[args.target_model]["name"],
        "target_request_cost": TARGET_MODELS[args.target_model]["request_cost"],
        "target_temp": args.target_temp,
        "attacker_temp": args.attacker_temp,
        "attacker_model": args.attacker_model,
        "turns": args.turns,
        "strongreject_threshold": args.threshold,
        "target_memory_enabled": args.memory,
        "sample_size": args.sample_size,
        "max_workers": args.workers,
        "verbosity_level": args.verbose,
        "verbosity_level_name": VERBOSE_LEVEL_NAMES[args.verbose],
        "adversarial_prompts": args.prompts,
        "harmbench_prompts": args.harmbench_prompts,
        "prompt_sources": args.prompt_sources,
        "prompt_mix_ratio": args.prompt_mix,
        "system_prompt": args.system_prompt,
        "system_prompt_followup": args.followup_prompt,
        "logs_directory": args.logs_dir,
        "save_temp_files": args.save_temp,
        "use_pattern_memory": not args.no_patterns,
        "initial_prompt": initial_prompt,
        "followup_prompt": followup_prompt,
    }

    # Display configuration
    display_config(config)

    # Run experiment with pattern-enhanced system prompts
    conversation_logs, success_rate = run_experiment(config, pattern_memory)

    # Save results
    if conversation_logs:
        # Create logs directory if it doesn't exist
        logs_dir = ensure_directory_exists(config["logs_directory"])

        # Get next available filename
        output_file = get_next_filename(logs_dir, "conversationLog")

        # Create run info
        run_info = {
            "Attacker Temperature": config["attacker_temp"],
            "Target Temperature": config["target_temp"],
            "Number of Turns": config["turns"],
            "Initial Attacker Rewrite Prompt": config["initial_prompt"],
            "Followup Attacker Rewrite Prompt": config["followup_prompt"],
            "StrongReject Threshold": config["strongreject_threshold"],
            "Target Model Key": config["target_model"],
            "Target Model": TARGET_MODELS[config["target_model"]]["name"],
            "Attacker Model": config["attacker_model"],
        }

        # Save logs
        success_rate_str = f"{(success_rate * 100):.2f}%"
        save_conversation_log(
            run_info, conversation_logs, success_rate_str, output_file
        )
        log(f"All conversation logs saved to {output_file}", "success")

    return True


if __name__ == "__main__":
    # Clear screen for fresh start
    os.system("cls" if os.name == "nt" else "clear")

    # Run the main function
    main()
