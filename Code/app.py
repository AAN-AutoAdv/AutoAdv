import os
import sys
import random
import time
import concurrent.futures
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.insert(
    1,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Helpers"
    ),
)

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


from logging_utils import log as _log, display_config, ensure_directory_exists
from utils import is_model_available, validate_all_required_apis
from attacker_llm import AttackerLLM
from target_llm import TargetLLM
from conversation import multi_turn_conversation, save_conversation_log
from pattern_manager import PatternManager
from prompt_enhancer import enhance_prompt_with_patterns


def log(message, *args, **kwargs):
    return _log(f"App: {message}", *args, **kwargs)


def load_prompts(filepath, sample_size=None):
    log(f"Loading prompts from {filepath}", "info")

    try:
        df_prompts = pd.read_csv(filepath)
        all_prompts = df_prompts["prompt"].tolist()

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


def _build_source_weights(source_prompts, mix_ratio):
    sources = list(source_prompts.keys())
    if not sources:
        return {}

    if mix_ratio == "advbench_heavy" and "advbench" in source_prompts:
        others = [source for source in sources if source != "advbench"]
        if not others:
            return {"advbench": 1.0}

        weights = {"advbench": 0.7}
        other_weight = 0.3 / len(others)
        for source in others:
            weights[source] = other_weight
        return weights

    if mix_ratio == "harmbench_heavy" and "harmbench" in source_prompts:
        others = [source for source in sources if source != "harmbench"]
        if not others:
            return {"harmbench": 1.0}

        weights = {"harmbench": 0.7}
        other_weight = 0.3 / len(others)
        for source in others:
            weights[source] = other_weight
        return weights

    equal_weight = 1.0 / len(sources)
    return {source: equal_weight for source in sources}


def _allocate_prompt_counts(source_prompts, source_weights, total_needed):
    counts = {source: 0 for source in source_prompts}
    if total_needed <= 0:
        return counts

    available_counts = {source: len(prompts) for source, prompts in source_prompts.items()}
    fractional_parts = []

    for source in source_prompts:
        weighted_target = source_weights.get(source, 0.0) * total_needed
        floor_target = int(weighted_target)
        assigned = min(floor_target, available_counts[source])
        counts[source] = assigned
        fractional_parts.append((weighted_target - floor_target, source))

    assigned_total = sum(counts.values())
    remaining = total_needed - assigned_total

    if remaining > 0:
        fractional_parts.sort(key=lambda item: item[0], reverse=True)
        for _, source in fractional_parts:
            if remaining <= 0:
                break
            if counts[source] < available_counts[source]:
                counts[source] += 1
                remaining -= 1

    if remaining > 0:
        sources_by_spare_capacity = sorted(
            source_prompts.keys(),
            key=lambda source: available_counts[source] - counts[source],
            reverse=True,
        )
        while remaining > 0:
            filled_any = False
            for source in sources_by_spare_capacity:
                if counts[source] < available_counts[source]:
                    counts[source] += 1
                    remaining -= 1
                    filled_any = True
                    if remaining <= 0:
                        break
            if not filled_any:
                break

    return counts


def load_multi_source_prompts(config):
    all_prompts = []
    prompt_sources = config.get("prompt_sources", ["advbench"])
    mix_ratio = config.get("prompt_mix_ratio", "equal")
    sample_size = config.get("sample_size")

    source_files = {
        "advbench": config.get("adversarial_prompts", DEFAULT_PATHS["adversarial_prompts"]),
        "harmbench": config.get("harmbench_prompts", DEFAULT_PATHS["harmbench_prompts"]),
    }

    source_prompts = {}
    for source in prompt_sources:
        if source in source_files:
            filepath = source_files[source]
            if os.path.exists(filepath):
                prompts = load_prompts(filepath, sample_size=None)
                if prompts:
                    source_prompts[source] = prompts
                    log(f"Loaded {len(prompts)} prompts from {source}", "info")
            else:
                log(f"Warning: {source} file not found at {filepath}", "warning")

    if not source_prompts:
        log("No prompt sources could be loaded", "error")
        return []

    total_available = sum(len(prompts) for prompts in source_prompts.values())
    if sample_size and sample_size > total_available:
        log(
            f"Requested sample size {sample_size} exceeds available prompts {total_available}. Using all available prompts.",
            "warning",
        )

    if mix_ratio == "custom" or not sample_size:
        for source, prompts in source_prompts.items():
            all_prompts.extend([(prompt, source) for prompt in prompts])
            log(f"Added all {len(prompts)} prompts from {source} (custom mix)", "info")

        random.shuffle(all_prompts)

        if sample_size and sample_size < len(all_prompts):
            all_prompts = random.sample(all_prompts, sample_size)
            log(
                f"Final sampling: randomly selected {sample_size} prompts from combined sources",
                "info",
            )
    else:
        target_total = min(sample_size, total_available)
        source_weights = _build_source_weights(source_prompts, mix_ratio)
        selected_counts = _allocate_prompt_counts(
            source_prompts, source_weights, target_total
        )

        for source, count in selected_counts.items():
            if count <= 0:
                continue
            selected = random.sample(source_prompts[source], count)
            all_prompts.extend([(prompt, source) for prompt in selected])

            if mix_ratio == "equal":
                mix_label = "equal mix"
            elif mix_ratio == "advbench_heavy":
                mix_label = "70/30 mix"
            elif mix_ratio == "harmbench_heavy":
                mix_label = "70/30 mix"
            else:
                mix_label = "weighted mix"
            log(
                f"Randomly selected {count} prompts from {source} ({mix_label})",
                "info",
            )

        random.shuffle(all_prompts)

        if len(all_prompts) != target_total:
            log(
                f"Expected {target_total} prompts after allocation but got {len(all_prompts)}.",
                "warning",
            )

    final_prompts = [prompt for prompt, source in all_prompts]

    log(f"Total prompts loaded: {len(final_prompts)}", "success")
    return final_prompts


def load_system_prompts(initial_prompt_path, followup_prompt_path=None, pattern_manager=None, target_model=None):
    try:
        with open(initial_prompt_path, "r") as f:
            initial_prompt = f.read()
    except Exception as e:
        log(f"Error loading initial system prompt: {e}", "error")
        initial_prompt = None

    followup_prompt = None
    if followup_prompt_path and os.path.exists(followup_prompt_path):
        try:
            with open(followup_prompt_path, "r") as f:
                followup_prompt = f.read()
        except Exception as e:
            log(f"Error loading followup system prompt: {e}", "error")

    if pattern_manager and initial_prompt:
        enhance_enabled = getattr(pattern_manager, '_enhance_enabled', True)
        if enhance_enabled:
            initial_prompt = enhance_prompt_with_patterns(initial_prompt, pattern_manager, target_model, "initial")
            if followup_prompt:
                followup_prompt = enhance_prompt_with_patterns(followup_prompt, pattern_manager, target_model, "followup")

    return initial_prompt, followup_prompt


def process_prompt(prompt, config, pattern_manager=None, no_temperature_learning=False, baseline_mode=False):
    disable_temp_manager = no_temperature_learning or baseline_mode

    initial_prompt = config["initial_prompt"]
    followup_prompt = config["followup_prompt"]
    if pattern_manager and not baseline_mode and config.get("use_pattern_memory", True):
        enhance_enabled = getattr(pattern_manager, "_enhance_enabled", True)
        if enhance_enabled:
            initial_prompt = enhance_prompt_with_patterns(
                initial_prompt,
                pattern_manager,
                config["target_model"],
                "initial",
            )
            if followup_prompt:
                followup_prompt = enhance_prompt_with_patterns(
                    followup_prompt,
                    pattern_manager,
                    config["target_model"],
                    "followup",
                )

    attacker = AttackerLLM(
        temperature=config["attacker_temp"],
        instructions=initial_prompt,
        followup_instructions=followup_prompt,
        attacker_model_key=config["attacker_model"],
        enable_temperature_manager=not disable_temp_manager,
    )

    target = TargetLLM(
        temperature=config["target_temp"],
        target_model_key=config["target_model"],
        memory_enabled=config["target_memory_enabled"],
    )

    prompt_summary = prompt[:30] + "..." if len(prompt) > 30 else prompt
    log(f"Processing: {prompt_summary}", "info", VERBOSE_DETAILED)

    try:
        # Baseline mode keeps the multi-turn loop, but disables adaptive learning features.
        effective_no_temperature_learning = no_temperature_learning or baseline_mode
        effective_pattern_manager = None if baseline_mode else pattern_manager
        conversation_log = multi_turn_conversation(
            attacker,
            target,
            prompt,
            config["turns"],
            config["strongreject_threshold"],
            effective_pattern_manager,
            effective_no_temperature_learning,
        )
        if baseline_mode:
            conversation_log["baseline_mode"] = True

        is_success = conversation_log.get("status") == "success"

        conversation_log["targetModel"] = target.model_key
        conversation_log["attackerTemp"] = attacker.temperature

        if is_success:
            log(
                f"Success! [{conversation_log.get('processing_time', 0):.2f}s] Prompt: {prompt_summary}",
                "success",
            )
        else:
            log(
                f"Failed. [{conversation_log.get('processing_time', 0):.2f}s] Prompt: {prompt_summary}",
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


def calculate_cumulative_asr(conversation_logs, max_turns=10):
    total_prompts = len(conversation_logs)
    if total_prompts == 0:
        return {}
    
    cumulative_successes = {}
    turn_successes = {}
    
    for turn in range(1, max_turns + 1):
        cumulative_successes[turn] = 0
        turn_successes[turn] = 0
    
    for conv_log in conversation_logs:
        status = conv_log.get("status", "failure")
        
        success_turn = conv_log.get("successTurn")
        
        if success_turn and status == "success":
            for turn in range(success_turn, max_turns + 1):
                cumulative_successes[turn] += 1
            
            if success_turn <= max_turns:
                turn_successes[success_turn] += 1
    
    cumulative_asr = {}
    for turn in range(1, max_turns + 1):
        cumulative_asr[turn] = {
            "cumulative_successes": cumulative_successes[turn],
            "cumulative_asr": (cumulative_successes[turn] / total_prompts) * 100,
            "turn_successes": turn_successes[turn],
            "turn_asr": (turn_successes[turn] / total_prompts) * 100
        }
    
    return cumulative_asr


def run_experiment(config, pattern_memory=None, no_temperature_learning=False, baseline_mode=False):
    start_time = time.time()

    if pattern_memory is None:
        pattern_memory = (
            PatternManager() if config.get("use_pattern_memory", True) else None
        )

    if config.get("prompt_sources") and len(config["prompt_sources"]) > 1:
        prompts = load_multi_source_prompts(config)
        log(f"Using multi-source prompt loading: {config['prompt_sources']}", "info")
    else:
        prompts = load_prompts(config["adversarial_prompts"], config["sample_size"])
        log("Using single-source prompt loading", "info")
    
    if not prompts:
        log("No prompts to process.", "error")
        return [], 0, {}

    conversation_logs = []
    successes = 0
    total = len(prompts)
    online_pattern_learning = bool(
        pattern_memory
        and config.get("online_pattern_learning", True)
        and not baseline_mode
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["max_workers"]
    ) as executor:
        future_to_prompt = {
            executor.submit(process_prompt, prompt, config, pattern_memory, no_temperature_learning, baseline_mode): prompt
            for prompt in prompts
        }

        with tqdm(total=total, desc="Processing prompts") as progress_bar:
            for future in concurrent.futures.as_completed(future_to_prompt):
                try:
                    conv_log, is_success = future.result()
                    conversation_logs.append(conv_log)

                    if is_success:
                        successes += 1
                        if online_pattern_learning:
                            pattern_memory.analyze_conversation(conv_log)

                    progress_bar.update(1)

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

    success_rate = successes / total if total > 0 else 0
    end_time = time.time()
    total_time = end_time - start_time

    cumulative_asr = calculate_cumulative_asr(conversation_logs, config["turns"])

    log("\nEXECUTION SUMMARY", "result")
    log(f"Success rate: {successes}/{total} ({success_rate * 100:.2f}%)", "result")
    log(f"Total execution time: {total_time:.2f} seconds", "result")
    
    log("\nCUMULATIVE ASR BY TURN", "result")
    log("Turn | Cumulative Successes | Cumulative ASR | Turn Successes | Turn ASR", "result")
    log("-" * 80, "result")
    for turn in range(1, config["turns"] + 1):
        if turn in cumulative_asr:
            data = cumulative_asr[turn]
            log(f"{turn:4d} | {data['cumulative_successes']:19d} | {data['cumulative_asr']:13.2f}% | {data['turn_successes']:14d} | {data['turn_asr']:7.2f}%", "result")

    if pattern_memory:
        if online_pattern_learning:
            if pattern_memory.save():
                log("Updated pattern memory with successful patterns", "success")
        else:
            if pattern_memory.analyze_logs(conversation_logs):
                log("Updated pattern memory with successful patterns", "success")

    return conversation_logs, success_rate, cumulative_asr


def save_intermediate_results(config, logs, successes, count):
    logs_dir = ensure_directory_exists(
        config.get("logs_directory", DEFAULT_PATHS["logs_directory"])
    )

    temp_file = os.path.join(logs_dir, f"temp_results_{count}.csv")

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

    success_rate_str = f"{(successes / count * 100):.2f}%" if count > 0 else "0.00%"
    save_conversation_log(run_info, logs, success_rate_str, temp_file)
    log(f"Saved intermediate results to {temp_file}", "info")


def generate_descriptive_filename(logs_dir, config, success_rate, file_type="LOG"):
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    target_model = config.get("target_model", "unknown")
    sample_size = config.get("sample_size", 0)
    turns = config.get("turns", 0)
    pattern_enabled = config.get("use_pattern_memory", False)
    attacker_model = config.get("attacker_model", "unknown")
    
    success_pct = f"{success_rate * 100:.1f}pct" if success_rate is not None else "unknown"
    
    components = [
        file_type,
        target_model,
        f"{sample_size}prompts",
        f"{turns}turns",
        success_pct,
        "pattern" if pattern_enabled else "nopattern",
        timestamp
    ]
    
    filename = "_".join(components) + ".csv"
    
    filename = filename.replace(" ", "").replace("/", "-").replace("\\", "-")
    
    full_path = os.path.join(logs_dir, filename)
    
    counter = 1
    original_path = full_path
    while os.path.exists(full_path):
        name_part = original_path.replace(".csv", "")
        full_path = f"{name_part}_{counter}.csv"
        counter += 1
    
    return full_path


def save_cumulative_asr_data(cumulative_asr, config, output_file):
    import csv
    
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        header_fields = [
            f"Target Model = {config.get('target_model_name', 'Unknown')}",
            f"Target Model Key = {config.get('target_model', 'Unknown')}",
            f"Attacker Model = {config.get('attacker_model', 'Unknown')}",
            f"Attacker Temperature = {config.get('attacker_temp', 'Unknown')}",
            f"Target Temperature = {config.get('target_temp', 'Unknown')}",
            f"Number of Turns = {config.get('turns', 'Unknown')}",
            f"StrongReject Threshold = {config.get('strongreject_threshold', 'Unknown')}",
            f"Sample Size = {config.get('sample_size', 'Unknown')}",
        ]
        
        writer.writerow(header_fields)
        writer.writerow([])
        
        writer.writerow([
            "Turn", "Cumulative Successes", "Cumulative ASR (%)", 
            "Turn Successes", "Turn ASR (%)"
        ])
        
        for turn in sorted(cumulative_asr.keys()):
            data = cumulative_asr[turn]
            writer.writerow([
                turn,
                data["cumulative_successes"],
                f"{data['cumulative_asr']:.2f}",
                data["turn_successes"],
                f"{data['turn_asr']:.2f}"
            ])
        
        writer.writerow([])
        writer.writerow(["SUMMARY STATISTICS"])
        
        if cumulative_asr:
            final_turn = max(cumulative_asr.keys())
            final_data = cumulative_asr[final_turn]
            writer.writerow([
                f"Final ASR (Turn {final_turn})",
                f"{final_data['cumulative_asr']:.2f}%",
                f"({final_data['cumulative_successes']} successes)"
            ])
            
            max_asr_turn = max(cumulative_asr.keys(), 
                             key=lambda t: cumulative_asr[t]["cumulative_asr"])
            max_asr_data = cumulative_asr[max_asr_turn]
            writer.writerow([
                f"Peak ASR (Turn {max_asr_turn})",
                f"{max_asr_data['cumulative_asr']:.2f}%",
                f"({max_asr_data['cumulative_successes']} successes)"
            ])


def main():


    parser = argparse.ArgumentParser(description="Multi-Turn Prompting Framework")

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
        help="Unified evaluator threshold for success",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_CONFIG["sample_size"],
        help="number of prompts to sample (none for all)",
    )
    parser.add_argument(
        "--no-patterns",
        action="store_true",
        help="Disable pattern memory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["max_workers"],
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--no-temperature-learning", action="store_true", help="Disable temperature adjustments and learning"
    )
    parser.add_argument(
        "--baseline-mode", action="store_true", help="Use baseline ablation mode (multi-turn, no pattern memory, no temperature learning)"
    )
    prompt_ablation_group = parser.add_mutually_exclusive_group()
    prompt_ablation_group.add_argument(
        "--no-fewshot-learning",
        action="store_true",
        help="Use no-fewshot attacker prompt templates (ablation mode)",
    )
    prompt_ablation_group.add_argument(
        "--no-seed-techniques",
        action="store_true",
        help="Use no-seed-techniques attacker prompt templates (ablation mode)",
    )

    args = parser.parse_args()

    global VERBOSE_LEVEL
    VERBOSE_LEVEL = DEFAULT_CONFIG["verbosity_level"]

    log(f"Checking if model '{args.target_model}' is available...", "info")
    
    models_to_validate = [args.target_model, args.attacker_model]
    validation_results = validate_all_required_apis(models_to_validate)
    
    if not validation_results.get(args.target_model, {}).get("available", False):
        error_msg = validation_results.get(args.target_model, {}).get("error", "Unknown error")
        log(f"Target model '{args.target_model}' is not available: {error_msg}", "error")
        return False
    
    if not validation_results.get(args.attacker_model, {}).get("available", False):
        error_msg = validation_results.get(args.attacker_model, {}).get("error", "Unknown error")
        log(f"Attacker model '{args.attacker_model}' is not available: {error_msg}", "error")
        return False
    
    log("All required APIs validated successfully", "success")

    system_prompt_path = DEFAULT_PATHS["system_prompt"]
    followup_prompt_path = DEFAULT_PATHS["system_prompt_followup"]

    if args.baseline_mode:
        system_prompt_path = os.path.join(
            os.path.dirname(__file__), "../Files/system_prompt_baseline.md"
        )
        followup_prompt_path = None
        pattern_memory = None
        log("Baseline mode: Using baseline system prompt template", "info")
        log("Baseline mode: Pattern learning disabled", "info")
    elif args.no_fewshot_learning:
        system_prompt_path = os.path.join(
            os.path.dirname(__file__), "../Files/system_prompt_no_fewshot.md"
        )
        log("Ablation mode: using no-fewshot system prompt template", "info")
        pattern_memory = PatternManager() if not args.no_patterns else None
    elif args.no_seed_techniques:
        system_prompt_path = os.path.join(
            os.path.dirname(__file__), "../Files/system_prompt_no_seed_techniques.md"
        )
        followup_prompt_path = os.path.join(
            os.path.dirname(__file__), "../Files/followup_prompt_no_seed_techniques.md"
        )
        log("Ablation mode: using no-seed-techniques prompt templates", "info")
        pattern_memory = PatternManager() if not args.no_patterns else None
    else:
        pattern_memory = PatternManager() if not args.no_patterns else None

    if pattern_memory:
        pattern_memory._enhance_enabled = DEFAULT_CONFIG.get("pattern_enhanced_prompts", True)

    if args.baseline_mode:
        initial_prompt, followup_prompt = load_system_prompts(
            system_prompt_path, followup_prompt_path, None, args.target_model
        )
        log("Baseline mode: Pattern enhancement disabled", "info")
        log("Baseline mode: Temperature manager disabled (multi-turn still enabled)", "info")
    else:
        initial_prompt, followup_prompt = load_system_prompts(
            system_prompt_path, followup_prompt_path, None, args.target_model
        )
    if not initial_prompt:
        log("Failed to load system prompt.", "error")
        return False

    if pattern_memory:
        pattern_count = len(pattern_memory.patterns.get("effective_prompts", []))
        if pattern_count > 0:
            log(
                f"Loaded {pattern_count} learned patterns (applied dynamically per prompt during run)",
                "success",
            )
        else:
            log("No learned patterns available - using base system prompts", "info")

    disable_temperature_learning = args.no_temperature_learning or args.baseline_mode

    config = {
        "target_model": args.target_model,
        "target_model_name": TARGET_MODELS[args.target_model]["name"],
        "target_request_cost": TARGET_MODELS[args.target_model]["request_cost"],
        "target_temp": args.target_temp,
        "attacker_temp": args.attacker_temp,
        "attacker_model": args.attacker_model,
        "turns": args.turns,
        "strongreject_threshold": args.threshold,
        "target_memory_enabled": True,
        "sample_size": args.sample_size,
        "max_workers": args.workers,
        "verbosity_level": DEFAULT_CONFIG["verbosity_level"],
        "verbosity_level_name": VERBOSE_LEVEL_NAMES[DEFAULT_CONFIG["verbosity_level"]],
        "adversarial_prompts": DEFAULT_PATHS["adversarial_prompts"],
        "harmbench_prompts": DEFAULT_PATHS["harmbench_prompts"],
        "prompt_sources": DEFAULT_CONFIG["prompt_sources"],
        "prompt_mix_ratio": DEFAULT_CONFIG["prompt_mix_ratio"],
        "system_prompt": system_prompt_path,
        "system_prompt_followup": followup_prompt_path,
        "logs_directory": DEFAULT_PATHS["logs_directory"],
        "save_temp_files": False,
        "use_pattern_memory": (not args.no_patterns) and (not args.baseline_mode),
        "online_pattern_learning": DEFAULT_CONFIG.get("online_pattern_learning", True),
        "baseline_mode": args.baseline_mode,
        "no_temperature_learning": disable_temperature_learning,
        "no_fewshot_learning": args.no_fewshot_learning,
        "no_seed_techniques": args.no_seed_techniques,
        "initial_prompt": initial_prompt,
        "followup_prompt": followup_prompt,
    }

    display_config(config)

    conversation_logs, success_rate, cumulative_asr = run_experiment(
        config,
        pattern_memory,
        disable_temperature_learning,
        args.baseline_mode,
    )

    if conversation_logs:
        logs_dir = ensure_directory_exists(config["logs_directory"])

        output_file = generate_descriptive_filename(logs_dir, config, success_rate)

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
            "Sample Size": config["sample_size"],
            "Pattern Memory Enabled": config.get("use_pattern_memory", False),
            "Prompt Sources": ", ".join(config.get("prompt_sources", [])),
            "Prompt Mix Ratio": config.get("prompt_mix_ratio", "unknown"),
            "Temperature Strategy": config.get("temperature_strategy", "unknown"),
            "Max Workers": config.get("max_workers", 1),
        }

        success_rate_str = f"{(success_rate * 100):.2f}%"
        save_conversation_log(
            run_info, conversation_logs, success_rate_str, output_file
        )
        log(f"All conversation logs saved to {output_file}", "success")
        
        asr_output_file = generate_descriptive_filename(logs_dir, config, success_rate, file_type="ASR")
        save_cumulative_asr_data(cumulative_asr, config, asr_output_file)
        log(f"Cumulative ASR data saved to {asr_output_file}", "success")

    return True


if __name__ == "__main__":
    os.system("cls" if os.name == "nt" else "clear")

    main()
