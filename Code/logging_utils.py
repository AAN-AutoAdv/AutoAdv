import datetime
import colorama
from colorama import Fore, Style, Back
import os

colorama.init(autoreset=True)

VERBOSE_NONE = 0
VERBOSE_NORMAL = 1
VERBOSE_DETAILED = 2

VERBOSE_LEVEL = VERBOSE_NORMAL

VERBOSE_LEVEL_NAMES = {
    VERBOSE_NONE: "Minimal",
    VERBOSE_NORMAL: "Normal",
    VERBOSE_DETAILED: "Detailed",
}

def log(message, type="info", verbose_level=VERBOSE_NORMAL):
    from config import VERBOSE_LEVEL
    
    if verbose_level > VERBOSE_LEVEL:
        return

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    if type == "info":
        prefix = f"{Fore.BLUE}[INFO]{Style.RESET_ALL}"
    elif type == "success":
        prefix = f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL}"
    elif type == "error":
        prefix = f"{Fore.RED}[ERROR]{Style.RESET_ALL}"
    elif type == "warning":
        prefix = f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL}"
    elif type == "debug":
        prefix = f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL}"
    elif type == "config":
        prefix = f"{Fore.CYAN}[CONFIG]{Style.RESET_ALL}"
    elif type == "result":
        prefix = f"{Back.BLUE}{Fore.WHITE}[RESULT]{Style.RESET_ALL}"
    else:
        prefix = f"[{type.upper()}]"

    print(f"{prefix} {timestamp} | {message}")
    
def print_header(title, width=60):
    print(f"\n{Fore.CYAN}{'=' * width}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{title.center(width)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * width}{Style.RESET_ALL}\n")
    
def print_section(title, width=60):
    print(f"\n{Fore.CYAN}{title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'-' * width}{Style.RESET_ALL}")


def _display_provider_details(label, model_key, model_name, provider_info):
    print(f"{Fore.CYAN}{label}:{Style.RESET_ALL} {model_key} - {model_name}")
    if not provider_info:
        return

    print(f"  - Provider: {provider_info.get('provider_display_name', 'Unknown')}")
    print(f"  - SDK: {provider_info.get('sdk_display_name', 'Unknown')}")
    print(f"  - Endpoint: {provider_info.get('endpoint', 'Unknown')}")
    print(f"  - API Key Env: {provider_info.get('api_key_env', 'Unknown')}")
    
def display_config(config_dict, width=80):
    print_header("CONFIGURATION", width)

    _display_provider_details(
        "TARGET MODEL",
        config_dict["target_model"],
        config_dict.get("target_model_name", "Unknown"),
        config_dict.get("target_provider_info"),
    )
    _display_provider_details(
        "ATTACKER MODEL",
        config_dict.get("attacker_model", "gpt-4o-mini"),
        config_dict.get("attacker_model_name", "Unknown"),
        config_dict.get("attacker_provider_info"),
    )

    print(f"  - Attacker Temperature: {config_dict['attacker_temp']}")
    print(f"  - Target Temperature: {config_dict['target_temp']}")
    print(f"  - Number of Turns: {config_dict['turns']}")
    print(f"  - StrongReject Threshold: {config_dict['strongreject_threshold']}")
    print(
        f"  - Target Memory Enabled: {'Yes' if config_dict.get('target_memory_enabled', False) else 'No'}"
    )

    print(f"  - Sample Size: {config_dict.get('sample_size', 'All prompts')}")
    print(f"  - Parallel Workers: {config_dict.get('max_workers', 1)}")
    print(f"  - Verbosity Level: {VERBOSE_LEVEL_NAMES.get(config_dict.get('verbosity_level', VERBOSE_NORMAL), 'Detailed')}")

    print(f"  - Prompts: {config_dict.get('adversarial_prompts', 'Not specified')}")
    print(f"  - System Prompt: {config_dict.get('system_prompt', 'Not specified')}")
    print(
        f"  - Followup Prompt: {config_dict.get('system_prompt_followup', 'Not Used')}"
    )

    print(f"{Fore.CYAN}{'=' * width}{Style.RESET_ALL}\n")

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        log(f"Created directory: {directory}", "info", VERBOSE_DETAILED)
    return directory
