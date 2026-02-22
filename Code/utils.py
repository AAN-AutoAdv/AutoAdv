from __future__ import print_function
from dotenv import load_dotenv
import os, time
import builtins as __builtin__
import re

from config import TARGET_MODELS, ATTACKER_MODELS, API_KEYS, DISCLAIMER_PATTERNS
from logging_utils import log, VERBOSE_NORMAL, VERBOSE_DETAILED

load_dotenv()

def check_api_key_existence(apiKeyName):
    apiKey = os.getenv(apiKeyName)

    if apiKey is None:
        log(f"API key '{apiKeyName}' is missing from your environment variables (or .env file).", "warning")
        log("You can add it to a .env file in the project root and restart.", "info")
        log("Alternatively, you can enter it now (will not be saved).", "info")

        try:
            import getpass
            apiKey = getpass.getpass(f"Please enter your {apiKeyName} key: ")
        except ImportError:
             apiKey = input(f"Please enter your {apiKeyName} key: ")


        if not apiKey:
             error_msg = f"No API key provided for {apiKeyName}. Exiting."
             log(error_msg, "error")
             raise ValueError(error_msg)

        log(f"Using provided API key for {apiKeyName} for this session.", "info")
        return apiKey
    else:
        return apiKey

def api_call_with_retry(api_func, *args, **kwargs):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            log(f"API call failed, retrying in {retry_delay}s", "warning")
            time.sleep(retry_delay)
            retry_delay *= 2

def check_file_existence(filepath):
    if not os.path.exists(filepath):
        error_msg = f"File '{filepath}' not found!"
        log(error_msg, "error")
        raise FileNotFoundError(error_msg)
    elif not os.path.isfile(filepath):
         error_msg = f"Path '{filepath}' exists but is not a file!"
         log(error_msg, "error")
         raise IsADirectoryError(error_msg)
    else:
        return filepath

def check_directory_existence(directory, autoCreate=True):
    if not os.path.exists(directory):
        if autoCreate:
            try:
                os.makedirs(directory)
                log(f"Created directory: {directory}", "info", VERBOSE_DETAILED)
            except OSError as e:
                 log(f"Error creating directory {directory}: {e}", "error")
                 raise
        else:
            error_msg = f"Directory '{directory}' not found and autoCreate is False!"
            log(error_msg, "error")
            raise FileNotFoundError(error_msg)
    elif not os.path.isdir(directory):
         error_msg = f"Path '{directory}' exists but is not a directory!"
         log(error_msg, "error")
         raise NotADirectoryError(error_msg)

    return directory

def ensure_directory_exists(directory):
    return check_directory_existence(directory, autoCreate=True)


def print(*args, type=None, **kwargs):
    color_code = ""
    type_tag_map = {
        "success": ("\033[92m", "SUCCESS"),
        "error":   ("\033[91m", "  ERROR"),
        "warning": ("\033[93m", "WARNING"),
        "info":    ("\033[95m", "   INFO"),
        "debug":   ("\033[96m", "  DEBUG"),
        "result":  ("\033[94m", " RESULT"),
    }

    if type in type_tag_map:
        color_code, type_tag = type_tag_map[type]
    else:
        return __builtin__.print(*args, **kwargs)

    reset_code = "\033[0m"
    message = " ".join(map(str, args))

    formatted_message = f"{color_code}[{type_tag.strip()}] {message}{reset_code}"

    return __builtin__.print(formatted_message, **kwargs)


def strip_disclaimers(text):
    from config import DISCLAIMER_PATTERNS
    import re

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    original_length = len(text)
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    text = text.strip()
    
    if len(text) < original_length * 0.8:
        from logging_utils import log
        from config import VERBOSE_DETAILED
        log(f"Stripped disclaimer from response (removed {original_length - len(text)} chars)", "debug", VERBOSE_DETAILED)
    
    return text

def is_model_available(model_key):
    model_config = TARGET_MODELS.get(model_key) or ATTACKER_MODELS.get(model_key)

    if not model_config:
        log(f"Model key '{model_key}' not found in TARGET_MODELS or ATTACKER_MODELS.", "error")
        return False

    api_type = model_config.get("api")
    if not api_type:
         log(f"API type not defined for model '{model_key}' in config.", "error")
         return False

    api_key_name = None
    if api_type == "openai":
        api_key_name = "OPENAI_API_KEY"
    elif api_type == "together":
        api_key_name = "TOGETHER_API_KEY"
    elif api_type == "xai" or api_type == "grok":
        api_key_name = "XAI_API_KEY"
    elif api_type == "anthropic":
        api_key_name = "ANTHROPIC_API_KEY"
    if not api_key_name:
        log(f"No known API key variable associated with API type '{api_type}' for model '{model_key}'.", "error")
        return False

    if not os.getenv(api_key_name):
         log(f"Required API key '{api_key_name}' for model '{model_key}' (API: {api_type}) is not set in the environment. Will prompt if used.", "warning", VERBOSE_NORMAL)

    log(f"Model '{model_key}' appears to be configured.", "info", VERBOSE_DETAILED)
    return True


def validate_api_key_format(api_key, api_type):
    if not api_key or not isinstance(api_key, str):
        return False
    
    if api_type == "openai":
        return api_key.startswith('sk-') and len(api_key) >= 40
    elif api_type == "together":
        return len(api_key) >= 20 and api_key.replace('-', '').replace('_', '').isalnum()
    elif api_type == "xai" or api_type == "grok":
        return len(api_key) >= 20
    elif api_type == "anthropic":
        return api_key.startswith('sk-ant-') and len(api_key) >= 30
    
    return len(api_key) >= 10


def test_api_connectivity(model_key, test_prompt="Hello"):
    try:
        model_config = TARGET_MODELS.get(model_key) or ATTACKER_MODELS.get(model_key)
        if not model_config:
            return False
            
        api_type = model_config.get("api")
        if not api_type:
            return False
            
        api_key_name = None
        if api_type == "openai":
            api_key_name = "OPENAI_API_KEY"
        elif api_type == "together":
            api_key_name = "TOGETHER_API_KEY"
        elif api_type == "xai":
            api_key_name = "XAI_API_KEY"
        elif api_type == "anthropic":
            api_key_name = "ANTHROPIC_API_KEY"
            
        if not api_key_name:
            return False
            
        api_key = os.getenv(api_key_name)
        if not api_key:
            return False
            
        if not validate_api_key_format(api_key, api_type):
            log(f"API key format appears invalid for {api_type}", "warning")
            return False
            
        if api_type in ["openai", "together"]:
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1" if api_type == "together" else None
            )
            response = client.chat.completions.create(
                model=model_config["name"],
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=10,
                temperature=0.1
            )
            return bool(response.choices[0].message.content)
        elif api_type == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_config["name"],
                max_tokens=10,
                messages=[{"role": "user", "content": test_prompt}]
            )
            return bool(response.content[0].text)
        elif api_type == "xai":
            return True
            
    except Exception as e:
        log(f"API connectivity test failed for {model_key}: {e}", "debug", VERBOSE_DETAILED)
        return False
    
    return False


def validate_all_required_apis(model_keys):
    results = {}
    
    for model_key in model_keys:
        log(f"Validating API for model: {model_key}", "info")
        
        if not is_model_available(model_key):
            results[model_key] = {"available": False, "error": "Model not configured"}
            continue
            
        if model_key in ["grok-3-mini-beta"]:
            from config import TARGET_MODELS, ATTACKER_MODELS
            api_type = (TARGET_MODELS.get(model_key) or ATTACKER_MODELS.get(model_key)).get("api")
            if api_type == "grok" or api_type == "xai":
                api_key = os.getenv("XAI_API_KEY")
            elif api_type == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif api_type == "together":
                api_key = os.getenv("TOGETHER_API_KEY")
            elif api_type == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            else:
                api_key = None
                
            if api_key:
                results[model_key] = {"available": True, "error": None}
                log(f"[OK] {model_key} API key found (skipping connectivity test)", "success")
            else:
                results[model_key] = {"available": False, "error": "API key not found"}
                log(f"[FAIL] {model_key} API key not found", "error")
        else:
            if test_api_connectivity(model_key):
                results[model_key] = {"available": True, "error": None}
                log(f"[OK] {model_key} API validation successful", "success")
            else:
                results[model_key] = {"available": False, "error": "API connectivity test failed"}
                log(f"[FAIL] {model_key} API validation failed", "error")
    
    return results
