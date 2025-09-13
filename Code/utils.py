"""
General utility functions for the codebase.
Contains helper functions for file operations, API key checking, display formatting, etc.
"""
from __future__ import print_function
from dotenv import load_dotenv
import os, glob, textwrap, shutil, time
import builtins as __builtin__
import re # Import re for strip_disclaimers
import traceback # Import traceback for logging

# Make sure progress bar library is available
try:
    from progress.bar import ChargingBar
except ImportError:
    ChargingBar = None # Set to None if library not installed

# Import necessary items from local modules
from config import VERBOSE_DETAILED, TARGET_MODELS, ATTACKER_MODELS, API_KEYS, DISCLAIMER_PATTERNS # Import necessary configs
from logging_utils import log, VERBOSE_NORMAL, VERBOSE_DETAILED # Make sure all levels used are imported

# Load .env file at the start
load_dotenv()

# Check for the existence of a specified API key and return the key (if available)
def check_api_key_existence(apiKeyName):
    """
    Check for the existence of a specified API key in environment variables.
    Uses the value from the environment if found, otherwise prompts the user.

    Args:
        apiKeyName (str): Name of the API key environment variable (e.g., "OPENAI_API_KEY", "XAI_API_KEY")

    Returns:
        str: The API key if found or entered.

    Raises:
        ValueError: If the user doesn't provide a key when prompted.
    """
    apiKey = os.getenv(apiKeyName)

    if apiKey is None:
        log(f"API key '{apiKeyName}' is missing from your environment variables (or .env file).", "warning")
        log("You can add it to a .env file in the project root and restart.", "info")
        log("Alternatively, you can enter it now (will not be saved).", "info")

        # Prompt the user securely if possible, otherwise fallback to standard input
        try:
            import getpass
            apiKey = getpass.getpass(f"Please enter your {apiKeyName} key: ")
        except ImportError:
             apiKey = input(f"Please enter your {apiKeyName} key: ")


        if not apiKey:
             error_msg = f"No API key provided for {apiKeyName}. Exiting."
             log(error_msg, "error")
             raise ValueError(error_msg) # Raise an error instead of returning None implicitly

        # Optionally store it in the environment for the current session only
        # os.environ[apiKeyName] = apiKey
        log(f"Using provided API key for {apiKeyName} for this session.", "info")
        return apiKey
    else:
        # log(f"Found API key '{apiKeyName}' in environment.", "debug", VERBOSE_DETAILED+1) # Too verbose maybe
        return apiKey

def api_call_with_retry(api_func, *args, **kwargs):
    """Make an API call with exponential backoff retry"""
    max_retries = 3
    retry_delay = 1  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise  # Re-raise the exception if all retries failed
            
            log(f"API call failed, retrying in {retry_delay}s", "warning")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

# Check for the existence of a specified file
def check_file_existence(filepath):
    """
    Check if a file exists at the specified path.

    Args:
        filepath (str): Path to the file to check

    Returns:
        str: The filepath if the file exists

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(filepath):
        error_msg = f"File '{filepath}' not found!"
        log(error_msg, "error")
        raise FileNotFoundError(error_msg)
    elif not os.path.isfile(filepath): # Added check to ensure it's a file
         error_msg = f"Path '{filepath}' exists but is not a file!"
         log(error_msg, "error")
         raise IsADirectoryError(error_msg) # Or FileNotFoundError? IsADirectoryError is more specific
    else:
        # log(f"File '{filepath}' found.", "debug", VERBOSE_DETAILED+1)
        return filepath

# Check for the existence of a specified directory
def check_directory_existence(directory, autoCreate=True):
    """
    Check if a directory exists, optionally creating it if it doesn't.

    Args:
        directory (str): Path to the directory to check
        autoCreate (bool): Whether to create the directory if it doesn't exist

    Returns:
        str: The directory path

    Raises:
        FileNotFoundError: If the directory doesn't exist and autoCreate is False.
        NotADirectoryError: If the path exists but is not a directory.
        OSError: If directory creation fails.
    """
    if not os.path.exists(directory):
        if autoCreate:
            try:
                os.makedirs(directory)
                log(f"Created directory: {directory}", "info", VERBOSE_DETAILED)
            except OSError as e:
                 log(f"Error creating directory {directory}: {e}", "error")
                 raise # Re-raise the error
        else:
            error_msg = f"Directory '{directory}' not found and autoCreate is False!"
            log(error_msg, "error")
            raise FileNotFoundError(error_msg)
    elif not os.path.isdir(directory):
         error_msg = f"Path '{directory}' exists but is not a directory!"
         log(error_msg, "error")
         raise NotADirectoryError(error_msg)

    # log(f"Directory '{directory}' exists.", "debug", VERBOSE_DETAILED+1)
    return directory

# Ensure a directory exists (alias for check_directory_existence for compatibility)
def ensure_directory_exists(directory):
    """
    Ensure a directory exists, creating it if necessary.
    This is an alias for check_directory_existence with default parameters.

    Args:
        directory (str): Path to the directory to check/create

    Returns:
        str: The directory path
    """
    return check_directory_existence(directory, autoCreate=True)


# Override the default print function to have custom types
# DEPRECATED in favor of log function. Keep for compatibility if needed, but prefer log.
def print(*args, type=None, **kwargs):
    """
    Enhanced print function with colored output for different message types.
    Prefer using the `log` function instead.

    Args:
        *args: Values to print
        type (str, optional): Message type ('success', 'error', 'warning', 'info')
        **kwargs: Additional print function arguments
    """
    color_code = ""
    type_tag_map = {
        "success": ("\033[92m", "SUCCESS"),
        "error":   ("\033[91m", "  ERROR"),
        "warning": ("\033[93m", "WARNING"),
        "info":    ("\033[95m", "   INFO"),
        "debug":   ("\033[96m", "  DEBUG"), # Added debug
        "result":  ("\033[94m", " RESULT"), # Changed result color
    }

    if type in type_tag_map:
        color_code, type_tag = type_tag_map[type]
    else:
        # Default print behavior if type is None or unrecognized
        return __builtin__.print(*args, **kwargs)

    reset_code = "\033[0m"
    message = " ".join(map(str, args))

    # Basic formatting, no complex wrapping here - use `log` for better formatting
    formatted_message = f"{color_code}[{type_tag.strip()}] {message}{reset_code}"

    # Use built-in print for actual output
    return __builtin__.print(formatted_message, **kwargs)


# Get the next file name
def get_next_filename(directory, baseName="data", fileExtension=".csv"):
    """
    Generate the next sequential filename in a directory (e.g., data0.csv, data1.csv).
    """
    check_directory_existence(directory)

    # Match filenames like data0.csv, data1.csv, etc.
    pattern = re.compile(rf"{re.escape(baseName)}(\d+){re.escape(fileExtension)}")

    existing_files = glob.glob(os.path.join(directory, f"{baseName}*{fileExtension}"))

    numbers = []
    for file_path in existing_files:
        filename = os.path.basename(file_path)  # Extract filename only (no full path)
        match = pattern.match(filename)
        if match:
            try:
                numbers.append(int(match.group(1)))
            except (ValueError, IndexError):
                pass

    next_number = max(numbers) + 1 if numbers else 0
    next_filename = f"{baseName}{next_number}{fileExtension}"
    full_path = os.path.join(directory, next_filename)
    log(f"Next filename generated: {full_path}", "debug", VERBOSE_DETAILED + 1)
    return full_path


# Progress bar to tell how far along the code is
# Consider removing if tqdm is preferred (used in app.py)
class ProgressBar:
    """
    Simple progress bar for displaying status of operations.
    NOTE: `tqdm` is generally preferred for more complex scenarios.

    Attributes:
        userInfo (str): Information to display alongside the progress
        total (int): Total number of steps
        bar (ChargingBar): The progress bar instance from the `progress` library
    """

    def __init__(self, userInfo, total):
        """
        Initialize a new progress bar.

        Args:
            userInfo (str): Text to display with the progress bar
            total (int): Total number of steps
        """
        if total <= 0:
             log("Progress bar total must be positive.", "warning")
             self.bar = None
             return

        self.userInfo = userInfo
        self.total = total

        # Create the bar at the start
        # Ensure ChargingBar is available or handle import error
        if ChargingBar:
             try:
                 self.bar = ChargingBar(
                     message=f"{userInfo} %(index)d/%(max)d", # Use %d for integers
                     max=total,
                     suffix="%(percent).1f%% (ETA %(eta)ds)",
                 )
             except Exception as e:
                 log(f"Failed to initialize ChargingBar: {e}", "warning")
                 self.bar = None
        else:
             log("`progress` library not installed? Progress bar disabled.", "warning")
             self.bar = None


    def continue_progress(self):
        """Increment the progress bar by one step."""
        if self.bar:
            try:
                self.bar.next()
            except Exception as e:
                 log(f"Progress bar error: {e}", "warning")

    def end_progress(self):
        """Complete and close the progress bar."""
        if self.bar:
             try:
                 self.bar.finish()
             except Exception as e:
                 log(f"Progress bar finish error: {e}", "warning")


def categorize_prompt(prompt):
    try:
        from technique_analyzer import categorize_prompt as _categorize_prompt
        return _categorize_prompt(prompt)
    except ImportError:
        log("technique_analyzer module not found for categorize_prompt", "error")
        return "general" # Fallback


# Get a specialized strategy (DUPLICATED - import from technique_analyzer.py)
def get_specialized_strategy(category):
    try:
        from technique_analyzer import get_specialized_strategy as _get_specialized_strategy
        return _get_specialized_strategy(category)
    except ImportError:
        log("technique_analyzer module not found for get_specialized_strategy", "error")
        # Fallback strategy
        return {
            "framing": "Academic or educational research",
            "persona": "Researcher or educator",
            "format": "Analytical framework or case study",
            "technique": "Theoretical exploration with educational purpose"
        }

# Identify working technique (DUPLICATED - import from technique_analyzer.py)
def identify_working_technique(target_response, prompt_category):
    try:
        from technique_analyzer import identify_working_technique as _identify_working_technique
        return _identify_working_technique(target_response, prompt_category)
    except ImportError:
        log("technique_analyzer module not found for identify_working_technique", "error")
        return "educational_framing" # Fallback


# Strip disclaimers (Seems appropriate for utils.py)
def strip_disclaimers(text):
    """
    Strip disclaimer statements from the beginning of AI responses.
    
    Args:
        text (str): The response text to clean
        
    Returns:
        str: The response with disclaimers removed
    """
    from config import DISCLAIMER_PATTERNS
    import re
    
    # Apply each pattern to strip disclaimers
    original_length = len(text)
    for pattern in DISCLAIMER_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Clean up any leading whitespace or newlines
    text = text.strip()
    
    # If significant content was removed, log it
    if len(text) < original_length * 0.8:  # If more than 20% was removed
        from logging_utils import log
        from config import VERBOSE_DETAILED
        log(f"Stripped disclaimer from response (removed {original_length - len(text)} chars)", "debug", VERBOSE_DETAILED)
    
    return text

# Check model availability (Seems appropriate for utils.py)
def is_model_available(model_key):
    """
    Basic check if a model key exists in config and its API key is available.

    Args:
        model_key (str): The key for the model in TARGET_MODELS or ATTACKER_MODELS

    Returns:
        bool: True if the model seems configured and API key exists, False otherwise.
    """
    # from config import TARGET_MODELS, ATTACKER_MODELS, API_KEYS # Import necessary configs - already imported at top

    model_config = TARGET_MODELS.get(model_key) or ATTACKER_MODELS.get(model_key)

    if not model_config:
        log(f"Model key '{model_key}' not found in TARGET_MODELS or ATTACKER_MODELS.", "error")
        return False

    api_type = model_config.get("api")
    if not api_type:
         log(f"API type not defined for model '{model_key}' in config.", "error")
         return False

    # Map API type to the expected environment variable name
    api_key_name = None
    if api_type == "openai":
        api_key_name = "OPENAI_API_KEY"
    elif api_type == "together":
        api_key_name = "TOGETHER_API_KEY"
    elif api_type == "grok":
        api_key_name = "XAI_API_KEY"
    elif api_type == "anthropic":
        api_key_name = "ANTHROPIC_API_KEY"
    # Add other mappings if needed

    if not api_key_name:
        log(f"No known API key variable associated with API type '{api_type}' for model '{model_key}'.", "error")
        return False

    # Check if the required API key exists in the environment (don't prompt here)
    # check_api_key_existence will handle prompting later if needed, this is just a quick check
    if not os.getenv(api_key_name):
         log(f"Required API key '{api_key_name}' for model '{model_key}' (API: {api_type}) is not set in the environment. Will prompt if used.", "warning", VERBOSE_NORMAL)
         # Return True, as the prompt later will handle it.
         # If strict check desired: return False

    log(f"Model '{model_key}' appears to be configured.", "info", VERBOSE_DETAILED)
    return True # Basic configuration exists
