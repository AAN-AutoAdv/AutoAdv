import os
import sys
import argparse

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_manager import PatternManager
from logging_utils import log
from config import SUCCESSFUL_PATTERNS_PATH

def reset_patterns(filepath=None):
    try:
        pattern_manager = PatternManager(filepath)
        pattern_manager.reset()
        return True
    except Exception as e:
        log(f"Error resetting patterns: {e}", "error")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset pattern memory to default values")
    parser.add_argument("--file", type=str, default=SUCCESSFUL_PATTERNS_PATH,
                        help="Path to the pattern storage file")
    parser.add_argument("--confirm", action="store_true",
                        help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    if not args.confirm:
        confirm = input("This will reset all learned patterns. Are you sure? (y/n): ")
        if confirm.lower() != 'y':
            log("Reset cancelled.", "info")
            sys.exit(0)
    
    if reset_patterns(args.file):
        log("Successfully reset pattern memory.", "success")
    else:
        log("Failed to reset pattern memory.", "error")
        sys.exit(1)