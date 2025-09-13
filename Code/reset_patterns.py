#!/usr/bin/env python
"""
Pattern memory reset utility.

This script resets the pattern memory to its default state. Use it when you want to 
start fresh without any learned patterns from previous runs.

USAGE:
    python reset_patterns.py [--file PATH] [--confirm]
    
    --file PATH: Path to the successful_patterns.json file (optional)
    --confirm: Skip confirmation prompt (optional)

NOTE: This is a standalone utility script that is not called by the main codebase.
It's used for manual maintenance of the pattern learning system.
"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern_manager import PatternManager
from logging_utils import log
from config import SUCCESSFUL_PATTERNS_PATH

def reset_patterns(filepath=None):
    """
    Reset pattern memory to default values.
    
    Args:
        filepath (str, optional): Path to the pattern storage file
        
    Returns:
        bool: Whether the reset was successful
    """
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