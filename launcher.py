#!/usr/bin/env python3
"""
Launcher script for PyInstaller packaging.
Imports and runs the OmniVoice Gradio Demo.
"""
import os
import sys

# Ensure the root folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omnivoice.cli.demo import main

if __name__ == "__main__":
    main()
