"""
Pytest configuration for torch-projectors tests.
This file sets up the test environment and makes imports work correctly.
"""

import sys
import os

# Add the project root to Python path so imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the tests directory to Python path for test_utils imports
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)