#!/usr/bin/env python3
"""
Test runner script for ML charge modeling visualization tests.
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    # Discover and run all tests in the current directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())