import unittest
import sys
import os

# Add the parent directory (shjnn) to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    print("Running all tests for SHJNN library...")
    
    # Use unittest's discovery to find and run all tests
    test_directory = os.path.dirname(os.path.abspath(__file__))
    test_suite = unittest.defaultTestLoader.discover(test_directory, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Set exit code based on test results
    sys.exit(0 if result.wasSuccessful() else 1)