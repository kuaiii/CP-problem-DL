# -*- coding: utf-8 -*-
import sys
import os

# 自动将项目根目录添加到 sys.path，防止直接运行时找不到 'codes' 模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from main import main

class MockArgs:
    def __init__(self, dataset='GtsCe', synthetic=True, nodes=50, rate=0.1, batch_size=1, attack='target', metric='efficiency'):
        self.dataset = dataset
        self.synthetic = synthetic
        self.nodes = nodes
        self.rate = rate
        self.batch_size = batch_size
        self.attack = attack
        self.metric = metric

def test_main():
    """
    Function for debugging with fixed parameters.
    It bypasses command line argument parsing by patching sys.argv or mocking argparse.
    Here we will use a simpler approach: calling the internal logic of main if we could, 
    but since main() parses args inside, we need to trick it or refactor it.
    
    Given main() uses parser.parse_args(), we can mock sys.argv.
    """
    
    # Define fixed parameters for debugging
    DEBUG_PARAMS = [
        'main.py',
        '--synthetic',
        '--nodes', '50',
        '--attack', 'degree',
        '--metric', 'gcc',
        '--batch_size', '1'
    ]
    
    print(f"Running debug test with params: {DEBUG_PARAMS}")
    
    # Backup original sys.argv
    original_argv = sys.argv
    
    try:
        # Override sys.argv
        sys.argv = DEBUG_PARAMS
        
        # Run main
        main()
        
    except SystemExit as e:
        print(f"SystemExit caught: {e}")
    except Exception as e:
        print(f"Exception caught during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore sys.argv
        sys.argv = original_argv

if __name__ == "__main__":
    test_main()
