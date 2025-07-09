#%%
# debug_data_prep_0.py
import os
import sys
sys.path.append('.')

try:
    from config import config
    print(f"✅ Config loaded: {config.RUN_PATH} → {config.reference_port}")
    
    # Test the script step by step
    print("\n🔍 Testing imports...")
    
    # Test individual imports from the script
    import pandas as pd
    print("✅ pandas imported")
    
    import numpy as np
    print("✅ numpy imported")
    
    import matplotlib.pyplot as plt
    print("✅ matplotlib imported")
    
    import seaborn as sns
    print("✅ seaborn imported")
    
    from config import get_input_files, get_output_files
    print("✅ config functions imported")
    
    print("\n🔍 Testing file paths...")
    input_files = get_input_files()
    
    # Check what files we need vs what exists
    critical_files = ['closed_ports', 'wave_height_csv']
    
    for name in critical_files:
        path = input_files[name]
        exists = os.path.exists(path)
        print(f"  {name}: {'✅' if exists else '❌'} {path}")
        
        if not exists:
            # Check if directory exists
            dir_exists = os.path.exists(os.path.dirname(path))
            print(f"    Directory exists: {'✅' if dir_exists else '❌'} {os.path.dirname(path)}")
    
    print("\n🔍 Testing script execution...")
    
    # Try to import the main function
    sys.path.insert(0, 'scripts')
    import data_preparation_0
    print("✅ Script module imported successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
# %%
