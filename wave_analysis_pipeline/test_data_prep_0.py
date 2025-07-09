#%%
# test_data_prep_0.py
import os
import sys

# Add project root to path
sys.path.append('.')

# Import and test
try:
    from config import config
    print(f"✅ Config loaded: {config.RUN_PATH} → {config.reference_port}")
    
    # Check if required files exist
    from config import get_input_files
    input_files = get_input_files()
    
    print("\n📂 Checking input files:")
    for name, path in input_files.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {name}: {path}")
    
    # Run the script
    print("\n🚀 Running data_preparation_0.py...")
    exec(open('scripts/data_preparation_0.py').read())
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
# %%
