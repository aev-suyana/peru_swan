#%%
# test_data_preparation_0.py
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
    
    # Check if script exists with correct name
    script_path = 'scripts/data_preparation_0.py'  # Corrected filename
    
    if os.path.exists(script_path):
        print(f"\n✅ Script found: {script_path}")
        
        # Run the script
        print("\n🚀 Running data_preparation_0.py...")
        exec(open(script_path).read())
        
    else:
        print(f"\n❌ Script not found: {script_path}")
        print("Available scripts:")
        if os.path.exists('scripts/'):
            for f in os.listdir('scripts/'):
                print(f"  {f}")
        else:
            print("  scripts/ directory not found")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
# %%
