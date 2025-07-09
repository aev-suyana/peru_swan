#%%
# test_data_preparation_0.py
import os
import sys
import importlib

# Add project root to path
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force reload config module if it's already loaded
if 'config' in sys.modules:
    importlib.reload(sys.modules['config'])

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
    script_path = 'scripts/data_preparation_0.py'
    
    if os.path.exists(script_path):
        print(f"\n✅ Script found: {script_path}")
        
        # Add scripts directory to path for proper imports
        scripts_dir = os.path.abspath('scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        
        print("\n🚀 Running data_preparation_0.py...")
        print("="*60)
        
        # Import and run the main function directly
        try:
            import data_preparation_0
            print("✅ Script module imported and executed successfully!")
            
        except Exception as script_error:
            print(f"❌ Error running script: {script_error}")
            import traceback
            traceback.print_exc()
        
    else:
        print(f"\n❌ Script not found: {script_path}")
        print("Available scripts:")
        if os.path.exists('scripts/'):
            for f in sorted(os.listdir('scripts/')):
                if f.endswith('.py'):
                    print(f"  {f}")
        else:
            print("  scripts/ directory not found")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
# %%
