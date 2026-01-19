import sys
import os

# Add root to path
sys.path.append(os.getcwd())

try:
    import config
    print(f"✅ Config imported successfully.")
    if hasattr(config, "GEMINI_API_KEY"):
        print(f"✅ GEMINI_API_KEY found in config: {config.GEMINI_API_KEY[:5]}...")
        if config.GEMINI_API_KEY == "AIzaSyBIDJgxLuUouiBQrslV...":
             print("✅ GEMINI_API_KEY matches expected value.")
        else:
             print(f"❌ GEMINI_API_KEY does not match expected value. Got: {config.GEMINI_API_KEY}")
    else:
        print("❌ GEMINI_API_KEY NOT found in config.")

except ImportError as e:
    print(f"❌ Failed to import config: {e}")
