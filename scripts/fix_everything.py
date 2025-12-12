import os
import shutil

# --- CONFIGURATION ---
global_folder = r"C:\Users\Robert\.streamlit"
secrets_path = os.path.join(global_folder, "secrets.toml")
json_path = os.path.join(global_folder, "service_account.json")

print("\n--- 🛠️ STARTING REPAIR ---")

# 1. FIX THE SECRETS FILE NAME (The "Double Extension" Killer)
files = os.listdir(global_folder)
print(f"📂 Scanning {global_folder}...")
found_secrets = False

for filename in files:
    full_path = os.path.join(global_folder, filename)

    # Case A: Found the perfect file
    if filename == "secrets.toml":
        print("✅ Found correct 'secrets.toml'.")
        found_secrets = True

    # Case B: Found the "Hidden Extension" trap (secrets.toml.toml)
    elif filename == "secrets.toml.toml":
        print("⚠️ Found 'secrets.toml.toml'! Renaming it...")
        try:
            os.rename(full_path, secrets_path)
            print("✅ FIXED: Renamed to 'secrets.toml'")
            found_secrets = True
        except:
            print("❌ Error renaming. Close any open files and try again.")

    # Case C: Found a generic file named "secrets"
    elif filename == "secrets" and os.path.isfile(full_path):
        print("⚠️ Found file 'secrets' (no extension). Renaming...")
        os.rename(full_path, secrets_path)
        print("✅ FIXED: Renamed to 'secrets.toml'")
        found_secrets = True

if not found_secrets:
    print("❌ ERROR: Could not find any secrets file. Please create 'secrets.toml' manually.")

# 2. VERIFY THE KEY FILE
if os.path.exists(json_path):
    print("✅ Found 'service_account.json'.")
else:
    print("❌ ERROR: 'service_account.json' is MISSING.")
    print(f"👉 Please ensure you saved the JSON code into: {json_path}")

print("\n--- 📝 NEXT STEP: UPDATE YOUR APP CODE ---")
print("To verify the fix, STOP the current app (Trash Can icon) and run this script.")
print("Then, verify the output above is all GREEN CHECKMARKS.")