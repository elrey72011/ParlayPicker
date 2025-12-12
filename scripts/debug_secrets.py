import os

print("\n--- DIAGNOSTIC START ---")
current_dir = os.getcwd()
print(f"📂 Working in: {current_dir}")

# Define the expected path
expected_path = os.path.join(current_dir, "old_streamlit", "secrets.toml")
print(f"🔎 Looking for: {expected_path}")

if os.path.exists(expected_path):
    print("✅ SUCCESS: File found!")
    with open(expected_path, "r") as f:
        content = f.read()
        if "PROJECT_ID" in content:
            print("✅ CHECK: 'PROJECT_ID' is present inside the file.")
        else:
            print("⚠️ WARNING: File exists but might be empty or missing PROJECT_ID.")
else:
    print("❌ FAILURE: File NOT found.")
    print("👉 Check if the folder is named 'old_streamlit' (with a dot).")
    print("👉 Check if the file is named 'secrets.toml.txt' by mistake.")

print("--- DIAGNOSTIC END ---\n")