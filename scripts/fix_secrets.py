import os

# The folder where the file lives
folder = r"C:\Users\Robert\old_streamlit"
bad_file = os.path.join(folder, "secrets")  # The file with no extension
correct_file = os.path.join(folder, "secrets.toml")  # The file we WANT

print(f"📂 Checking folder: {folder}")

# 1. Check if the "bad" file exists
if os.path.exists(bad_file):
    print("Found the file named 'secrets' (Type: File)...")

    # Read the keys you already pasted inside it
    with open(bad_file, "r") as f:
        my_keys = f.read()

    # Write them into the NEW correct file
    with open(correct_file, "w") as f:
        f.write(my_keys)

    print(f"✅ SUCCESS: Created '{correct_file}' with your keys inside!")

    # Optional: Remove the old confusing one
    try:
        os.remove(bad_file)
        print("🗑️ Deleted the old confusing file.")
    except:
        pass

elif os.path.exists(correct_file):
    print("✅ The file 'secrets.toml' already exists! You should be good.")

else:
    print("❌ Could not find the file 'secrets'. Make sure you didn't delete it!")

print("\n👉 NOW: Go restart 'streamlit run streamlit_app.py'")