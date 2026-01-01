import re

with open("streamlit_app.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if ".apply(" in line:
        # Check next few lines for copy()
        found_copy = False
        for j in range(1, 5): # Check next 4 lines
            if i + j >= len(lines): break
            next_line = lines[i+j].strip()
            if ".copy()" in next_line:
                found_copy = True
                break

        if not found_copy:
            print(f"Line {i+1}: {line.strip()}")
            for k in range(1, 4):
                if i+k < len(lines):
                    print(f"  + {lines[i+k].strip()}")
