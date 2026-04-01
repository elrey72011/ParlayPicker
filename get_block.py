with open('streamlit_app.py', 'r') as f:
    lines = f.readlines()
for i, line in enumerate(lines[715:735]):
    print(f"{i+715}: {line.rstrip()}")
