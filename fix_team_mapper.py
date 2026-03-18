import re

file_path = 'core/team_mapper.py'
with open(file_path, 'r') as f:
    content = f.read()

# Add aliases to TEAM_MAP
replacements = {
    '"golden state warriors": "Golden State",': '"golden state warriors": "Golden State",\n    "g-state": "Golden State",\n    "g-state warriors": "Golden State",',
    '"lehigh mountain hawks": "Lehigh",': '"lehigh mountain hawks": "Lehigh",\n    "lehigh": "Lehigh",',
    '"navy midshipmen": "Navy",': '"navy midshipmen": "Navy",\n    "navy": "Navy",',
    '"george washington revolutionaries": "George Washington",': '"george washington revolutionaries": "George Washington",\n    "gw revolutionaries": "George Washington",\n    "revolutionaries": "George Washington",',
    '"uic flames": "UIC",': '"uic flames": "UIC",\n    "uic": "UIC",',
}

for old, new in replacements.items():
    if old in content and new not in content:
        content = content.replace(old, new)
    else:
        print(f"Skipping {old} in {file_path}")

with open(file_path, 'w') as f:
    f.write(content)

print("Updated team_mapper.py")
