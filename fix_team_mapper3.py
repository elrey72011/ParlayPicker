with open("core/team_mapper.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if '"Golden State Warriors": "Golden State",' in line:
        new_lines.append('    "G-State": "Golden State",\n')
        new_lines.append('    "G-State Warriors": "Golden State",\n')
    elif '"Lehigh Mountain Hawks": "Lehigh",' in line:
        new_lines.append('    "Lehigh": "Lehigh",\n')
    elif '"Navy Midshipmen": "Navy",' in line:
        new_lines.append('    "Navy": "Navy",\n')
    elif '"George Washington Revolutionaries": "George Washington",' in line:
        new_lines.append('    "GW Revolutionaries": "George Washington",\n')
        new_lines.append('    "Revolutionaries": "George Washington",\n')
    elif '"UIC Flames": "UIC",' in line:
        new_lines.append('    "UIC": "UIC",\n')

with open("core/team_mapper.py", "w") as f:
    f.writelines(new_lines)
