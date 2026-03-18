with open("core/team_mapper.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)

    if '"golden state warriors": "Golden State"' in line:
        new_lines.append('    "g-state": "Golden State",\n')
        new_lines.append('    "g-state warriors": "Golden State",\n')

    elif '"lehigh mountain hawks": "Lehigh"' in line:
        new_lines.append('    "lehigh": "Lehigh",\n')

    elif '"navy midshipmen": "Navy"' in line:
        new_lines.append('    "navy": "Navy",\n')

    elif '"george washington revolutionaries": "George Washington"' in line:
        new_lines.append('    "gw revolutionaries": "George Washington",\n')
        new_lines.append('    "revolutionaries": "George Washington",\n')

    elif '"uic flames": "UIC"' in line:
        new_lines.append('    "uic": "UIC",\n')

with open("core/team_mapper.py", "w") as f:
    f.writelines(new_lines)
