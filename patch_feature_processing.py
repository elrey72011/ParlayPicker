import re

file_path = "app_core/feature_processing.py"

with open(file_path, "r") as f:
    content = f.read()

search_block = """    # Proactive NHL Location-Only mapping validation logging
    if is_nhl:
        # Check if the name matches one of the known risky location-only names
        nhl_location_check = {"COLORADO", "FLORIDA", "CAROLINA", "TAMPA BAY", "NEW JERSEY", "SAN JOSE", "VEGAS", "DALLAS"}

        # We only log mapping success for these explicit shorthand names, not every normal NHL team
        temp_norm = normalize_team(name_upper)
        if name_upper in nhl_location_check or temp_norm in nhl_location_check:
            mapped_val = MANUAL_TEAM_OVERRIDES.get(name_upper, MANUAL_TEAM_OVERRIDES.get(temp_norm))
            if mapped_val:
                # Explicitly format as requested by the user
                logger.info(f"NHL TEAM MAPPING: '{name}' -> '{mapped_val}'")
                return mapped_val
            else:
                logger.warning(f"⚠️ NHL Mapping Validation: '{name}' normalized to raw location '{temp_norm}'. We expect this to be mapped to the full team name in MANUAL_TEAM_OVERRIDES.")"""

replace_block = """    # Proactive NHL Location-Only mapping validation logging
    if is_nhl:
        # Check if the name matches one of the known risky location-only names
        nhl_location_check = {"COLORADO", "FLORIDA", "CAROLINA", "TAMPA BAY", "NEW JERSEY", "SAN JOSE", "VEGAS", "DALLAS"}

        # We only log mapping success for these explicit shorthand names, not every normal NHL team
        # NOTE: name is the original string provided (e.g., 'Colorado')
        # name_upper is the stripped/upper version
        temp_norm = normalize_team(name_upper)

        # Check if the original name, upper name, or normalized name is just the location
        # A lot of times, `name` comes in as "Colorado" or "Florida"
        if name_upper in nhl_location_check or temp_norm in nhl_location_check or name.upper().strip() in nhl_location_check:
            mapped_val = MANUAL_TEAM_OVERRIDES.get(name_upper, MANUAL_TEAM_OVERRIDES.get(temp_norm, MANUAL_TEAM_OVERRIDES.get(name.upper().strip())))
            if mapped_val:
                # Explicitly format as requested by the user
                logger.info(f"NHL TEAM MAPPING: '{name}' -> '{mapped_val}'")
                return mapped_val
            else:
                logger.warning(f"⚠️ NHL Mapping Validation: '{name}' normalized to raw location '{temp_norm}'. We expect this to be mapped to the full team name in MANUAL_TEAM_OVERRIDES.")"""

content = content.replace(search_block, replace_block)

with open(file_path, "w") as f:
    f.write(content)
