"""NFL identities shared by schedule statistics and input matching."""
NFL_TEAMS = dict(zip(
    'ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LAR MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS'.split(),
    ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
     'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
     'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
     'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
     'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
     'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
     'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
     'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders']))


def nfl_stats_identity(value, *, schedule_code=False):
    key = ' '.join(str(value).upper().replace('.', '').split())
    # nfl_data_py uses LA for the Rams. A bare city from an upload is ambiguous.
    if schedule_code and key == 'LA':
        key = 'LAR'
    key = {'WSH': 'WAS', 'JAC': 'JAX', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LAR'}.get(key, key)
    if key in NFL_TEAMS:
        return NFL_TEAMS[key].upper()
    for full in NFL_TEAMS.values():
        full = full.upper()
        city = full.rsplit(' ', 1)[0]
        if key == full or (city not in {'NEW YORK', 'LOS ANGELES'} and key == city):
            return full
    return key
