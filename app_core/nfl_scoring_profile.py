"""Explicit NFL-only release CSV profile. No permissions or authority creation."""
PROFILE = 'nflverse-schedules-scoring-4m-v1'
MAX_WIRE = 4 * 1024 * 1024
MAX_BODY = 4 * 1024 * 1024
MAX_TOTAL = 6 * 1024 * 1024
MAX_PACKET = 12 * 1024 * 1024
MAX_SELECTED = 16 * 1024 * 1024
MAX_OBJECTS = 16
MAX_HEADER_BYTES = 16 * 1024
ENDPOINT = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv'
SOURCE_ID = 'nflverse:nflverse-data:schedules:games.csv'
SOURCE = dict(repository='nflverse/nflverse-data', repository_id=452908115,
              release_tag='schedules', asset_name='games.csv', endpoint=ENDPOINT,
              license='CC-BY-4.0', license_url='https://github.com/nflverse/nflverse-data/blob/a78da18ca93f167bd0a2af274df48f539fa4153c/LICENSE.md',
              attribution='NFL game/schedule data maintained by Lee Sharpe; distributed by nflverse/nflverse-data.',
              modifications='Original CSV unchanged; separately identified projections and derived features.',
              publisher_availability='UNKNOWN')
