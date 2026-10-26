"""Session-local saved-viewer cleanup; no database or archive-reader imports."""


def revoke(state):
    for key in list(state):
        if str(key).startswith('nfl_saved_'):
            del state[key]


def clear_payloads(state):
    for key in list(state):
        if str(key).startswith('nfl_saved_zip_') and key != 'nfl_saved_zip_generation':
            del state[key]
    state['nfl_saved_loaded'] = []
    state['nfl_saved_selected'] = []
    state['nfl_saved_zip_generation'] = state.get('nfl_saved_zip_generation', 0) + 1
