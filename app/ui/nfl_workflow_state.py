"""Pure owner-session cleanup. Never opens storage or imports a provider."""


def revoke(state, *, keep_enabled=False):
    for key in list(state):
        if str(key).startswith('nfl_workflow_') and not (keep_enabled and key in {
                'nfl_workflow_enabled', 'nfl_workflow_diag_enabled',
                'nfl_workflow_diag_spent', 'nfl_workflow_diag_result'}):
            del state[key]


def invalidate_review(state):
    for key in list(state):
        if str(key).startswith(('nfl_workflow_review_', 'nfl_workflow_pair_', 'nfl_workflow_result_')):
            del state[key]
