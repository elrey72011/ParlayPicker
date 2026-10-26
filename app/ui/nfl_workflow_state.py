"""Pure owner-session cleanup. Never opens storage or imports a provider."""


def revoke(state, *, keep_enabled=False):
    for key in list(state):
        if str(key).startswith('nfl_workflow_') and not (keep_enabled and key=='nfl_workflow_enabled'):
            del state[key]


def invalidate_review(state):
    for key in list(state):
        if str(key).startswith(('nfl_workflow_review_', 'nfl_workflow_pair_', 'nfl_workflow_result_')):
            del state[key]
