from copy import deepcopy
from app_core.public_record import current_records
from scripts.publish_board import render
from test_public_history import pub, report


def test_cutoff_covers_every_group_and_category_without_mutation():
    rows=[{'date':day,'category':category,'group':group,'id':day+category+group}
          for day in ('2026-09-10','2026-09-11','2026-09-12')
          for category in ('overall','sides','totals','parlays','props')
          for group in ('Locked','Approved','Research','Imported research')]
    before=deepcopy(rows)
    selected=current_records(rows)
    assert len(selected)==40
    assert all(r['date']>='2026-09-11' for r in selected)
    assert rows==before


def test_render_excludes_older_results_without_rewriting_archive():
    package=pub()['package']
    package.update(schema_version=5,parlays=[],results=report([pub()],[]))
    today={**package['results'][0],'id':'new-record','date':'2026-09-11'}
    package['results'].append(today)
    before=deepcopy(package)
    html=render(package)
    assert 'Fresh record' in html
    # Public results are removed from the serialized package, not just hidden by CSS.
    assert 'new-record' in html
    assert package['results'][0]['id'] not in html
    assert package==before


def test_update_leaves_pre_start_grades_archived(monkeypatch):
    from app.ui import public_results
    from types import SimpleNamespace
    saved={'publications':[pub()],'rows':report([pub()],[]),'revisions':[],'locks':[],'imports':[]}
    monkeypatch.setattr(public_results,'history',lambda setting:SimpleNamespace())
    def unexpected(*args):raise AssertionError('Older results must not be regraded by the fresh-record action')
    monkeypatch.setattr(public_results,'fetch_scores',unexpected)
    before=deepcopy(saved)
    public_results.update_pending_results(lambda key:'',saved)
    assert saved==before
