from copy import deepcopy
from app_core.public_history import digest
from app_core.locked_picks import lock_candidates
from app_core.public_reconciliation import reconcile, counts, load_source
from test_public_history import pub, scores, Memory, History


def source():
    publication = pub()
    for rows in publication["package"]["games"].values():
        for leg in rows:
            leg["start"] = leg["start"].replace("09-09","09-15")
            leg["as_of"] = leg["as_of"].replace("09-09","09-15")
    publication["confirmed_at"] = publication["confirmed_at"].replace("09-09","09-15")
    publication["package_hash"] = digest(publication["package"])
    locks = lock_candidates(publication["package"],publication["confirmed_at"])
    for item in locks:
        item["legs"][0]["pick"] = "Boston -1.5"
        item["legs"][0]["odds"] = -180
    finals = scores()
    finals[0]["start"] = finals[0]["start"].replace("09-09","09-15")
    return dict(publications=[publication],locks=locks,revisions=[dict(recorded_at="2026-09-16T03:00:00Z",scores=finals)])


def test_separate_original_locks_read_only():
    original = source(); before = deepcopy(original)
    result = reconcile(original,"2026-09-15")
    assert original == before
    assert result["cohorts"]["Locked Overall"]["losses"] == 1
    assert result["cohorts"]["Published Overall"]["wins"] == 1
    locked = next(r for r in result["records"] if r["group"] == "Locked")
    assert locked["legs"][0]["line"] == -1.5 and locked["legs"][0]["odds"] == -180
    client = Memory(); store = History("site-1234","folder",client)
    p = original["publications"][0]; h = store.archive(p["package"])
    store.confirm("deploy-123",h,p["confirmed_at"])
    data_before = deepcopy(client.data)
    assert load_source(store)["publications"] == original["publications"]
    assert client.data == data_before


def test_duplicates_conflicts_and_denominators():
    original = source()
    original["locks"].append(deepcopy(original["locks"][0]))
    result = reconcile(original,"2026-09-15")
    assert result["cohorts"]["Locked Overall"]["records"] == 1
    original["locks"][1]["legs"][0]["odds"] = -150
    result = reconcile(original,"2026-09-15")
    assert result["cohorts"]["Locked Overall"]["needs_review"] == 1
    assert result["cohorts"]["Locked Overall"]["win_rate"] is None
    assert counts([{"outcome":o} for o in ["WIN","LOSS","PUSH","PENDING","NEEDS_REVIEW"]],"2026-09-15")["win_rate"] == .5


def test_corrupt_publication_is_not_treated_as_verified():
    original = source();original["publications"][0]["package"]["games"]["overall"][0]["odds"] = 120
    result = reconcile(original,"2026-09-15")
    assert result["status"] == "NEEDS_REVIEW" and result["source_errors"]


def test_first_publication_identity_collision_needs_review():
    original = source()
    publication = original['publications'][0]
    conflicting = deepcopy(publication['package']['games']['overall'][0])
    conflicting['odds'] = -120
    publication['package']['games']['overall'].append(conflicting)
    publication['package_hash'] = digest(publication['package'])
    result = reconcile(original,'2026-09-15')
    assert result['cohorts']['Published Overall']['needs_review'] == 1
    assert result['cohorts']['Published Overall']['win_rate'] is None
