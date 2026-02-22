"""
test_variants.py - Verify primary team code resolution for Michigan St, Penn St, Ohio St.
Run: python -m pytest tests/test_variants.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_core.kalshi_integrator import generate_comprehensive_team_variants

def test_michigan_st_primary():
    """Michigan St Spartans must resolve to MSU as index 0."""
    variants = generate_comprehensive_team_variants("Michigan St Spartans", "NCAAB")
    assert len(variants) > 0, "No variants generated"
    assert variants[0] == "MSU", (
        f"Expected 'MSU' at index 0, got '{variants[0]}'. Full list: {variants[:10]}"
    )

def test_michigan_state_primary():
    """Michigan State (full name) must also resolve to MSU at index 0."""
    variants = generate_comprehensive_team_variants("Michigan State", "NCAAB")
    assert len(variants) > 0, "No variants generated"
    assert variants[0] == "MSU", (
        f"Expected 'MSU' at index 0, got '{variants[0]}'. Full list: {variants[:10]}"
    )

def test_penn_st_primary():
    """Penn St Nittany Lions must resolve to PSU as index 0."""
    variants = generate_comprehensive_team_variants("Penn St Nittany Lions", "NCAAB")
    assert len(variants) > 0, "No variants generated"
    assert variants[0] == "PSU", (
        f"Expected 'PSU' at index 0, got '{variants[0]}'. Full list: {variants[:10]}"
    )

def test_penn_state_primary():
    """Penn State (full name) must also resolve to PSU at index 0."""
    variants = generate_comprehensive_team_variants("Penn State", "NCAAB")
    assert len(variants) > 0, "No variants generated"
    assert variants[0] == "PSU", (
        f"Expected 'PSU' at index 0, got '{variants[0]}'. Full list: {variants[:10]}"
    )

def test_ohio_st_primary():
    """Ohio St Buckeyes must resolve to OSU as index 0."""
    variants = generate_comprehensive_team_variants("Ohio St Buckeyes", "NCAAB")
    assert len(variants) > 0, "No variants generated"
    assert variants[0] == "OSU", (
        f"Expected 'OSU' at index 0, got '{variants[0]}'. Full list: {variants[:10]}"
    )

def test_ohio_state_primary():
    """Ohio State (full name) must also resolve to OSU at index 0."""
    variants = generate_comprehensive_team_variants("Ohio State", "NCAAB")
    assert len(variants) > 0, "No variants generated"
    assert variants[0] == "OSU", (
        f"Expected 'OSU' at index 0, got '{variants[0]}'. Full list: {variants[:10]}"
    )

def test_msu_in_variants_for_michigan_st():
    """MSU must appear somewhere in the variants list for any Michigan St input."""
    for name in ["Michigan St", "Michigan St Spartans", "Michigan State", "Michigan State Spartans"]:
        variants = generate_comprehensive_team_variants(name, "NCAAB")
        assert "MSU" in variants, f"MSU not in variants for '{name}': {variants[:15]}"

def test_psu_in_variants_for_penn_st():
    """PSU must appear somewhere in the variants list for any Penn St input."""
    for name in ["Penn St", "Penn St Nittany Lions", "Penn State", "Penn State Nittany Lions"]:
        variants = generate_comprehensive_team_variants(name, "NCAAB")
        assert "PSU" in variants, f"PSU not in variants for '{name}': {variants[:15]}"

def test_osu_in_variants_for_ohio_st():
    """OSU must appear somewhere in the variants list for any Ohio St input."""
    for name in ["Ohio St", "Ohio St Buckeyes", "Ohio State", "Ohio State Buckeyes"]:
        variants = generate_comprehensive_team_variants(name, "NCAAB")
        assert "OSU" in variants, f"OSU not in variants for '{name}': {variants[:15]}"

def test_oregon_state_not_osu():
    """Oregon State should NOT have OSU as its primary code (Kalshi uses ORST)."""
    variants = generate_comprehensive_team_variants("Oregon State", "NCAAB")
    assert len(variants) > 0, "No variants generated for Oregon State"
    assert variants[0] != "OSU", (
        f"Oregon State incorrectly resolved to OSU at index 0. Full list: {variants[:10]}"
    )
    assert "ORST" in variants, (
        f"ORST not found in Oregon State variants: {variants[:15]}"
    )

if __name__ == "__main__":
    import traceback
    tests = [
        test_michigan_st_primary,
        test_michigan_state_primary,
        test_penn_st_primary,
        test_penn_state_primary,
        test_ohio_st_primary,
        test_ohio_state_primary,
        test_msu_in_variants_for_michigan_st,
        test_psu_in_variants_for_penn_st,
        test_osu_in_variants_for_ohio_st,
        test_oregon_state_not_osu,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {t.__name__}")
            print(f"     {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 ERROR: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
