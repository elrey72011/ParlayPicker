"""Typed consumer extension over unchanged native discrete-margin mathematics."""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from app_core import ncaaf_model_compatibility as model, ncaaf_research as native
from app_core.ncaaf_research_contract import target_contract as old_target

VERSION='ncaaf-native-discrete-spread-masses-v1'
NATIVE_SHA256='e85a82eb585967c0f9fda8d483aecb8155d982a471b20a4359eda340046d579b'
TOLERANCE=1e-12
require=model.require


def selected_policy(policy):
    from app_core.ncaaf_owner_spread import ReviewPolicy
    return type(policy) is ReviewPolicy


def target_contract(kind,line,*,integer):
    require(kind in {'spread_home','spread_away'} and type(line) in (int,float) and math.isfinite(line), 'NCAAF_SPREAD_TARGET')
    require(float(line).is_integer() if integer else abs(line%1)==.5,'NCAAF_SPREAD_LINE_CONTRACT')
    result=old_target(kind,line)
    result['version']='ncaaf-integer-spread-target-v1' if integer else 'ncaaf-named-half-point-spread-target-v1'
    return result


def normalize(value):
    require(isinstance(value,dict) and set(value)=={'cover','push','loss'}
        and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1 for x in value.values())
        and math.isclose(sum(value.values()),1.,rel_tol=0,abs_tol=TOLERANCE), 'NCAAF_SPREAD_MASS_CONFLICT')
    return deepcopy(value)


def conditional_no_push(mass):
    mass=normalize(mass);denominator=1-mass['push']
    return dict(semantics='cover_conditional_on_no_push',state='VALUE' if denominator>TOLERANCE else 'UNKNOWN',
        value=mass['cover']/denominator if denominator>TOLERANCE else None)


def from_native(mass,kind):
    require(kind in {'spread_home','spread_away'} and isinstance(mass,dict) and set(mass)=={'over','under','push'},'NCAAF_SPREAD_MASS_CONFLICT')
    return normalize(dict(cover=mass['over' if kind=='spread_home' else 'under'],push=mass['push'],loss=mass['under' if kind=='spread_home' else 'over']))


def consumer_binding():
    require(hashlib.sha256(Path(native.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==NATIVE_SHA256,'NCAAF_COMPAT_MATH_CHANGED')
    return dict(version=VERSION,native_sha256=NATIVE_SHA256,primitive='ncaaf_research.probabilities',
        original_model_declarations_preserved=True,mathematics_changed=False,calibrated=False)


def inspect_receipt(receipt,kind):
    mass=normalize(receipt.get('outcome_probabilities'))
    require(kind in {'spread_home','spread_away'} and receipt.get('target',{}).get('selection')==kind.rsplit('_',1)[-1]
        and receipt.get('probability_semantics')=='unconditional_cover_push_loss'
        and receipt.get('native_mass_binding')==consumer_binding()
        and receipt.get('raw_probability')==mass['cover'] and receipt.get('push_probability')==mass['push']
        and receipt.get('conditional_no_push')==conditional_no_push(mass), 'NCAAF_SPREAD_MASS_CONFLICT')
    if receipt['target']['push']=='structural_zero_at_half_point': require(mass['push']==0,'NCAAF_SPREAD_MASS_CONFLICT')
    return mass


def compare(home,away):
    home,away=normalize(home),normalize(away)
    require(all(math.isclose(x,y,rel_tol=0,abs_tol=TOLERANCE) for x,y in
        ((home['cover'],away['loss']),(home['loss'],away['cover']),(home['push'],away['push'])))
        and math.isclose(home['cover']+away['cover']+home['push'],1.,rel_tol=0,abs_tol=TOLERANCE), 'NCAAF_PAIR_COMPLEMENT_CONFLICT')
    delta=home['cover']-away['cover']
    return dict(home_cover=home['cover'],away_cover=away['cover'],shared_push=home['push'],difference=delta,
        higher_cover='tie' if abs(delta)<=TOLERANCE else 'home' if delta>0 else 'away',comparison_tolerance=TOLERANCE,
        meaning='higher_unconditional_model_cover_estimate_not_EV_or_validated_recommendation')
