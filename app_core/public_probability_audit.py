"""Read-only diagnostics of original public estimates; never wager authority."""
from collections import Counter, defaultdict
from itertools import combinations
import math
from app_core.result_reconciliation import stamp


def probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 < value < 1


def summary(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    wins = sum(y for p, y in rows)
    rate = wins / n
    z = 1.96
    center = (rate + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))/(1+z*z/n)
    return dict(n=n, wins=wins, actual_win_rate=rate,
                mean_estimate=sum(p for p,y in rows)/n,
                brier_score=sum((p-y)**2 for p,y in rows)/n,
                descriptive_wilson_95=[center-half, center+half])


def build(reconciliations, *, holdout_start=None):
    """Separate groups/categories; deduplicate repeated exports and reject conflicts."""
    from app_core.public_history import digest
    if holdout_start is not None:
        from datetime import date
        date.fromisoformat(holdout_start)
    baselines = defaultdict(list)
    variants = defaultdict(dict)
    for report in reconciliations:
        if report.get('source_errors'):
            continue
        for row in report.get('records', []):
            variants[(row.get('date'), row.get('id'))][digest(row)] = row
    exclusions = Counter()
    cohorts, slates = defaultdict(list), defaultdict(list)
    pairs, exposure = {}, Counter()
    comparisons = defaultdict(lambda: defaultdict(list))
    for key, versions in variants.items():
        if not key[1] or len(versions) != 1:
            exclusions['conflicting_or_missing_record_identity'] += 1
            continue
        row = next(iter(versions.values()))
        if row.get('outcome') not in {'WIN', 'LOSS'}:
            exclusions['push_pending_or_review'] += 1
            continue
        legs = row.get('legs', [])
        at = stamp(row.get('published_at'))
        if not legs or not at or any(not stamp(l.get('start')) or at >= stamp(l['start']) for l in legs):
            exclusions['unverified_pregame_record'] += 1
            continue
        if any(not probability(l.get('original_win_estimate')) or l.get('outcome') not in {'WIN','LOSS'} for l in legs):
            exclusions['missing_probability_or_decisive_leg_outcome'] += 1
            continue
        group = row.get('group')
        if len(legs) == 1 and row.get('category') != 'parlays':
            leg = legs[0]
            family = str(leg.get('market_type','')).split('_')[0]
            cohort = (group, row.get('category'), leg.get('league'), family)
            value = (leg['original_win_estimate'], int(leg['outcome']=='WIN'))
            cohorts[cohort].append(value)
            odds = leg.get('odds')
            if isinstance(odds, (int, float)) and not isinstance(odds, bool) and math.isfinite(odds) and abs(odds) >= 100:
                implied = -odds/(100-odds) if odds < 0 else 100/(100+odds)
                baselines[cohort].append((row['date'], value[0], value[1], implied))
            if row.get('category') == 'overall':
                label = 'locked' if group == 'Locked' else 'published'
                if group in {'Locked', 'Approved', 'Research'}:
                    comparisons[(row['date'], leg.get('league'), leg.get('game'), leg.get('start'))][label].append(leg)
            slates[(cohort, row['date'])].append(value)
        elif row.get('category') == 'parlays':
            def identity(l):
                return (l.get('league'), l.get('game'), l.get('start'), l.get('selection'), l.get('odds'))
            for leg in legs:
                exposure[(group, row['date'], identity(leg))] += 1
            for a,b in combinations(sorted(legs,key=lambda l:str(identity(l))),2):
                pair_key = (group, row['date'], identity(a), identity(b))
                value = (a['original_win_estimate'], b['original_win_estimate'], int(a['outcome']=='WIN'), int(b['outcome']=='WIN'))
                if pair_key in pairs and pairs[pair_key] != value:
                    pairs[pair_key] = None
                else:
                    pairs[pair_key] = value
    results = []
    for cohort, rows in sorted(cohorts.items(), key=lambda x:str(x[0])):
        bands = defaultdict(list)
        for p,y in rows:
            bands[min(19,int(p*20))].append((p,y))
        concordant = discordant = 0
        days = 0
        for (c,day), values in slates.items():
            if c != cohort:
                continue
            days += 1
            for (p,y),(q,z) in combinations(values,2):
                if p == q or y == z:
                    continue
                if (p-q)*(y-z)>0: concordant += 1
                else: discordant += 1
        n = concordant + discordant
        results.append(dict(group=cohort[0], category=cohort[1], sport=cohort[2], market=cohort[3],
                            **summary(rows), dates=days,
                            price_baseline=baseline_comparison(baselines[cohort]),
                            chronological_comparison=(dict(
                                before_cutoff=baseline_comparison([r for r in baselines[cohort] if r[0] < holdout_start]),
                                on_or_after_cutoff=baseline_comparison([r for r in baselines[cohort] if r[0] >= holdout_start])) if holdout_start else None),
                            probability_bands=[dict(lower=b/20, upper=(b+1)/20, **summary(v)) for b,v in sorted(bands.items())],
                            ranking_comparable_pairs=n,
                            ranking_concordance=concordant/n if n else None))
    pair_groups = defaultdict(list)
    for key, value in pairs.items():
        if value is not None:
            pair_groups[key[0]].append((key[1],value))
    dependence=[]
    for group, values in sorted(pair_groups.items()):
        v=[x[1] for x in values]; n=len(v)
        x=sum(a for p,q,a,b in v)/n; y=sum(b for p,q,a,b in v)/n
        joint=sum(a*b for p,q,a,b in v)/n
        denom=math.sqrt(x*(1-x)*y*(1-y))
        dependence.append(dict(group=group, unique_pairs=n, dates=len({d for d,_ in values}),
            both_win_rate=joint, both_loss_rate=sum((1-a)*(1-b) for p,q,a,b in v)/n,
            independence_estimated_both_win=sum(p*q for p,q,a,b in v)/n,
            pooled_outcome_phi=(joint-x*y)/denom if denom else None))
    changes = []
    for event, sides in comparisons.items():
        if len(sides['locked']) == len(sides['published']) == 1:
            a, b = sides['published'][0], sides['locked'][0]
            if any(a.get(k) != b.get(k) for k in ('selection','odds','original_win_estimate')):
                changes.append(dict(date=event[0], sport=event[1], game=event[2], published=a, locked=b))
    return dict(holdout_start=holdout_start, selection_changes=changes, schema_version=2, descriptive_only=True, cohorts=results, parlay_dependence=dependence,
                repeated_leg_exposure=[dict(group=k[0],date=k[1],leg=list(k[2]),tickets=v) for k,v in exposure.items() if v>1],
                exclusions=dict(exclusions),
                limitations=['Single-sided implied-price baseline includes vig; it is not a no-vig fair probability.',
                  'Chronological comparison is exploratory unless the cutoff and model were frozen before the evaluation period; it never authorizes promotion.',
                  'No ranking, probability, stake, or validation changes.',
                  'Categories overlap; compare separately. Pushes excluded; probability semantics may be unverified.',
                  'Wilson intervals assume independent games and are descriptive; shared-day/model dependence can widen uncertainty.',
                  'Pair observations share legs. Pooled phi is descriptive, not a validated correlation adjustment or causal finding.',
                  'Joint-probability discrepancies can reflect miscalibration as well as dependence. Multiple dates and independent holdout validation are required.'])


def baseline_comparison(rows):
    if not rows:
        return {'n': 0, 'status': 'NO_PAIRED_PRICE_EVIDENCE'}
    n = len(rows)
    model = sum((p-y)**2 for d,p,y,q in rows)/n
    price = sum((q-y)**2 for d,p,y,q in rows)/n
    return dict(n=n, dates=len({r[0] for r in rows}), model_brier=model,
                single_sided_price_brier=price, model_minus_price_brier=model-price,
                status='DESCRIPTIVE_NOT_VALIDATION')
