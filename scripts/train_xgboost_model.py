import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json
from pathlib import Path
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, brier_score_loss
from collections import Counter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import VERTEX_CONFIG
from core.model_validation import compare_candidate_to_market, select_best_candidate
from core.walk_forward import chronological_split

VERTEX_FEATURE_COLUMNS = VERTEX_CONFIG['feature_cols']

def build_features_from_master(df: pd.DataFrame) -> pd.DataFrame:
    """Maps columns from master_all_sports.csv to the strict 21 Vertex schema."""
    out = pd.DataFrame(index=df.index)

    mapping = {
        "implied_home_prob": "implied_home_prob",
        "sentiment_diff": "sentiment_diff",
        "kalshi_prob": "kalshi_prob",
        "injuries_home_count": "injuries_impact",
        "injuries_away_count": "injuries_away_count",
        "weather_flag": "weather_factor",
        "feature_home_win_pct": "home_win_pct",
        "feature_home_ppg": "home_ppg",
        "feature_home_oppg": "home_oppg",
        "feature_home_streak": "home_streak",
        "feature_away_win_pct": "away_win_pct",
        "feature_away_ppg": "away_ppg",
        "feature_away_oppg": "away_oppg",
        "feature_away_streak": "away_streak",
        "feature_diff_win_pct": "feature_diff_win_pct",
        "feature_diff_ppg": "feature_diff_ppg",
        "feature_diff_oppg": "feature_diff_oppg",
        "feature_diff_last5": "feature_diff_last5",
        "feature_diff_streak": "feature_diff_streak",
        "feature_home_rest_days": "rest_advantage",
        "feature_away_rest_days": "feature_away_rest_days"
    }

    # Determine the league for scaling if present
    if "league" in df.columns:
        is_nhl = df["league"].str.upper() == "NHL"
        is_mlb = df["league"].str.upper() == "MLB"
        is_ncaab = df["league"].str.upper() == "NCAAB"
        is_ncaaf = df["league"].str.upper() == "NCAAF"
    else:
        is_nhl = pd.Series(False, index=df.index)
        is_mlb = pd.Series(False, index=df.index)
        is_ncaab = pd.Series(False, index=df.index)
        is_ncaaf = pd.Series(False, index=df.index)

    for col in ["home_ppg", "home_oppg", "away_ppg", "away_oppg"]:
        if col in df.columns:
            # Scale NHL by 35x
            if is_nhl.any():
                df[col] = df[col].mask(is_nhl, df[col] * 35.0)
            # Scale MLB by 25x
            if is_mlb.any():
                df[col] = df[col].mask(is_mlb, df[col] * 25.0)
            # Scale NCAAB by 1.55x
            if is_ncaab.any():
                df[col] = df[col].mask(is_ncaab, df[col] * 1.55)
            # Scale NCAAF by 4.0x
            if is_ncaaf.any():
                df[col] = df[col].mask(is_ncaaf, df[col] * 4.0)

    # Calculate derived/synthetic columns if missing
    if "feature_diff_win_pct" not in df.columns:
        df["feature_diff_win_pct"] = df["home_win_pct"].fillna(0.5) - df["away_win_pct"].fillna(0.5)
    if "feature_diff_ppg" not in df.columns:
        df["feature_diff_ppg"] = df["home_ppg"].fillna(0.0) - df["away_ppg"].fillna(0.0)
    if "feature_diff_oppg" not in df.columns:
        df["feature_diff_oppg"] = df.get("home_oppg", pd.Series(0.0, index=df.index)).fillna(0.0) - df.get("away_oppg", pd.Series(0.0, index=df.index)).fillna(0.0)
    if "feature_diff_streak" not in df.columns:
         df["feature_diff_streak"] = df.get("home_streak", pd.Series(0.0, index=df.index)).fillna(0.0) - df.get("away_streak", pd.Series(0.0, index=df.index)).fillna(0.0)
    if "feature_diff_last5" not in df.columns:
         # Use win pct diff as proxy for last 5 diff if missing
         df["feature_diff_last5"] = df["feature_diff_win_pct"]

    # Default Kalshi
    if "kalshi_prob" not in df.columns:
         df["kalshi_prob"] = df.get("theover_probability", pd.Series(0.5, index=df.index))

    if "sentiment_diff" not in df.columns:
         df["sentiment_diff"] = df.get("sharp_vs_public", pd.Series(0.0, index=df.index))

    for target_col in VERTEX_FEATURE_COLUMNS:
        source_col = mapping.get(target_col)

        # Probs default to 0.5, counts/stats/diffs default to 0.0, just like production `app_core/prediction_engine.py`
        default_val = 0.5 if "prob" in target_col else 0.0

        if source_col and source_col in df.columns:
            out[target_col] = pd.to_numeric(df[source_col], errors='coerce').fillna(default_val)
        else:
            out[target_col] = default_val

    # Specific fallbacks for implied prob if missing or exactly 0.5
    if "implied_home_prob" in df.columns:
         mask = out["implied_home_prob"] == 0.5
         out.loc[mask, "implied_home_prob"] = pd.to_numeric(df.loc[mask, "home_win_pct"], errors='coerce').fillna(0.5)

    # Ensure all data is float to prevent XGBoost DMatrix issues
    out = out.astype(float)
    return out[VERTEX_FEATURE_COLUMNS]

def evaluate_probabilities(probabilities, outcomes):
    """Return proper probability scores plus diagnostic distribution metrics."""
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(outcomes, dtype=int)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    acc = float(accuracy_score(y_true, probs > 0.5))
    ll = float(log_loss(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))
    try:
        auc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc = 0.5

    unique_probs = int(len(np.unique(probs)))
    unique_ratio = float(unique_probs / len(probs)) if len(probs) else 0.0
    return {
        "acc": acc,
        "ll": ll,
        "brier": brier,
        "auc": auc,
        "unique_count": unique_probs,
        "unique_ratio": unique_ratio,
    }


def evaluate_model(model, X_val, y_val, name="Model"):
    """Evaluate a candidate on a future holdout using proper scoring rules."""
    dval = xgb.DMatrix(X_val)
    probs = model.predict(dval)
    metrics = evaluate_probabilities(probs, y_val)

    counts = Counter(probs)
    top_3 = sorted(
        [(float(p), int(c)) for p, c in counts.items()],
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    percentiles = np.percentile(probs, [0, 25, 50, 75, 100])
    spread_range = float(percentiles[4] - percentiles[0])

    print("\n" + "=" * 50)
    print(f"--- {name} Results ---")
    print(f"Validation Row Count: {len(probs)}")
    print(
        f"Accuracy: {metrics['acc']:.4f} | Log Loss: {metrics['ll']:.4f} | "
        f"Brier: {metrics['brier']:.4f} | AUC: {metrics['auc']:.4f}"
    )
    print(f"Raw Unique Count: {metrics['unique_count']} / {len(probs)}")
    print(f"Raw Unique Ratio (diagnostic only): {metrics['unique_ratio']:.1%}")
    print(f"Probability Spread (Max - Min): {spread_range:.4f}")
    print(
        "Distribution [Min, 25th, Median, 75th, Max]:\n"
        f"  [{percentiles[0]:.4f}, {percentiles[1]:.4f}, {percentiles[2]:.4f}, "
        f"{percentiles[3]:.4f}, {percentiles[4]:.4f}]"
    )
    print(f"Top 3 repeated probabilities and counts: {top_3}")
    print("=" * 50)

    return metrics["unique_ratio"], model, metrics


def main():
    print("Loading historical training data...")
    try:
        df = pd.read_csv('data/master_all_sports.csv')
    except Exception as e:
        print(f"ERROR: Could not load data/master_all_sports.csv: {e}")
        return

    # Sort by commence_time if available to do time-based split
    if 'commence_time' in df.columns:
        df['commence_time'] = pd.to_datetime(df['commence_time'], errors='coerce')
        df = df.sort_values('commence_time')

    # Target
    if 'home_won' not in df.columns:
        print("ERROR: 'home_won' target not found in historical data.")
        return

    y = df['home_won'].astype(int)
    X = build_features_from_master(df)

    # Honest future holdout. This uses the shared chronological helper and
    # deliberately keeps the final 25% out of every candidate fit.
    evaluation_frame = X.copy()
    evaluation_frame["home_won"] = y.to_numpy()
    evaluation_frame["commence_time"] = df["commence_time"].to_numpy()
    evaluation_frame["market_implied_prob"] = pd.to_numeric(
        df["implied_home_prob"], errors="coerce"
    ).to_numpy()
    if evaluation_frame["market_implied_prob"].isna().any():
        print("ERROR: implied_home_prob contains missing/non-numeric values; market benchmark is required.")
        return

    train_frame, val_frame = chronological_split(
        evaluation_frame,
        "commence_time",
        test_fraction=0.25,
        min_train_rows=100,
    )
    X_train = train_frame[VERTEX_FEATURE_COLUMNS]
    X_val = val_frame[VERTEX_FEATURE_COLUMNS]
    y_train = train_frame["home_won"].astype(int)
    y_val = val_frame["home_won"].astype(int)

    print(f"Schema exactly matches production 21-feature schema: {list(X.columns) == VERTEX_FEATURE_COLUMNS}")
    print(f"Train set: {X_train.shape}, chronological validation set: {X_val.shape}")
    print(
        f"Train end: {pd.to_datetime(train_frame['commence_time'], utc=True).max()} | "
        f"Validation start: {pd.to_datetime(val_frame['commence_time'], utc=True).min()}"
    )

    market_metrics = evaluate_probabilities(
        val_frame["market_implied_prob"],
        y_val,
    )
    print(
        "Market baseline on the same holdout: "
        f"Log Loss={market_metrics['ll']:.4f} | "
        f"Brier={market_metrics['brier']:.4f} | "
        f"AUC={market_metrics['auc']:.4f}"
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)

    models = {}

    # Config 1: Baseline approximation (Shallow, few trees - similar to what might be causing current coarseness)
    params_baseline = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 1: Baseline (Low Capacity)...")
    model_base = xgb.train(params_baseline, dtrain, num_boost_round=50)
    ratio, _, metrics = evaluate_model(model_base, X_val, y_val, "Candidate 1 (depth=3, n=50)")
    models['candidate_1_baseline'] = {'ratio': ratio, 'model': model_base, 'metrics': metrics}

    # Config 2: Medium Capacity
    params_med = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 2: Medium Capacity...")
    model_med = xgb.train(params_med, dtrain, num_boost_round=150)
    ratio, _, metrics = evaluate_model(model_med, X_val, y_val, "Candidate 2 (depth=5, n=150)")
    models['candidate_2_medium'] = {'ratio': ratio, 'model': model_med, 'metrics': metrics}

    # Config 3: High Capacity (More expressive, deeper, more trees)
    params_high = {
        'objective': 'binary:logistic',
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 2,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 3: High Capacity...")
    model_high = xgb.train(params_high, dtrain, num_boost_round=300)
    ratio, _, metrics = evaluate_model(model_high, X_val, y_val, "Candidate 3 (depth=7, n=300, lr=0.02)")
    models['candidate_3_high'] = {'ratio': ratio, 'model': model_high, 'metrics': metrics}

    # Config 4: Ultra Capacity with Regularization (Deepest, most trees, but reg_alpha/lambda)
    params_ultra = {
        'objective': 'binary:logistic',
        'max_depth': 9,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'logloss',
        'seed': 42
    }
    print("\nTraining Candidate 4: Ultra Capacity w/ L1/L2 Regularization...")
    model_ultra = xgb.train(params_ultra, dtrain, num_boost_round=500)
    ratio, _, metrics = evaluate_model(model_ultra, X_val, y_val, "Candidate 4 (depth=9, n=500, reg)")
    models['candidate_4_ultra'] = {'ratio': ratio, 'model': model_ultra, 'metrics': metrics}

    # Save candidates for inspection, but choose by proper scores rather than
    # probability uniqueness. A candidate cannot be promoted unless it beats
    # the raw market probability on the same future holdout.
    os.makedirs("models", exist_ok=True)
    for name, data in models.items():
        save_path = f"models/{name}.json"
        data["model"].save_model(save_path)
        metrics = data["metrics"]
        print(
            f"- Saved {name} to {save_path} "
            f"(Log Loss: {metrics['ll']:.4f}, Brier: {metrics['brier']:.4f}, "
            f"AUC: {metrics['auc']:.4f}, Unique Ratio: {data['ratio']:.1%})"
        )

    best_candidate_name = select_best_candidate(models)
    best_metrics = models[best_candidate_name]["metrics"]
    promotion = compare_candidate_to_market(best_metrics, market_metrics)

    sport_col = "sport" if "sport" in df.columns else "league" if "league" in df.columns else None
    sport_counts = (
        df[sport_col].astype("string").str.upper().value_counts().to_dict()
        if sport_col
        else {}
    )
    constant_features = [
        column
        for column in VERTEX_FEATURE_COLUMNS
        if X_train[column].nunique(dropna=False) <= 1
    ]
    integrity_reasons = []
    if len(df) < 1000:
        integrity_reasons.append(
            f"training set has only {len(df)} rows; at least 1000 are required for production promotion"
        )
    if len(sport_counts) != 1:
        integrity_reasons.append(
            "mixed-sport global model is not production eligible; train and validate one model per sport"
        )
    if len(constant_features) > len(VERTEX_FEATURE_COLUMNS) * 0.25:
        integrity_reasons.append(
            f"{len(constant_features)}/{len(VERTEX_FEATURE_COLUMNS)} model features are constant"
        )

    if integrity_reasons:
        promotion["promotable"] = False
        promotion["reasons"] = list(promotion.get("reasons", [])) + integrity_reasons

    report = {
        "out_of_sample": True,
        "target": "home_won",
        "supported_market_families": ["moneyline", "h2h"],
        "selection_metric_order": ["log_loss", "brier", "auc"],
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "train_end": str(pd.to_datetime(train_frame["commence_time"], utc=True).max()),
        "validation_start": str(pd.to_datetime(val_frame["commence_time"], utc=True).min()),
        "sport_counts": {str(key): int(value) for key, value in sport_counts.items()},
        "constant_features": constant_features,
        "market_baseline": market_metrics,
        "candidates": {
            name: data["metrics"]
            for name, data in models.items()
        },
        "selected_candidate": best_candidate_name,
        "promotion_gate": promotion,
    }
    report_path = Path("models/training_validation_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("RETRAINING HARNESS SUMMARY")
    print("=" * 50)
    print(
        f"Best candidate by log loss/Brier/AUC: {best_candidate_name.upper()} "
        f"(Log Loss: {best_metrics['ll']:.4f}, Brier: {best_metrics['brier']:.4f}, "
        f"AUC: {best_metrics['auc']:.4f})"
    )
    print(f"Validation report: {report_path}")
    if promotion["promotable"]:
        print("\nREADY: candidate beat the market benchmark and passed data-integrity gates.")
        print(
            f"Command to promote for moneyline/H2H only: "
            f"cp models/{best_candidate_name}.json models/xgboost_model_v2.json"
        )
    else:
        print("\nBLOCKED: do not promote this model.")
        for reason in promotion.get("reasons", []):
            print(f"- {reason}")


if __name__ == "__main__":
    main()
