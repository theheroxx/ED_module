"""
STEP 2 - DECISION TREE ANALYSIS  (refactored)
=============================================
Role in the ED[0-100] system:
  * SURROGATE / EXPLAINER for the ED baseline -- NOT an independent predictor.
    It fits a shallow, interpretable REGRESSION tree on the raw weather
    features to recover the effective piecewise THRESHOLDS ("risk rises above
    ~32C apparent") and FEATURE IMPORTANCE used by the baseline.
  * Confirms the baseline is monotonic/consistent and gives the report its
    "IF temp > X THEN high risk" decision paths.

Decoupled by design (per the plan):
  * It regenerates the ED label itself via common.ed_baseline (through
    common.preprocessing). It does NOT read Step 1's rules or pickle.
  * Target is CONTINUOUS ed_score (0-100), matching the final ED parameter,
    rather than 5 arbitrary classes.

UPDATED (2026-07-24): Optimized features based on decision tree results:
    - Added air_quality_PM2.5 (7.7% importance, used as root split)
    - Removed humidity, wind_kph, wind_chill_c, pressure_mb (0% importance)
    - Kept EPA index for regulatory alignment (57.7% importance)
    - Kept temperature (Celsius) and apparent_temp_c for splits
    - Kept uv_index (0.5% importance)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from common.preprocessing import get_processed_data
from common.io_utils import save_df, save_json, write_manifest, ensure_dir


# OPTIMIZED FEATURES - Based on decision tree importance

# DROPPED (0% importance): humidity, wind_kph, wind_chill_c, pressure_mb
# KEPT: EPA (57.7%), temperature (17.1%), apparent_temp (16.2%), 
#       PM2.5 (7.7%), uv_index (0.5%)

CANDIDATE_FEATURES = [
    # --- HIGH IMPORTANCE (≥ 7%) ---
    "air_quality_us-epa-index",        # 57.7% ← #1 (regulatory alignment)
    "temperature_celsius",             # 17.1% (used in tree splits)
    "apparent_temp_c",                 # 16.2% (used in tree splits)
    "air_quality_PM2.5",               # 7.7% ← root split in new tree
    
    # --- LOW IMPORTANCE (< 1%) ---
    "uv_index",                        # 0.5% (keep for completeness)
    
    # --- DROPPED (0% importance) ---
    # "humidity",                      # 0.0%
    # "wind_kph",                     # 0.0%
    # "wind_chill_c",                 # 0.0%
    # "pressure_mb",                  # 0.0%
    # "temperature_fahrenheit",       # Dropped (redundant with Celsius)
]


class ExerciseDangerDecisionTree:
    def __init__(self, max_depth=config.TREE_MAX_DEPTH,
                 min_samples_leaf=config.TREE_MIN_SAMPLES_LEAF):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.features = None
        self.metrics = {}

    def prepare(self, df):
        feats = [f for f in CANDIDATE_FEATURES if f in df.columns]
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        y = pd.to_numeric(df["ed_score"], errors="coerce").fillna(0.0)
        self.features = feats
        print(f"   Using {len(feats)} features: {feats}")
        return X, y

    def fit(self, X, y):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE)
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            random_state=config.RANDOM_STATE)
        self.model.fit(Xtr, ytr)
        pred = self.model.predict(Xte)
        self.metrics = {
            "r2": float(r2_score(yte, pred)),
            "mae": float(mean_absolute_error(yte, pred)),
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        }
        return self.model

    def feature_importance(self):
        imp = pd.DataFrame({"feature": self.features,
                            "importance": self.model.feature_importances_})
        return imp.sort_values("importance", ascending=False).reset_index(drop=True)

    def critical_thresholds(self):
        """Extract the split thresholds the tree uses -> danger boundaries."""
        t = self.model.tree_
        rows = []
        for i in range(t.node_count):
            if t.children_left[i] != t.children_right[i]:  # internal node
                rows.append({
                    "feature": self.features[t.feature[i]],
                    "threshold": float(t.threshold[i]),
                    "samples": int(t.n_node_samples[i]),
                })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (df.groupby("feature")
                  .agg(n_splits=("threshold", "size"),
                       min_threshold=("threshold", "min"),
                       median_threshold=("threshold", "median"),
                       max_threshold=("threshold", "max"))
                  .reset_index().sort_values("n_splits", ascending=False))

    def rules_text(self):
        return export_text(self.model, feature_names=list(self.features))


def main():
    print("=" * 70, "\nSTEP 2: DECISION TREE (baseline surrogate/explainer)\n", "=" * 70)
    df = get_processed_data()               # decoupled: recompute label itself
    dt = ExerciseDangerDecisionTree()
    X, y = dt.prepare(df)
    dt.fit(X, y)
    imp = dt.feature_importance()
    thr = dt.critical_thresholds()

    out = ensure_dir(config.STEP2_OUT)
    a = {}
    a["feature_importances"] = save_df(imp, out / "feature_importances.csv")
    a["critical_thresholds"] = save_df(thr, out / "critical_thresholds.csv")
    a["decision_rules_text"] = (out / "decision_rules.txt")
    (out / "decision_rules.txt").write_text(dt.rules_text(), encoding="utf-8")
    a["metrics"] = save_json(dt.metrics, out / "metrics.json")
    write_manifest(out, "step2_decision_tree", a, extra=dt.metrics)

    print(f"Surrogate fit  R2={dt.metrics['r2']:.3f}  MAE={dt.metrics['mae']:.2f} ED pts")
    print("Top features:\n", imp.head(5).to_string(index=False))
    print(f"Artifacts -> {out}")
    return dt


if __name__ == "__main__":
    main()