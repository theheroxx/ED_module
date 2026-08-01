"""
STEP 5 - ANOMALY DETECTION  (PIPELINE‑INTEGRATED, FIXED)
==========================================================
Uses the processed parquet data (same as all other steps).
Detects anomalies using:
  - Statistical outliers (IQR on weather features)
  - Contextual anomalies (ED component mismatches) - FIXED
  - ML-based (Isolation Forest)
Outputs: anomalies.csv, flags, summary.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from common.preprocessing import get_processed_data
from common.io_utils import save_df, save_json, write_manifest, ensure_dir

# ----------------------------------------------------------------------------
# Features to check for anomalies
# ----------------------------------------------------------------------------
WEATHER_FEATURES = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "uv_index",
    "apparent_temp_c",
    "pressure_mb",
    "air_quality_us-epa-index",
    "air_quality_PM2.5",
]


class AnomalyDetector:
    def __init__(self, df, contamination=0.05):
        self.df = df.reset_index(drop=True).copy()
        self.contamination = contamination
        self.flags = pd.DataFrame(index=self.df.index)

    # --------------------------------------------------------------------
    # 1. Statistical Outliers (IQR)
    # --------------------------------------------------------------------
    def statistical_outliers(self):
        """Flag extreme values using IQR with 2.5 * IQR."""
        for col in WEATHER_FEATURES:
            if col not in self.df.columns:
                continue
            data = self.df[col].dropna()
            if len(data) < 10:
                continue
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 2.5 * IQR
            upper = Q3 + 2.5 * IQR
            # Create a boolean series aligned with self.df.index
            outlier_series = pd.Series(False, index=self.df.index)
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            outlier_series[mask] = True
            self.flags[f"stat_{col}"] = outlier_series.fillna(False)
            if outlier_series.any():
                print(f"   • {col}: {outlier_series.sum()} statistical outliers")

    # --------------------------------------------------------------------
    # 2. Contextual Anomalies (ED component mismatches) - FIXED
    # --------------------------------------------------------------------
    def contextual_anomalies(self):
        """
        Flag records where ED components don't match the category.
        e.g., Very high ED but all components are low = data error.
        Uses safe boolean operations.
        """
        d = self.df

        # Check if we have ED components
        ed_cols = ["ed_f_heat", "ed_f_air", "ed_f_uv", "ed_score", "ed_category"]
        available = [c for c in ed_cols if c in d.columns]
        if not available:
            print("   ⚠️ No ED component columns available for contextual anomalies.")
            return

        # We'll build flags safely
        flags = {}

        # Rule 1: High ED but low components (data error)
        if "ed_score" in d.columns and "ed_f_heat" in d.columns and "ed_f_air" in d.columns:
            high_ed = d["ed_score"] > 70
            low_heat = d["ed_f_heat"].fillna(0) < 20
            low_air = d["ed_f_air"].fillna(0) < 20
            low_uv = d["ed_f_uv"].fillna(0) < 10 if "ed_f_uv" in d.columns else pd.Series(True, index=d.index)
            # Combine: high_ed AND (low_heat OR low_air OR low_uv)
            low_any = low_heat | low_air | low_uv
            flag = high_ed & low_any
            flags["context_high_ed_low_comp"] = flag
            if flag.any():
                print(f"   • High ED but low components: {flag.sum()}")

        # Rule 2: ED_Category says DANGEROUS but score is low
        if "ed_category" in d.columns and "ed_score" in d.columns:
            dangerous = d["ed_category"].isin(["ED_DANGEROUS", "ED_VERY_DANGEROUS"])
            low_score = d["ed_score"] < 40
            flag = dangerous & low_score
            flags["context_dangerous_low_score"] = flag
            if flag.any():
                print(f"   • Dangerous category but low score: {flag.sum()}")

        # Rule 3: ED_Category says SAFE but score is high
        if "ed_category" in d.columns and "ed_score" in d.columns:
            safe = d["ed_category"].isin(["ED_VERY_SAFE", "ED_MODERATE_SAFE"])
            high_score = d["ed_score"] > 60
            flag = safe & high_score
            flags["context_safe_high_score"] = flag
            if flag.any():
                print(f"   • Safe category but high score: {flag.sum()}")

        # Add flags to self.flags
        for key, flag in flags.items():
            self.flags[key] = flag.fillna(False)

    # --------------------------------------------------------------------
    # 3. ML Outliers (Isolation Forest)
    # --------------------------------------------------------------------
    def ml_outliers(self):
        """Use Isolation Forest on weather features."""
        feats = [f for f in WEATHER_FEATURES if f in self.df.columns]
        if len(feats) < 3:
            print("   ⚠️ Not enough features for ML detection")
            return

        X = self.df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())

        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=config.RANDOM_STATE,
            n_estimators=200,
        )
        preds = iso.fit_predict(Xs)
        self.flags["ml_isolated_forest"] = preds == -1
        if self.flags["ml_isolated_forest"].any():
            print(f"   • ML (Isolation Forest): {self.flags['ml_isolated_forest'].sum()}")

    # --------------------------------------------------------------------
    # 4. Data Quality (sensor range checks)
    # --------------------------------------------------------------------
    def data_quality(self):
        """Flag records with physically impossible values."""
        issues = []
        checks = {
            "temperature_celsius": (-70, 60),
            "humidity": (0, 100),
            "wind_kph": (0, 300),
            "uv_index": (0, 16),
            "pressure_mb": (850, 1100),
            "air_quality_us-epa-index": (1, 6),
            "air_quality_PM2.5": (0, 500),
        }
        total_oob = pd.Series(False, index=self.df.index)
        for col, (lo, hi) in checks.items():
            if col not in self.df.columns:
                continue
            v = pd.to_numeric(self.df[col], errors="coerce")
            oob = (v < lo) | (v > hi)
            oob = oob.fillna(False)
            if oob.any():
                issues.append({"issue": "out_of_range", "feature": col, "count": int(oob.sum())})
            total_oob |= oob
        self.flags["data_out_of_range"] = total_oob
        if total_oob.any():
            print(f"   • Out of range: {total_oob.sum()}")
        return pd.DataFrame(issues)

    # --------------------------------------------------------------------
    # 5. Combine and Finalize
    # --------------------------------------------------------------------
    def combine(self):
        """Sum all flags and classify anomalies."""
        flag_cols = [c for c in self.flags.columns]
        if not flag_cols:
            print("   ⚠️ No anomaly flags generated.")
            return pd.DataFrame()

        self.df["anomaly_score"] = self.flags[flag_cols].sum(axis=1).astype(int)
        self.df["anomaly_sources"] = self.flags[flag_cols].apply(
            lambda r: ", ".join([c for c in flag_cols if r[c]]), axis=1
        )

        # Anomaly = score >= 2 (detected by at least 2 methods)
        self.df["is_anomaly"] = self.df["anomaly_score"] >= 2

        # Also flag any record with data_out_of_range
        if "data_out_of_range" in self.flags.columns:
            self.df["is_anomaly"] = self.df["is_anomaly"] | self.flags["data_out_of_range"]

        anomalies = self.df[self.df["is_anomaly"]].copy()

        # Keep relevant columns
        keep = [
            "location_name", "country", "latitude", "longitude",
            "temperature_celsius", "humidity", "wind_kph", "uv_index",
            "apparent_temp_c", "pressure_mb",
            "air_quality_us-epa-index", "air_quality_PM2.5",
            "ed_score", "ed_category",
            "anomaly_score", "anomaly_sources", "is_anomaly",
        ]
        keep = [c for c in keep if c in anomalies.columns]

        if not anomalies.empty:
            return anomalies[keep].sort_values("anomaly_score", ascending=False).reset_index(drop=True)
        return pd.DataFrame()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 5: ANOMALY DETECTION (pipeline‑integrated, fixed)")
    print("=" * 70)

    # 1. Load processed data (same as all other steps)
    df = get_processed_data()
    print(f"Loaded {len(df)} records from {config.PROCESSED_DATA_PATH}")

    # 2. Initialize detector
    det = AnomalyDetector(df, contamination=0.05)

    # 3. Run all detection methods
    print("\n🔍 Running anomaly detection methods:")
    det.statistical_outliers()
    det.contextual_anomalies()
    det.ml_outliers()
    dq_issues = det.data_quality()

    # 4. Combine and finalize
    anomalies = det.combine()

    # 5. Save outputs
    out = ensure_dir(config.STEP5_OUT)
    artifacts = {}

    artifacts["anomalies"] = save_df(anomalies, out / "confirmed_anomalies.csv")

    # Also save the full dataset with flags
    full_df = det.df[["location_name", "country", "temperature_celsius", "humidity",
                      "ed_score", "ed_category", "anomaly_score", "anomaly_sources", "is_anomaly"]]
    full_df = full_df.fillna("")
    artifacts["full_with_flags"] = save_df(full_df, out / "dataset_with_anomaly_flags.csv")

    if not dq_issues.empty:
        artifacts["data_quality_issues"] = save_df(dq_issues, out / "data_quality_issues.csv")

    # 6. Summary
    summary = {
        "n_records": int(len(df)),
        "n_anomalies": int(len(anomalies)),
        "anomaly_rate_pct": round(100 * len(anomalies) / max(1, len(df)), 2),
        "flagged_by_multiple_methods": int((anomalies["anomaly_score"] > 1).sum()) if not anomalies.empty else 0,
        "data_quality_issues": int(len(dq_issues)),
    }
    artifacts["summary"] = save_json(summary, out / "anomaly_summary.json")

    # 7. Manifest
    write_manifest(out, "step5_anomaly", artifacts, extra=summary)

    print(f"\n📊 Anomaly Detection Summary:")
    print(f"   • Records: {summary['n_records']:,}")
    print(f"   • Anomalies: {summary['n_anomalies']:,} ({summary['anomaly_rate_pct']}%)")
    print(f"   • Multi-method flags: {summary['flagged_by_multiple_methods']}")
    print(f"   • Data quality issues: {summary['data_quality_issues']}")
    print(f"\n📁 Artifacts saved to {out}")

    return det


if __name__ == "__main__":
    main()