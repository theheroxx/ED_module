"""
STEP 4 - CORRELATION & INTERACTION ANALYSIS  (refactored)
=========================================================
Role in the ED[0-100] system:
  * Tells the MATH MODEL how the raw weather inputs relate to each other and to
    the ED baseline, so it can avoid double-counting and handle interactions:
      - correlation matrix (Pearson & Spearman) over raw features + ed_score
      - MULTICOLLINEARITY via VIF  -> which inputs are redundant (non-circular;
        computed on raw inputs only, no ed_score)
      - COMPONENT DECOMPOSITION    -> how much each ed_f_* contributes to ed_score
      - STRATIFIED INTERACTION     -> does pollution's danger rise with heat?
  * Emits machine-readable model_insights.json consumed by math_model/.

Decoupled: reads common.preprocessing.get_processed_data() and reuses
common.ed_baseline component columns. It does NOT read Step 3's output and has
no hard-coded paths.

Note on honesty: correlations WITH ed_score are partly mechanical (ed_score is
built from these factors). The genuinely new signal here is (a) multicollinearity
AMONG RAW INPUTS and (b) the stratified heat x pollution interaction.

UPDATED (2026-07-24): Optimized features based on decision tree results:
    - Added air_quality_PM2.5 (64.2% importance)
    - Removed humidity, wind_kph, wind_chill_c, pressure_mb (0% importance)
    - Kept EPA, temperature, apparent_temp, uv_index
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from common.preprocessing import get_processed_data
from common.io_utils import save_df, save_json, write_manifest, ensure_dir
from common.ed_baseline import COMPONENT_COLS

# OPTIMIZED FEATURES - Based on decision tree importance
# -------------------------------------------------------
# DROPPED (0% importance): humidity, wind_kph, wind_chill_c, pressure_mb
# KEPT: EPA (5.4%), temperature (16.5%), apparent_temp (13.4%), 
#       PM2.5 (64.2%), uv_index (0.5%)
# -------------------------------------------------------
RAW_FEATURES = [
    # --- HIGH IMPORTANCE (≥ 7%) ---
    "air_quality_PM2.5",               # 64.2% ← #1 (root split)
    "temperature_celsius",             # 16.5% (used in tree splits)
    "apparent_temp_c",                 # 13.4% (used in tree splits)
    "air_quality_us-epa-index",        # 5.4% (regulatory alignment)
    
    # --- LOW IMPORTANCE (< 1%) ---
    "uv_index",                        # 0.5% (keep for completeness)
    
    # --- DROPPED (0% importance) ---
    # "humidity",                      # 0.0%
    # "wind_kph",                     # 0.0%
    # "wind_chill_c",                 # 0.0%
    # "pressure_mb",                  # 0.0%
]


class CorrelationAnalyzer:
    def __init__(self, df):
        self.df = df
        self.features = [f for f in RAW_FEATURES if f in df.columns]
        self.insights = []
        print(f"   Using {len(self.features)} features: {self.features}")

    # -- (1) correlation matrices ------------------------------------------
    def correlation_matrix(self):
        cols = self.features + (["ed_score"] if "ed_score" in self.df.columns else [])
        num = self.df[cols].apply(pd.to_numeric, errors="coerce")
        pear = num.corr(method="pearson")
        spear = num.corr(method="spearman")
        # record dominant drivers (|Pearson| with ed_score)
        if "ed_score" in pear.columns:
            drivers = (pear["ed_score"].drop("ed_score").abs()
                       .sort_values(ascending=False))
            for f, v in drivers.items():
                if v > 0.5:
                    self.insights.append(f"Primary driver of ED: {f} (|r|={v:.2f})")
        return pear, spear

    # -- (2) VIF multicollinearity among RAW inputs ------------------------
    def vif(self):
        X = self.df[self.features].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median())
        X = X.loc[:, X.std() > 1e-9]                 # drop constant columns
        rows = []
        for col in X.columns:
            y = X[col].values
            others = X.drop(columns=[col]).values
            if others.shape[1] == 0:
                vif = 1.0
            else:
                r2 = LinearRegression().fit(others, y).score(others, y)
                vif = float(1.0 / max(1e-9, 1.0 - r2))
            rows.append({"feature": col, "VIF": round(vif, 2)})
        vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
        redundant = vif_df.loc[vif_df["VIF"] > 5, "feature"].tolist()
        if redundant:
            self.insights.append(
                f"Redundant inputs (VIF>5) -> math model should keep only one of: {redundant}")
        return vif_df, redundant

    # -- (3) component decomposition of ed_score ---------------------------
    def component_decomposition(self):
        comps = [c for c in COMPONENT_COLS if c in self.df.columns]
        if not comps or "ed_score" not in self.df.columns:
            return pd.DataFrame()
        tot = self.df["ed_score"].replace(0, np.nan)
        rows = []
        for c in comps:
            mean_pts = float(self.df[c].mean())
            share = float((self.df[c] / tot).replace([np.inf, -np.inf], np.nan).mean())
            rows.append({"component": c.replace("ed_f_", ""),
                         "mean_points": round(mean_pts, 2),
                         "avg_share_of_score": round(share, 3)})
        return pd.DataFrame(rows).sort_values("mean_points", ascending=False)

    # -- (4) stratified heat x pollution interaction -----------------------
    def stratified_interaction(self):
        if "temperature_celsius" not in self.df.columns:
            return pd.DataFrame()
        d = self.df.copy()
        d["temp_band"] = pd.cut(d["temperature_celsius"],
                                bins=[-100, 15, 28, 200],
                                labels=["COLD(<15)", "MILD(15-28)", "HOT(>28)"])
        # Use PM2.5 as the air quality measure (more granular than EPA)
        air = "air_quality_PM2.5" if "air_quality_PM2.5" in d else "air_quality_us-epa-index"
        if air not in d.columns:
            return pd.DataFrame()
        
        rows = []
        for band, g in d.groupby("temp_band", observed=True):
            if len(g) < 10:
                continue
            r_air = (g[air].corr(g["ed_score"]) if air and air in g else np.nan)
            rows.append({"temp_band": str(band), "n": int(len(g)),
                         "mean_ed": round(float(g["ed_score"].mean()), 2),
                         "corr_air_vs_ed": round(float(r_air), 3) if pd.notna(r_air) else None})
        out = pd.DataFrame(rows)
        if not out.empty and out["corr_air_vs_ed"].notna().any():
            hot = out[out["temp_band"].str.startswith("HOT")]
            cold = out[out["temp_band"].str.startswith("COLD")]
            if not hot.empty and not cold.empty:
                if (hot["corr_air_vs_ed"].iloc[0] or 0) > (cold["corr_air_vs_ed"].iloc[0] or 0) + 0.05:
                    self.insights.append(
                        "Heat x pollution interaction confirmed -> keep synergy term in ED baseline")
        return out


def _heatmap(pear, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list(pear.columns)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pear.values, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{pear.values[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Feature Correlation Matrix (incl. ED score)")
        fig.tight_layout()
        p = Path(out_dir) / "correlation_heatmap.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        return p
    except Exception as e:
        print(f"   (heatmap skipped: {e})")
        return None


def main():
    print("=" * 70, "\nSTEP 4: CORRELATION & INTERACTION ANALYSIS\n", "=" * 70)
    df = get_processed_data()
    an = CorrelationAnalyzer(df)

    pear, spear = an.correlation_matrix()
    vif_df, redundant = an.vif()
    decomp = an.component_decomposition()
    inter = an.stratified_interaction()

    out = ensure_dir(config.STEP4_OUT)
    a = {}
    a["correlation_matrix_pearson"] = save_df(pear.reset_index(), out / "correlation_matrix_pearson.csv")
    a["correlation_matrix_spearman"] = save_df(spear.reset_index(), out / "correlation_matrix_spearman.csv")
    a["vif_multicollinearity"] = save_df(vif_df, out / "vif_multicollinearity.csv")
    a["component_decomposition"] = save_df(decomp, out / "component_decomposition.csv")
    a["interaction_strength"] = save_df(inter, out / "interaction_strength.csv")
    a["model_insights"] = save_json(
        {"insights": an.insights, "redundant_features": redundant},
        out / "model_insights.json")
    hm = _heatmap(pear, out)
    if hm:
        a["correlation_heatmap"] = hm
    write_manifest(out, "step4_correlation", a,
                   extra={"redundant_features": redundant,
                          "n_insights": len(an.insights)})

    print(f"VIF redundant inputs: {redundant or 'none'}")
    if not decomp.empty:
        print("Component decomposition (mean pts):\n", decomp.to_string(index=False))
    print("Insights:", *[f"\n  - {i}" for i in an.insights])
    print(f"Artifacts -> {out}")
    return an


if __name__ == "__main__":
    main()