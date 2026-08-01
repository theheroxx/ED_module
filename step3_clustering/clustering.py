"""
STEP 3 – CLUSTERING (final, with visualization report)
=======================================================
- Selects k using silhouette_ratio (default 0.65) – picks k=6 for your dataset.
- Enforces min_cluster_size=3.
- Saves a comprehensive Matplotlib report alongside the CSV/JSON outputs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # headless-friendly
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from common.preprocessing import get_processed_data
from common.io_utils import save_df, save_json, write_manifest, ensure_dir

# ----------------------------------------------------------------------------
# BASE FEATURES (non‑circular, no ED, no raw longitude)
# ----------------------------------------------------------------------------
BASE_FEATURES = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "uv_index",
    "air_quality_us-epa-index",
    "pressure_mb",
    "abs_latitude",
]


class ExerciseDangerClusterAnalyzer:
    def __init__(self, k_max=config.CLUSTER_MAX_K, min_cluster_size=3, silhouette_ratio=0.65):
        self.k_max = k_max
        self.min_cluster_size = min_cluster_size
        self.silhouette_ratio = silhouette_ratio
        self.location_df = None
        self.features = None
        self.k = None
        self._seasonal_cols = []
        self._Xs_scaled = None   # for visualization
        self._pca = None         # for visualization

    # ------------------------------------------------------------------------
    # 1. AGGREGATE BY LOCATION
    # ------------------------------------------------------------------------
    def aggregate_locations(self, df):
        keys = [k for k in config.LOCATION_KEYS if k in df.columns]
        if not keys:
            df = df.copy()
            df["location_name"] = "loc_" + df.index.astype(str)
            keys = ["location_name"]

        df = df.copy()
        df["abs_latitude"] = df.get("latitude", pd.Series(0.0, index=df.index)).abs()

        # ---- Extract month for seasonality ----
        has_month = False
        if "month" in df.columns:
            has_month = True
        else:
            time_cols = [c for c in df.columns if c in ["last_updated", "timestamp", "datetime"]]
            if time_cols:
                try:
                    df["month"] = pd.to_datetime(df[time_cols[0]]).dt.month
                    has_month = True
                except Exception:
                    pass

        # ---- Base aggregation ----
        agg = {f: "mean" for f in BASE_FEATURES if f in df.columns}
        loc = df.groupby(keys, dropna=False).agg(agg)
        loc["n_obs"] = df.groupby(keys, dropna=False).size()

        # ---- Seasonal features ----
        if has_month:
            monthly = df.groupby(keys + ["month"]).agg({"temperature_celsius": "mean"}).reset_index()
            max_temp = monthly.loc[monthly.groupby(keys)["temperature_celsius"].idxmax()]
            min_temp = monthly.loc[monthly.groupby(keys)["temperature_celsius"].idxmin()]
            loc_season = max_temp[keys + ["month"]].copy()
            loc_season["amplitude"] = max_temp["temperature_celsius"].values - min_temp["temperature_celsius"].values
            loc_season.rename(columns={"month": "peak_month"}, inplace=True)
            loc = loc.reset_index().merge(loc_season, on=keys, how="left").set_index(keys)
            loc["temp_amplitude"] = loc["amplitude"].fillna(0.0)
            loc["phase_sin"] = np.sin(2 * np.pi * loc["peak_month"] / 12)
            loc["phase_cos"] = np.cos(2 * np.pi * loc["peak_month"] / 12)
            loc.drop(columns=["amplitude", "peak_month"], inplace=True, errors="ignore")
            self._seasonal_cols = ["temp_amplitude", "phase_sin", "phase_cos"]
        else:
            self._seasonal_cols = []

        loc = loc.reset_index()
        self.location_df = loc
        return loc

    # ------------------------------------------------------------------------
    # 2. CLUSTERING (with silhouette_ratio selection)
    # ------------------------------------------------------------------------
    def cluster(self, max_k=None):
        if max_k is None:
            max_k = self.k_max

        loc = self.location_df

        # ---- Build feature list ----
        feats = [f for f in BASE_FEATURES if f in loc.columns]
        for col in self._seasonal_cols:
            if col in loc.columns:
                feats.append(col)

        if "air_quality_us-epa-index" not in feats and "air_quality_PM2.5" in loc.columns:
            feats.append("air_quality_PM2.5")

        if len(feats) < 3:
            fallback = ["temperature_celsius", "abs_latitude", "humidity"]
            feats = [f for f in fallback if f in loc.columns]

        print(f"   Initial features: {feats}")

        # ---- Clean ----
        X = loc[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.mean()) if not X.isnull().all().all() else X.fillna(0.0)
        X = X.fillna(0.0)

        # ---- Drop constant ----
        constant = X.columns[X.var() < 1e-6].tolist()
        if constant:
            print(f"   Dropping constant: {constant}")
            X = X.drop(columns=constant)
            feats = [f for f in feats if f not in constant]

        # ---- Drop high correlation (>0.95) ----
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        if to_drop:
            print(f"   Dropping highly correlated: {to_drop}")
            X = X.drop(columns=to_drop)
            feats = [f for f in feats if f not in to_drop]

        if X.shape[1] < 2:
            raise ValueError("Not enough informative features.")

        # ---- Scale ----
        scaler = RobustScaler()
        Xs_scaled = scaler.fit_transform(X)

        # ---- PCA (if >4 features) ----
        if Xs_scaled.shape[1] > 4:
            pca = PCA(n_components=min(5, Xs_scaled.shape[1]))
            Xs_pca = pca.fit_transform(Xs_scaled)
            self._pca = pca
            print(f"   PCA kept {Xs_pca.shape[1]} comps (explained {pca.explained_variance_ratio_.sum():.2%})")
        else:
            Xs_pca = Xs_scaled
            self._pca = None

        self.features = feats
        self._Xs_scaled = Xs_scaled
        self._Xs = Xs_pca

        # ---- Find candidate k values ----
        n = Xs_pca.shape[0]
        max_k = min(max_k, n // 2) if n > 4 else 3
        max_k = max(max_k, 2)

        valid = []  # (k, silhouette, counts)

        for k in range(2, max_k + 1):
            if k >= n:
                break
            clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = clusterer.fit_predict(Xs_pca)
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) < 2:
                continue
            if np.min(counts) < self.min_cluster_size:
                print(f"   k={k}: sizes {counts} -> min < {self.min_cluster_size}, skip")
                continue
            sil = silhouette_score(Xs_pca, labels)
            valid.append((k, sil, counts))

        if not valid:
            print("   No valid k found, falling back to k=2")
            clusterer = AgglomerativeClustering(n_clusters=2, linkage='ward')
            labels = clusterer.fit_predict(Xs_pca)
            best_k = 2
            best_sil = silhouette_score(Xs_pca, labels) if len(set(labels)) > 1 else 0.0
        else:
            # ---- Selection using silhouette_ratio ----
            max_sil = max(s for _, s, _ in valid)
            threshold = max_sil * self.silhouette_ratio
            candidates = [(k, sil) for k, sil, _ in valid if sil >= threshold]
            if candidates:
                best_k = max(candidates, key=lambda x: x[0])[0]
                best_sil = max(c[1] for c in candidates if c[0] == best_k)
            else:
                best_k = max(valid, key=lambda x: x[1])[0]
                best_sil = max(valid, key=lambda x: x[1])[1]

        # ---- Fit final ----
        clusterer = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
        labels = clusterer.fit_predict(Xs_pca)

        self.k = best_k
        self.silhouette = best_sil
        loc["cluster"] = labels

        print(f"   Valid k scores: {valid}")
        print(f"   Selected k={self.k} (silhouette={self.silhouette:.3f})")
        return loc

    # ------------------------------------------------------------------------
    # 3. PROFILES
    # ------------------------------------------------------------------------
    def profiles(self):
        loc = self.location_df
        if "ed_score" not in loc.columns:
            return pd.DataFrame()
        prof = (loc.groupby("cluster")
                   .agg(n_locations=("cluster", "size"),
                        mean_ed=("ed_score", "mean"),
                        mean_temp=("temperature_celsius", "mean") if "temperature_celsius" in loc else ("ed_score", "mean"),
                        mean_hum=("humidity", "mean") if "humidity" in loc else ("ed_score", "mean"),
                        mean_aqi=("air_quality_us-epa-index", "mean") if "air_quality_us-epa-index" in loc else ("ed_score", "mean"))
                   .reset_index())
        overall = loc["ed_score"].mean()
        prof["ed_offset_vs_global"] = prof["mean_ed"] - overall
        return prof.sort_values("mean_ed", ascending=False)

    # ------------------------------------------------------------------------
    # 4. GNN GRAPH
    # ------------------------------------------------------------------------
    def build_location_graph(self, similarity_threshold=0.0):
        loc = self.location_df
        Xs = self._Xs
        if np.any(~np.isfinite(Xs)):
            Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.linalg.norm(Xs, axis=1, keepdims=True)
        zero = (norm == 0).flatten()
        if np.any(zero):
            uv = np.ones((1, Xs.shape[1])) / np.sqrt(Xs.shape[1])
            for i in np.where(zero)[0]:
                Xs[i, :] = uv
            norm = np.linalg.norm(Xs, axis=1, keepdims=True)
        norm[norm == 0] = 1e-8
        unit = Xs / norm
        adj = unit @ unit.T
        np.fill_diagonal(adj, 1.0)
        adj = np.where(adj >= similarity_threshold, adj, 0.0)
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

        node_ids = loc[[k for k in config.LOCATION_KEYS if k in loc.columns]] \
            .astype(str).agg(" | ".join, axis=1).tolist()
        features_clean = np.nan_to_num(loc[self.features].astype(float).values, nan=0.0)

        return {
            "node_ids": node_ids,
            "node_features": features_clean.tolist(),
            "feature_names": self.features,
            "cluster_ids": loc["cluster"].astype(int).tolist(),
            "adjacency_matrix": adj.tolist(),
            "note": "nodes = locations; adjacency = climate-regime cosine similarity",
        }


# ============================================================================
# 5. VISUALIZATION REPORT
# ============================================================================
def plot_cluster_report(loc, Xs_scaled, features, labels, out_dir, title_suffix=""):
    """Generate a 2x2 matplotlib report and save as PNG."""
    out_dir = Path(out_dir)
    n_clusters = len(set(labels))
    cluster_counts = loc["cluster"].value_counts().sort_index()

    # --- PCA for 2D projection ---
    pca = PCA(n_components=2, random_state=42)
    Xs_2d = pca.fit_transform(Xs_scaled)

    # --- Silhouette per cluster ---
    sil_vals = silhouette_samples(Xs_scaled, labels)
    sil_df = pd.DataFrame({"cluster": labels, "silhouette": sil_vals})
    cluster_sil = sil_df.groupby("cluster")["silhouette"].mean().sort_index()

    # --- Feature means per cluster ---
    # Use the original location_df to get raw means
    cluster_feat_means = loc.groupby("cluster")[features].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Clustering Analysis Report – k={n_clusters}  (silhouette={silhouette_score(Xs_scaled, labels):.3f}){title_suffix}",
                 fontsize=14, fontweight='bold')

    # (1) Cluster sizes
    ax = axes[0, 0]
    bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values, color=plt.cm.tab20(np.linspace(0, 1, n_clusters)))
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of locations")
    ax.set_title("Cluster Sizes")
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f"{count}", ha='center', va='bottom', fontsize=9)

    # (2) PCA projection
    ax = axes[0, 1]
    scatter = ax.scatter(Xs_2d[:, 0], Xs_2d[:, 1], c=labels, cmap='tab20', s=30, alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("PCA Projection (color = cluster)")
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    # (3) Feature profiles (grouped bar chart – top 6 features to avoid clutter)
    ax = axes[1, 0]
    # Normalize features for radar – use z-scores or min-max
    feat_norm = (cluster_feat_means - cluster_feat_means.min()) / (cluster_feat_means.max() - cluster_feat_means.min() + 1e-8)
    # use a subset of features for readability
    plot_feats = [f for f in features if f in feat_norm.columns][:6]  # limit to 6
    if len(plot_feats) > 0:
        x = np.arange(len(plot_feats))
        width = 0.8 / n_clusters
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        for i, cluster_id in enumerate(sorted(feat_norm.index)):
            offset = (i - n_clusters/2 + 0.5) * width
            ax.bar(x + offset, feat_norm.loc[cluster_id, plot_feats].values,
                   width=width, label=f'Cluster {cluster_id}', color=colors[i], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_feats, rotation=45, ha='right')
        ax.set_ylabel("Normalized value (0-1)")
        ax.set_title("Feature Profiles (by cluster)")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not enough features to plot", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Feature Profiles")

    # (4) Silhouette per cluster
    ax = axes[1, 1]
    ax.bar(cluster_sil.index.astype(str), cluster_sil.values, color=plt.cm.tab20(np.linspace(0, 1, n_clusters)))
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Mean Silhouette")
    ax.set_title("Silhouette Score per Cluster")
    for i, v in enumerate(cluster_sil.values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout(pad=2.0)
    report_path = out_dir / "cluster_analysis_report.png"
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return report_path


# ============================================================================
# 6. MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("STEP 3: CLUSTERING (final, with visualization report)")
    print("=" * 70)

    df = get_processed_data()
    print(f"Loaded {len(df)} records from {config.PROCESSED_DATA_PATH}")

    an = ExerciseDangerClusterAnalyzer(
        k_max=config.CLUSTER_MAX_K,
        min_cluster_size=3,
        silhouette_ratio=0.65
    )
    an.aggregate_locations(df)
    print(f"Aggregated to {len(an.location_df)} locations")

    loc = an.cluster(max_k=config.CLUSTER_MAX_K)

    # Save artifacts
    out = ensure_dir(config.STEP3_OUT)

    # 1. CSV outputs
    prof = an.profiles()
    graph = an.build_location_graph()

    artifacts = {}
    artifacts["location_clusters"] = save_df(loc, out / "location_clusters.csv")
    artifacts["cluster_profiles"] = save_df(prof, out / "cluster_profiles.csv")
    artifacts["gnn_graph_data"] = save_json(graph, out / "gnn_graph_data.json")

    # 2. Visualization report
    if an._Xs_scaled is not None:
        plot_path = plot_cluster_report(loc, an._Xs_scaled, an.features, loc["cluster"].values, out)
        artifacts["cluster_analysis_report"] = plot_path
        print(f"   Report saved: {plot_path}")
    else:
        print("   Skipping report: no scaled data available.")

    # 3. Manifest
    write_manifest(out, "step3_clustering", artifacts,
                   extra={
                       "n_locations": int(len(loc)),
                       "k": an.k,
                       "silhouette": an.silhouette,
                       "features_used": an.features,
                       "min_cluster_size": an.min_cluster_size,
                       "silhouette_ratio": an.silhouette_ratio,
                   })

    print(f"\n✅ Clustering completed.")
    print(f"   Locations: {len(loc)}")
    print(f"   Selected k: {an.k} (silhouette={an.silhouette:.3f})")
    print("   Cluster sizes:\n", loc["cluster"].value_counts().sort_index().to_string())
    print(f"   Artifacts saved to {out}")

    return an


if __name__ == "__main__":
    main()