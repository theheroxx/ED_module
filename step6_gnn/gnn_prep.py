# step6_gnn/prepare_gnn_inputs.py
import sys
from pathlib import Path
import pandas as pd
import json
import shutil
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from common.preprocessing import get_processed_data

OPTIMIZED_FEATURES = [
    "air_quality_PM2.5",
    "temperature_celsius",
    "air_quality_us-epa-index",
    "uv_index",
]


def prepare_gnn_inputs_from_grid():
    print("=" * 70)
    print("STEP 6: GNN PREPARATION (Using Grid Data)")
    print("=" * 70)
    
    grid_dir = Path(config.OUTPUTS_DIR) / "gnn_grid_data"
    grid_features_path = grid_dir / "grid_features.csv"
    if not grid_features_path.exists():
        raise FileNotFoundError(f"Grid features not found: {grid_features_path}")
    
    grid_features = pd.read_csv(grid_features_path)
    print(f"Loaded {len(grid_features)} grid points")
    
    targets_path = grid_dir / "targets.npy"
    if targets_path.exists():
        math_scores = np.load(targets_path)
        grid_features['math_danger_score'] = math_scores
    else:
        raise FileNotFoundError(f"Targets not found: {targets_path}")
    
    cluster_path = Path(config.STEP3_OUT) / "location_clusters.csv"
    if cluster_path.exists():
        cluster_df = pd.read_csv(cluster_path)
        if 'node_id' not in grid_features.columns:
            grid_features['node_id'] = grid_features.apply(
                lambda r: f"{r['latitude']:.2f}_{r['longitude']:.2f}", axis=1
            )
        if 'node_id' not in cluster_df.columns:
            cluster_df['node_id'] = cluster_df.apply(
                lambda r: f"{r['latitude']:.2f}_{r['longitude']:.2f}", axis=1
            )
        grid_features = grid_features.merge(
            cluster_df[['node_id', 'cluster']], on='node_id', how='left'
        )
        grid_features['cluster'] = grid_features['cluster'].fillna(-1).astype(int)
        print(f"Merged clusters: {grid_features['cluster'].nunique()} unique")
    else:
        print("No cluster assignments found. Using cluster=-1 for all.")
        grid_features['cluster'] = -1
    
    cluster_assignments = pd.DataFrame({
        'location_name': grid_features['node_id'],
        'cluster': grid_features['cluster'],
        'air_quality_PM2.5': grid_features.get('pm25_mean', 10.0),
        'temperature_celsius': grid_features.get('temp_mean', 20.0),
        'air_quality_us-epa-index': grid_features.get('epa_mean', 1.0),
        'uv_index': 3.0,
    })
    
    out_dir = Path(config.OUTPUTS_DIR) / "step6_gnn_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    math_df = grid_features[['node_id', 'math_danger_score']].rename(
        columns={'node_id': 'location_name'}
    )
    math_df.to_csv(out_dir / "math_model_predictions.csv", index=False)
    print(f"math_model_predictions.csv ({len(math_df)} rows)")
    
    cluster_assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    print(f"cluster_assignments.csv ({len(cluster_assignments)} rows)")
    
    graph_src = grid_dir / "gnn_graph_data.json"
    graph_dst = out_dir / "gnn_graph_data.json"
    if graph_src.exists():
        shutil.copy(graph_src, graph_dst)
        print(f"gnn_graph_data.json (copied from grid)")
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import RobustScaler
        
        feature_cols = [c for c in grid_features.columns if c not in ['node_id', 'latitude', 'longitude', 'cluster', 'math_danger_score']]
        X = grid_features[feature_cols].values
        X_scaled = RobustScaler().fit_transform(X)
        
        sim = cosine_similarity(X_scaled)
        threshold = np.percentile(sim, 70)
        sim[sim < threshold] = 0
        np.fill_diagonal(sim, 1)
        row_sums = sim.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        adj = sim / row_sums
        
        graph = {
            "node_ids": grid_features['node_id'].tolist(),
            "node_features": X_scaled.tolist(),
            "feature_names": feature_cols,
            "cluster_ids": grid_features['cluster'].tolist(),
            "adjacency_matrix": adj.tolist(),
        }
        with open(graph_dst, 'w') as f:
            json.dump(graph, f, indent=2)
        print(f"gnn_graph_data.json (generated from grid)")
    
    print("\nGNN preparation complete (grid data).")
    print(f"Output: {out_dir}")
    return out_dir


def prepare_gnn_inputs_from_cities():
    print("=" * 70)
    print("STEP 6: GNN PREPARATION (City Data)")
    print("=" * 70)
    print(f"Optimized features: {OPTIMIZED_FEATURES}")
    print("Based on decision tree importance (PM2.5=64.2%, Temp=16.5%)")
    print()
    
    step3_dir = Path(config.STEP3_OUT)
    clusters_path = step3_dir / "location_clusters.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Clustering output not found: {clusters_path}")
    clusters_df = pd.read_csv(clusters_path)
    print(f"Loaded clusters: {len(clusters_df)} locations")
    
    graph_path = step3_dir / "gnn_graph_data.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph data not found: {graph_path}")
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    print(f"Loaded graph: {len(graph_data['node_ids'])} nodes")
    
    df = get_processed_data()
    agg_dict = {feature: 'mean' for feature in OPTIMIZED_FEATURES if feature in df.columns}
    location_features = df.groupby('location_name', as_index=False).agg(agg_dict)
    print(f"Loaded weather for {len(location_features)} locations")
    
    math_path = Path(config.OUTPUTS_DIR) / "math_model" / "math_model_predictions.csv"
    if not math_path.exists():
        raise FileNotFoundError(f"Math predictions not found: {math_path}")
    math_df = pd.read_csv(math_path)
    print(f"Loaded math predictions: {len(math_df)} locations")
    
    clusters_df['location_name'] = clusters_df['location_name'].str.strip()
    location_features['location_name'] = location_features['location_name'].str.strip()
    math_df['location_name'] = math_df['location_name'].str.strip()
    
    merged_df = clusters_df.merge(location_features, on='location_name', how='left')
    merged_df = merged_df.merge(math_df, on='location_name', how='left')
    merged_df = merged_df.dropna(subset=['cluster', 'math_danger_score'])
    print(f"After merging: {len(merged_df)} complete locations")
    
    for feat in OPTIMIZED_FEATURES:
        if feat not in merged_df.columns:
            if feat == 'air_quality_PM2.5':
                merged_df[feat] = 10.0
            elif feat == 'temperature_celsius':
                merged_df[feat] = 20.0
            elif feat == 'air_quality_us-epa-index':
                merged_df[feat] = 1.0
            elif feat == 'uv_index':
                merged_df[feat] = 3.0
    
    cluster_cols = ['location_name', 'cluster'] + OPTIMIZED_FEATURES
    cluster_assignments = merged_df[cluster_cols]
    
    out_dir = Path(config.OUTPUTS_DIR) / "step6_gnn_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    math_output = merged_df[['location_name', 'math_danger_score']]
    math_output.to_csv(out_dir / "math_model_predictions.csv", index=False)
    print(f"math_model_predictions.csv ({len(math_output)} rows)")
    
    cluster_assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    print(f"cluster_assignments.csv ({len(cluster_assignments)} rows)")
    
    shutil.copy(graph_path, out_dir / "gnn_graph_data.json")
    print(f"gnn_graph_data.json (copied from step3)")
    
    print("\nGNN preparation complete (city data).")
    print(f"Output: {out_dir}")
    return out_dir


def prepare_gnn_inputs():
    grid_dir = Path(config.OUTPUTS_DIR) / "gnn_grid_data"
    if grid_dir.exists() and (grid_dir / "grid_features.csv").exists():
        return prepare_gnn_inputs_from_grid()
    else:
        return prepare_gnn_inputs_from_cities()


def verify_gnn_inputs():
    pass


if __name__ == "__main__":
    prepare_gnn_inputs()
    verify_gnn_inputs()