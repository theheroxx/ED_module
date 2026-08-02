"""
step6_gnn/prepare_grid_for_gnn.py
==================================
Prepare grid data for GNN training.

Generates:
    - outputs/gnn_grid_data/node_features.npy
    - outputs/gnn_grid_data/adjacency_matrix.npy
    - outputs/gnn_grid_data/targets.npy
    - outputs/gnn_grid_data/node_ids.json
    - outputs/gnn_grid_data/grid_features.csv
    - outputs/gnn_grid_data/gnn_graph_data.json (compatible with GNN)

Usage:
    python -m step6_gnn.prepare_grid_for_gnn
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from common.grid_processor import convert_grid_data, aggregate_grid_points
from common.pollution import pm25_to_aqi, pm10_to_aqi, aqi_to_epa_index
from math_model.math_model import ExerciseDangerMathModel
from common.io_utils import save_json, write_manifest, ensure_dir


def prepare_grid_for_gnn():
    """
    Full pipeline: convert, aggregate, compute math scores, build GNN inputs.
    """
    print("=" * 70)
    print("PREPARE GRID DATA FOR GNN")
    print("=" * 70)
    
    # 1. Load raw grid data
    grid_path = Path(config.DATA_DIR) / "output_data.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid data not found: {grid_path}")
    
    print(f"📊 Loading grid data from {grid_path}")
    df = pd.read_csv(grid_path)
    
    # 2. Convert units
    df = convert_grid_data(df)
    print(f"   ✅ Converted: {len(df)} records")
    
    # 3. Aggregate to grid point features
    grid_features = aggregate_grid_points(df)
    print(f"   ✅ Aggregated to {len(grid_features)} unique grid points")
    
    # 4. Compute EPA index for each grid point (already in grid_features as epa_mean/epa_max)
    # Use epa_max for worst-case, or epa_mean for average
    # We'll use epa_max for safety
    grid_features['epa_index'] = grid_features['epa_max'].fillna(1).astype(int)
    
    # 5. Compute math scores for each grid point
    print("   🧮 Computing math scores...")
    model = ExerciseDangerMathModel()
    
    math_scores = []
    for _, row in grid_features.iterrows():
        try:
            result = model.predict(
                temperature_celsius=row['temp_mean'],
                humidity=50,          # placeholder
                wind_kph=10,          # placeholder
                uv_index=3,           # placeholder
                air_quality_us_epa_index=row['epa_index'],
                air_quality_PM2_5=row['pm25_mean'],
                air_quality_PM10=row['pm10_mean'],
                cluster_id=None,
                anomaly_flag=False,
            )
            math_scores.append(result['ED'])
        except Exception as e:
            print(f"   ⚠️ Error for grid point {row['node_id']}: {e}")
            math_scores.append(50.0)  # fallback
    
    grid_features['math_danger_score'] = math_scores
    print(f"   ✅ Math scores computed")
    
    # 6. Prepare node features (weather only)
    feature_cols = [
        'temp_mean', 'temp_std', 'temp_max', 'temp_min',
        'pm25_mean', 'pm25_std', 'pm25_max', 'pm25_min',
        'pm10_mean', 'pm10_std', 'pm10_max', 'pm10_min',
        'epa_mean', 'epa_max'
    ]
    # Include AOD if present
    if 'duaod550_mean' in grid_features.columns:
        feature_cols.extend(['duaod550_mean', 'duaod550_std'])
    
    # Ensure all features exist; fill missing with 0
    for col in feature_cols:
        if col not in grid_features.columns:
            grid_features[col] = 0.0
    
    X = grid_features[feature_cols].values
    
    # Normalize features (Robust scaling)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. Build adjacency matrix (cosine similarity on normalized features)
    print("   🔗 Building adjacency matrix...")
    similarity = cosine_similarity(X_scaled)
    # Keep only top 30% strongest connections to create sparse graph
    threshold = np.percentile(similarity, 70)
    similarity[similarity < threshold] = 0
    np.fill_diagonal(similarity, 1)
    # Row-normalize
    row_sums = similarity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency = similarity / row_sums
    
    # 8. Save outputs
    out_dir = Path(config.OUTPUTS_DIR) / "gnn_grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numpy arrays
    np.save(out_dir / "node_features.npy", X_scaled)
    np.save(out_dir / "adjacency_matrix.npy", adjacency)
    np.save(out_dir / "targets.npy", grid_features['math_danger_score'].values)
    
    # Save node IDs
    with open(out_dir / "node_ids.json", 'w') as f:
        json.dump(grid_features['node_id'].tolist(), f)
    
    # Save full grid features for reference
    grid_features.to_csv(out_dir / "grid_features.csv", index=False)
    
    # 9. Build gnn_graph_data.json (compatible with existing GNN)
    graph_data = {
        "node_ids": grid_features['node_id'].tolist(),
        "node_features": X_scaled.tolist(),
        "feature_names": feature_cols,
        "cluster_ids": [],  # will be filled later by clustering
        "adjacency_matrix": adjacency.tolist(),
        "note": "Grid points from time-series data. Clustering not yet applied."
    }
    graph_json_path = out_dir / "gnn_graph_data.json"
    with open(graph_json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"\n   💾 Saved to {out_dir}")
    print(f"      - node_features.npy ({X_scaled.shape})")
    print(f"      - adjacency_matrix.npy ({adjacency.shape})")
    print(f"      - targets.npy ({len(grid_features)})")
    print(f"      - node_ids.json ({len(grid_features)} nodes)")
    print(f"      - grid_features.csv")
    print(f"      - gnn_graph_data.json")
    
    # 10. Write manifest
    try:
        artifacts = {
            "node_features": out_dir / "node_features.npy",
            "adjacency_matrix": out_dir / "adjacency_matrix.npy",
            "targets": out_dir / "targets.npy",
            "node_ids": out_dir / "node_ids.json",
            "grid_features": out_dir / "grid_features.csv",
            "gnn_graph_data": graph_json_path,
        }
        write_manifest(out_dir, "gnn_grid_data", artifacts, extra={
            "n_nodes": len(grid_features),
            "features": feature_cols,
            "source_data": str(grid_path),
        })
    except Exception as e:
        print(f"   ⚠️ Could not write manifest: {e}")
    
    print("\n✅ Grid data preparation complete!")
    return out_dir


if __name__ == "__main__":
    prepare_grid_for_gnn()