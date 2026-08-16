# step6_gnn/prepare_grid_for_gnn.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler

import config
from common.grid_processor import convert_grid_data, aggregate_grid_points
from math_model.math_model import ExerciseDangerMathModel
from common.io_utils import save_json, write_manifest, ensure_dir


def prepare_grid_for_gnn():
    print("=" * 70)
    print("PREPARE GRID DATA FOR GNN")
    print("=" * 70)
    
    grid_path = Path(config.DATA_DIR) / "output_data.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid data not found: {grid_path}")
    
    print(f"Loading grid data from {grid_path}")
    df = pd.read_csv(grid_path)
    print(f"Loaded {len(df)} records")
    
    df = convert_grid_data(df)
    print(f"Converted and cleaned: {len(df)} records remain")
    
    if len(df) == 0:
        raise ValueError("No valid data after cleaning missing values")
    
    grid_features = aggregate_grid_points(df)
    print(f"Aggregated to {len(grid_features)} unique grid points")
    
    grid_features['epa_index'] = grid_features['epa_max'].fillna(1).astype(int)
    
    print("Computing math scores...")
    model = ExerciseDangerMathModel()
    
    math_scores = []
    for _, row in grid_features.iterrows():
        try:
            result = model.predict(
                temperature_celsius=row['temp_mean'],
                humidity=50,
                wind_kph=10,
                uv_index=3,
                air_quality_us_epa_index=row['epa_index'],
                air_quality_PM2_5=row['pm25_mean'],
                air_quality_PM10=row['pm10_mean'],
                cluster_id=None,
                anomaly_flag=False,
            )
            math_scores.append(result['ED'])
        except Exception as e:
            print(f"Error for grid point {row['node_id']}: {e}")
            math_scores.append(50.0)
    
    grid_features['math_danger_score'] = math_scores
    print("Math scores computed")
    
    feature_cols = [
        'temp_mean', 'temp_std', 'temp_max', 'temp_min',
        'pm25_mean', 'pm25_std', 'pm25_max', 'pm25_min',
        'pm10_mean', 'pm10_std', 'pm10_max', 'pm10_min',
        'epa_mean', 'epa_max'
    ]
    
    if 'duaod550_mean' in grid_features.columns:
        feature_cols.extend(['duaod550_mean', 'duaod550_std'])
    
    for col in feature_cols:
        if col not in grid_features.columns:
            grid_features[col] = 0.0
    
    X = grid_features[feature_cols].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Building adjacency matrix...")
    similarity = cosine_similarity(X_scaled)
    threshold = np.percentile(similarity, 70)
    similarity[similarity < threshold] = 0
    np.fill_diagonal(similarity, 1)
    row_sums = similarity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adjacency = similarity / row_sums
    
    out_dir = Path(config.OUTPUTS_DIR) / "gnn_grid_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "node_features.npy", X_scaled)
    np.save(out_dir / "adjacency_matrix.npy", adjacency)
    np.save(out_dir / "targets.npy", grid_features['math_danger_score'].values)
    
    with open(out_dir / "node_ids.json", 'w') as f:
        json.dump(grid_features['node_id'].tolist(), f)
    
    grid_features.to_csv(out_dir / "grid_features.csv", index=False)
    
    graph_data = {
        "node_ids": grid_features['node_id'].tolist(),
        "node_features": X_scaled.tolist(),
        "feature_names": feature_cols,
        "cluster_ids": [],
        "adjacency_matrix": adjacency.tolist(),
        "note": "Grid points from time-series data. Clustering not yet applied."
    }
    graph_json_path = out_dir / "gnn_graph_data.json"
    with open(graph_json_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"\nSaved to {out_dir}")
    print(f"      - node_features.npy ({X_scaled.shape})")
    print(f"      - adjacency_matrix.npy ({adjacency.shape})")
    print(f"      - targets.npy ({len(grid_features)})")
    print(f"      - node_ids.json ({len(grid_features)} nodes)")
    print(f"      - grid_features.csv")
    print(f"      - gnn_graph_data.json")
    
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
        print(f"Could not write manifest: {e}")
    
    print("\nGrid data preparation complete!")
    return out_dir


if __name__ == "__main__":
    prepare_grid_for_gnn()