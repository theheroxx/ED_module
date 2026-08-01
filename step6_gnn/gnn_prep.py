"""
STEP 6 - GNN PREPARATION
========================
Prepares inputs for the GNN from existing pipeline outputs.

OPTIMIZED FEATURES (based on decision tree importance):
    1. air_quality_PM2.5      (64.2%)  ← Primary
    2. temperature_celsius     (16.5%)  ← Primary temperature
    3. air_quality_us-epa-index (5.4%)  ← Regulatory alignment
    4. uv_index               (0.5%)   ← UV risk

Files created:
    - math_model_predictions.csv
    - cluster_assignments.csv  (with optimized features)
    - gnn_graph_data.json

Usage:
    python -m step6_gnn.gnn_prep
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import json
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from common.preprocessing import get_processed_data

# ============================================================================
# OPTIMIZED FEATURES - Based on decision tree importance
# ============================================================================
# 1. air_quality_PM2.5    (64.2%)  ← Primary air quality
# 2. temperature_celsius  (16.5%)  ← Primary temperature
# 3. air_quality_us-epa-index (5.4%) ← Regulatory alignment
# 4. uv_index             (0.5%)   ← UV risk
#
# DROPPED (0% importance or redundant):
# - apparent_temp_c (redundant with temperature_celsius, VIF > 5)
# - humidity (0%)
# - wind_kph (0%)
# - pressure_mb (0%)
# ============================================================================

OPTIMIZED_FEATURES = [
    "air_quality_PM2.5",
    "temperature_celsius",
    "air_quality_us-epa-index",
    "uv_index",
]

# Additional features needed for clustering (used in graph, not as separate features)
GRAPH_FEATURES = [
    "temperature_celsius",
    "humidity",
    "wind_kph",
    "uv_index",
    "air_quality_us-epa-index",
    "pressure_mb",
    "abs_latitude",
]


def prepare_gnn_inputs():
    """
    Prepare all three files needed by the GNN using optimized features.
    
    Outputs:
        outputs/step6_gnn_prep/
        ├── math_model_predictions.csv
        ├── cluster_assignments.csv  (with optimized features)
        └── gnn_graph_data.json
    """
    print("=" * 70)
    print("STEP 6: GNN PREPARATION (Optimized Features)")
    print("=" * 70)
    print(f"   📊 Optimized features: {OPTIMIZED_FEATURES}")
    print("   ℹ️  Based on decision tree importance (PM2.5=64.2%, Temp=16.5%)")
    print()
    
    # 1. Load clustering outputs
    step3_dir = Path(config.STEP3_OUT)
    
    # Load cluster assignments
    clusters_path = step3_dir / "location_clusters.csv"
    if not clusters_path.exists():
        raise FileNotFoundError(f"Clustering output not found: {clusters_path}")
    
    clusters_df = pd.read_csv(clusters_path)
    print(f"   ✅ Loaded clusters: {len(clusters_df)} locations")
    
    # Load graph data
    graph_path = step3_dir / "gnn_graph_data.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph data not found: {graph_path}")
    
    with open(graph_path, 'r') as f:
        graph_data = json.load(f)
    print(f"   ✅ Loaded graph: {len(graph_data['node_ids'])} nodes")
    
    # 2. Get weather features for each location from processed data
    print("   📊 Loading weather data...")
    df = get_processed_data()
    
    # Get mean weather per location - ONLY optimized features
    agg_dict = {feature: 'mean' for feature in OPTIMIZED_FEATURES if feature in df.columns}
    # Also need location_name for merging
    location_features = df.groupby('location_name', as_index=False).agg(agg_dict)
    print(f"   ✅ Loaded weather for {len(location_features)} locations")
    print(f"      Features: {list(location_features.columns)}")
    
    # 3. Load math model predictions
    math_path = Path(config.OUTPUTS_DIR) / "math_model" / "math_model_predictions.csv"
    if not math_path.exists():
        raise FileNotFoundError(f"Math predictions not found: {math_path}")
    
    math_df = pd.read_csv(math_path)
    print(f"   ✅ Loaded math predictions: {len(math_df)} locations")
    
    # 4. Merge everything
    # First, ensure location_name matches (strip whitespace)
    clusters_df['location_name'] = clusters_df['location_name'].str.strip()
    location_features['location_name'] = location_features['location_name'].str.strip()
    math_df['location_name'] = math_df['location_name'].str.strip()
    
    # Merge clusters with optimized features
    merged_df = clusters_df.merge(location_features, on='location_name', how='left')
    print(f"   After merging clusters + features: {len(merged_df)} rows")
    
    # Merge with math scores
    merged_df = merged_df.merge(math_df, on='location_name', how='left')
    print(f"   After merging with math scores: {len(merged_df)} rows")
    
    # Drop rows with missing data
    initial_len = len(merged_df)
    merged_df = merged_df.dropna(subset=['cluster', 'math_danger_score'])
    print(f"   ✅ After dropping missing: {len(merged_df)} complete locations (dropped {initial_len - len(merged_df)})")
    
    # Check if we have the required optimized features
    available_features = [f for f in OPTIMIZED_FEATURES if f in merged_df.columns]
    missing_features = [f for f in OPTIMIZED_FEATURES if f not in merged_df.columns]
    if missing_features:
        print(f"   ⚠️ Missing features: {missing_features}")
        # Add fallback values for missing features
        for feat in missing_features:
            if feat == 'air_quality_PM2.5':
                merged_df[feat] = 10.0
            elif feat == 'temperature_celsius':
                merged_df[feat] = 20.0
            elif feat == 'air_quality_us-epa-index':
                merged_df[feat] = 1.0
            elif feat == 'uv_index':
                merged_df[feat] = 3.0
    
    # 5. Create cluster_assignments.csv with optimized features
    cluster_cols = ['location_name', 'cluster'] + OPTIMIZED_FEATURES
    cluster_assignments = merged_df[cluster_cols]
    
    # 6. Create output directory
    out_dir = Path(config.OUTPUTS_DIR) / "step6_gnn_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 7. Save files
    print("\n   💾 Saving files...")
    
    # File 1: math_model_predictions.csv
    math_output = merged_df[['location_name', 'math_danger_score']]
    math_output.to_csv(out_dir / "math_model_predictions.csv", index=False)
    print(f"      ✅ math_model_predictions.csv ({len(math_output)} rows)")
    
    # File 2: cluster_assignments.csv (with optimized features)
    cluster_assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    print(f"      ✅ cluster_assignments.csv ({len(cluster_assignments)} rows)")
    print(f"         Columns: {cluster_assignments.columns.tolist()}")
    
    # File 3: gnn_graph_data.json (copy from step3)
    shutil.copy(graph_path, out_dir / "gnn_graph_data.json")
    print(f"      ✅ gnn_graph_data.json (copied from step3)")
    
    # 8. Verify the files
    print("\n   🔍 Verification:")
    print(f"      - math_model_predictions.csv: {len(math_output)} rows")
    print(f"      - cluster_assignments.csv: {len(cluster_assignments)} rows")
    print(f"      - gnn_graph_data.json: {len(graph_data['node_ids'])} nodes")
    
    # Show sample of cluster_assignments
    print("\n   📋 Sample of cluster_assignments.csv:")
    print(cluster_assignments.head(5).to_string(index=False))
    
    # 9. Write manifest
    try:
        from common.io_utils import write_manifest
        artifacts = {
            "math_model_predictions": out_dir / "math_model_predictions.csv",
            "cluster_assignments": out_dir / "cluster_assignments.csv",
            "gnn_graph_data": out_dir / "gnn_graph_data.json",
        }
        write_manifest(out_dir, "step6_gnn_prep", artifacts, extra={
            "n_locations": len(merged_df),
            "optimized_features": OPTIMIZED_FEATURES,
            "source_clusters": str(clusters_path),
            "source_graph": str(graph_path),
            "source_math": str(math_path),
        })
    except Exception as e:
        print(f"   ⚠️ Could not write manifest: {e}")
    
    print(f"\n✅ GNN preparation complete!")
    print(f"   Files saved to: {out_dir}")
    print(f"   Features used: {OPTIMIZED_FEATURES}")
    return out_dir


def verify_gnn_inputs():
    """
    Verify that all GNN inputs are valid and compatible.
    """
    print("\n" + "=" * 70)
    print("VERIFYING GNN INPUTS")
    print("=" * 70)
    
    gnn_dir = Path(config.OUTPUTS_DIR) / "step6_gnn_prep"
    
    # Check all files exist
    required = [
        "math_model_predictions.csv",
        "cluster_assignments.csv",
        "gnn_graph_data.json",
    ]
    
    all_exist = True
    for file in required:
        path = gnn_dir / file
        if path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n   ❌ Some files are missing. Run prepare_gnn_inputs() first.")
        return False
    
    # Verify cluster assignments
    clusters = pd.read_csv(gnn_dir / "cluster_assignments.csv")
    print(f"\n   📊 cluster_assignments.csv:")
    print(f"      Shape: {clusters.shape}")
    print(f"      Columns: {clusters.columns.tolist()}")
    print(f"      Unique clusters: {sorted(clusters['cluster'].unique())}")
    
    # Verify math predictions
    math = pd.read_csv(gnn_dir / "math_model_predictions.csv")
    print(f"\n   📊 math_model_predictions.csv:")
    print(f"      Shape: {math.shape}")
    print(f"      Math score range: {math['math_danger_score'].min():.1f} - {math['math_danger_score'].max():.1f}")
    print(f"      Mean math score: {math['math_danger_score'].mean():.1f}")
    
    # Verify graph
    with open(gnn_dir / "gnn_graph_data.json", 'r') as f:
        graph = json.load(f)
    print(f"\n   📊 gnn_graph_data.json:")
    print(f"      Nodes: {len(graph['node_ids'])}")
    print(f"      Features: {len(graph['feature_names'])}")
    print(f"      Feature names: {graph['feature_names']}")
    
    return all_exist


if __name__ == "__main__":
    # Run preparation
    prepare_gnn_inputs()
    
    # Verify
    verify_gnn_inputs()