"""
common/grid_processor.py
========================
Process spatial grid data for the ED system.
Converts units, aggregates to grid-point features, and computes EPA index.
No city mapping needed – works directly with grid points.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.pollution import pm25_to_aqi, pm10_to_aqi, aqi_to_epa_index


def convert_grid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert units for grid data.
    Input columns: pm2p5 (kg/m³), pm10 (kg/m³), t2m (K)
    Output: pm2p5 (µg/m³), pm10 (µg/m³), t2m (°C)
    """
    df = df.copy()
    if 'pm2p5' in df.columns:
        df['pm2p5'] = df['pm2p5'] * 1_000_000_000
    if 'pm10' in df.columns:
        df['pm10'] = df['pm10'] * 1_000_000_000
    if 't2m' in df.columns:
        df['t2m'] = df['t2m'] - 273.15
    return df


def compute_epa_index_from_grid(row):
    """Compute EPA index from PM2.5 and PM10 in a row."""
    aqi_values = {}
    if pd.notna(row.get('pm2p5')):
        aqi_values["PM2.5"] = pm25_to_aqi(row['pm2p5'])
    if pd.notna(row.get('pm10')):
        aqi_values["PM10"] = pm10_to_aqi(row['pm10'])
    if aqi_values:
        max_aqi = max(aqi_values.values())
        return aqi_to_epa_index(max_aqi)
    return 1


def aggregate_grid_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate time-series grid data to point-level features.
    Returns one row per (latitude, longitude) with weather statistics.
    """
    # Ensure converted units exist
    if 'pm2p5' not in df.columns or 't2m' not in df.columns:
        df = convert_grid_data(df)
    
    # Compute EPA index per row
    df['epa_index'] = df.apply(compute_epa_index_from_grid, axis=1)
    
    # Define aggregation dictionary
    agg_dict = {
        't2m': ['mean', 'std', 'max', 'min'],
        'pm2p5': ['mean', 'std', 'max', 'min'],
        'pm10': ['mean', 'std', 'max', 'min'],
        'epa_index': ['mean', 'max'],
    }
    if 'duaod550' in df.columns:
        agg_dict['duaod550'] = ['mean', 'std']
    
    # Group by grid point
    grid_features = df.groupby(['latitude', 'longitude']).agg(agg_dict).reset_index()
    
    # Flatten column names
    flat_cols = ['latitude', 'longitude']
    for col, stats in agg_dict.items():
        for stat in stats:
            flat_cols.append(f'{col}_{stat}')
    grid_features.columns = flat_cols
    
    # Rename for clarity
    rename_map = {
        't2m_mean': 'temp_mean',
        't2m_std': 'temp_std',
        't2m_max': 'temp_max',
        't2m_min': 'temp_min',
        'pm2p5_mean': 'pm25_mean',
        'pm2p5_std': 'pm25_std',
        'pm2p5_max': 'pm25_max',
        'pm2p5_min': 'pm25_min',
        'pm10_mean': 'pm10_mean',
        'pm10_std': 'pm10_std',
        'pm10_max': 'pm10_max',
        'pm10_min': 'pm10_min',
        'epa_index_mean': 'epa_mean',
        'epa_index_max': 'epa_max',
    }
    grid_features.rename(columns=rename_map, inplace=True)
    
    # Add node_id as string of lat_lon
    grid_features['node_id'] = grid_features.apply(
        lambda row: f"{row['latitude']:.2f}_{row['longitude']:.2f}", axis=1
    )
    
    return grid_features


def get_clustering_features(grid_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for clustering (weather only, no lat/lon).
    Returns DataFrame with node_id and weather features.
    """
    # List of weather feature columns (exclude lat, lon, node_id)
    feature_cols = [c for c in grid_features.columns 
                    if c not in ['latitude', 'longitude', 'node_id']]
    return grid_features[['node_id', 'latitude', 'longitude'] + feature_cols]


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    from pathlib import Path
    
    test_path = Path(__file__).resolve().parent.parent / "data" / "output_data.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        df_converted = convert_grid_data(df)
        grid_feats = aggregate_grid_points(df_converted)
        print("✅ Grid features:")
        print(grid_feats.head())
        print(f"\nFeatures: {grid_feats.columns.tolist()}")
    else:
        print("Test data not found.")