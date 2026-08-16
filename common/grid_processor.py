import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent.parent))


def _force_numeric(series):
    return pd.to_numeric(series, errors='coerce')


def convert_grid_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    cols = ['pm2p5', 'pm10', 't2m', 'latitude', 'longitude', 'duaod550']
    for col in cols:
        if col in df.columns:
            df[col] = _force_numeric(df[col])
    
    key_cols = ['pm2p5', 'pm10', 't2m']
    present = [c for c in key_cols if c in df.columns]
    if present:
        df = df.dropna(subset=present)
    
    if 'pm2p5' in df.columns:
        df['pm2p5'] = df['pm2p5'] * 1_000_000_000
    if 'pm10' in df.columns:
        df['pm10'] = df['pm10'] * 1_000_000_000
    if 't2m' in df.columns:
        df['t2m'] = df['t2m'] - 273.15
    
    return df



def aggregate_grid_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    for col in df.columns:
        try:
            df[col] = _force_numeric(df[col])
        except Exception:
            pass
    
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("Missing latitude/longitude columns")
    
    df = df.dropna(subset=['latitude', 'longitude'])
    


    def safe_epa(row):
        try:
            from common.pollution import pm25_to_aqi, pm10_to_aqi, aqi_to_epa_index
            aqis = {}
            if pd.notna(row['pm2p5']) and row['pm2p5'] > 0:
                aqis['PM2.5'] = pm25_to_aqi(row['pm2p5'])
            if pd.notna(row['pm10']) and row['pm10'] > 0:
                aqis['PM10'] = pm10_to_aqi(row['pm10'])
            if aqis:
                return aqi_to_epa_index(max(aqis.values()))
            return 1
        except Exception:
            return 1
    
    df['epa_index'] = df.apply(safe_epa, axis=1)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    agg_dict = {
        't2m': ['mean', 'std', 'max', 'min'],
        'pm2p5': ['mean', 'std', 'max', 'min'],
        'pm10': ['mean', 'std', 'max', 'min'],
        'epa_index': ['mean', 'max'],
    }
    if 'duaod550' in df.columns:
        agg_dict['duaod550'] = ['mean', 'std']
    
    grid_features = df.groupby(['latitude', 'longitude']).agg(agg_dict).reset_index()
    


    flat_cols = ['latitude', 'longitude']
    for col, stats in agg_dict.items():
        for stat in stats:
            flat_cols.append(f'{col}_{stat}')
    grid_features.columns = flat_cols
    
    for col in grid_features.columns:
        if col not in ['latitude', 'longitude']:
            grid_features[col] = grid_features[col].fillna(0)
    
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
    
    grid_features['node_id'] = grid_features.apply(
        lambda r: f"{r['latitude']:.2f}_{r['longitude']:.2f}", axis=1
    )
    
    return grid_features




def get_clustering_features(grid_features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in grid_features.columns 
                    if c not in ['latitude', 'longitude', 'node_id']]
    return grid_features[['node_id', 'latitude', 'longitude'] + feature_cols]


if __name__ == "__main__":
    test_path = Path(__file__).resolve().parent.parent / "data" / "output_data.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        df_conv = convert_grid_data(df)
        grid_feats = aggregate_grid_points(df_conv)
        print("Grid features:")
        print(grid_feats.head())
        print(f"\nFeatures: {grid_feats.columns.tolist()}")
    else:
        print("Test data not found.")