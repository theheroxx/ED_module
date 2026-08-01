"""
math_model.py
=============
Thin wrapper around ed_baseline.py.
Adds GNN regional adjustment (Step 3) and anomaly override (Step 5).
All core scoring is delegated to common.ed_baseline.

UPDATED (2026-07-24 - FIXED):
  - Uses ed_category from baseline (single source of truth)
  - Safety floor flag computed after regional adjustment
  - No duplicate risk category logic
  - All pollutants passed through
  - Consistent type handling

UPDATED (2026-07-26):
  - Added generate_predictions_for_gnn() to export math scores for GNN
  - Fixed output path to match config

UPDATED (2026-07-27):
  - Added sigma_map for confidence ranges
  - Improved bias map loading with validation
  - Added cluster prediction fallback
  - Fixed path resolution (now correctly points to project root)
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))  # add project root

from common.ed_baseline import compute_ed_baseline

# Correct path: project root is parent of math_model folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class ExerciseDangerMathModel:
    def __init__(
        self,
        gnn_min: float = -10.0,
        gnn_max: float = 15.0,
        bias_map: dict | None = None,
    ):
        self.gnn_min = gnn_min
        self.gnn_max = gnn_max
        self.bias_map = bias_map or self._load_bias_map()
        self.sigma_map = self._load_sigma_map()

    def _load_bias_map(self) -> dict:
        """
        Load GNN bias map from GNN output CSV.
        
        Priority:
            1. GNN output: outputs/gnn/FINAL_EXERCISE_DANGER_SCORES_R.csv
            2. Fallback: clustering profiles
        
        Returns:
            dict: {cluster: bias}
        """
        # Priority 1: Use GNN output
        gnn_path = OUTPUTS_DIR / "gnn" / "FINAL_EXERCISE_DANGER_SCORES_R.csv"
        print(f"   🔍 Looking for GNN bias map at: {gnn_path}")
        if gnn_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(gnn_path)
                if "cluster" in df.columns and "gnn_bias" in df.columns:
                    # Group by cluster and take mean bias
                    bias_map = df.groupby("cluster")["gnn_bias"].mean().to_dict()
                    print(f"   ✅ GNN bias map loaded from: {gnn_path}")
                    print(f"   📊 Biases: {bias_map}")
                    return bias_map
                else:
                    print(f"   ⚠️ GNN file missing required columns: cluster, gnn_bias")
                    print(f"      Found columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"   ⚠️ Could not load GNN bias map: {e}")
        else:
            print(f"   ⚠️ GNN file not found at: {gnn_path}")
        
        # Fallback: Use clustering profiles
        profiles_path = OUTPUTS_DIR / "step3_clustering" / "cluster_profiles.csv"
        if profiles_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(profiles_path)
                if "cluster" in df.columns and "ed_offset_vs_global" in df.columns:
                    bias_map = dict(zip(df["cluster"], df["ed_offset_vs_global"]))
                    print(f"   ✅ Fallback bias map loaded from: {profiles_path}")
                    return bias_map
            except Exception as e:
                print(f"   ⚠️ Could not load fallback bias map: {e}")
        
        print("   ⚠️ No bias map found. Using no regional adjustment.")
        return {}

    def _load_sigma_map(self) -> dict:
        """
        Load per-cluster confidence (sigma) for confidence ranges.
        
        Returns:
            dict: {cluster: sigma}
        """
        gnn_path = OUTPUTS_DIR / "gnn" / "FINAL_EXERCISE_DANGER_SCORES_R.csv"
        if gnn_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(gnn_path)
                if "cluster" in df.columns and "final_danger_score" in df.columns:
                    sigma_map = df.groupby("cluster")["final_danger_score"].std().to_dict()
                    # Ensure minimum sigma of 3.0
                    sigma_map = {k: max(v, 3.0) for k, v in sigma_map.items()}
                    return sigma_map
            except Exception as e:
                print(f"   ⚠️ Could not load sigma map: {e}")
        
        # Default sigma per cluster
        return {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0}

    def predict(
        self,
        temperature_celsius: float,
        humidity: float,
        wind_kph: float,
        uv_index: float,
        air_quality_us_epa_index: float,
        # ---- ALL POLLUTANTS ----
        air_quality_PM2_5: float | None = None,
        air_quality_PM10: float | None = None,
        air_quality_Ozone: float | None = None,
        air_quality_Nitrogen_dioxide: float | None = None,
        air_quality_Sulphur_dioxide: float | None = None,
        air_quality_Carbon_Monoxide: float | None = None,
        # -------------------------
        cluster_id: int | None = None,
        anomaly_flag: bool = False,
    ) -> dict:
        """
        Single‑observation prediction.

        Parameters:
        -----------
        temperature_celsius, humidity, wind_kph, uv_index
            Raw weather inputs.
        air_quality_us_epa_index
            EPA Index (1-6).
        air_quality_PM2_5, air_quality_PM10, air_quality_Ozone,
        air_quality_Nitrogen_dioxide, air_quality_Sulphur_dioxide,
        air_quality_Carbon_Monoxide
            Raw pollutant concentrations. All are optional.
        cluster_id : int, optional
            Cluster ID from Step 3 (used to look up regional adjustment).
            If None, no regional adjustment is applied.
        anomaly_flag : bool
            If True, returns ED=100 (Step 5 override).

        Returns:
        --------
        dict with ED, Risk_Level, Category, breakdown, regional_adjustment,
        safety_floor_activated, max_component, and confidence_range.
        """
        # --- 1. Anomaly override (Step 5) ---
        if anomaly_flag:
            return {
                "ED": 100.0,
                "Risk_Level": "EXTREME",
                "Category": "ED_VERY_DANGEROUS",
                "breakdown": {"anomaly_override": 100.0},
                "regional_adjustment": 0.0,
                "safety_floor_activated": False,
                "max_component": 0.0,
                "confidence_range": "85 - 100",
                "note": "Anomaly detected – exercise not advised.",
            }

        # --- 2. Call the baseline (core scoring) with ALL pollutants ---
        weather = {
            "temperature_celsius": temperature_celsius,
            "humidity": humidity,
            "wind_kph": wind_kph,
            "uv_index": uv_index,
            "air_quality_us-epa-index": air_quality_us_epa_index,
        }
        
        # Add all pollutants if provided
        if air_quality_PM2_5 is not None:
            weather["air_quality_PM2.5"] = air_quality_PM2_5
        if air_quality_PM10 is not None:
            weather["air_quality_PM10"] = air_quality_PM10
        if air_quality_Ozone is not None:
            weather["air_quality_Ozone"] = air_quality_Ozone
        if air_quality_Nitrogen_dioxide is not None:
            weather["air_quality_Nitrogen_dioxide"] = air_quality_Nitrogen_dioxide
        if air_quality_Sulphur_dioxide is not None:
            weather["air_quality_Sulphur_dioxide"] = air_quality_Sulphur_dioxide
        if air_quality_Carbon_Monoxide is not None:
            weather["air_quality_Carbon_Monoxide"] = air_quality_Carbon_Monoxide

        base_result = compute_ed_baseline(weather)
        baseline_ed = base_result["ed_score"]
        components = base_result["components"]
        baseline_category = base_result["ed_category"]  # Single source of truth

        # --- 3. Regional adjustment (Step 3: GNN) ---
        regional_adj = 0.0
        if cluster_id is not None and self.bias_map:
            regional_adj = self.bias_map.get(cluster_id, 0.0)
        adj = float(max(self.gnn_min, min(self.gnn_max, regional_adj)))

        # --- 4. Final assembly ---
        final_ed = float(np.clip(baseline_ed + adj, 0, 100))

        # --- 5. Re-evaluate category from baseline's single source of truth ---
        from common.ed_baseline import _score_to_category
        final_category = _score_to_category(np.array([final_ed]))[0]

        # --- 6. Safety floor detection (after regional adjustment) ---
        max_component = max(components.values())
        safety_floor_activated = max_component > 70

        # --- 7. Confidence range ---
        sigma = self.sigma_map.get(cluster_id, 5.0) if cluster_id is not None else 5.0
        range_min = max(0, int(final_ed - sigma))
        range_max = min(100, int(final_ed + sigma))
        confidence_range = f"{range_min} - {range_max}"

        # --- 8. Risk level (mapping from category for backward compatibility) ---
        risk_map = {
            "ED_VERY_DANGEROUS": "EXTREME",
            "ED_DANGEROUS": "DANGEROUS",
            "ED_CAUTION": "CAUTION",
            "ED_MODERATE_SAFE": "MODERATE SAFE",
            "ED_VERY_SAFE": "VERY SAFE",
        }
        risk_level = risk_map.get(final_category, "UNKNOWN")

        return {
            "ED": round(final_ed, 2),
            "Risk_Level": risk_level,
            "Category": final_category,
            "breakdown": components,
            "baseline_ed": round(baseline_ed, 2),
            "regional_adjustment": round(adj, 2),
            "safety_floor_activated": safety_floor_activated,
            "max_component": round(max_component, 2),
            "confidence_range": confidence_range,
        }


def generate_predictions_for_gnn():
    """
    Generate math_model_predictions.csv for the GNN.
    
    Outputs:
        outputs/math_model/math_model_predictions.csv
    """
    import pandas as pd
    from common.preprocessing import get_processed_data
    import config
    
    print("📊 Generating math model predictions for GNN...")
    
    # 1. Load processed data
    df = get_processed_data()
    
    # 2. Get unique locations with their average weather
    location_data = df.groupby('location_name', as_index=False).agg({
        'temperature_celsius': 'mean',
        'humidity': 'mean',
        'air_quality_PM2.5': 'mean',
        'air_quality_us-epa-index': 'mean',
        'uv_index': 'mean',
        'wind_kph': 'mean',
    })
    
    # 3. Create model and predict for each location
    model = ExerciseDangerMathModel()
    
    math_scores = []
    for _, row in location_data.iterrows():
        try:
            result = model.predict(
                temperature_celsius=row['temperature_celsius'],
                humidity=row['humidity'],
                wind_kph=row['wind_kph'],
                uv_index=row['uv_index'],
                air_quality_us_epa_index=round(row['air_quality_us-epa-index']),
                air_quality_PM2_5=row['air_quality_PM2.5'],
                cluster_id=None,
                anomaly_flag=False,
            )
            math_scores.append(result['ED'])
        except Exception as e:
            print(f"   ⚠️ Error for {row['location_name']}: {e}")
            math_scores.append(50.0)  # Fallback
    
    location_data['math_danger_score'] = math_scores
    
    # 4. Save to CSV using config
    output_dir = Path(config.OUTPUTS_DIR) / "math_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "math_model_predictions.csv"
    location_data[['location_name', 'math_danger_score']].to_csv(output_path, index=False)
    
    print(f"✅ Saved to {output_path}")
    return output_path


# --------------------------------------------------------------------
# Compatibility wrapper for old interface.py
# --------------------------------------------------------------------
class ExerciseDangerPredictor(ExerciseDangerMathModel):
    def predict(self, inp: dict) -> dict:
        return super().predict(
            temperature_celsius=inp.get("temperature_celsius", inp.get("temp", 20)),
            humidity=inp.get("humidity", inp.get("humid", 50)),
            wind_kph=inp.get("wind_kph", inp.get("wind", 10)),
            uv_index=inp.get("uv_index", inp.get("uv", 0)),
            air_quality_us_epa_index=inp.get("air_quality_us-epa-index", inp.get("epa_index", 1)),
            air_quality_PM2_5=inp.get("air_quality_PM2.5", inp.get("pm25", None)),
            air_quality_PM10=inp.get("air_quality_PM10", inp.get("pm10", None)),
            air_quality_Ozone=inp.get("air_quality_Ozone", inp.get("o3", None)),
            air_quality_Nitrogen_dioxide=inp.get("air_quality_Nitrogen_dioxide", inp.get("no2", None)),
            air_quality_Sulphur_dioxide=inp.get("air_quality_Sulphur_dioxide", inp.get("so2", None)),
            air_quality_Carbon_Monoxide=inp.get("air_quality_Carbon_Monoxide", inp.get("co", None)),
            cluster_id=inp.get("cluster_id", None),
            anomaly_flag=inp.get("anomaly_flag", False),
        )


def create_model_from_pipeline() -> ExerciseDangerMathModel:
    """Factory to build the model (reads bias map from Step 3 outputs)."""
    return ExerciseDangerMathModel()


if __name__ == "__main__":
    # Quick test
    model = ExerciseDangerMathModel()
    result = model.predict(
        temperature_celsius=22.0,
        humidity=50,
        wind_kph=10,
        uv_index=3,
        air_quality_us_epa_index=1,
        cluster_id=None,
        anomaly_flag=False,
    )
    print("Test prediction:", result)
    
    # Generate GNN predictions
    generate_predictions_for_gnn()