"""
run_pipeline.py
===============
Orchestrator. Builds the shared processed dataset, runs all mining steps,
and then builds the math model dynamically from the reports.

UPDATED (2026-08-01):
  - Added prepare_grid_data() to process time-series grid data if available
  - Grid data is prepared before clustering so clustering can use it
  - Pipeline now auto-detects and uses grid data
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config
from common.preprocessing import get_processed_data


def build_common():
    print("\n[common] building shared processed dataset...")
    df = get_processed_data(use_cache=False)
    print(f"[common] processed data cached: {df.shape} -> {config.PROCESSED_DATA_PATH}")


def prepare_grid_data():
    """
    Prepare grid data if output_data.csv exists.
    This must run before clustering so that clustering can use grid features.
    """
    grid_path = Path(config.DATA_DIR) / "output_data.csv"
    if not grid_path.exists():
        print("\n[grid] No grid data found (output_data.csv missing). Skipping grid preparation.")
        return False

    print("\n[grid] Preparing grid data from time-series...")
    try:
        from step6_gnn.prepare_grid_for_gnn import prepare_grid_for_gnn
        prepare_grid_for_gnn()
        print("[grid] Grid data preparation complete.")
        return True
    except Exception as e:
        print(f"[grid] ERROR preparing grid data: {e}")
        return False


def build_math_model():
    """Build the math model and generate predictions for GNN."""
    print("\n[math_model] building from pipeline reports...")
    from math_model.math_model import create_model_from_pipeline, ExerciseDangerPredictor, generate_predictions_for_gnn

    model = create_model_from_pipeline()
    predictor = ExerciseDangerPredictor()

    # Test prediction
    test_result = model.predict(
        temperature_celsius=22.0,
        humidity=50,
        wind_kph=10,
        uv_index=3,
        air_quality_us_epa_index=1,
        cluster_id=None,
        anomaly_flag=False,
    )
    print(f"[math_model] sample prediction: ED={test_result['ED']} ({test_result['Risk_Level']})")
    print(f"   Regional adjustment: {test_result['regional_adjustment']:+.2f}")
    print(f"   Confidence range: {test_result.get('confidence_range', 'N/A')}")
    print("[math_model] ready.")
    
    # Generate predictions for GNN
    generate_predictions_for_gnn()
    
    return model, predictor


def build_gnn_inputs():
    """Prepare GNN inputs from pipeline outputs."""
    print("\n[step6_gnn_prep] preparing GNN inputs...")
    from step6_gnn.gnn_prep import prepare_gnn_inputs, verify_gnn_inputs
    
    prepare_gnn_inputs()
    verify_gnn_inputs()
    print("[step6_gnn_prep] GNN inputs ready.")


def main():
    # 1. Build common processed data
    build_common()

    # 2. Prepare grid data if available (must happen before clustering)
    prepare_grid_data()

    # 3. Import and run all steps
    from step1_association_rules import association_rules
    from step2_decision_tree import decision_tree
    from step3_clustering import clustering
    from step4_correlation import correlation_analysis
    from step5_anomaly import anomaly_detection

    association_rules.main()
    decision_tree.main()
    clustering.main()
    correlation_analysis.main()
    anomaly_detection.main()

    # 4. Build Math Model (also generates math_model_predictions.csv)
    model, predictor = build_math_model()

    # 5. Step 6: GNN Preparation
    build_gnn_inputs()

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)
    print(f"   Model: baseline + GNN adjustment (bias map loaded: {bool(model.bias_map)})")
    if model.bias_map:
        print(f"   📊 GNN Bias Map: {model.bias_map}")
    print(f"   Regional adjustment bounds: [{model.gnn_min}, {model.gnn_max}]")
    print(f"   GNN inputs ready in: outputs/step6_gnn_prep/")
    print("=" * 70)

    # Optionally save the model for later use
    # import pickle
    # import os
    # os.makedirs("outputs/math_model", exist_ok=True)
    # with open("outputs/math_model/model.pkl", "wb") as f:
    #     pickle.dump(model, f)


if __name__ == "__main__":
    main()