# Exercise Danger Prediction System

Produces a continuous **Exercise Danger score, ED[0–100]**, from weather &
air-quality data. Data mining does **not** invent ED — it *shapes, explains and
regionalizes* an evidence-based baseline.

```
Final_ED = clip( ED_baseline(0–100)  +  GNN_regional_adjustment(−10..+15) , 0, 100 )
           └── common/ed_baseline.py ─┘   └── gnn_adjuster (Step 3 graph) ──┘
```

## Why this design
`GlobalWeatherRepository.csv` has **no ground-truth ED label**, so ED cannot be
supervised-learned; and the plan requires an *evidence-based, interpretable*
score. Therefore ED is a **knowledge-anchored index** (`common/ed_baseline.py`),
and every mining step consumes that single source of truth. This removes the
circularity of the original code (models re-learning a hand-coded formula).

## Folder structure
```
exercise_danger_system/
├── config.py                     # paths + params (no hard-coded OS paths)
├── data/                         # put GlobalWeatherRepository.csv here
│   └── location_enrichment.csv   # optional: elevation_m, population per location
├── common/                       # SHARED core (imported by every step)
│   ├── ed_baseline.py            # ★ ED[0-100] source of truth (heat/air/cold/uv/synergy)
│   ├── preprocessing.py          # load→clean→enrich→cache (the only cross-step handoff)
│   └── io_utils.py               # save/manifest helpers
├── step1_association_rules/      # rules + danger-frequency + interaction evidence
├── step2_decision_tree/          # regression-tree surrogate/explainer (decoupled)
├── step3_clustering/             # per-LOCATION regime clusters + GNN graph
├── step4_correlation/            # correlation matrix + VIF + stratified interaction
├── step5_anomaly/                # IsolationForest/LOF + contextual + DQ + regional residuals
├── math_model/                   # "core brain": assembles Final_ED
├── gnn_adjuster/                 # stub: regional adjustment (−10..+15)
├── outputs/                      # all generated artifacts (per step)
└── run_pipeline.py               # orchestrator
```

## Run
```bash
pip install -r requirements.txt
# place GlobalWeatherRepository.csv in ./data/
python run_pipeline.py            # runs steps 1–3
python math_model/math_model.py   # demo single prediction
```

## How the steps connect (decoupled, no pickle chain)
- `common/preprocessing.get_processed_data()` builds/caches one processed table
  (raw features + ED baseline columns). **Every step reads this** — no step
  depends on another step's *analytical* output.
- Step 1 → explanation rules + synergy justification (for report & math model).
- Step 2 → interpretable thresholds & feature importance (surrogate for baseline).
- Step 3 → per-location clusters + `gnn_graph_data.json` (for the GNN).
- Step 4 → VIF/redundancy + interaction evidence → `model_insights.json` (for the math model).
- Step 5 → anomalies + data-quality flags + per-location regional residuals (for the GNN).
- GNN → per-location adjustment; `math_model.assemble_final_ed()` combines it.

## Notes for later steps
- **Elevation/Population**: add `data/location_enrichment.csv` (keys matching
  `config.LOCATION_KEYS`, columns `elevation_m`, `population`). Step 3 joins it
  automatically and flags when missing.
- **GNN supervision**: no ground-truth ED exists — plan a spatial-smoothing /
  self-supervised objective bounded to [−10, +15] (see `gnn_adjuster/`).
