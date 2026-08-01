"""
GNN REGIONAL ADJUSTER  (interface stub)
=======================================
Planned role: produce the bounded regional adjustment (-10..+15) added to the
ED baseline. Operates on the LOCATION graph built by Step 3.

Contract:
  * INPUT : config.STEP3_OUT / gnn_graph_data.json
              -> node_ids, node_features, cluster_ids, adjacency_matrix
            (node features should include elevation_m & population once the
             data/location_enrichment.csv join is in place)
  * OUTPUT: config.GNN_OUT / {regional_adjustments.csv, gnn_model.pt, manifest.json}
            regional_adjustments.csv: node_id -> adjustment in [-10, +15]

  * The math model consumes the per-location adjustment:
        math_model.assemble_final_ed(weather, regional_adjustment=adj)

OPEN QUESTION to settle when we build this (there is no ground-truth ED):
  supervision signal for the GNN. Likely a spatial-smoothing / self-supervised
  objective that nudges a location's residual toward its cluster/neighbour
  statistics, bounded to [-10, +15]; switch to supervised if a real outcome
  (e.g. heat-related incidents, activity data) becomes available.

TODO: implement graph conv -> message passing -> bounded adjustment head.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: F401

ADJ_MIN, ADJ_MAX = -10.0, 15.0


def main():
    raise NotImplementedError("GNN adjuster not implemented yet - awaiting spec.")


if __name__ == "__main__":
    main()
