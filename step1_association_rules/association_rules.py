"""
STEP 1 - ASSOCIATION RULE MINING  (refactored)
==============================================
Role in the ED[0-100] system:
  * Provides HUMAN-READABLE rules for the explainable report
      e.g. {TEMP_HOT, AQI_UNHEALTHY} -> ED_DANGEROUS  (conf 0.83)
  * NON-CIRCULAR contribution: mines the *frequency* of dangerous weather
    combinations and quantifies the HEAT x POLLUTION interaction. This is
    information the baseline formula does not already contain, and it is what
    justifies keeping the synergy term in common/ed_baseline.py.

It does NOT define danger -- the ED label comes from common.ed_baseline via
common.preprocessing. Steps 2 and 3 do not consume this step's output.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from common.preprocessing import get_processed_data
from common.io_utils import save_df, save_json, write_manifest, ensure_dir

DANGER_LABELS = {"ED_CAUTION", "ED_DANGEROUS", "ED_VERY_DANGEROUS"}
ITEM_COLS = ["temp_bin", "aqi_bin", "hum_bin", "wind_bin", "uv_bin", "climate_zone"]


class ExerciseDangerAssociationMiner:
    def __init__(self, min_support=config.ASSOC_MIN_SUPPORT,
                 min_confidence=config.ASSOC_MIN_CONFIDENCE):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.df = None
        self.rules = None
        self.frequent_itemsets = None

    # -- transactions ------------------------------------------------------
    def _transactions(self, df, include_ed=True):
        cols = ITEM_COLS + (["ed_category"] if include_ed else [])
        return df[cols].astype(str).values.tolist()

    def _encode(self, transactions):
        te = TransactionEncoder()
        arr = te.fit(transactions).transform(transactions)
        return pd.DataFrame(arr, columns=te.columns_)

    # -- (A) explainable rules with ED consequent ---------------
    def mine_rules(self, df):
        trans = self._transactions(df, include_ed=True)
        enc = self._encode(trans)
        self.frequent_itemsets = apriori(enc, min_support=self.min_support,
                                         use_colnames=True, max_len=3)
        if self.frequent_itemsets.empty:
            self.rules = pd.DataFrame()
            return self.rules
        rules = association_rules(self.frequent_itemsets, metric="confidence",
                                  min_threshold=self.min_confidence)
        # keep single ED_* consequents only -> readable safety rules
        def _is_ed_consequent(cons):
            return len(cons) == 1 and str(list(cons)[0]).startswith("ED_")
        rules = rules[rules["consequents"].apply(_is_ed_consequent)]
        rules = rules[rules["lift"] > 1.0]
        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: list(s)[0])
        self.rules = rules.sort_values(["confidence", "lift"], ascending=False)
        return self.rules

    # - (B) NON-CIRCULAR: danger-scenario frequency ----------------
    def danger_frequency(self, df):
        """How often each raw weather combination actually occurs, and what
        share of those occurrences are dangerous. Uses the raw items only."""
        d = df.copy()
        d["is_dangerous"] = d["ed_category"].isin(DANGER_LABELS)
        grp = (d.groupby(["temp_bin", "aqi_bin"])
                 .agg(n=("is_dangerous", "size"),
                      danger_rate=("is_dangerous", "mean"),
                      mean_ed=("ed_score", "mean"))
                 .reset_index())
        grp["support"] = grp["n"] / len(d)
        return grp.sort_values(["danger_rate", "support"], ascending=False)

    # -- (B) NON-CIRCULAR: heat x pollution interaction ---------------------
    def interaction_evidence(self, df):
        """Quantifies whether heat+pollution together are MORE dangerous than
        the sum of their parts -> empirical justification for the synergy term."""
        d = df.copy()
        hot = d["temperature_celsius"] > 28 if "temperature_celsius" in d else pd.Series(False, index=d.index)
        polluted = d["ed_f_air"] > 18
        base = d["ed_score"].mean()
        only_hot = d.loc[hot & ~polluted, "ed_score"].mean()
        only_pol = d.loc[~hot & polluted, "ed_score"].mean()
        both = d.loc[hot & polluted, "ed_score"].mean()
        expected_additive = (np.nan_to_num(only_hot - base) +
                             np.nan_to_num(only_pol - base) + base)
        return {
            "mean_ed_overall": float(base),
            "mean_ed_hot_only": float(np.nan_to_num(only_hot)),
            "mean_ed_polluted_only": float(np.nan_to_num(only_pol)),
            "mean_ed_hot_and_polluted": float(np.nan_to_num(both)),
            "expected_if_additive": float(expected_additive),
            "synergy_excess": float(np.nan_to_num(both) - expected_additive),
            "n_hot_and_polluted": int((hot & polluted).sum()),
        }


def main():
    print("=" * 70, "\nSTEP 1: ASSOCIATION RULE MINING\n", "=" * 70)
    df = get_processed_data()
    print(f"Loaded processed data: {df.shape}")

    miner = ExerciseDangerAssociationMiner()
    rules = miner.mine_rules(df)
    freq = miner.danger_frequency(df)
    inter = miner.interaction_evidence(df)

    out = ensure_dir(config.STEP1_OUT)
    a = {}
    a["association_rules"] = save_df(
        rules[["antecedents", "consequents", "support", "confidence", "lift"]]
        if not rules.empty else pd.DataFrame(),
        out / "association_rules.csv")
    a["danger_scenario_frequency"] = save_df(freq, out / "danger_scenario_frequency.csv")
    a["interaction_evidence"] = save_json(inter, out / "interaction_evidence.json")
    write_manifest(out, "step1_association_rules", a,
                   extra={"n_rules": int(len(rules)),
                          "synergy_excess": inter["synergy_excess"]})

    print(f"Rules found: {len(rules)} | synergy excess: {inter['synergy_excess']:.2f} ED pts")
    print(f"Artifacts -> {out}")
    return miner


if __name__ == "__main__":
    main()
