"""Stage 4 — connectivity scoring (A3, the core).

For every L1000 compound signature, compute how strongly it reverses the aging
query (CMap weighted-KS). Then aggregate across cell lines and doses per
compound (CLUE tau vs the reference distribution), up-weighting skin-relevant
cell lines and recording cell-line provenance for every score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import Store, get_logger
from ..stats.connectivity_score import (
    connectivity_score,
    rank_genes_by_expression,
    tau_score,
)
from . import ingest

log = get_logger("m3.connectivity")
PRIMARY = "compound_scores.csv"


def run(cfg, force: bool = False) -> Store:
    store = Store(cfg.cache_dir, "connectivity")
    if store.exists(PRIMARY) and not force:
        log.info("cached — skipping")
        return store

    ing = Store(cfg.cache_dir, "ingest")
    harm = Store(cfg.cache_dir, "harmonize")

    Z, meta, landmark = ingest.read_l1000(ing)
    query = harm.read_json("query.json")
    up_set, down_set = query["up"], query["down"]
    skin_relevant = set(cfg.get("connectivity.skin_relevant_cell_lines", []))

    # ---- per-signature connectivity (per compound x cell line x dose) -------
    n = Z.shape[0]
    cs = np.empty(n, dtype=float)
    for i in range(n):
        ranked = rank_genes_by_expression(landmark, Z[i])
        cs[i] = connectivity_score(ranked, up_set, down_set)
    meta = meta.copy()
    meta["cs"] = cs
    meta["skin_relevant"] = meta["cell_line"].isin(skin_relevant)
    store.write_csv(PRIMARY.replace("compound_scores", "per_signature"), meta, index=False)
    log.info("scored %d signatures (cs<0 => reverses aging)", n)

    # ---- aggregate to one score per compound (tau vs Touchstone) ------------
    reference = cs                                   # empirical Touchstone distribution
    rows = []
    for compound, grp in meta.groupby("compound"):
        w = np.where(grp["skin_relevant"].to_numpy(), 1.0, 0.5)  # up-weight skin lines
        agg_cs = float(np.average(grp["cs"].to_numpy(), weights=w))
        tau = tau_score(agg_cs, reference)
        # provenance: which cell lines contributed, and the dominant one
        by_line = grp.groupby("cell_line")["cs"].mean().sort_values()
        top_line = by_line.index[0]                  # most-reversing cell line
        line_share = _dominance(grp)
        rows.append({
            "compound": compound,
            "cs": agg_cs,
            "tau": tau,
            "n_signatures": int(grp.shape[0]),
            "n_cell_lines": int(grp["cell_line"].nunique()),
            "top_cell_line": top_line,
            "top_cell_line_skin_relevant": bool(top_line in skin_relevant),
            "max_cell_line_share": line_share,
        })

    scores = pd.DataFrame(rows).set_index("compound").sort_values("cs")  # most-negative first
    store.write_csv(PRIMARY, scores, index=True)
    log.info("aggregated to %d compounds; top reverser cs=%.3f (tau=%.0f)",
             scores.shape[0], scores["cs"].iloc[0], scores["tau"].iloc[0])
    return store


def _dominance(grp: pd.DataFrame) -> float:
    """Share of the (negative) reversal signal owned by a single cell line —
    used downstream for the A4 leakage check."""
    rev = grp[grp["cs"] < 0]
    if rev.empty:
        return 0.0
    by_line = (-rev.groupby("cell_line")["cs"].sum())
    total = by_line.sum()
    return float(by_line.max() / total) if total > 0 else 0.0
