"""Stage 5 — validation gates (A4).

Face validity (known actives near the top) is necessary but not sufficient. The
real gate is *blind recovery*: curate a positive set, hide it, and measure
whether the engine retrieves it. We also run null/FDR, directional controls
(aging-promoters must rank at the bottom — they catch sign bugs), signature
concordance, and a leakage check, then emit a single PASS/FAIL against the
§A4 go/no-go thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import Store, get_logger
from ..stats.connectivity_score import (
    empirical_qvalues,
    permutation_null,
    rank_genes_by_expression,
)
from ..stats.metrics import (
    enrichment_in_top_fraction,
    precision_at_k,
    roc_auc,
)
from . import ingest

log = get_logger("m3.validate")
PRIMARY = "report.json"


def run(cfg, force: bool = False) -> Store:
    store = Store(cfg.cache_dir, "validate")
    if store.exists(PRIMARY) and not force:
        log.info("cached — skipping")
        return store

    ing = Store(cfg.cache_dir, "ingest")
    conn = Store(cfg.cache_dir, "connectivity")
    harm = Store(cfg.cache_dir, "harmonize")
    sigs = Store(cfg.cache_dir, "signatures")

    scores = conn.read_csv(PRIMARY.replace("report.json", "compound_scores.csv"), index_col=0)
    truth = ingest.read_truth(ing)
    coverage = harm.read_json("coverage.json")
    concordance = sigs.read_json("concordance.json")

    compounds = list(scores.index)
    cs = scores["cs"].to_numpy()
    rank_frac = scores["cs"].rank(method="average").to_numpy() / len(scores)  # 0=top reverser

    known = set(truth["known_actives"])
    promoters = set(truth["aging_promoters"])
    labels = np.array([c in known for c in compounds])

    gonogo = cfg.get("validation.go_no_go")
    top_decile = float(cfg.get("validation.top_decile", 0.10))

    # ---- Gate 1: held-out recovery (PRIMARY) --------------------------------
    auc = roc_auc(cs, labels, lower_is_positive=True)
    p_at_known = precision_at_k(cs, labels, len(known), lower_is_positive=True)
    p_at_10 = precision_at_k(cs, labels, 10, lower_is_positive=True)
    enr = enrichment_in_top_fraction(cs, labels, top_decile, lower_is_positive=True)
    recovery = {
        "roc_auc": auc,
        "precision_at_k_known": p_at_known,
        "precision_at_10": p_at_10,
        "top_decile_enrichment": enr,
        "pass": bool(auc >= gonogo["recovery_auc_min"]),
    }

    # ---- Gate 2: face validity (sanity) -------------------------------------
    known_ranks = {c: int(scores.index.get_loc(c)) + 1 for c in known if c in scores.index}
    n_top_decile = max(int(round(len(scores) * top_decile)), 1)
    in_top_decile = sorted([c for c, r in known_ranks.items() if r <= n_top_decile],
                           key=lambda c: known_ranks[c])
    face = {
        "known_in_top_decile": len(in_top_decile),
        "n_known": len(known),
        "examples": in_top_decile[:8],
        "pass": bool(len(in_top_decile) >= max(1, len(known) // 2)),
    }

    # ---- Gate 3: null / FDR -------------------------------------------------
    qvals = _permutation_qvalues(cfg, ing, harm, cs, compounds)
    q_by_compound = dict(zip(compounds, qvals))
    enr_q = enr["pval"]                       # over-representation p as the enrichment q
    nullfdr = {
        "n_permutations": int(cfg.get("validation.permutations")),
        "n_compounds_q_lt_0.1": int((qvals < 0.1).sum()),
        "median_q_known": float(np.median([q_by_compound[c] for c in known if c in q_by_compound]) or 1.0),
        "top_decile_enrichment_q": float(enr_q),
        "pass": bool(enr_q < gonogo["top_decile_q_max"]),
    }

    # ---- Gate 4: directional controls (aging-promoters at the bottom) -------
    prom_rank = {c: float(rank_frac[compounds.index(c)]) for c in promoters if c in compounds}
    prom_in_top_decile = [c for c, rf in prom_rank.items() if rf <= top_decile]
    mean_prom_rank = float(np.mean(list(prom_rank.values()))) if prom_rank else 0.0
    directional = {
        "n_promoters": len(prom_rank),
        "mean_rank_fraction": mean_prom_rank,         # ~1.0 == bottom (good)
        "promoters_in_top_decile": len(prom_in_top_decile),
        # promoters must rank near the bottom and none may sneak into the top decile
        "pass": bool(mean_prom_rank >= 0.60 and len(prom_in_top_decile) == 0),
    }

    # ---- Gate 5: signature concordance (A vs C; B context only) -------------
    conc_pass = bool(
        concordance["A_vs_C_jaccard"] > 0.02
        and concordance["A_vs_B_jaccard"] <= max(concordance["A_vs_C_jaccard"] * 1.5, 0.05)
    )
    concordance_gate = {**concordance, "pass": conc_pass}

    # ---- Gate 6: leakage control (no single cell line drives top hits) ------
    top = scores.head(n_top_decile)
    line_counts = top["top_cell_line"].value_counts()
    dominant_share = float(line_counts.iloc[0] / line_counts.sum()) if len(line_counts) else 1.0
    leakage = {
        "top_decile_cell_line_counts": line_counts.to_dict(),
        "dominant_cell_line_share": dominant_share,
        "mean_max_cell_line_share": float(top["max_cell_line_share"].mean()),
        "pass": bool(dominant_share < 0.70),
    }

    # ---- Gate G0 (only if ML g(x,u) added) ----------------------------------
    g0 = {"status": "not_applicable", "note": "no ML g(x,u) module in M3 baseline"}

    # ---- coverage gate (A3.1) ----------------------------------------------
    coverage_gate = {**coverage, "pass": bool(not coverage["low_coverage_flag"])}

    # ---- go / no-go decision ------------------------------------------------
    decision_pass = bool(
        recovery["pass"]
        and nullfdr["pass"]
        and (directional["pass"] or not gonogo.get("require_directional_controls", True))
    )
    report = {
        "decision": "PASS" if decision_pass else "FAIL",
        "go_no_go_rule": {
            "recovery_auc_min": gonogo["recovery_auc_min"],
            "top_decile_q_max": gonogo["top_decile_q_max"],
            "require_directional_controls": gonogo.get("require_directional_controls", True),
        },
        "gates": {
            "held_out_recovery": recovery,
            "face_validity": face,
            "null_fdr": nullfdr,
            "directional_controls": directional,
            "signature_concordance": concordance_gate,
            "leakage_control": leakage,
            "gene_space_coverage": coverage_gate,
            "g0_ml": g0,
        },
        "compound_qvalues": q_by_compound,
    }
    store.write_json(PRIMARY, report)

    log.info("RECOVERY  ROC-AUC=%.3f (min %.2f)  p@k=%.2f  -> %s",
             auc, gonogo["recovery_auc_min"], p_at_known,
             "PASS" if recovery["pass"] else "FAIL")
    log.info("DIRECTIONAL  promoter mean-rank=%.2f  in-top-decile=%d  -> %s",
             mean_prom_rank, len(prom_in_top_decile), "PASS" if directional["pass"] else "FAIL")
    log.info("ENRICHMENT  top-decile q=%.3g  -> %s", enr_q, "PASS" if nullfdr["pass"] else "FAIL")
    log.info("DECISION  %s", report["decision"])
    return store


def _permutation_qvalues(cfg, ing, harm, cs, compounds) -> np.ndarray:
    """Per-compound empirical q-values from a gene-label permutation null."""
    Z, meta, landmark = ingest.read_l1000(ing)
    query = harm.read_json("query.json")
    rng = np.random.default_rng(cfg.seed + 7)
    n_perm = int(cfg.get("validation.permutations"))

    # build a modest pool of ranked lists for the null (sampling rows of Z)
    n_pool = min(300, Z.shape[0])
    pool_idx = rng.choice(Z.shape[0], size=n_pool, replace=False)
    ranked_lists = [rank_genes_by_expression(landmark, Z[i]) for i in pool_idx]
    null = permutation_null(ranked_lists, query["up"], query["down"], n_perm, rng)
    return empirical_qvalues(np.asarray(cs), null)
