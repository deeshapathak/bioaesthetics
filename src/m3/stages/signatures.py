"""Stage 2 — build the old→young skin signature (A2, A2.1).

Signature A (PRIMARY): intrinsic age, site-controlled — age-DE within each GTEx
skin site, covarying sex/RIN/ischemic time, kept concordant across both sites,
then strengthened by cross-cohort meta-analysis (A2.1). This is the reversal
target.

Signature B (CONTEXTUAL): site / photoexposure axis — robustness check only.
Signature C (CROSS-CHECK): curated programs (SenMayo, ECM/collagen, ...), used
to annotate A with a 'why' (which aesthetic axis each gene belongs to).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import Store, get_logger
from ..stats.de import concordant, differential_expression
from ..stats.meta import random_effects_meta, robust_rank_aggregation
from . import ingest

log = get_logger("m3.signatures")
PRIMARY = "signatureA.json"


def run(cfg, force: bool = False) -> Store:
    store = Store(cfg.cache_dir, "signatures")
    if store.exists(PRIMARY) and not force:
        log.info("cached — skipping")
        return store

    ing = Store(cfg.cache_dir, "ingest")
    padj = float(cfg.get("signatures.padj"))
    top_n = int(cfg.get("signatures.top_n"))
    covars = list(cfg.get("signatures.covariates", []))
    min_repl = int(cfg.get("signatures.meta.min_replication", 2))
    meta_method = cfg.get("signatures.meta.method", "random_effects")

    cohorts = ingest.read_cohorts_index(ing)
    gtex = [c["name"] for c in cohorts if c["role"] == "gtex_site"]
    externals = [c["name"] for c in cohorts if c["role"] == "external"]

    # ---- per-cohort age-DE (old vs young), covarying nuisance ---------------
    de_by_cohort: dict[str, pd.DataFrame] = {}
    for name in gtex + externals:
        expr, meta = ingest.read_cohort(ing, name)
        cov = _covariate_frame(meta, covars)
        de = differential_expression(expr, meta["group"].to_numpy(), cov)
        de_by_cohort[name] = de
        log.info("age-DE %-22s: %d genes padj<%.2g", name, int((de["padj"] < padj).sum()), padj)

    # ---- Signature A: concordant across the two GTEx sites ------------------
    site_a, site_b = gtex[0], gtex[1]
    up_gtex = set(concordant(de_by_cohort[site_a], de_by_cohort[site_b], "up", padj))
    down_gtex = set(concordant(de_by_cohort[site_a], de_by_cohort[site_b], "down", padj))

    # ---- A2.1: strengthen across cohorts via meta-analysis ------------------
    all_de = [de_by_cohort[n] for n in gtex + externals]
    meta_tbl = random_effects_meta(all_de)
    # per-gene replication: # cohorts where padj<cut and sign matches meta sign
    repl = _replication_count(de_by_cohort, meta_tbl, padj)
    meta_tbl["replication"] = repl.reindex(meta_tbl.index).fillna(0).astype(int)

    if meta_method == "rank_aggregation":
        rra_up = robust_rank_aggregation(all_de, "up")
        rra_down = robust_rank_aggregation(all_de, "down")
        log.info("rank-aggregation computed (up=%d, down=%d genes scored)",
                 len(rra_up), len(rra_down))

    # final A = GTEx-concordant AND replicated in >= min_repl cohorts
    def finalize(direction: str, gtex_set: set[str]) -> list[str]:
        sign = 1.0 if direction == "up" else -1.0
        cand = [g for g in gtex_set
                if meta_tbl.loc[g, "replication"] >= min_repl
                and np.sign(meta_tbl.loc[g, "effect"]) == sign
                and meta_tbl.loc[g, "padj"] < padj]
        cand.sort(key=lambda g: abs(meta_tbl.loc[g, "effect"]), reverse=True)
        return cand[:top_n]

    up = finalize("up", up_gtex)
    down = finalize("down", down_gtex)
    log.info("Signature A: %d up / %d down genes (replicated >=%d cohorts)",
             len(up), len(down), min_repl)

    # ---- Signature B: site / photoexposure (contextual) ---------------------
    up_b, down_b = _signature_b(ing, gtex, covars, padj, top_n)

    # ---- Signature C: curated programs + per-gene annotation ----------------
    gene_sets = ingest.read_gene_sets(ing)
    annotation = _annotate(up + down, gene_sets)

    # ---- concordance summary (A vs C agree, B contextual) -------------------
    concordance = _concordance_summary(up, down, up_b, down_b, gene_sets)

    # ---- persist ------------------------------------------------------------
    store.write_json(PRIMARY, {"up": up, "down": down})
    store.write_json("signatureB.json", {"up": up_b, "down": down_b})
    store.write_json("signatureC.json", {"programs": gene_sets, "annotation": annotation})
    store.write_json("concordance.json", concordance)
    # per-gene table for the audit trail
    sigA_genes = pd.DataFrame(
        {"direction": (["up"] * len(up)) + (["down"] * len(down))},
        index=up + down,
    )
    sigA_genes["effect"] = meta_tbl.reindex(sigA_genes.index)["effect"]
    sigA_genes["padj"] = meta_tbl.reindex(sigA_genes.index)["padj"]
    sigA_genes["replication"] = meta_tbl.reindex(sigA_genes.index)["replication"]
    sigA_genes["programs"] = [", ".join(annotation.get(g, [])) for g in sigA_genes.index]
    store.write_csv("signatureA_genes.csv", sigA_genes, index=True)
    log.info("A↔C concordance=%.2f  B-overlap=%.2f (context only)",
             concordance["A_vs_C_jaccard"], concordance["A_vs_B_jaccard"])
    return store


# --------------------------------------------------------------------- helpers
def _covariate_frame(meta: pd.DataFrame, covars: list[str]) -> pd.DataFrame | None:
    cols = [c for c in covars if c in meta.columns]
    if not cols:
        return None
    return meta[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _replication_count(de_by_cohort, meta_tbl, padj) -> pd.Series:
    genes = meta_tbl.index
    meta_sign = np.sign(meta_tbl["effect"])
    counts = pd.Series(0, index=genes, dtype=int)
    for de in de_by_cohort.values():
        de = de.reindex(genes)
        hit = (de["padj"] < padj) & (np.sign(de["effect"]) == meta_sign)
        counts = counts + hit.fillna(False).astype(int)
    return counts


def _signature_b(ing, gtex, covars, padj, top_n):
    """Sun-exposed (lower_leg) vs non-sun (suprapubic) — contextual axis."""
    # site is the group here; covary age + sex to isolate the site contrast
    frames, groups, metas = [], [], []
    for gi, name in enumerate(gtex):
        expr, meta = ingest.read_cohort(ing, name)
        frames.append(expr)
        metas.append(meta)
        groups.append(np.full(expr.shape[0], gi))  # 0 = first site, 1 = second
    expr = pd.concat(frames, axis=0)
    meta = pd.concat(metas, axis=0)
    group = np.concatenate(groups)
    cov = _covariate_frame(meta, ["age", "sex"])
    de = differential_expression(expr, group, cov)
    up = de[(de["effect"] > 0) & (de["padj"] < padj)].sort_values("effect", ascending=False)
    down = de[(de["effect"] < 0) & (de["padj"] < padj)].sort_values("effect")
    return up.index[:top_n].tolist(), down.index[:top_n].tolist()


def _annotate(genes: list[str], gene_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    members = {name: set(g) for name, g in gene_sets.items()}
    out: dict[str, list[str]] = {}
    for g in genes:
        progs = [name for name, s in members.items() if g in s]
        if progs:
            out[g] = progs
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _concordance_summary(up, down, up_b, down_b, gene_sets) -> dict:
    A = set(up) | set(down)
    B = set(up_b) | set(down_b)
    C = set().union(*[set(v) for v in gene_sets.values()]) if gene_sets else set()
    return {
        "A_size": len(A),
        "B_size": len(B),
        "C_size": len(C),
        "A_vs_C_jaccard": _jaccard(A, C),   # should be appreciable (A grounded in curated axes)
        "A_vs_B_jaccard": _jaccard(A, B),   # B context only — must not dominate A
        "note": "Reversal target = A, cross-checked against C; B supports/contextualizes only.",
    }
