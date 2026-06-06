"""Stage 3 — gene-space harmonization (A3.1, do not skip).

L1000 measures only ~1,000 landmark genes, but Signature A spans the full
transcriptome. A naive overlap silently drops most signature genes and corrupts
the connectivity score. So before scoring we: map both sides to a common
identifier, restrict the query to the L1000-measured (landmark) space, and
report coverage — how many Signature-A genes survive. If coverage is low we
keep the landmark genes that remain and flag the limitation rather than hiding
it.
"""
from __future__ import annotations

from ..artifacts import Store, get_logger
from . import ingest

log = get_logger("m3.harmonize")
PRIMARY = "query.json"


def run(cfg, force: bool = False) -> Store:
    store = Store(cfg.cache_dir, "harmonize")
    if store.exists(PRIMARY) and not force:
        log.info("cached — skipping")
        return store

    ing = Store(cfg.cache_dir, "ingest")
    sig = Store(cfg.cache_dir, "signatures")

    landmark = set(ingest.read_landmark(ing))
    sigA = sig.read_json("signatureA.json")
    up_full, down_full = sigA["up"], sigA["down"]

    landmark_only = bool(cfg.get("harmonize.landmark_only", True))
    min_cov = float(cfg.get("harmonize.min_coverage", 0.25))

    if landmark_only:
        up = [g for g in up_full if g in landmark]
        down = [g for g in down_full if g in landmark]
    else:
        up, down = list(up_full), list(down_full)

    n_in = len(up_full) + len(down_full)
    n_out = len(up) + len(down)
    coverage = (n_out / n_in) if n_in else 0.0
    low = coverage < min_cov

    store.write_json(PRIMARY, {"up": up, "down": down})
    cov = {
        "namespace": cfg.get("harmonize.id_namespace", "hgnc_symbol"),
        "landmark_only": landmark_only,
        "signatureA_genes_in": n_in,
        "signatureA_genes_scored": n_out,
        "coverage": coverage,
        "up_scored": len(up),
        "down_scored": len(down),
        "min_coverage": min_cov,
        "low_coverage_flag": low,
    }
    store.write_json("coverage.json", cov)

    msg = "Signature-A coverage in L1000 space: %d/%d (%.1f%%)"
    if low:
        log.warning(msg + "  — BELOW %.0f%% threshold; weighting by surviving landmark genes",
                    n_out, n_in, 100 * coverage, 100 * min_cov)
    else:
        log.info(msg, n_out, n_in, 100 * coverage)
    return store
