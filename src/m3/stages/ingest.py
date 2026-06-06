"""Stage 1 — ingest (A1).

In synthetic mode, generate the planted dataset; in real mode, call the
adapters. Persist everything to the cache in plain formats so every downstream
stage reads from disk (A6: staged, cached, independently rerunnable).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import Store, get_logger
from ..data import synthetic as syn

log = get_logger("m3.ingest")
PRIMARY = "manifest.json"


def run(cfg, force: bool = False) -> Store:
    store = Store(cfg.cache_dir, "ingest")
    if store.exists(PRIMARY) and not force:
        log.info("cached — skipping")
        return store

    if cfg.mode == "synthetic":
        data = syn.generate(cfg)
        provenance = "synthetic (planted signal)"
    else:
        from ..data.adapters import load_real
        data = load_real(cfg)
        provenance = "real adapters"

    # gene universe + L1000 landmark space
    store.write_json("genes.json", data.gene_ids)
    store.write_json("landmark.json", data.landmark_genes)

    # cohorts (GTEx sites + external aging series)
    cohorts: list[dict] = []
    for site, cohort in data.gtex_sites.items():
        _write_cohort(store, cohort, role="gtex_site")
        cohorts.append({"name": cohort.name, "role": "gtex_site"})
    for cohort in data.external_cohorts:
        _write_cohort(store, cohort, role="external")
        cohorts.append({"name": cohort.name, "role": "external"})
    store.write_json("cohorts.json", cohorts)

    # curated programs (Signature C)
    store.write_json("gene_sets.json", data.gene_sets)

    # L1000 signatures
    store.write_npz("l1000.npz", Z=data.l1000_signatures.astype(np.float32))
    store.write_csv("l1000_meta.csv", data.l1000_meta, index=False)
    store.write_csv("compound_meta.csv", data.compound_meta, index=True)

    # ground truth — used ONLY by the validation stage (blind recovery)
    store.write_json("truth.json", data.truth)

    manifest = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "provenance": provenance,
        "n_genes": len(data.gene_ids),
        "n_landmark": len(data.landmark_genes),
        "cohorts": cohorts,
        "n_compounds": int(data.compound_meta.shape[0]),
        "n_l1000_signatures": int(data.l1000_signatures.shape[0]),
        "accessions": cfg.get("datasets"),
    }
    store.write_json(PRIMARY, manifest)
    log.info("ingested %d genes, %d compounds, %d L1000 signatures (%s)",
             manifest["n_genes"], manifest["n_compounds"],
             manifest["n_l1000_signatures"], provenance)
    return store


# ---------------------------------------------------------------- writers/readers
def _write_cohort(store: Store, cohort: syn.Cohort, role: str) -> None:
    store.write_npz(f"cohort_{cohort.name}.npz", X=cohort.expr.to_numpy().astype(np.float32))
    store.write_csv(f"cohort_{cohort.name}_meta.csv", cohort.meta, index=True)


def read_genes(store: Store) -> list[str]:
    return store.read_json("genes.json")


def read_landmark(store: Store) -> list[str]:
    return store.read_json("landmark.json")


def read_cohort(store: Store, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    genes = read_genes(store)
    meta = store.read_csv(f"cohort_{name}_meta.csv", index_col=0)
    X = store.read_npz(f"cohort_{name}.npz")["X"]
    expr = pd.DataFrame(X, columns=genes, index=meta.index)
    return expr, meta


def read_cohorts_index(store: Store) -> list[dict]:
    return store.read_json("cohorts.json")


def read_l1000(store: Store) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    Z = store.read_npz("l1000.npz")["Z"]
    meta = store.read_csv("l1000_meta.csv", index_col=None)
    landmark = read_landmark(store)
    return Z, meta, landmark


def read_compound_meta(store: Store) -> pd.DataFrame:
    return store.read_csv("compound_meta.csv", index_col=0)


def read_gene_sets(store: Store) -> dict[str, list[str]]:
    return store.read_json("gene_sets.json")


def read_truth(store: Store) -> dict:
    return store.read_json("truth.json")
