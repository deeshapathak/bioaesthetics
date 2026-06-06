"""Cross-cohort meta-analysis of age-DE (A2.1).

GTEx within-site skin is underpowered, so Signature A from GTEx alone is noisy.
We meta-analyze *at the statistic level* — never pool raw counts — to avoid the
cross-platform batch artifacts that pooling introduces.

Two methods, matching the recipe:
  * random_effects   — DerSimonian-Laird effect-size meta-analysis.
  * rank_aggregation — robust rank aggregation across studies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import benjamini_hochberg


def random_effects_meta(deg_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """DerSimonian-Laird random-effects meta-analysis per gene.

    Each input is a DE table (index = gene) with 'effect' and 'se' columns.
    Genes are aligned on their intersection. Returns a table with the combined
    effect, se, z, p, q and the per-gene replication count (k studies).
    """
    if not deg_tables:
        raise ValueError("need at least one DE table")
    genes = deg_tables[0].index
    for d in deg_tables[1:]:
        genes = genes.intersection(d.index)
    genes = list(genes)

    effects = np.column_stack([d.loc[genes, "effect"].to_numpy() for d in deg_tables])  # (g, k)
    ses = np.column_stack([d.loc[genes, "se"].to_numpy() for d in deg_tables])
    var = np.maximum(ses ** 2, 1e-12)
    w = 1.0 / var                                  # fixed-effect weights

    # fixed-effect mean and Cochran's Q
    mean_fe = (w * effects).sum(axis=1) / w.sum(axis=1)
    Q = (w * (effects - mean_fe[:, None]) ** 2).sum(axis=1)
    k = effects.shape[1]
    c = w.sum(axis=1) - (w ** 2).sum(axis=1) / w.sum(axis=1)
    tau2 = np.maximum((Q - (k - 1)) / np.maximum(c, 1e-12), 0.0)  # between-study var

    w_re = 1.0 / (var + tau2[:, None])
    eff_re = (w_re * effects).sum(axis=1) / w_re.sum(axis=1)
    se_re = np.sqrt(1.0 / w_re.sum(axis=1))
    z = eff_re / se_re
    pval = 2.0 * stats.norm.sf(np.abs(z))
    qval = benjamini_hochberg(pval)

    return pd.DataFrame(
        {
            "effect": eff_re,
            "se": se_re,
            "z": z,
            "pval": pval,
            "padj": qval,
            "replication": int(k),
        },
        index=genes,
    )


def robust_rank_aggregation(deg_tables: list[pd.DataFrame], direction: str) -> pd.DataFrame:
    """Robust rank aggregation (RRA) across studies for one direction.

    For each study, genes are ranked by signed evidence in the requested
    direction (most up/most down first), normalized to (0, 1]. The RRA score is
    a simple order-statistic p-value (beta-distribution) on the per-gene rank
    vector — small score = consistently near the top across studies.
    """
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    sign = 1.0 if direction == "up" else -1.0

    genes = deg_tables[0].index
    for d in deg_tables[1:]:
        genes = genes.intersection(d.index)
    genes = list(genes)
    n = len(genes)

    norm_ranks = []
    for d in deg_tables:
        score = sign * d.loc[genes, "effect"].to_numpy() / np.maximum(
            d.loc[genes, "se"].to_numpy(), 1e-12)
        order = np.argsort(-score, kind="mergesort")
        r = np.empty(n, dtype=float)
        r[order] = (np.arange(1, n + 1)) / n          # in (0, 1]
        norm_ranks.append(r)
    R = np.column_stack(norm_ranks)                   # (g, k)

    # RRA: min over j of Beta(j, k-j+1).cdf( j-th smallest normalized rank )
    R_sorted = np.sort(R, axis=1)
    k = R.shape[1]
    rho = np.ones(n)
    for j in range(1, k + 1):
        beta_cdf = stats.beta.cdf(R_sorted[:, j - 1], j, k - j + 1)
        rho = np.minimum(rho, beta_cdf)
    rho = np.clip(rho * k, 0, 1)                       # Bonferroni over order stats
    qval = benjamini_hochberg(rho)
    return pd.DataFrame({"rra_score": rho, "padj": qval}, index=genes).sort_values("rra_score")
