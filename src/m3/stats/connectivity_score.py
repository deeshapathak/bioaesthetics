"""Connectivity / reversal scoring — the core of M3 (A3).

For every L1000 compound signature we compute how strongly it *reverses* the
aging signature. We implement the CMap connectivity score: a Kolmogorov-Smirnov
running-sum statistic of the query UP / DOWN tag sets against the compound's
rank-ordered gene list.

Convention (sign matters — A4 directional controls catch bugs here):
    cs < 0  => the compound pushes old→young  => candidate rejuvenator.

A reverser drives the aging-UP genes DOWN (to the bottom of its ranked list)
and the aging-DOWN genes UP (to the top), so ES_up < 0, ES_down > 0, and the
combined connectivity is strongly negative.
"""
from __future__ import annotations

import numpy as np


def _ks_enrichment(rank_of_gene: dict[str, int], tag_set: list[str], n_total: int) -> float:
    """Signed CMap KS statistic for one tag set against one ranked list.

    rank_of_gene : gene -> rank position (1 = top / most up-regulated by the
                   compound, n_total = bottom / most down-regulated).
    Returns ES in [-1, 1].
    """
    positions = sorted(rank_of_gene[g] for g in tag_set if g in rank_of_gene)
    t = len(positions)
    if t == 0:
        return 0.0
    n = n_total
    a = 0.0
    b = 0.0
    for j, V in enumerate(positions, start=1):
        a = max(a, j / t - V / n)
        b = max(b, V / n - (j - 1) / t)
    return a if a > b else -b


def connectivity_score(
    ranked_genes: list[str],
    up_set: list[str],
    down_set: list[str],
) -> float:
    """CMap connectivity of one compound signature vs the aging query.

    ranked_genes : compound's genes ordered most-UP-regulated → most-DOWN.
    up_set / down_set : the aging signature (high-in-old / low-in-old).
    """
    n = len(ranked_genes)
    rank_of_gene = {g: i + 1 for i, g in enumerate(ranked_genes)}
    es_up = _ks_enrichment(rank_of_gene, up_set, n)
    es_down = _ks_enrichment(rank_of_gene, down_set, n)
    # CMap rule: only count when the two tag sets pull in opposite directions.
    if np.sign(es_up) == np.sign(es_down):
        return 0.0
    return (es_up - es_down) / 2.0


def rank_genes_by_expression(genes: list[str], values: np.ndarray) -> list[str]:
    """Order genes most-up-regulated (highest z) → most-down-regulated."""
    order = np.argsort(-np.asarray(values, dtype=float), kind="mergesort")
    return [genes[i] for i in order]


def tau_score(compound_cs: float, reference_cs: np.ndarray) -> float:
    """CLUE-style tau: signed percentile of a compound's connectivity vs a
    reference (Touchstone) distribution, in [-100, 100].

    tau = -100 means more strongly reversing (negative) than ~all references;
    tau = +100 means more strongly mimicking than ~all references.
    """
    ref = np.asarray(reference_cs, dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return 0.0
    # signed: percentile of |cs| among references with the same sign, signed
    same_sign = ref[np.sign(ref) == np.sign(compound_cs)] if compound_cs != 0 else ref
    base = same_sign if same_sign.size else ref
    pct = 100.0 * (np.abs(base) <= abs(compound_cs)).mean()
    return float(np.sign(compound_cs) * pct)


def permutation_null(
    ranked_lists: list[list[str]],
    up_set: list[str],
    down_set: list[str],
    n_perm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null connectivity distribution by permuting the signature gene labels
    (A4: 'Permute signature genes → null connectivity').

    Returns an array of null connectivity scores (one per permutation, using a
    randomly chosen compound each time) to compare top hits against.
    """
    if not ranked_lists:
        return np.array([])
    universe = list({g for lst in ranked_lists for g in lst})
    n_up, n_down = len(up_set), len(down_set)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        ranked = ranked_lists[rng.integers(len(ranked_lists))]
        perm = rng.choice(universe, size=n_up + n_down, replace=False)
        fake_up = list(perm[:n_up])
        fake_down = list(perm[n_up:])
        null[i] = connectivity_score(ranked, fake_up, fake_down)
    return null


def empirical_qvalues(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Per-compound q-value: empirical left-tail rate in the null (reversers
    are negative), BH-adjusted across compounds."""
    from .metrics import benjamini_hochberg

    s = np.asarray(scores, dtype=float)
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return np.ones_like(s)
    m = null.size
    # left-tail p: fraction of null <= score (more extreme negative)
    pvals = np.array([(np.sum(null <= sc) + 1) / (m + 1) for sc in s])
    return benjamini_hochberg(pvals)
