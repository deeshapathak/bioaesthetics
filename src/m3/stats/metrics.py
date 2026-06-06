"""Validation metrics: BH-FDR, ROC-AUC, precision@k (A4)."""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment. Returns q-values, same shape."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p downward
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def roc_auc(scores: np.ndarray, labels: np.ndarray, lower_is_positive: bool = True) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), tie-aware.

    For connectivity, the most-negative scores are the best candidates, so
    ``lower_is_positive=True``: a positive (true rejuvenator) should get a
    *lower* score than a negative.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    if lower_is_positive:
        s = -s
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks (ties get mean rank)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    # tie correction: assign average rank within equal-value groups
    _assign_tie_ranks(s, ranks)
    sum_ranks_pos = ranks[y].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(values: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(values, kind="mergesort")
    sv = values[order]
    i = 0
    n = sv.size
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i:j + 1]]).mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1


def precision_at_k(scores: np.ndarray, labels: np.ndarray, k: int,
                   lower_is_positive: bool = True) -> float:
    """Fraction of the top-k ranked items that are true positives."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    k = min(k, s.size)
    if k == 0:
        return float("nan")
    order = np.argsort(s if lower_is_positive else -s, kind="mergesort")
    topk = order[:k]
    return float(y[topk].sum() / k)


def enrichment_in_top_fraction(scores: np.ndarray, labels: np.ndarray,
                               fraction: float, lower_is_positive: bool = True) -> dict:
    """How enriched are positives in the top fraction (e.g., top decile)?

    Returns observed count, expected-by-chance, fold-enrichment, and a
    hypergeometric p-value for the over-representation.
    """
    from scipy.stats import hypergeom

    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    n = s.size
    k = max(int(round(n * fraction)), 1)
    order = np.argsort(s if lower_is_positive else -s, kind="mergesort")
    top = set(order[:k].tolist())
    n_pos = int(y.sum())
    observed = int(sum(1 for i in top if y[i]))
    expected = n_pos * k / n
    fold = observed / expected if expected > 0 else float("nan")
    # P(X >= observed) drawing k from n with n_pos successes
    pval = float(hypergeom.sf(observed - 1, n, n_pos, k)) if n_pos > 0 else float("nan")
    return {
        "k": k,
        "n_positives": n_pos,
        "observed": observed,
        "expected": float(expected),
        "fold_enrichment": float(fold),
        "pval": pval,
    }
