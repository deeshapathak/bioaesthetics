"""Unit tests for the stats core — the math the gates depend on."""
import numpy as np
import pandas as pd

from m3.stats.connectivity_score import connectivity_score, rank_genes_by_expression
from m3.stats.de import concordant, differential_expression
from m3.stats.meta import random_effects_meta
from m3.stats.metrics import (
    benjamini_hochberg,
    enrichment_in_top_fraction,
    precision_at_k,
    roc_auc,
)


def test_connectivity_sign_convention():
    """A reverser must score negative; an aging-mimic must score positive."""
    genes = [f"G{i}" for i in range(60)]
    up = genes[:6]
    down = genes[-6:]
    vals = np.zeros(60)
    for g in up:
        vals[genes.index(g)] = -3.0      # compound pushes aging-up genes DOWN
    for g in down:
        vals[genes.index(g)] = +3.0      # and aging-down genes UP
    ranked = rank_genes_by_expression(genes, vals)
    assert connectivity_score(ranked, up, down) < -0.5
    ranked_mimic = rank_genes_by_expression(genes, -vals)
    assert connectivity_score(ranked_mimic, up, down) > 0.5


def test_connectivity_zero_when_same_sign():
    """CMap rule: when both tag sets pull the same way, connectivity is 0."""
    genes = [f"G{i}" for i in range(40)]
    up, down = genes[:4], genes[4:8]
    vals = np.zeros(40)
    for g in up + down:
        vals[genes.index(g)] = 3.0       # both sets at the top -> same sign
    ranked = rank_genes_by_expression(genes, vals)
    assert connectivity_score(ranked, up, down) == 0.0


def test_roc_auc_and_precision():
    scores = np.array([-0.9, -0.8, 0.1, 0.3, 0.7])
    labels = np.array([1, 1, 0, 0, 0])
    assert roc_auc(scores, labels, lower_is_positive=True) == 1.0
    assert precision_at_k(scores, labels, 2, lower_is_positive=True) == 1.0
    # reversed convention should flip a perfect score to 0
    assert roc_auc(scores, labels, lower_is_positive=False) == 0.0


def test_benjamini_hochberg_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.2, 0.5, 0.9])
    q = benjamini_hochberg(p)
    assert np.all((q >= 0) & (q <= 1))
    # q is monotone in the original p ordering here (already sorted)
    assert np.all(np.diff(q) >= -1e-12)
    assert q[0] >= p[0]


def test_enrichment_top_fraction():
    scores = np.linspace(-1, 1, 100)        # most-negative first = best
    labels = np.zeros(100, dtype=bool)
    labels[:10] = True                       # all positives are the top reversers
    enr = enrichment_in_top_fraction(scores, labels, 0.10, lower_is_positive=True)
    assert enr["observed"] == 10
    assert enr["fold_enrichment"] > 5
    assert enr["pval"] < 1e-6


def test_differential_expression_recovers_planted_effect():
    rng = np.random.default_rng(0)
    n, g = 60, 20
    X = rng.normal(size=(n, g))
    group = np.array([0] * 30 + [1] * 30)
    X[30:, 0] += 2.0                         # gene0 up in old
    X[30:, 1] -= 2.0                         # gene1 down in old
    de = differential_expression(pd.DataFrame(X, columns=[f"g{i}" for i in range(g)]), group)
    assert de.loc["g0", "effect"] > 1.0 and de.loc["g0", "padj"] < 0.05
    assert de.loc["g1", "effect"] < -1.0 and de.loc["g1", "padj"] < 0.05
    assert de.loc["g5", "padj"] > 0.05       # a null gene stays non-significant (usually)


def test_concordant_intersection():
    idx = [f"g{i}" for i in range(10)]
    a = pd.DataFrame({"effect": [2, 2, -2, 1, 0, 0, 0, 0, 0, 0],
                      "padj": [0.01, 0.01, 0.01, 0.2, 1, 1, 1, 1, 1, 1]}, index=idx)
    b = pd.DataFrame({"effect": [3, -1, -3, 2, 0, 0, 0, 0, 0, 0],
                      "padj": [0.01, 0.01, 0.01, 0.2, 1, 1, 1, 1, 1, 1]}, index=idx)
    up = concordant(a, b, "up", padj=0.05)
    down = concordant(a, b, "down", padj=0.05)
    assert "g0" in up          # up in both, significant
    assert "g1" not in up      # discordant direction
    assert "g2" in down        # down in both, significant
    assert "g3" not in up      # not significant (padj 0.2)


def test_random_effects_meta_combines_evidence():
    idx = ["g0", "g1"]
    studies = [
        pd.DataFrame({"effect": [1.0, 0.0], "se": [0.3, 0.3]}, index=idx),
        pd.DataFrame({"effect": [1.2, 0.0], "se": [0.3, 0.3]}, index=idx),
        pd.DataFrame({"effect": [0.9, 0.0], "se": [0.3, 0.3]}, index=idx),
    ]
    meta = random_effects_meta(studies)
    # combined SE should shrink below any single-study SE for the real effect
    assert meta.loc["g0", "se"] < 0.3
    assert meta.loc["g0", "padj"] < 0.05
    assert meta.loc["g1", "padj"] > 0.05
