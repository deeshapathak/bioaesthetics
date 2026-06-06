"""Differential expression: vectorized OLS with covariates (A2).

Age-DE is computed *within each skin site separately* (old vs young), covarying
sex, RIN, ischemic time. We fit, per gene, a linear model

    y = b0 + b_group * group + sum_k b_k * covariate_k + e

and report the group coefficient (the log fold-change of old vs young), its
standard error, t-statistic, p-value, and BH-adjusted q-value. Fitting all
genes at once via a shared design matrix keeps this fast without statsmodels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import benjamini_hochberg


def differential_expression(
    expr: pd.DataFrame,
    group: np.ndarray,
    covariates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-gene old-vs-young DE.

    Parameters
    ----------
    expr : DataFrame, shape (n_samples, n_genes)
        Expression matrix (rows = samples, columns = genes).
    group : array, shape (n_samples,)
        Binary group label; 1 = old (the "high" level), 0 = young.
    covariates : DataFrame, optional, shape (n_samples, n_cov)
        Numeric nuisance covariates (encode categoricals upstream).

    Returns
    -------
    DataFrame indexed by gene with columns:
        effect (b_group, = logFC old vs young), se, t, pval, padj.
    """
    Y = np.asarray(expr.values, dtype=float)          # (n, g)
    n, g = Y.shape
    group = np.asarray(group, dtype=float).reshape(n, 1)

    cols = [np.ones((n, 1)), group]                   # intercept, group
    if covariates is not None and covariates.shape[1] > 0:
        C = np.asarray(covariates.values, dtype=float)
        # center covariates for numerical stability; doesn't change b_group
        C = C - C.mean(axis=0, keepdims=True)
        cols.append(C)
    D = np.hstack(cols)                               # (n, p)
    p = D.shape[1]
    group_idx = 1

    # least squares for all genes at once: B = (D^T D)^-1 D^T Y
    XtX = D.T @ D
    XtX_inv = np.linalg.pinv(XtX)
    B = XtX_inv @ (D.T @ Y)                           # (p, g)
    resid = Y - D @ B
    dof = max(n - p, 1)
    sigma2 = (resid ** 2).sum(axis=0) / dof           # (g,)
    var_beta = sigma2 * XtX_inv[group_idx, group_idx]
    se = np.sqrt(np.maximum(var_beta, 1e-12))
    effect = B[group_idx]
    t = effect / se
    pval = 2.0 * stats.t.sf(np.abs(t), df=dof)
    padj = benjamini_hochberg(pval)

    return pd.DataFrame(
        {"effect": effect, "se": se, "t": t, "pval": pval, "padj": padj},
        index=expr.columns,
    )


def concordant(
    deg_a: pd.DataFrame,
    deg_b: pd.DataFrame,
    direction: str,
    padj: float = 0.05,
    top_n: int | None = None,
) -> list[str]:
    """Genes moving the *same* way in two DE tables (A2 concordance).

    The intersection across both skin sites is the robust intrinsic-aging
    signal, with anatomical site held constant.

    direction : 'up'   -> higher in old in both (effect > 0)
                'down' -> lower in old in both (effect < 0)
    """
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    sign = 1.0 if direction == "up" else -1.0
    shared = deg_a.index.intersection(deg_b.index)
    a = deg_a.loc[shared]
    b = deg_b.loc[shared]
    mask = (
        (np.sign(a["effect"]) == sign)
        & (np.sign(b["effect"]) == sign)
        & (a["padj"] < padj)
        & (b["padj"] < padj)
    )
    hits = shared[mask.values]
    # rank by combined evidence (mean absolute effect, larger first)
    strength = (a.loc[hits, "effect"].abs() + b.loc[hits, "effect"].abs()) / 2.0
    ordered = strength.sort_values(ascending=False).index.tolist()
    return ordered[:top_n] if top_n else ordered
