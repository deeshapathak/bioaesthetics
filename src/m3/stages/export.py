"""Stage 6 — export the deliverables (A5).

Two artifacts (A6):
  1. the ranked table — compound | reversal_score (tau) | q-value | top
     aesthetic axis | known/novel | cell-line provenance;
  2. a one-page validation report — recovery ROC-AUC, top-decile enrichment,
     directional-control checks, signature gene-coverage, and the PASS/FAIL
     decision against the §A4 go/no-go thresholds.

Every output carries the honest claim: mechanistic hypotheses from public data,
not proof of human efficacy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts import Store, get_logger
from . import ingest

log = get_logger("m3.export")

SCOPE_GUARDRAIL = (
    "Mechanistic hypotheses from public data — NOT proof of human efficacy. "
    "L1000 cell lines are mostly not primary skin; signatures are cross-sectional "
    "(association, not causation)."
)


def run(cfg, force: bool = False) -> dict:
    ing = Store(cfg.cache_dir, "ingest")
    conn = Store(cfg.cache_dir, "connectivity")
    val = Store(cfg.cache_dir, "validate")
    sigs = Store(cfg.cache_dir, "signatures")
    out_dir = cfg.outputs_dir

    scores = conn.read_csv("compound_scores.csv", index_col=0)
    report = val.read_json("report.json")
    truth = ingest.read_truth(ing)
    known = set(truth["known_actives"])
    qmap = report["compound_qvalues"]

    # ---- top aesthetic axis per compound (Signature C annotation) -----------
    axis = _top_axis(cfg, ing, sigs)

    # ---- assemble the ranked table -----------------------------------------
    tbl = pd.DataFrame(index=scores.index)
    tbl["reversal_score_tau"] = scores["tau"].round(1)
    tbl["connectivity"] = scores["cs"].round(4)
    tbl["q_value"] = [round(float(qmap.get(c, 1.0)), 4) for c in scores.index]
    tbl["top_aesthetic_axis"] = [axis.get(c, "general") for c in scores.index]
    tbl["known_or_novel"] = ["known" if c in known else "novel" for c in scores.index]
    tbl["cell_line_provenance"] = [
        f"{row.top_cell_line} ({row.n_cell_lines} lines"
        + ("; skin-relevant)" if row.top_cell_line_skin_relevant else ")")
        for row in scores.itertuples()
    ]
    tbl = tbl.sort_values("connectivity")            # most-negative = top candidate
    tbl.index.name = "compound"

    table_path = out_dir / "ranked_table.csv"
    tbl.to_csv(table_path)

    # ---- one-page validation report ----------------------------------------
    report_path = out_dir / "validation_report.md"
    report_path.write_text(_render_report(cfg, report, tbl, truth), encoding="utf-8")

    decision = report["decision"]
    log.info("wrote %s (%d compounds)", table_path, len(tbl))
    log.info("wrote %s", report_path)
    log.info("DECISION: %s — %s", decision,
             "ship to demo" if decision == "PASS" else "iterate signatures, not messaging")
    return {"decision": decision, "ranked_table": str(table_path),
            "validation_report": str(report_path)}


# --------------------------------------------------------------------- helpers
def _top_axis(cfg, ing, sigs) -> dict[str, str]:
    """Assign each compound the curated program it most strongly reverses."""
    Z, meta, landmark = ingest.read_l1000(ing)
    lm_index = {g: i for i, g in enumerate(landmark)}
    query = Store(cfg.cache_dir, "harmonize").read_json("query.json")
    aging_sign = {g: 1.0 for g in query["up"]}
    aging_sign.update({g: -1.0 for g in query["down"]})

    programs = sigs.read_json("signatureC.json")["programs"]
    # program -> list of (landmark_index, aging_sign) for genes in the aging signature
    prog_idx: dict[str, list[tuple[int, float]]] = {}
    for name, genes in programs.items():
        items = [(lm_index[g], aging_sign[g]) for g in genes
                 if g in lm_index and g in aging_sign]
        if items:
            prog_idx[name] = items

    skin = set(cfg.get("connectivity.skin_relevant_cell_lines", []))
    out: dict[str, str] = {}
    for compound, grp in meta.groupby("compound"):
        w = np.where(grp["cell_line"].isin(skin).to_numpy(), 1.0, 0.5)
        rows = grp.index.to_numpy()
        sig = np.average(Z[rows], axis=0, weights=w)   # mean compound signature
        best, best_score = "general", 0.0
        for name, items in prog_idx.items():
            idx = np.array([i for i, _ in items])
            sgn = np.array([s for _, s in items])
            # reversal of this program = aging_sign * (-z), averaged
            rscore = float(np.mean(sgn * (-sig[idx])))
            if rscore > best_score:
                best, best_score = name, rscore
        out[compound] = best
    return out


def _render_report(cfg, report, tbl, truth) -> str:
    g = report["gates"]
    rec = g["held_out_recovery"]
    enr = rec["top_decile_enrichment"]
    cov = g["gene_space_coverage"]
    dirc = g["directional_controls"]
    rule = report["go_no_go_rule"]
    decision = report["decision"]

    def yn(b):
        return "✅ PASS" if b else "❌ FAIL"

    top = tbl.head(15).reset_index()
    lines = []
    L = lines.append
    L("# M3 Rejuvenation Screen — Validation Report")
    L("")
    L(f"**Decision: {decision}** — "
      + ("clears the §A4 go/no-go bar; safe to ship to a demo."
         if decision == "PASS"
         else "below the §A4 bar; iterate the signatures, not the messaging."))
    L("")
    L(f"_Mode: {cfg.mode} · seed: {cfg.seed} · {len(tbl)} compounds scored._")
    L("")
    L("## Go / no-go gates")
    L("")
    L("| Gate | Result | Pass condition | Status |")
    L("|------|--------|----------------|--------|")
    L(f"| Held-out recovery (PRIMARY) | ROC-AUC **{rec['roc_auc']:.3f}**, "
      f"precision@k {rec['precision_at_k_known']:.2f} | AUC ≥ {rule['recovery_auc_min']:.2f} "
      f"| {yn(rec['pass'])} |")
    L(f"| Top-decile enrichment | {enr['fold_enrichment']:.1f}× "
      f"(q={g['null_fdr']['top_decile_enrichment_q']:.2g}) | q < {rule['top_decile_q_max']:.2f} "
      f"| {yn(g['null_fdr']['pass'])} |")
    L(f"| Directional controls | promoter mean-rank {dirc['mean_rank_fraction']:.2f}, "
      f"{dirc['promoters_in_top_decile']} in top decile | promoters at bottom, none in top "
      f"| {yn(dirc['pass'])} |")
    L(f"| Face validity (sanity) | {g['face_validity']['known_in_top_decile']}/"
      f"{g['face_validity']['n_known']} known actives in top decile | necessary, not sufficient "
      f"| {yn(g['face_validity']['pass'])} |")
    L(f"| Signature concordance | A↔C Jaccard {g['signature_concordance']['A_vs_C_jaccard']:.2f}, "
      f"A↔B {g['signature_concordance']['A_vs_B_jaccard']:.2f} | A and C agree; B context only "
      f"| {yn(g['signature_concordance']['pass'])} |")
    L(f"| Leakage control | dominant cell-line share "
      f"{g['leakage_control']['dominant_cell_line_share']:.2f} | no single line drives top hits "
      f"| {yn(g['leakage_control']['pass'])} |")
    L(f"| Gene-space coverage (A3.1) | {cov['signatureA_genes_scored']}/"
      f"{cov['signatureA_genes_in']} Sig-A genes in L1000 space ({cov['coverage']*100:.0f}%) "
      f"| ≥ {cov['min_coverage']*100:.0f}% | {yn(cov['pass'])} |")
    L("")
    L("## The recovery test, concretely")
    L("")
    L(f"A positive set of {rec['top_decile_enrichment']['n_positives']} known "
      "dermatology/longevity interventions (retinoids, senolytics, rapamycin/metformin) "
      "was withheld from all tuning. Running the full pipeline blind, the hidden positives "
      f"recover at ROC-AUC **{rec['roc_auc']:.3f}** and enrich "
      f"**{enr['fold_enrichment']:.1f}×** in the top decile "
      f"(observed {enr['observed']} vs {enr['expected']:.1f} expected by chance).")
    L("")
    L("## Top 15 candidates")
    L("")
    L("| # | compound | tau | q | top axis | known/novel | provenance |")
    L("|---|----------|-----|---|----------|-------------|------------|")
    for i, r in top.iterrows():
        L(f"| {i+1} | {r['compound']} | {r['reversal_score_tau']:.0f} | "
          f"{r['q_value']:.3f} | {r['top_aesthetic_axis']} | {r['known_or_novel']} | "
          f"{r['cell_line_provenance']} |")
    L("")
    L("## Scope guardrail")
    L("")
    L(f"> {SCOPE_GUARDRAIL}")
    L("")
    L("Known actives recovering blind is the proof the engine is real; the novel "
      "high-rankers are the first hypotheses.")
    L("")
    return "\n".join(lines)
