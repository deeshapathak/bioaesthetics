# M3 Rejuvenation Screen

> Build an old→young skin signature, then score which public compound
> perturbations *reverse* it. Negative connectivity = candidate rejuvenator.

A build of **Genesis Appendix A** — the M3 module of the Genesis v1 baseline
engine. One engineer, a few weeks, public data → a ranked, mechanistically
annotated skin-rejuvenation shortlist that validates against known actives.

The pipeline runs **end-to-end out of the box** on a synthetic dataset with a
planted aging signal and known rejuvenators, so every stage executes and the
§A4 validation gates exercise real signal. Real-data adapters (CLUE.io / GTEx)
are stubbed and clearly marked for plugging in the pinned accessions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # add .[dev] for pytest, .[real] for real-data adapters

m3 run                      # ingest → … → export; writes outputs/, exits non-zero on FAIL
cat outputs/validation_report.md
open outputs/ranked_table.csv
```

A green run prints `M3 decision: PASS` and produces:

* `outputs/ranked_table.csv` — `compound | reversal_score (tau) | q-value | top
  aesthetic axis | known/novel | cell-line provenance`
* `outputs/validation_report.md` — recovery ROC-AUC, top-decile enrichment,
  directional-control checks, signature gene-coverage, and the PASS/FAIL call.

## The method (one line)

Build an old→young skin signature, then score which L1000 compound
perturbations reverse it (CMap weighted-KS connectivity); the most-negative
scores are the candidate rejuvenators.

## Pipeline stages (A6)

`ingest → signatures (A/B/C) → harmonize → connectivity → validate → export`

| Stage | Appendix | What it does |
|-------|----------|--------------|
| `ingest` | A1 | Load datasets (synthetic generator or real adapters); cache to disk. |
| `signatures` | A2, A2.1 | **Signature A** (intrinsic age, site-controlled, concordant across GTEx sites + cross-cohort meta-analysis), **B** (site/photoexposure, contextual), **C** (curated programs: SenMayo, ECM/collagen, …). |
| `harmonize` | A3.1 | Map to a common identifier, restrict the query to L1000 landmark space, report coverage. |
| `connectivity` | A3 | CMap weighted-KS reversal score per compound × cell line × dose; aggregate to one tau per compound; record cell-line provenance. |
| `validate` | A4 | Blind held-out recovery (PRIMARY), face validity, null/FDR, directional controls, signature concordance, leakage — then PASS/FAIL vs go/no-go. |
| `export` | A5 | The ranked table + the one-page validation report. |

Each stage is independently rerunnable and cached under `.cache/m3/`.

```bash
m3 signatures --force        # recompute one stage
m3 run --from connectivity   # recompute from a stage onward
m3 clean                     # wipe the stage cache
```

## Configuration

Everything is driven by the single pinned config `config/m3.yaml` (A6): the
random seed, dataset accessions + versions (GSE92742, GSE70138, GTEx v8, …),
and the §A4 go/no-go thresholds. No hard-coded paths or unverified IDs live in
the code.

```yaml
mode: synthetic            # synthetic | real
validation:
  go_no_go:
    recovery_auc_min: 0.70   # blind held-out recovery ROC-AUC (PRIMARY gate)
    top_decile_q_max: 0.10   # known actives enrich in top decile at q < this
    require_directional_controls: true
```

## Go / no-go (A4)

Ship to a demo only if **all** hold:

1. **Held-out recovery (PRIMARY)** — withhold a labeled positive set (retinoids,
   senolytics, rapamycin/metformin), run blind, recover it: ROC-AUC ≥ ~0.70.
2. **Top-decile enrichment** — known actives enrich in the top decile at q < 0.1.
3. **Directional controls** — aging-promoters (genotoxic / senescence-inducing)
   rank at the **bottom**. If one scores as "rejuvenating", the sign convention
   or gene-space mapping is broken — fixed before reading any hit.

Below that, the signatures or gene-space need work before any novel hit is
trustworthy. `m3 run` exits non-zero on FAIL so CI can gate on it.

## Going to real data

Set `mode: real` and implement the adapters in
[`src/m3/data/adapters.py`](src/m3/data/adapters.py) (each documents the exact
steps and raises with the next action until wired):

* **L1000** — CLUE.io Level 5 (`GSE92742` + `GSE70138`), parsed with `cmapPy`.
* **GTEx skin v8** — two tissues (Suprapubic, Lower leg) with age brackets.
* **External cohorts** — verified GEO skin-aging series for the A2.1 meta.
* **Gene sets** — SenMayo (Saul 2022), CellAge, MSigDB hallmark GMTs.

`pip install -e .[real]` pulls the optional `cmapPy` / `gseapy` / `scanpy`.

## Layout

```
config/m3.yaml             # the single pinned config
src/m3/
  config.py  artifacts.py  pipeline.py  cli.py
  data/      synthetic.py (planted signal)  adapters.py (real-data stubs)
  stages/    ingest  signatures  harmonize  connectivity  validate  export
  stats/     de  meta  connectivity_score  metrics
tests/                     # stats unit tests + end-to-end PASS test
```

```bash
pytest -q                  # 11 tests: math + end-to-end PASS
```

## Honest limitations (A5)

* **Cell-line mismatch** — L1000 lines are mostly not primary skin; scores are
  mechanistic hypotheses; provenance is annotated and skin-relevant lines are
  up-weighted.
* **Cross-sectional signatures** — old-vs-young is a snapshot; connectivity
  shows association, not causation.
* **In vitro ≠ in vivo ≠ aesthetic outcome** — a reversed transcriptomic
  signature is a lead, not a proven treatment.

Every output carries that scope guardrail. Known actives recovering blind is
the proof the engine is real; the novel high-rankers are the first hypotheses.
