"""Real-data adapters (mode: real).

These are intentionally thin stubs that document exactly how to wire each
pinned accession (A1) to the same in-memory shapes the synthetic generator
produces, so the downstream pipeline is identical in both modes. Implementing
them requires external access (a CLUE.io account, GTEx/dbGaP) and the optional
`real` extras (`pip install -e .[real]`: cmapPy, gseapy, scanpy).

Each raises NotImplementedError with the concrete next step rather than failing
silently — A6: "No hard-coded paths or unverified IDs."
"""
from __future__ import annotations

from .synthetic import Cohort, SyntheticData


class AdapterNotImplemented(NotImplementedError):
    pass


def load_gtex_skin(cfg) -> dict[str, Cohort]:
    """GTEx v8 skin → two Cohorts keyed 'suprapubic' / 'lower_leg'.

    Steps:
      1. Pull the GTEx v8 gene TPM matrix + sample attributes (SMTSD) and
         subject phenotypes (AGE bracket, SEX) from the GTEx portal / dbGaP.
      2. Subset to 'Skin - Not Sun Exposed (Suprapubic)' and
         'Skin - Sun Exposed (Lower leg)'.
      3. Build per-site (samples x genes) log-expression + a meta frame with
         group (old vs young from AGE bracket), sex, RIN (SMRIN), ischemic_time
         (SMTSISCH), site.
    """
    raise AdapterNotImplemented(
        "GTEx adapter not implemented. See config.datasets.gtex_skin; "
        "download GTEx v8 skin TPMs + sample attributes, then return "
        "{'suprapubic': Cohort, 'lower_leg': Cohort}."
    )


def load_external_cohorts(cfg) -> list[Cohort]:
    """Additional human skin aging series for the A2.1 meta-cohort.

    Screen GEO for verified skin RNA-seq/array aging series; for each, return a
    Cohort with an 'group' (old vs young) meta column. Verify accessions +
    sample metadata before ingest — do NOT hard-code unverified IDs.
    """
    raise AdapterNotImplemented(
        "External-cohort adapter not implemented. Add verified GEO accessions "
        "and return a list[Cohort]; meta-analysis combines them (m3.stats.meta)."
    )


def load_gene_sets(cfg) -> dict[str, list[str]]:
    """Signature C curated programs: SenMayo, CellAge, MSigDB hallmarks.

    SenMayo: Saul 2022 supplement. CellAge: genomics.senescence.info.
    MSigDB hallmarks (inflammation, ECM/collagen): MSigDB GMT files.
    Return {set_name: [HGNC symbols]}.
    """
    raise AdapterNotImplemented(
        "Gene-set adapter not implemented. Load SenMayo / CellAge / MSigDB GMTs "
        "and return {name: [symbols]}."
    )


def load_l1000(cfg):
    """LINCS L1000 Level 5 signatures (GSE92742 + GSE70138) via CLUE.io.

    Steps:
      1. Authenticate to CLUE.io (free academic account).
      2. Pull Level 5 replicate-collapsed signatures (moderated z) + the
         Touchstone reference; parse .gctx with cmapPy (parse_gctx).
      3. Return (signatures: ndarray [n_sig x n_landmark],
                 meta: DataFrame[compound, cell_line, dose],
                 landmark_genes: list[str]).
    """
    raise AdapterNotImplemented(
        "L1000 adapter not implemented. Use cmapPy to parse CLUE.io Level 5 "
        ".gctx for GSE92742 + GSE70138; return (signatures, meta, landmark_genes)."
    )


def load_real(cfg) -> SyntheticData:
    """Assemble a SyntheticData-shaped bundle from real sources (mode: real)."""
    gtex = load_gtex_skin(cfg)
    externals = load_external_cohorts(cfg)
    gene_sets = load_gene_sets(cfg)
    sigs, meta, landmark = load_l1000(cfg)
    raise AdapterNotImplemented(
        "Wire load_real() once the individual adapters above are implemented."
    )
