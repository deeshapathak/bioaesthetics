"""End-to-end pipeline test: the synthetic run must reach a PASS decision and
produce the two deliverables, and stages must cache."""
import json

import pytest

from m3.config import load_config
from m3 import pipeline


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    base = load_config("config/m3.yaml")
    # redirect cache/outputs into a temp dir so the test is hermetic
    tmp = tmp_path_factory.mktemp("m3run")
    base.data["paths"]["cache"] = str(tmp / "cache")
    base.data["paths"]["outputs"] = str(tmp / "outputs")
    base.cache_dir.mkdir(parents=True, exist_ok=True)
    base.outputs_dir.mkdir(parents=True, exist_ok=True)
    return base


def test_end_to_end_pass(cfg):
    result = pipeline.run_all(cfg)
    assert result["decision"] == "PASS"

    val_report = json.loads((cfg.cache_dir / "validate" / "report.json").read_text())
    rec = val_report["gates"]["held_out_recovery"]
    assert rec["roc_auc"] >= cfg.get("validation.go_no_go.recovery_auc_min")
    assert val_report["gates"]["directional_controls"]["pass"] is True
    assert val_report["gates"]["null_fdr"]["pass"] is True

    # deliverables exist
    table = cfg.outputs_dir / "ranked_table.csv"
    assert table.exists()
    header = table.read_text().splitlines()[0]
    for col in ["reversal_score_tau", "q_value", "top_aesthetic_axis",
                "known_or_novel", "cell_line_provenance"]:
        assert col in header


def test_known_actives_rank_above_promoters(cfg):
    # pipeline already ran in the previous test (module-scoped cfg/cache)
    from m3.artifacts import Store
    conn = Store(cfg.cache_dir, "connectivity")
    truth = json.loads((cfg.cache_dir / "ingest" / "truth.json").read_text())
    import pandas as pd
    scores = pd.read_csv(conn.file("compound_scores.csv"), index_col=0)
    known_cs = scores.loc[[c for c in truth["known_actives"] if c in scores.index], "cs"].mean()
    prom_cs = scores.loc[[c for c in truth["aging_promoters"] if c in scores.index], "cs"].mean()
    assert known_cs < 0 < prom_cs          # reversers negative, promoters positive
    assert known_cs < prom_cs


def test_stage_caching(cfg):
    store = pipeline.ingest.run(cfg)            # already cached from run_all
    manifest = json.loads((store.dir / "manifest.json").read_text())
    mtime_before = (store.dir / "manifest.json").stat().st_mtime
    pipeline.ingest.run(cfg)                     # should be a no-op (cached)
    mtime_after = (store.dir / "manifest.json").stat().st_mtime
    assert mtime_before == mtime_after
