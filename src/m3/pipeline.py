"""Stage orchestration (A6: ingest → signatures → harmonize → connectivity →
validation → ranked-table export). Each stage is independently rerunnable and
cached; ``force`` recomputes from the named stage onward.
"""
from __future__ import annotations

from .artifacts import get_logger
from .stages import connectivity, export, harmonize, ingest, signatures, validate

log = get_logger("m3.pipeline")

# ordered stages; export is terminal (writes deliverables, returns decision)
STAGES = [
    ("ingest", ingest.run),
    ("signatures", signatures.run),
    ("harmonize", harmonize.run),
    ("connectivity", connectivity.run),
    ("validate", validate.run),
    ("export", export.run),
]
STAGE_NAMES = [name for name, _ in STAGES]


def run_stage(cfg, name: str, force: bool = False):
    fn = dict(STAGES)[name]
    log.info("── stage: %s ──", name)
    return fn(cfg, force=force)


def run_all(cfg, force: bool = False, force_from: str | None = None):
    """Run the whole pipeline. ``force_from`` recomputes from that stage on."""
    result = None
    forcing = force
    for name, fn in STAGES:
        if force_from and name == force_from:
            forcing = True
        log.info("── stage: %s ──", name)
        result = fn(cfg, force=forcing)
    return result
