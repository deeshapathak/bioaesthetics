"""M3 command-line interface.

    m3 run                 # full pipeline: ingest → … → export
    m3 run --force         # recompute everything
    m3 run --from signatures   # recompute from a stage onward
    m3 ingest | signatures | harmonize | connectivity | validate | export
    m3 clean               # wipe the stage cache

All commands take --config (default config/m3.yaml).
"""
from __future__ import annotations

import argparse
import shutil
import sys

from .artifacts import get_logger
from .config import load_config
from .pipeline import STAGE_NAMES, run_all, run_stage

log = get_logger("m3")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="m3", description="M3 Rejuvenation Screen pipeline")
    parser.add_argument("--config", default=None, help="path to config (default config/m3.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline")
    p_run.add_argument("--force", action="store_true", help="recompute all stages")
    p_run.add_argument("--from", dest="force_from", choices=STAGE_NAMES,
                       help="recompute from this stage onward")

    for name in STAGE_NAMES:
        ps = sub.add_parser(name, help=f"run only the '{name}' stage")
        ps.add_argument("--force", action="store_true")

    sub.add_parser("clean", help="wipe the stage cache")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "run":
        result = run_all(cfg, force=args.force, force_from=args.force_from)
        return _finish(result)

    if args.command == "clean":
        if cfg.cache_dir.exists():
            shutil.rmtree(cfg.cache_dir)
            log.info("cleaned cache: %s", cfg.cache_dir)
        return 0

    if args.command in STAGE_NAMES:
        result = run_stage(cfg, args.command, force=getattr(args, "force", False))
        return _finish(result if args.command == "export" else None)

    parser.error(f"unknown command {args.command!r}")
    return 2


def _finish(result) -> int:
    """Return a non-zero exit code on a FAIL decision so CI can gate on it."""
    if isinstance(result, dict) and "decision" in result:
        decision = result["decision"]
        print(f"\nM3 decision: {decision}", file=sys.stderr)
        return 0 if decision == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
