#!/usr/bin/env python3
"""
batch_run_all_cities.py

Production batch runner for the Brazil 26-city Sentinel-2 multi-scene V3 pipeline.

This script orchestrates:
1. search_and_download_multiscene.py
2. build_cloud_reduced_composite_v3.py
3. render_composite_preview_v3.py

Features
--------
- Runs all 26 configured cities or a user-selected subset
- Supports stage selection: download / composite / render
- Continues on failure and records per-city status
- Writes per-city logs
- Writes global JSON + TXT summaries

Expected sibling files
----------------------
In the same folder:
- cities_config_26.py
- search_and_download_multiscene.py
- build_cloud_reduced_composite_v3.py
- render_composite_preview_v3.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# -----------------------------------------------------------------------------
# Robust local import of cities_config_26.py
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from cities_config_26 import CITY_CONFIGS_26
except ImportError as exc:
    raise ImportError(
        f"Could not import CITY_CONFIGS_26 from cities_config_26.py. "
        f"Expected file at: {SCRIPT_DIR / 'cities_config_26.py'}"
    ) from exc


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DOWNLOAD_SCRIPT = SCRIPT_DIR / "search_and_download_multiscene.py"
COMPOSITE_SCRIPT = SCRIPT_DIR / "build_cloud_reduced_composite_v3.py"
RENDER_SCRIPT = SCRIPT_DIR / "render_composite_preview_v3.py"

DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_PROCESSED_ROOT = Path("data/processed")
DEFAULT_LOG_DIR = Path("logs/brazil_26cities_2022_s2_multiscene_v3")

VALID_STAGES = {"download", "composite", "render"}


# -----------------------------------------------------------------------------
# Logging / utils
# -----------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_required_scripts() -> None:
    missing = []
    for path in [DOWNLOAD_SCRIPT, COMPOSITE_SCRIPT, RENDER_SCRIPT]:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing required pipeline script(s):\n" + "\n".join(missing)
        )


# -----------------------------------------------------------------------------
# Command builders
# -----------------------------------------------------------------------------

def build_download_command(
    city_key: str,
    raw_root: Path,
    log_level: str,
) -> List[str]:
    return [
        sys.executable,
        str(DOWNLOAD_SCRIPT),
        "--city",
        city_key,
        "--output-root",
        str(raw_root),
        "--log-level",
        log_level,
    ]


def build_composite_command(
    city_key: str,
    raw_root: Path,
    processed_root: Path,
    residual_threshold: float,
    valid_scl_values: List[int],
    log_level: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(COMPOSITE_SCRIPT),
        "--city",
        city_key,
        "--raw-root",
        str(raw_root),
        "--processed-root",
        str(processed_root),
        "--residual-threshold",
        str(residual_threshold),
        "--log-level",
        log_level,
    ]
    if valid_scl_values:
        cmd.extend(["--valid-scl", *[str(v) for v in valid_scl_values]])
    return cmd


def build_render_command(
    city_key: str,
    processed_root: Path,
    p_low: float,
    p_high: float,
    save_rgb_tif: bool,
    log_level: str,
) -> List[str]:
    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "--city",
        city_key,
        "--processed-root",
        str(processed_root),
        "--p-low",
        str(p_low),
        "--p-high",
        str(p_high),
        "--log-level",
        log_level,
    ]
    if save_rgb_tif:
        cmd.append("--save-rgb-tif")
    return cmd


# -----------------------------------------------------------------------------
# Subprocess runner
# -----------------------------------------------------------------------------

def run_command_with_log(
    cmd: List[str],
    log_path: Path,
    workdir: Path,
) -> Tuple[bool, float, str]:
    """
    Run a subprocess, teeing stdout/stderr to a log file.

    Returns
    -------
    success : bool
    elapsed_seconds : float
    error_message : str
    """
    ensure_dir(log_path.parent)

    start = time.time()
    error_message = ""

    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"[START] {utc_now_iso()}\n")
        log_f.write(f"[CWD] {workdir}\n")
        log_f.write(f"[CMD] {' '.join(shlex.quote(x) for x in cmd)}\n\n")
        log_f.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

        elapsed = time.time() - start

        log_f.write(f"\n[END] {utc_now_iso()}\n")
        log_f.write(f"[RETURN_CODE] {proc.returncode}\n")
        log_f.write(f"[ELAPSED_SECONDS] {elapsed:.2f}\n")

    success = proc.returncode == 0
    if not success:
        error_message = f"Command failed with return code {proc.returncode}"

    return success, elapsed, error_message


# -----------------------------------------------------------------------------
# City selection
# -----------------------------------------------------------------------------

def resolve_city_keys(city_arg: str) -> List[str]:
    if city_arg == "all":
        return list(CITY_CONFIGS_26.keys())

    requested = [c.strip() for c in city_arg.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CITY_CONFIGS_26]
    if unknown:
        raise KeyError(
            f"Unknown city key(s): {unknown}. "
            f"Available: {list(CITY_CONFIGS_26.keys())}"
        )
    return requested


def resolve_stages(stages_arg: str) -> List[str]:
    requested = [s.strip().lower() for s in stages_arg.split(",") if s.strip()]
    invalid = [s for s in requested if s not in VALID_STAGES]
    if invalid:
        raise ValueError(
            f"Invalid stage(s): {invalid}. Valid stages: {sorted(VALID_STAGES)}"
        )
    return requested


# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------

def run_stage_for_city(
    stage: str,
    city_key: str,
    raw_root: Path,
    processed_root: Path,
    residual_threshold: float,
    valid_scl_values: List[int],
    p_low: float,
    p_high: float,
    save_rgb_tif: bool,
    log_level: str,
    log_dir: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    city_slug = CITY_CONFIGS_26[city_key]["slug"]

    if stage == "download":
        cmd = build_download_command(
            city_key=city_key,
            raw_root=raw_root,
            log_level=log_level,
        )
    elif stage == "composite":
        cmd = build_composite_command(
            city_key=city_key,
            raw_root=raw_root,
            processed_root=processed_root,
            residual_threshold=residual_threshold,
            valid_scl_values=valid_scl_values,
            log_level=log_level,
        )
    elif stage == "render":
        cmd = build_render_command(
            city_key=city_key,
            processed_root=processed_root,
            p_low=p_low,
            p_high=p_high,
            save_rgb_tif=save_rgb_tif,
            log_level=log_level,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    stage_log_dir = log_dir / city_slug
    ensure_dir(stage_log_dir)
    log_path = stage_log_dir / f"{stage}.log"

    logging.info("Running %s for %s", stage, city_key)
    success, elapsed, error_message = run_command_with_log(
        cmd=cmd,
        log_path=log_path,
        workdir=repo_root,
    )

    result = {
        "stage": stage,
        "success": success,
        "elapsed_seconds": round(elapsed, 2),
        "log_path": str(log_path),
        "error": error_message,
        "command": cmd,
    }

    if success:
        logging.info("Stage %s succeeded for %s (%.2fs)", stage, city_key, elapsed)
    else:
        logging.error("Stage %s failed for %s", stage, city_key)

    return result


def run_pipeline_for_city(
    city_key: str,
    stages: List[str],
    raw_root: Path,
    processed_root: Path,
    residual_threshold: float,
    valid_scl_values: List[int],
    p_low: float,
    p_high: float,
    save_rgb_tif: bool,
    log_level: str,
    log_dir: Path,
    repo_root: Path,
    stop_on_stage_failure: bool,
) -> Dict[str, Any]:
    city_cfg = CITY_CONFIGS_26[city_key]

    city_result: Dict[str, Any] = {
        "city_key": city_key,
        "city_slug": city_cfg["slug"],
        "city_display_name": city_cfg["display_name"],
        "stages": {},
        "success": True,
    }

    for stage in stages:
        stage_result = run_stage_for_city(
            stage=stage,
            city_key=city_key,
            raw_root=raw_root,
            processed_root=processed_root,
            residual_threshold=residual_threshold,
            valid_scl_values=valid_scl_values,
            p_low=p_low,
            p_high=p_high,
            save_rgb_tif=save_rgb_tif,
            log_level=log_level,
            log_dir=log_dir,
            repo_root=repo_root,
        )

        city_result["stages"][stage] = stage_result

        if not stage_result["success"]:
            city_result["success"] = False
            if stop_on_stage_failure:
                break

    return city_result


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def build_text_summary(run_summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Brazil 26 Cities - Sentinel-2 V3 Batch Run Summary")
    lines.append("=" * 56)
    lines.append("")
    lines.append(f"Run timestamp UTC: {run_summary['run_timestamp_utc']}")
    lines.append(f"Stages: {', '.join(run_summary['stages'])}")
    lines.append(f"Cities requested: {len(run_summary['cities'])}")
    lines.append("")

    success_count = sum(1 for c in run_summary["cities"].values() if c["success"])
    fail_count = len(run_summary["cities"]) - success_count

    lines.append(f"Successful cities: {success_count}")
    lines.append(f"Failed cities: {fail_count}")
    lines.append("")

    for city_key, city_data in run_summary["cities"].items():
        lines.append(
            f"{city_data['city_display_name']} ({city_key}) - "
            f"{'SUCCESS' if city_data['success'] else 'FAILED'}"
        )
        for stage_name, stage_data in city_data["stages"].items():
            status = "OK" if stage_data["success"] else "FAIL"
            lines.append(
                f"  - {stage_name}: {status} | "
                f"{stage_data['elapsed_seconds']:.2f}s | "
                f"log={stage_data['log_path']}"
            )
            if stage_data["error"]:
                lines.append(f"      error: {stage_data['error']}")
        lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for Brazil 26-city Sentinel-2 V3 pipeline."
    )

    parser.add_argument(
        "--cities",
        type=str,
        default="all",
        help="Comma-separated city keys, or 'all'.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="download,composite,render",
        help="Comma-separated stages: download,composite,render",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Base raw data root.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Base processed data root.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for batch logs.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=0.02,
        help="Residual threshold for V3 composite stage.",
    )
    parser.add_argument(
        "--valid-scl",
        nargs="+",
        type=int,
        default=[4, 5, 6],
        help="Valid SCL classes for composite stage.",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=2.0,
        help="Lower percentile for render stage.",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=98.0,
        help="Upper percentile for render stage.",
    )
    parser.add_argument(
        "--save-rgb-tif",
        action="store_true",
        help="Also save stretched RGB GeoTIFFs in render stage.",
    )
    parser.add_argument(
        "--stop-on-stage-failure",
        action="store_true",
        help="Stop remaining stages for a city if one stage fails.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level for the batch runner.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.residual_threshold <= 1.0):
        raise ValueError("--residual-threshold must be between 0 and 1.")

    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100.")

    return args


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    validate_required_scripts()

    repo_root = SCRIPT_DIR.parent.parent.parent.resolve()

    city_keys = resolve_city_keys(args.cities)
    stages = resolve_stages(args.stages)

    ensure_dir(args.log_dir)

    run_summary: Dict[str, Any] = {
        "run_timestamp_utc": utc_now_iso(),
        "repo_root": str(repo_root),
        "script_dir": str(SCRIPT_DIR),
        "stages": stages,
        "cities": {},
        "parameters": {
            "raw_root": str(args.raw_root),
            "processed_root": str(args.processed_root),
            "log_dir": str(args.log_dir),
            "residual_threshold": args.residual_threshold,
            "valid_scl_values": args.valid_scl,
            "p_low": args.p_low,
            "p_high": args.p_high,
            "save_rgb_tif": args.save_rgb_tif,
            "stop_on_stage_failure": args.stop_on_stage_failure,
        },
    }

    logging.info("Starting batch run for %d city/cities", len(city_keys))
    logging.info("Stages: %s", stages)

    for city_key in city_keys:
        city_result = run_pipeline_for_city(
            city_key=city_key,
            stages=stages,
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            residual_threshold=args.residual_threshold,
            valid_scl_values=args.valid_scl,
            p_low=args.p_low,
            p_high=args.p_high,
            save_rgb_tif=args.save_rgb_tif,
            log_level=args.log_level,
            log_dir=args.log_dir,
            repo_root=repo_root,
            stop_on_stage_failure=args.stop_on_stage_failure,
        )
        run_summary["cities"][city_key] = city_result

    summary_json_path = args.log_dir / "batch_run_summary.json"
    summary_txt_path = args.log_dir / "batch_run_summary.txt"

    json_dump(run_summary, summary_json_path)
    write_text(summary_txt_path, build_text_summary(run_summary))

    logging.info("Batch run completed.")
    logging.info("Summary JSON: %s", summary_json_path)
    logging.info("Summary TXT: %s", summary_txt_path)


if __name__ == "__main__":
    main()