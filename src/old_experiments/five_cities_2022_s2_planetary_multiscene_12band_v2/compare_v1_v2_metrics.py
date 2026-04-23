#!/usr/bin/env python3
"""
compare_v1_v2_metrics.py

Compare V1 and V2 composite diagnostics for the five-city Sentinel-2 experiment.

What this script does
---------------------
For each city, it reads:

V1:
    data/processed/<city_slug>/2022/s2_planetary_multiscene_12band/
        selected_scenes_diagnostics.json

V2:
    data/processed/<city_slug>/2022/s2_planetary_multiscene_12band_v2/
        selected_scenes_diagnostics.json

It then extracts comparable metrics, creates a per-city comparison table,
and writes:

1. CSV summary
2. JSON summary
3. Human-readable TXT report

Comparison logic
----------------
Primary objective:
- lower final_unresolved_ratio is better

Secondary objectives:
- >= 2 contributing scenes in V2 is desirable
- no increase in unresolved ratio
- cleaner hard-city behavior

This script does not inspect imagery directly; it compares diagnostics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# Robust local import of cities_config.py
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from cities_config import CITY_CONFIGS
except ImportError as exc:
    raise ImportError(
        f"Could not import CITY_CONFIGS from cities_config.py. "
        f"Expected file at: {SCRIPT_DIR / 'cities_config.py'}"
    ) from exc


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class CityVersionMetrics:
    city_key: str
    city_slug: str
    version: str
    num_input_scenes: Optional[int]
    num_scenes_used_for_fill: Optional[int]
    aoi_pixels: Optional[int]
    final_filled_pixels: Optional[int]
    final_unresolved_pixels: Optional[int]
    final_unresolved_ratio: Optional[float]
    valid_scl_values: Optional[List[int]]
    residual_threshold: Optional[float]
    min_scenes_required: Optional[int]
    diagnostics_path: str


@dataclass
class CityComparison:
    city_key: str
    city_slug: str

    v1_num_input_scenes: Optional[int]
    v2_num_input_scenes: Optional[int]

    v1_num_scenes_used_for_fill: Optional[int]
    v2_num_scenes_used_for_fill: Optional[int]

    v1_final_unresolved_pixels: Optional[int]
    v2_final_unresolved_pixels: Optional[int]

    v1_final_unresolved_ratio: Optional[float]
    v2_final_unresolved_ratio: Optional[float]

    unresolved_ratio_delta_v2_minus_v1: Optional[float]
    unresolved_pixels_delta_v2_minus_v1: Optional[int]

    v1_valid_scl_values: Optional[List[int]]
    v2_valid_scl_values: Optional[List[int]]

    v1_min_scenes_required: Optional[int]
    v2_min_scenes_required: Optional[int]

    preferred_version: str
    decision_reason: str


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Diagnostics parsing
# -----------------------------------------------------------------------------

def get_diagnostics_path(
    processed_root: Path,
    city_slug: str,
    version_folder: str,
) -> Path:
    return (
        processed_root
        / city_slug
        / "2022"
        / version_folder
        / "selected_scenes_diagnostics.json"
    )


def safe_get(d: Dict[str, Any], *keys, default=None):
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_metrics(
    city_key: str,
    city_slug: str,
    version: str,
    diagnostics_path: Path,
) -> Optional[CityVersionMetrics]:
    if not diagnostics_path.exists():
        logging.warning("Missing diagnostics for %s %s: %s", city_key, version, diagnostics_path)
        return None

    data = load_json(diagnostics_path)

    return CityVersionMetrics(
        city_key=city_key,
        city_slug=city_slug,
        version=version,
        num_input_scenes=safe_get(data, "summary", "num_input_scenes"),
        num_scenes_used_for_fill=safe_get(data, "summary", "num_scenes_used_for_fill"),
        aoi_pixels=safe_get(data, "summary", "aoi_pixels"),
        final_filled_pixels=safe_get(data, "summary", "final_filled_pixels"),
        final_unresolved_pixels=safe_get(data, "summary", "final_unresolved_pixels"),
        final_unresolved_ratio=safe_get(data, "summary", "final_unresolved_ratio"),
        valid_scl_values=safe_get(data, "parameters", "valid_scl_values"),
        residual_threshold=safe_get(data, "parameters", "residual_threshold"),
        min_scenes_required=safe_get(data, "parameters", "min_scenes_required"),
        diagnostics_path=str(diagnostics_path),
    )


# -----------------------------------------------------------------------------
# Comparison logic
# -----------------------------------------------------------------------------

def compare_versions(
    city_key: str,
    city_slug: str,
    v1: Optional[CityVersionMetrics],
    v2: Optional[CityVersionMetrics],
) -> CityComparison:
    if v1 is None and v2 is None:
        return CityComparison(
            city_key=city_key,
            city_slug=city_slug,
            v1_num_input_scenes=None,
            v2_num_input_scenes=None,
            v1_num_scenes_used_for_fill=None,
            v2_num_scenes_used_for_fill=None,
            v1_final_unresolved_pixels=None,
            v2_final_unresolved_pixels=None,
            v1_final_unresolved_ratio=None,
            v2_final_unresolved_ratio=None,
            unresolved_ratio_delta_v2_minus_v1=None,
            unresolved_pixels_delta_v2_minus_v1=None,
            v1_valid_scl_values=None,
            v2_valid_scl_values=None,
            v1_min_scenes_required=None,
            v2_min_scenes_required=None,
            preferred_version="unknown",
            decision_reason="Neither V1 nor V2 diagnostics were found.",
        )

    if v1 is None:
        return CityComparison(
            city_key=city_key,
            city_slug=city_slug,
            v1_num_input_scenes=None,
            v2_num_input_scenes=v2.num_input_scenes,
            v1_num_scenes_used_for_fill=None,
            v2_num_scenes_used_for_fill=v2.num_scenes_used_for_fill,
            v1_final_unresolved_pixels=None,
            v2_final_unresolved_pixels=v2.final_unresolved_pixels,
            v1_final_unresolved_ratio=None,
            v2_final_unresolved_ratio=v2.final_unresolved_ratio,
            unresolved_ratio_delta_v2_minus_v1=None,
            unresolved_pixels_delta_v2_minus_v1=None,
            v1_valid_scl_values=None,
            v2_valid_scl_values=v2.valid_scl_values,
            v1_min_scenes_required=None,
            v2_min_scenes_required=v2.min_scenes_required,
            preferred_version="v2",
            decision_reason="Only V2 diagnostics were found.",
        )

    if v2 is None:
        return CityComparison(
            city_key=city_key,
            city_slug=city_slug,
            v1_num_input_scenes=v1.num_input_scenes,
            v2_num_input_scenes=None,
            v1_num_scenes_used_for_fill=v1.num_scenes_used_for_fill,
            v2_num_scenes_used_for_fill=None,
            v1_final_unresolved_pixels=v1.final_unresolved_pixels,
            v2_final_unresolved_pixels=None,
            v1_final_unresolved_ratio=v1.final_unresolved_ratio,
            v2_final_unresolved_ratio=None,
            unresolved_ratio_delta_v2_minus_v1=None,
            unresolved_pixels_delta_v2_minus_v1=None,
            v1_valid_scl_values=v1.valid_scl_values,
            v2_valid_scl_values=None,
            v1_min_scenes_required=v1.min_scenes_required,
            v2_min_scenes_required=None,
            preferred_version="v1",
            decision_reason="Only V1 diagnostics were found.",
        )

    ratio_delta = None
    if v1.final_unresolved_ratio is not None and v2.final_unresolved_ratio is not None:
        ratio_delta = v2.final_unresolved_ratio - v1.final_unresolved_ratio

    pixels_delta = None
    if v1.final_unresolved_pixels is not None and v2.final_unresolved_pixels is not None:
        pixels_delta = v2.final_unresolved_pixels - v1.final_unresolved_pixels

    preferred_version = "tie"
    decision_reason = "V1 and V2 are effectively equivalent by diagnostics."

    # Primary decision: unresolved ratio
    if (
        v1.final_unresolved_ratio is not None
        and v2.final_unresolved_ratio is not None
    ):
        eps = 1e-9

        if v2.final_unresolved_ratio + eps < v1.final_unresolved_ratio:
            preferred_version = "v2"
            decision_reason = "V2 has a lower final unresolved ratio."
        elif v1.final_unresolved_ratio + eps < v2.final_unresolved_ratio:
            preferred_version = "v1"
            decision_reason = "V1 has a lower final unresolved ratio."
        else:
            # Tie-breaker: if unresolved ratios are equal, prefer v2 when it uses >=2 scenes
            v2_multi = (v2.num_scenes_used_for_fill or 0) >= 2
            v1_multi = (v1.num_scenes_used_for_fill or 0) >= 2

            if v2_multi and not v1_multi:
                preferred_version = "v2"
                decision_reason = (
                    "Unresolved ratio is effectively tied, but V2 achieves real multi-scene filling."
                )
            elif v1_multi and not v2_multi:
                preferred_version = "v1"
                decision_reason = (
                    "Unresolved ratio is effectively tied, but only V1 achieves real multi-scene filling."
                )
            else:
                preferred_version = "tie"
                decision_reason = "Unresolved ratio and scene usage are effectively tied."

    return CityComparison(
        city_key=city_key,
        city_slug=city_slug,
        v1_num_input_scenes=v1.num_input_scenes,
        v2_num_input_scenes=v2.num_input_scenes,
        v1_num_scenes_used_for_fill=v1.num_scenes_used_for_fill,
        v2_num_scenes_used_for_fill=v2.num_scenes_used_for_fill,
        v1_final_unresolved_pixels=v1.final_unresolved_pixels,
        v2_final_unresolved_pixels=v2.final_unresolved_pixels,
        v1_final_unresolved_ratio=v1.final_unresolved_ratio,
        v2_final_unresolved_ratio=v2.final_unresolved_ratio,
        unresolved_ratio_delta_v2_minus_v1=ratio_delta,
        unresolved_pixels_delta_v2_minus_v1=pixels_delta,
        v1_valid_scl_values=v1.valid_scl_values,
        v2_valid_scl_values=v2.valid_scl_values,
        v1_min_scenes_required=v1.min_scenes_required,
        v2_min_scenes_required=v2.min_scenes_required,
        preferred_version=preferred_version,
        decision_reason=decision_reason,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def write_csv(rows: List[CityComparison], output_path: Path) -> None:
    ensure_dir(output_path.parent)

    fieldnames = [
        "city_key",
        "city_slug",
        "v1_num_input_scenes",
        "v2_num_input_scenes",
        "v1_num_scenes_used_for_fill",
        "v2_num_scenes_used_for_fill",
        "v1_final_unresolved_pixels",
        "v2_final_unresolved_pixels",
        "v1_final_unresolved_ratio",
        "v2_final_unresolved_ratio",
        "unresolved_ratio_delta_v2_minus_v1",
        "unresolved_pixels_delta_v2_minus_v1",
        "v1_valid_scl_values",
        "v2_valid_scl_values",
        "v1_min_scenes_required",
        "v2_min_scenes_required",
        "preferred_version",
        "decision_reason",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            writer.writerow(payload)


def build_human_report(rows: List[CityComparison]) -> str:
    lines: List[str] = []
    lines.append("V1 vs V2 Sentinel-2 Composite Comparison")
    lines.append("=" * 48)
    lines.append("")

    v1_better = sum(1 for r in rows if r.preferred_version == "v1")
    v2_better = sum(1 for r in rows if r.preferred_version == "v2")
    ties = sum(1 for r in rows if r.preferred_version == "tie")

    lines.append(f"Cities where V2 is preferred: {v2_better}")
    lines.append(f"Cities where V1 is preferred: {v1_better}")
    lines.append(f"Ties: {ties}")
    lines.append("")

    for row in rows:
        lines.append(f"City: {row.city_slug} ({row.city_key})")
        lines.append("-" * 48)
        lines.append(
            f"V1 unresolved ratio: {format_ratio(row.v1_final_unresolved_ratio)} | "
            f"V2 unresolved ratio: {format_ratio(row.v2_final_unresolved_ratio)}"
        )
        lines.append(
            f"V1 scenes used: {format_int(row.v1_num_scenes_used_for_fill)} | "
            f"V2 scenes used: {format_int(row.v2_num_scenes_used_for_fill)}"
        )
        lines.append(
            f"V1 unresolved pixels: {format_int(row.v1_final_unresolved_pixels)} | "
            f"V2 unresolved pixels: {format_int(row.v2_final_unresolved_pixels)}"
        )
        lines.append(
            f"Delta unresolved ratio (V2 - V1): {format_ratio(row.unresolved_ratio_delta_v2_minus_v1)}"
        )
        lines.append(
            f"Preferred version: {row.preferred_version}"
        )
        lines.append(
            f"Reason: {row.decision_reason}"
        )
        lines.append("")

    return "\n".join(lines)


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.6f}"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(value)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V1 and V2 Sentinel-2 composite diagnostics."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Base processed data root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/comparison_v1_v2"),
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    rows: List[CityComparison] = []
    detailed_json: Dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(args.processed_root),
        "cities": {},
    }

    for city_key, city_cfg in CITY_CONFIGS.items():
        city_slug = city_cfg["slug"]

        v1_path = get_diagnostics_path(
            processed_root=args.processed_root,
            city_slug=city_slug,
            version_folder="s2_planetary_multiscene_12band",
        )
        v2_path = get_diagnostics_path(
            processed_root=args.processed_root,
            city_slug=city_slug,
            version_folder="s2_planetary_multiscene_12band_v2",
        )

        logging.info("Comparing city: %s", city_key)

        v1_metrics = parse_metrics(city_key, city_slug, "v1", v1_path)
        v2_metrics = parse_metrics(city_key, city_slug, "v2", v2_path)

        comparison = compare_versions(city_key, city_slug, v1_metrics, v2_metrics)
        rows.append(comparison)

        detailed_json["cities"][city_key] = {
            "city_slug": city_slug,
            "v1": asdict(v1_metrics) if v1_metrics else None,
            "v2": asdict(v2_metrics) if v2_metrics else None,
            "comparison": asdict(comparison),
        }

    ensure_dir(args.output_dir)

    csv_path = args.output_dir / "v1_v2_comparison_summary.csv"
    json_path = args.output_dir / "v1_v2_comparison_summary.json"
    txt_path = args.output_dir / "v1_v2_comparison_summary.txt"

    write_csv(rows, csv_path)
    json_dump(detailed_json, json_path)

    report_text = build_human_report(rows)
    txt_path.write_text(report_text, encoding="utf-8")

    logging.info("Comparison CSV written to: %s", csv_path)
    logging.info("Comparison JSON written to: %s", json_path)
    logging.info("Comparison TXT written to: %s", txt_path)

    print("\n" + report_text)


if __name__ == "__main__":
    main()