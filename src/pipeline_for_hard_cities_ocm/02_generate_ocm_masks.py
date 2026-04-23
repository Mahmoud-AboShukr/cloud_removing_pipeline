# src/pipeline_for_hard_cities_ocm/02_generate_ocm_masks.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omnicloudmask import predict_from_array

from config_ocm_v1 import (
    DEFAULT_MASK_NODATA,
    LOGS_DIR,
    OCM_MASKS_DIR,
    REPORT_DIR,
    TARGET_CITIES,
    get_city_inventory_csv,
    get_city_log_dir,
    get_city_ocm_mask_dir,
    get_city_report_dir,
)
from utils_ocm import (
    ensure_dir,
    read_single_band,
    save_json,
    slugify_city_name,
    write_single_band,
)


def load_inventory(city_slug: str) -> pd.DataFrame:
    csv_path = get_city_inventory_csv(city_slug)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Inventory CSV not found for city '{city_slug}': {csv_path}\n"
            "Run 01_inventory_raw_backup.py first."
        )
    return pd.read_csv(csv_path)


def get_scene_dirs_from_inventory(df: pd.DataFrame) -> list[Path]:
    if "scene_dir" not in df.columns:
        raise KeyError("Inventory CSV does not contain a 'scene_dir' column.")
    return [Path(str(p)) for p in df["scene_dir"].dropna().tolist()]


def build_ocm_input_array(
    b04_path: Path,
    b03_path: Path,
    b08_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    b04, meta = read_single_band(b04_path)
    b03, _ = read_single_band(b03_path)
    b08, _ = read_single_band(b08_path)

    if b04.shape != b03.shape or b04.shape != b08.shape:
        raise ValueError(
            f"Band shape mismatch: B04={b04.shape}, B03={b03.shape}, B08={b08.shape}"
        )

    arr = np.stack(
        [
            b04.astype(np.float32),  # Red
            b03.astype(np.float32),  # Green
            b08.astype(np.float32),  # NIR
        ],
        axis=0,
    )
    return arr, meta


def run_ocm_on_array(arr: np.ndarray) -> np.ndarray:
    pred = predict_from_array(arr)
    pred = np.asarray(pred)

    if pred.ndim == 3:
        if pred.shape[0] == 1:
            pred = pred[0]
        else:
            raise ValueError(f"Unexpected OCM output shape: {pred.shape}")
    elif pred.ndim != 2:
        raise ValueError(f"Unexpected OCM output ndim: {pred.ndim}, shape={pred.shape}")

    return pred.astype(np.uint8)


def save_ocm_mask(
    mask_arr: np.ndarray,
    ref_meta: dict[str, Any],
    output_path: Path,
) -> None:
    write_single_band(
        arr=mask_arr,
        ref_meta=ref_meta,
        output_path=output_path,
        dtype="uint8",
        nodata=DEFAULT_MASK_NODATA,
    )


def summarize_mask(mask_arr: np.ndarray) -> dict[str, Any]:
    uniq, counts = np.unique(mask_arr, return_counts=True)
    total = int(mask_arr.size)

    return {
        "shape": [int(mask_arr.shape[0]), int(mask_arr.shape[1])],
        "unique_values": [int(v) for v in uniq.tolist()],
        "counts": {str(int(v)): int(c) for v, c in zip(uniq.tolist(), counts.tolist())},
        "fractions": {str(int(v)): float(c / total) for v, c in zip(uniq.tolist(), counts.tolist())},
        "total_pixels": total,
    }


def process_scene(
    city_slug: str,
    scene_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    scene_id = scene_dir.name
    out_dir = get_city_ocm_mask_dir(city_slug) / scene_id
    out_path = out_dir / "ocm_mask.tif"

    rec: dict[str, Any] = {
        "city_slug": city_slug,
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "output_mask": str(out_path),
        "status": None,
    }

    if out_path.exists() and not overwrite:
        rec["status"] = "exists"
        return rec

    b04_path = scene_dir / "B04.tif"
    b03_path = scene_dir / "B03.tif"
    b08_path = scene_dir / "B08.tif"

    if not (b04_path.exists() and b03_path.exists() and b08_path.exists()):
        raise FileNotFoundError(
            f"Missing required bands in {scene_dir}. "
            f"B04 exists={b04_path.exists()}, "
            f"B03 exists={b03_path.exists()}, "
            f"B08 exists={b08_path.exists()}"
        )

    arr, meta = build_ocm_input_array(
        b04_path=b04_path,
        b03_path=b03_path,
        b08_path=b08_path,
    )

    pred = run_ocm_on_array(arr)
    save_ocm_mask(pred, meta, out_path)

    rec["status"] = "ok"
    rec["input_B04"] = str(b04_path)
    rec["input_B03"] = str(b03_path)
    rec["input_B08"] = str(b08_path)
    rec["mask_summary"] = summarize_mask(pred)

    return rec


def process_city(
    city_slug: str,
    overwrite: bool,
) -> dict[str, Any]:
    print("-" * 80)
    print(f"[CITY] {city_slug}")

    city_log_dir = get_city_log_dir(city_slug)
    city_report_dir = get_city_report_dir(city_slug)

    ensure_dir(city_log_dir)
    ensure_dir(city_report_dir)
    ensure_dir(get_city_ocm_mask_dir(city_slug))

    df = load_inventory(city_slug)

    if "has_required_ocm_inputs" not in df.columns:
        raise KeyError(
            f"Inventory for {city_slug} does not contain 'has_required_ocm_inputs'. "
            "Please re-run 01_inventory_raw_backup.py."
        )

    df_valid = df[df["has_required_ocm_inputs"].fillna(False)].copy()
    scene_dirs = get_scene_dirs_from_inventory(df_valid)

    print(f"[INFO] Inventory rows                  : {len(df)}")
    print(f"[INFO] Scenes with required OCM inputs: {len(scene_dirs)}")

    results: list[dict[str, Any]] = []

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        try:
            rec = process_scene(
                city_slug=city_slug,
                scene_dir=scene_dir,
                overwrite=overwrite,
            )
            results.append(rec)

            if rec["status"] == "exists":
                print(f"[SKIP] {scene_id} already exists")
            else:
                print(f"[OK]   {scene_id} -> {rec['output_mask']}")

        except Exception as e:
            rec = {
                "city_slug": city_slug,
                "scene_id": scene_dir.name,
                "scene_dir": str(scene_dir),
                "status": "failed",
                "error": str(e),
            }
            results.append(rec)
            print(f"[FAIL] {scene_id}: {e}")

    n_ok = sum(r["status"] == "ok" for r in results)
    n_exists = sum(r["status"] == "exists" for r in results)
    n_failed = sum(r["status"] == "failed" for r in results)

    summary = {
        "city_slug": city_slug,
        "n_inventory_rows": int(len(df)),
        "n_scenes_with_required_ocm_inputs": int(len(scene_dirs)),
        "n_ok": int(n_ok),
        "n_exists": int(n_exists),
        "n_failed": int(n_failed),
        "overwrite": overwrite,
        "results_json": str(city_log_dir / f"{city_slug}_ocm_generation_results.json"),
        "summary_json": str(city_log_dir / f"{city_slug}_ocm_generation_summary.json"),
    }

    results_json = city_log_dir / f"{city_slug}_ocm_generation_results.json"
    summary_json = city_log_dir / f"{city_slug}_ocm_generation_summary.json"

    save_json(results, results_json)
    save_json(summary, summary_json)

    save_json(results, city_report_dir / f"{city_slug}_ocm_generation_results.json")
    save_json(summary, city_report_dir / f"{city_slug}_ocm_generation_summary.json")

    print(f"[INFO] n_ok     : {n_ok}")
    print(f"[INFO] n_exists : {n_exists}")
    print(f"[INFO] n_failed : {n_failed}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OCM masks for already-downloaded raw_backup scenes."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=TARGET_CITIES,
        help="City slugs to process. Example: recife belem sao_luis",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing OCM masks if present.",
    )
    args = parser.parse_args()

    ensure_dir(OCM_MASKS_DIR)
    ensure_dir(LOGS_DIR)
    ensure_dir(REPORT_DIR)

    print("=" * 80)
    print("GENERATE OCM MASKS")
    print("=" * 80)
    print(f"[INFO] Cities     : {args.cities}")
    print(f"[INFO] Overwrite  : {args.overwrite}")
    print(f"[INFO] Output dir : {OCM_MASKS_DIR}")
    print()

    all_summaries = []

    for city in args.cities:
        city_slug = slugify_city_name(city)
        summary = process_city(city_slug=city_slug, overwrite=args.overwrite)
        all_summaries.append(summary)

    global_summary = {
        "cities": all_summaries,
        "overwrite": args.overwrite,
        "global_summary_json": str(LOGS_DIR / "ocm_generation_global_summary.json"),
    }

    save_json(global_summary, LOGS_DIR / "ocm_generation_global_summary.json")
    save_json(global_summary, REPORT_DIR / "ocm_generation_global_summary.json")

    print()
    print("=" * 80)
    print("[DONE] OCM mask generation finished.")
    print(f"[DONE] Global summary: {LOGS_DIR / 'ocm_generation_global_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()