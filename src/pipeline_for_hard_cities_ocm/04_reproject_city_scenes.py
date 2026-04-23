# src/pipeline_for_hard_cities_ocm/04_reproject_city_scenes.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config_ocm_v1 import (
    ALIGNED_DIR,
    DEFAULT_MASK_NODATA,
    LOGS_DIR,
    REPORT_DIR,
    TARGET_CITIES,
    get_city_inventory_csv,
    get_city_log_dir,
    get_city_ocm_mask_dir,
    get_city_report_dir,
)
from utils_ocm import ensure_dir, load_json, save_json, slugify_city_name


BAND_FILES = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B11.tif",
    "B12.tif",
]

MASK_FILE = "ocm_mask.tif"


def load_inventory(city_slug: str) -> pd.DataFrame:
    csv_path = get_city_inventory_csv(city_slug)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Inventory CSV not found for city '{city_slug}': {csv_path}\n"
            "Run 01_inventory_raw_backup.py first."
        )
    return pd.read_csv(csv_path)


def load_grid_spec(city_slug: str) -> dict[str, Any]:
    path = ALIGNED_DIR.parent / "grid_specs" / city_slug / f"{city_slug}_grid_spec.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Grid spec not found for city '{city_slug}': {path}\n"
            "Run 03_prepare_city_grid.py first."
        )
    return load_json(path)


def make_transform(transform_values: list[float]) -> Affine:
    if len(transform_values) != 9:
        raise ValueError(f"Expected 9 transform values, got {len(transform_values)}")
    return Affine(*transform_values)


def choose_band_resampling(band_name: str) -> Resampling:
    """
    Use bilinear for continuous reflectance bands.
    """
    return Resampling.bilinear


def choose_mask_resampling() -> Resampling:
    """
    Use nearest neighbor for categorical masks.
    """
    return Resampling.nearest


def reproject_single_band_to_city_grid(
    src_path: Path,
    dst_path: Path,
    grid_spec: dict[str, Any],
    resampling: Resampling,
    dst_nodata: float | int | None = None,
) -> dict[str, Any]:
    """
    Reproject one single-band raster to the city grid.
    """
    ensure_dir(dst_path.parent)

    dst_transform = make_transform(grid_spec["grid_transform"])
    dst_crs = grid_spec["reference_crs"]
    dst_width = int(grid_spec["grid_width"])
    dst_height = int(grid_spec["grid_height"])

    with rasterio.open(src_path) as src:
        src_arr = src.read(1)

        out_nodata = dst_nodata
        if out_nodata is None:
            out_nodata = src.nodata if src.nodata is not None else 0

        dst_arr = np.full((dst_height, dst_width), out_nodata, dtype=src_arr.dtype)

        dst_meta = src.meta.copy()
        dst_meta.update(
            {
                "height": dst_height,
                "width": dst_width,
                "transform": dst_transform,
                "crs": dst_crs,
                "count": 1,
                "nodata": out_nodata,
                "compress": "deflate",
            }
        )

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=out_nodata,
            resampling=resampling,
        )

        with rasterio.open(dst_path, "w", **dst_meta) as dst:
            dst.write(dst_arr, 1)

    return {
        "src_path": str(src_path),
        "dst_path": str(dst_path),
        "resampling": resampling.name,
        "dst_width": dst_width,
        "dst_height": dst_height,
        "dst_crs": dst_crs,
        "dst_nodata": out_nodata,
    }


def process_scene(
    city_slug: str,
    scene_dir: Path,
    grid_spec: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    scene_id = scene_dir.name

    out_scene_dir = ALIGNED_DIR / city_slug / scene_id
    ensure_dir(out_scene_dir)

    ocm_mask_src = get_city_ocm_mask_dir(city_slug) / scene_id / MASK_FILE
    if not ocm_mask_src.exists():
        raise FileNotFoundError(
            f"OCM mask not found for city '{city_slug}', scene '{scene_id}': {ocm_mask_src}\n"
            "Run 02_generate_ocm_masks.py first."
        )

    scene_result: dict[str, Any] = {
        "city_slug": city_slug,
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "aligned_dir": str(out_scene_dir),
        "bands": [],
        "mask": None,
        "status": "ok",
    }

    # Reproject spectral bands
    for band_file in BAND_FILES:
        src_path = scene_dir / band_file
        dst_path = out_scene_dir / band_file

        rec = {
            "band_file": band_file,
            "src_exists": src_path.exists(),
            "dst_path": str(dst_path),
            "status": None,
        }

        if not src_path.exists():
            rec["status"] = "missing_source"
            scene_result["bands"].append(rec)
            continue

        if dst_path.exists() and not overwrite:
            rec["status"] = "exists"
            scene_result["bands"].append(rec)
            continue

        try:
            meta = reproject_single_band_to_city_grid(
                src_path=src_path,
                dst_path=dst_path,
                grid_spec=grid_spec,
                resampling=choose_band_resampling(band_file),
                dst_nodata=None,
            )
            rec["status"] = "ok"
            rec["meta"] = meta
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = str(e)
            scene_result["status"] = "partial_failure"

        scene_result["bands"].append(rec)

    # Reproject OCM mask
    ocm_mask_dst = out_scene_dir / MASK_FILE
    mask_rec = {
        "mask_file": MASK_FILE,
        "src_path": str(ocm_mask_src),
        "dst_path": str(ocm_mask_dst),
        "status": None,
    }

    if ocm_mask_dst.exists() and not overwrite:
        mask_rec["status"] = "exists"
    else:
        try:
            meta = reproject_single_band_to_city_grid(
                src_path=ocm_mask_src,
                dst_path=ocm_mask_dst,
                grid_spec=grid_spec,
                resampling=choose_mask_resampling(),
                dst_nodata=DEFAULT_MASK_NODATA,
            )
            mask_rec["status"] = "ok"
            mask_rec["meta"] = meta
        except Exception as e:
            mask_rec["status"] = "failed"
            mask_rec["error"] = str(e)
            scene_result["status"] = "partial_failure"

    scene_result["mask"] = mask_rec
    return scene_result


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
    ensure_dir(ALIGNED_DIR / city_slug)

    df = load_inventory(city_slug)
    grid_spec = load_grid_spec(city_slug)

    if "has_required_ocm_inputs" not in df.columns:
        raise KeyError(
            f"Inventory for {city_slug} does not contain 'has_required_ocm_inputs'. "
            "Please re-run 01_inventory_raw_backup.py."
        )

    df_valid = df[df["has_required_ocm_inputs"].fillna(False)].copy()
    scene_dirs = [Path(str(p)) for p in df_valid["scene_dir"].dropna().tolist()]

    print(f"[INFO] Valid scenes from inventory : {len(scene_dirs)}")
    print(f"[INFO] Grid size                  : {grid_spec['grid_width']} x {grid_spec['grid_height']}")
    print(f"[INFO] Grid CRS                   : {grid_spec['reference_crs']}")

    results = []

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        try:
            rec = process_scene(
                city_slug=city_slug,
                scene_dir=scene_dir,
                grid_spec=grid_spec,
                overwrite=overwrite,
            )
            results.append(rec)
            print(f"[OK]   {scene_id}")
        except Exception as e:
            results.append(
                {
                    "city_slug": city_slug,
                    "scene_id": scene_id,
                    "scene_dir": str(scene_dir),
                    "status": "failed",
                    "error": str(e),
                }
            )
            print(f"[FAIL] {scene_id}: {e}")

    n_ok = sum(r.get("status") == "ok" for r in results)
    n_partial = sum(r.get("status") == "partial_failure" for r in results)
    n_failed = sum(r.get("status") == "failed" for r in results)

    summary = {
        "city_slug": city_slug,
        "n_inventory_rows": int(len(df)),
        "n_valid_scenes": int(len(scene_dirs)),
        "n_ok": int(n_ok),
        "n_partial_failure": int(n_partial),
        "n_failed": int(n_failed),
        "overwrite": overwrite,
        "results_json": str(city_log_dir / f"{city_slug}_reproject_results.json"),
        "summary_json": str(city_log_dir / f"{city_slug}_reproject_summary.json"),
    }

    results_json = city_log_dir / f"{city_slug}_reproject_results.json"
    summary_json = city_log_dir / f"{city_slug}_reproject_summary.json"

    save_json(results, results_json)
    save_json(summary, summary_json)

    save_json(results, city_report_dir / f"{city_slug}_reproject_results.json")
    save_json(summary, city_report_dir / f"{city_slug}_reproject_summary.json")

    print(f"[INFO] n_ok             : {n_ok}")
    print(f"[INFO] n_partial       : {n_partial}")
    print(f"[INFO] n_failed        : {n_failed}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproject city scenes and OCM masks onto the common city grid."
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
        help="Overwrite existing aligned files if present.",
    )
    args = parser.parse_args()

    ensure_dir(ALIGNED_DIR)
    ensure_dir(LOGS_DIR)
    ensure_dir(REPORT_DIR)

    print("=" * 80)
    print("REPROJECT CITY SCENES FOR OCM_V1")
    print("=" * 80)
    print(f"[INFO] Cities      : {args.cities}")
    print(f"[INFO] Overwrite   : {args.overwrite}")
    print(f"[INFO] ALIGNED_DIR : {ALIGNED_DIR}")
    print()

    all_summaries = []

    for city in args.cities:
        city_slug = slugify_city_name(city)
        summary = process_city(city_slug=city_slug, overwrite=args.overwrite)
        all_summaries.append(summary)

    global_summary = {
        "cities": all_summaries,
        "overwrite": args.overwrite,
        "global_summary_json": str(LOGS_DIR / "reproject_global_summary.json"),
    }

    save_json(global_summary, LOGS_DIR / "reproject_global_summary.json")
    save_json(global_summary, REPORT_DIR / "reproject_global_summary.json")

    print()
    print("=" * 80)
    print("[DONE] Reprojection finished.")
    print(f"[DONE] Global summary: {LOGS_DIR / 'reproject_global_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()