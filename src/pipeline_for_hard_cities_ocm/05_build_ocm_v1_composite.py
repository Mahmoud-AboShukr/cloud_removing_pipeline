# src/pipeline_for_hard_cities_ocm/05_build_ocm_v1_composite.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation

from config_ocm_v1 import (
    ALIGNED_DIR,
    COMPOSITES_DIR,
    DEFAULT_COMPOSITE_NODATA,
    DEFAULT_DILATE_ITERS,
    DEFAULT_MASK_NODATA,
    DEFAULT_MAX_HOLE_PIXELS,
    LOGS_DIR,
    OCM_BAD_VALUES,
    OCM_CLEAR_VALUE,
    REPORT_DIR,
    TARGET_CITIES,
    get_city_log_dir,
    get_city_report_dir,
)
from utils_ocm import (
    ensure_dir,
    fill_small_holes_with_nearest,
    load_json,
    save_json,
    slugify_city_name,
)

BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


def load_grid_spec(city_slug: str) -> dict[str, Any]:
    path = ALIGNED_DIR.parent / "grid_specs" / city_slug / f"{city_slug}_grid_spec.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Grid spec not found for city '{city_slug}': {path}\n"
            "Run 03_prepare_city_grid.py first."
        )
    return load_json(path)


def discover_aligned_scene_dirs(city_slug: str) -> list[Path]:
    city_dir = ALIGNED_DIR / city_slug
    if not city_dir.exists():
        raise FileNotFoundError(
            f"Aligned city directory not found: {city_dir}\n"
            "Run 04_reproject_city_scenes.py first."
        )
    return sorted([p for p in city_dir.iterdir() if p.is_dir()])


def validate_scene_dir(scene_dir: Path) -> bool:
    """
    Require all 12 bands + ocm_mask.
    """
    for band in BAND_NAMES:
        if not (scene_dir / f"{band}.tif").exists():
            return False
    if not (scene_dir / "ocm_mask.tif").exists():
        return False
    return True


def read_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    return arr, meta


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def build_valid_mask_from_ocm(
    ocm_mask: np.ndarray,
    dilate_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    valid = OCM == 0
    bad   = OCM in {1,2,3}
    then dilate bad and exclude it.
    """
    clear = ocm_mask == OCM_CLEAR_VALUE
    bad = np.isin(ocm_mask, OCM_BAD_VALUES)

    if dilate_iters > 0:
        bad_dil = binary_dilation(bad, iterations=dilate_iters)
    else:
        bad_dil = bad

    valid = clear & (~bad_dil)
    return valid, bad_dil


def compute_scene_validity_diagnostics(
    scene_id: str,
    valid_mask: np.ndarray,
    bad_dil: np.ndarray,
) -> dict[str, Any]:
    total = int(valid_mask.size)
    valid_pixels = int(valid_mask.sum())
    bad_pixels = int(bad_dil.sum())

    return {
        "scene_id": scene_id,
        "total_pixels": total,
        "valid_pixels": valid_pixels,
        "valid_fraction": float(valid_pixels / total) if total > 0 else 0.0,
        "bad_dilated_pixels": bad_pixels,
        "bad_dilated_fraction": float(bad_pixels / total) if total > 0 else 0.0,
    }


def stack_band_across_scenes(
    scene_dirs: list[Path],
    valid_masks: list[np.ndarray],
    band_name: str,
) -> np.ndarray:
    """
    Return a stack of shape (n_scenes, H, W), with invalid pixels set to NaN.
    """
    stack = []
    for scene_dir, valid_mask in zip(scene_dirs, valid_masks):
        arr = read_band(scene_dir / f"{band_name}.tif")
        arr = arr.astype(np.float32)
        arr[~valid_mask] = np.nan
        stack.append(arr)

    return np.stack(stack, axis=0)


def median_composite(stack: np.ndarray) -> np.ndarray:
    """
    Per-band temporal median ignoring NaNs.
    """
    with np.errstate(all="ignore"):
        comp = np.nanmedian(stack, axis=0)
    return comp


def compute_observation_count(stack: np.ndarray) -> np.ndarray:
    """
    Count valid observations per pixel from stack (n_scenes, H, W).
    """
    return np.sum(np.isfinite(stack), axis=0).astype(np.uint16)


def write_multiband_geotiff(
    output_path: Path,
    bands_data: dict[str, np.ndarray],
    ref_meta: dict[str, Any],
    band_names: list[str],
    nodata_value: float | int,
) -> None:
    ensure_dir(output_path.parent)

    meta = ref_meta.copy()
    meta.update(
        {
            "count": len(band_names),
            "dtype": "float32",
            "nodata": nodata_value,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        for idx, band_name in enumerate(band_names, start=1):
            arr = bands_data[band_name].astype(np.float32)
            dst.write(arr, idx)
            dst.set_band_description(idx, band_name)


def write_singleband_geotiff(
    output_path: Path,
    arr: np.ndarray,
    ref_meta: dict[str, Any],
    dtype: str,
    nodata_value: float | int,
) -> None:
    ensure_dir(output_path.parent)

    meta = ref_meta.copy()
    meta.update(
        {
            "count": 1,
            "dtype": dtype,
            "nodata": nodata_value,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(arr.astype(dtype), 1)


def process_city(
    city_slug: str,
    dilate_iters: int,
    max_hole_pixels: int,
    overwrite: bool,
) -> dict[str, Any]:
    print("-" * 80)
    print(f"[CITY] {city_slug}")

    city_log_dir = get_city_log_dir(city_slug)
    city_report_dir = get_city_report_dir(city_slug)
    city_out_dir = COMPOSITES_DIR / city_slug

    ensure_dir(city_log_dir)
    ensure_dir(city_report_dir)
    ensure_dir(city_out_dir)

    grid_spec = load_grid_spec(city_slug)
    scene_dirs_all = discover_aligned_scene_dirs(city_slug)
    scene_dirs = [p for p in scene_dirs_all if validate_scene_dir(p)]

    if not scene_dirs:
        raise RuntimeError(f"No valid aligned scene directories found for city '{city_slug}'.")

    print(f"[INFO] n_scene_dirs discovered : {len(scene_dirs_all)}")
    print(f"[INFO] n_scene_dirs valid      : {len(scene_dirs)}")
    print(f"[INFO] dilate_iters           : {dilate_iters}")
    print(f"[INFO] max_hole_pixels        : {max_hole_pixels}")

    # Reference metadata from first scene B02
    with rasterio.open(scene_dirs[0] / "B02.tif") as src:
        ref_meta = src.meta.copy()

    scene_diags = []
    valid_masks = []

    # Build valid masks from OCM
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        ocm_mask, _ = read_mask(scene_dir / "ocm_mask.tif")
        valid_mask, bad_dil = build_valid_mask_from_ocm(ocm_mask, dilate_iters=dilate_iters)
        valid_masks.append(valid_mask)
        scene_diags.append(compute_scene_validity_diagnostics(scene_id, valid_mask, bad_dil))

    # Build composite band-by-band
    composite_bands: dict[str, np.ndarray] = {}
    composite_filled_bands: dict[str, np.ndarray] = {}

    # Use B02 stack for observation count and unresolved mask
    b02_stack = stack_band_across_scenes(scene_dirs, valid_masks, "B02")
    observation_count = compute_observation_count(b02_stack)
    unresolved_mask = observation_count == 0

    small_holes_mask_global = None
    large_holes_mask_global = None

    for band_name in BAND_NAMES:
        stack = stack_band_across_scenes(scene_dirs, valid_masks, band_name)
        comp = median_composite(stack)

        # Replace NaN with nodata for storage
        comp_out = comp.copy()
        comp_out[np.isnan(comp_out)] = DEFAULT_COMPOSITE_NODATA
        composite_bands[band_name] = comp_out.astype(np.float32)

        # Small-hole filling only
        valid_mask_band = np.isfinite(comp)
        filled, small_holes_mask, large_holes_mask = fill_small_holes_with_nearest(
            data=np.nan_to_num(comp, nan=DEFAULT_COMPOSITE_NODATA).astype(np.float32),
            valid_mask=valid_mask_band,
            max_hole_pixels=max_hole_pixels,
        )

        composite_filled_bands[band_name] = filled.astype(np.float32)

        if small_holes_mask_global is None:
            small_holes_mask_global = small_holes_mask
        if large_holes_mask_global is None:
            large_holes_mask_global = large_holes_mask

    unresolved_pixels = int(unresolved_mask.sum())
    total_pixels = int(unresolved_mask.size)
    unresolved_ratio = float(unresolved_pixels / total_pixels) if total_pixels > 0 else 0.0

    # Output files
    composite_tif = city_out_dir / f"{city_slug}_ocm_v1_composite.tif"
    composite_filled_tif = city_out_dir / f"{city_slug}_ocm_v1_composite_smallholes_filled.tif"
    observation_count_tif = city_out_dir / f"{city_slug}_ocm_v1_observation_count.tif"
    cloudmask_tif = city_out_dir / f"{city_slug}_ocm_v1_cloudmask.tif"
    small_holes_tif = city_out_dir / f"{city_slug}_ocm_v1_small_holes_mask.tif"
    diagnostics_json = city_out_dir / f"{city_slug}_ocm_v1_diagnostics.json"

    if overwrite or not composite_tif.exists():
        write_multiband_geotiff(
            output_path=composite_tif,
            bands_data=composite_bands,
            ref_meta=ref_meta,
            band_names=BAND_NAMES,
            nodata_value=DEFAULT_COMPOSITE_NODATA,
        )

    if overwrite or not composite_filled_tif.exists():
        write_multiband_geotiff(
            output_path=composite_filled_tif,
            bands_data=composite_filled_bands,
            ref_meta=ref_meta,
            band_names=BAND_NAMES,
            nodata_value=DEFAULT_COMPOSITE_NODATA,
        )

    if overwrite or not observation_count_tif.exists():
        write_singleband_geotiff(
            output_path=observation_count_tif,
            arr=observation_count,
            ref_meta=ref_meta,
            dtype="uint16",
            nodata_value=0,
        )

    if overwrite or not cloudmask_tif.exists():
        write_singleband_geotiff(
            output_path=cloudmask_tif,
            arr=unresolved_mask.astype(np.uint8),
            ref_meta=ref_meta,
            dtype="uint8",
            nodata_value=255,
        )

    if overwrite or not small_holes_tif.exists():
        write_singleband_geotiff(
            output_path=small_holes_tif,
            arr=small_holes_mask_global.astype(np.uint8),
            ref_meta=ref_meta,
            dtype="uint8",
            nodata_value=255,
        )

    diagnostics = {
        "city_slug": city_slug,
        "n_scene_dirs_discovered": int(len(scene_dirs_all)),
        "n_scene_dirs_valid": int(len(scene_dirs)),
        "dilate_iters": int(dilate_iters),
        "max_hole_pixels": int(max_hole_pixels),
        "total_pixels": total_pixels,
        "unresolved_pixels": unresolved_pixels,
        "unresolved_ratio": unresolved_ratio,
        "scene_diagnostics": scene_diags,
        "output_files": {
            "composite_tif": str(composite_tif),
            "composite_filled_tif": str(composite_filled_tif),
            "observation_count_tif": str(observation_count_tif),
            "cloudmask_tif": str(cloudmask_tif),
            "small_holes_tif": str(small_holes_tif),
        },
        "grid_spec": grid_spec,
    }

    save_json(diagnostics, diagnostics_json)
    save_json(diagnostics, city_log_dir / f"{city_slug}_ocm_v1_diagnostics.json")
    save_json(diagnostics, city_report_dir / f"{city_slug}_ocm_v1_diagnostics.json")

    print(f"[INFO] unresolved_ratio : {unresolved_ratio:.4f}")
    print(f"[INFO] composite       : {composite_tif}")
    print(f"[INFO] filled          : {composite_filled_tif}")

    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build OCM_V1 cloud-reduced composites from aligned scenes."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=TARGET_CITIES,
        help="City slugs to process. Example: recife belem sao_luis",
    )
    parser.add_argument(
        "--dilate-iters",
        type=int,
        default=DEFAULT_DILATE_ITERS,
        help="Binary dilation iterations applied to bad OCM classes.",
    )
    parser.add_argument(
        "--max-hole-pixels",
        type=int,
        default=DEFAULT_MAX_HOLE_PIXELS,
        help="Maximum hole size eligible for nearest-valid filling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    args = parser.parse_args()

    ensure_dir(COMPOSITES_DIR)
    ensure_dir(LOGS_DIR)
    ensure_dir(REPORT_DIR)

    print("=" * 80)
    print("BUILD OCM_V1 COMPOSITE")
    print("=" * 80)
    print(f"[INFO] Cities          : {args.cities}")
    print(f"[INFO] Dilate iters    : {args.dilate_iters}")
    print(f"[INFO] Max hole pixels : {args.max_hole_pixels}")
    print(f"[INFO] Overwrite       : {args.overwrite}")
    print()

    all_diags = []

    for city in args.cities:
        city_slug = slugify_city_name(city)
        diag = process_city(
            city_slug=city_slug,
            dilate_iters=args.dilate_iters,
            max_hole_pixels=args.max_hole_pixels,
            overwrite=args.overwrite,
        )
        all_diags.append(diag)

    global_diag = {
        "cities": all_diags,
        "dilate_iters": args.dilate_iters,
        "max_hole_pixels": args.max_hole_pixels,
        "overwrite": args.overwrite,
    }

    save_json(global_diag, LOGS_DIR / "ocm_v1_composite_global_summary.json")
    save_json(global_diag, REPORT_DIR / "ocm_v1_composite_global_summary.json")

    print()
    print("=" * 80)
    print("[DONE] OCM_V1 compositing finished.")
    print(f"[DONE] Global summary: {LOGS_DIR / 'ocm_v1_composite_global_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()