# src/pipeline_for_hard_cities_ocm/03_prepare_city_grid.py

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine

from config_ocm_v1 import (
    AOI_DIR,
    DEFAULT_COMPOSITE_NODATA,
    GRID_SPECS_DIR,
    REPORT_DIR,
    TARGET_CITIES,
    get_city_inventory_csv,
    get_city_log_dir,
    get_city_report_dir,
)
from utils_ocm import (
    ensure_dir,
    load_aoi,
    save_json,
    slugify_city_name,
)


def load_inventory(city_slug: str) -> pd.DataFrame:
    """
    Load inventory CSV for a city.
    """
    csv_path = get_city_inventory_csv(city_slug)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Inventory CSV not found for city '{city_slug}': {csv_path}\n"
            "Run 01_inventory_raw_backup.py first."
        )
    return pd.read_csv(csv_path)


def choose_reference_raster(scene_dir: Path) -> Path:
    """
    Choose a 10 m reference raster from the scene.
    Prefer B02, then B03, then B04, then B08.
    """
    candidates = ["B02.tif", "B03.tif", "B04.tif", "B08.tif"]
    for fname in candidates:
        p = scene_dir / fname
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No suitable 10 m reference raster found in scene directory: {scene_dir}"
    )


def snap_bounds_to_grid(
    bounds: tuple[float, float, float, float],
    transform: Affine,
    res_x: float,
    res_y: float,
) -> tuple[float, float, float, float]:
    """
    Snap projected AOI bounds outward to the reference grid.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in projected coordinates
    transform : Affine
        Reference raster transform
    res_x : float
        Pixel size in x
    res_y : float
        Pixel size in y (positive magnitude)

    Returns
    -------
    snapped_bounds : tuple
        (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds

    origin_x = transform.c
    origin_y = transform.f

    snapped_minx = origin_x + math.floor((minx - origin_x) / res_x) * res_x
    snapped_maxx = origin_x + math.ceil((maxx - origin_x) / res_x) * res_x

    # Y is north-up so transform.e is negative in most rasters.
    snapped_maxy = origin_y - math.floor((origin_y - maxy) / res_y) * res_y
    snapped_miny = origin_y - math.ceil((origin_y - miny) / res_y) * res_y

    return snapped_minx, snapped_miny, snapped_maxx, snapped_maxy


def build_city_grid_spec(
    city_slug: str,
) -> dict[str, Any]:
    """
    Build a city grid spec using:
    - buffered AOI
    - first valid scene's 10 m reference raster
    """
    aoi_path, aoi_gdf = load_aoi(AOI_DIR, city_slug)
    df = load_inventory(city_slug)

    if "has_required_ocm_inputs" not in df.columns:
        raise KeyError(
            f"Inventory for {city_slug} does not contain 'has_required_ocm_inputs'. "
            "Please re-run 01_inventory_raw_backup.py."
        )

    df_valid = df[df["has_required_ocm_inputs"].fillna(False)].copy()
    if df_valid.empty:
        raise ValueError(f"No valid OCM scenes found in inventory for city '{city_slug}'.")

    # Use first valid scene as reference for CRS/resolution/alignment
    first_scene_dir = Path(str(df_valid.iloc[0]["scene_dir"]))
    ref_raster_path = choose_reference_raster(first_scene_dir)

    with rasterio.open(ref_raster_path) as ref:
        ref_crs = ref.crs
        if ref_crs is None:
            raise ValueError(f"Reference raster CRS is missing: {ref_raster_path}")

        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        res_x = float(ref.res[0])
        res_y = abs(float(ref.res[1]))
        ref_dtype = ref.dtypes[0]
        ref_nodata = ref.nodata

    # Reproject AOI to the reference CRS
    aoi_proj = aoi_gdf.to_crs(ref_crs)
    proj_bounds = tuple(float(v) for v in aoi_proj.total_bounds)

    snapped_bounds = snap_bounds_to_grid(
        bounds=proj_bounds,
        transform=ref_transform,
        res_x=res_x,
        res_y=res_y,
    )

    minx, miny, maxx, maxy = snapped_bounds

    width = int(round((maxx - minx) / res_x))
    height = int(round((maxy - miny) / res_y))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid computed grid size for {city_slug}: width={width}, height={height}"
        )

    city_transform = Affine(res_x, 0.0, minx, 0.0, -res_y, maxy)

    spec = {
        "city_slug": city_slug,
        "aoi_path": str(aoi_path),
        "n_inventory_rows": int(len(df)),
        "n_valid_ocm_scenes": int(len(df_valid)),
        "reference_scene_dir": str(first_scene_dir),
        "reference_raster_path": str(ref_raster_path),
        "reference_crs": str(ref_crs),
        "reference_transform": list(ref_transform),
        "reference_res_x": res_x,
        "reference_res_y": res_y,
        "reference_width": int(ref_width),
        "reference_height": int(ref_height),
        "reference_dtype": str(ref_dtype),
        "reference_nodata": ref_nodata,
        "aoi_bounds_in_reference_crs": [float(v) for v in proj_bounds],
        "snapped_bounds": [float(v) for v in snapped_bounds],
        "grid_width": int(width),
        "grid_height": int(height),
        "grid_transform": list(city_transform),
        "grid_nodata": DEFAULT_COMPOSITE_NODATA,
    }

    return spec


def write_template_raster(spec: dict[str, Any], output_path: Path) -> None:
    """
    Write an empty template raster for the city grid.
    """
    ensure_dir(output_path.parent)

    transform = Affine(*spec["grid_transform"])
    width = int(spec["grid_width"])
    height = int(spec["grid_height"])
    crs = spec["reference_crs"]
    nodata = spec["grid_nodata"]

    arr = np.full((height, width), nodata, dtype=np.uint16)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(arr, 1)


def process_city(city_slug: str) -> dict[str, Any]:
    """
    Prepare one city grid and write outputs.
    """
    print("-" * 80)
    print(f"[CITY] {city_slug}")

    city_log_dir = get_city_log_dir(city_slug)
    city_report_dir = get_city_report_dir(city_slug)
    city_grid_dir = GRID_SPECS_DIR / city_slug

    ensure_dir(city_log_dir)
    ensure_dir(city_report_dir)
    ensure_dir(city_grid_dir)

    spec = build_city_grid_spec(city_slug)

    spec_json = city_grid_dir / f"{city_slug}_grid_spec.json"
    template_tif = city_grid_dir / f"{city_slug}_grid_template.tif"

    save_json(spec, spec_json)
    write_template_raster(spec, template_tif)

    # Save repo-side copy
    save_json(spec, city_report_dir / f"{city_slug}_grid_spec.json")

    summary = {
        "city_slug": city_slug,
        "grid_spec_json": str(spec_json),
        "grid_template_tif": str(template_tif),
        "reference_raster_path": spec["reference_raster_path"],
        "reference_crs": spec["reference_crs"],
        "grid_width": spec["grid_width"],
        "grid_height": spec["grid_height"],
        "snapped_bounds": spec["snapped_bounds"],
    }

    save_json(summary, city_log_dir / f"{city_slug}_grid_summary.json")
    save_json(summary, city_report_dir / f"{city_slug}_grid_summary.json")

    print(f"[INFO] Reference raster : {spec['reference_raster_path']}")
    print(f"[INFO] CRS              : {spec['reference_crs']}")
    print(f"[INFO] Grid size        : {spec['grid_width']} x {spec['grid_height']}")
    print(f"[INFO] Template raster  : {template_tif}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare common city grids for OCM_V1 compositing."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=TARGET_CITIES,
        help="City slugs to process. Example: recife belem sao_luis",
    )
    args = parser.parse_args()

    ensure_dir(GRID_SPECS_DIR)
    ensure_dir(REPORT_DIR)

    print("=" * 80)
    print("PREPARE CITY GRID FOR OCM_V1")
    print("=" * 80)
    print(f"[INFO] AOI_DIR       : {AOI_DIR}")
    print(f"[INFO] GRID_SPECS_DIR: {GRID_SPECS_DIR}")
    print(f"[INFO] REPORT_DIR    : {REPORT_DIR}")
    print(f"[INFO] Cities        : {args.cities}")
    print()

    all_summaries = []

    for city in args.cities:
        city_slug = slugify_city_name(city)
        summary = process_city(city_slug)
        all_summaries.append(summary)

    global_summary = {
        "cities": all_summaries,
        "global_summary_json": str(GRID_SPECS_DIR / "grid_global_summary.json"),
    }

    save_json(global_summary, GRID_SPECS_DIR / "grid_global_summary.json")
    save_json(global_summary, REPORT_DIR / "grid_global_summary.json")

    print()
    print("=" * 80)
    print("[DONE] City grid preparation finished.")
    print(f"[DONE] Global summary: {GRID_SPECS_DIR / 'grid_global_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()