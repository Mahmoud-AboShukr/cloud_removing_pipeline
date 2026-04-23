# src/pipeline_for_hard_cities_ocm/01_inventory_raw_backup.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config_ocm_v1 import (
    ALL_EXPECTED_BAND_FILES,
    ALL_EXPECTED_SCENE_FILES,
    AOI_DIR,
    OPTIONAL_SCL_FILE,
    RAW_BACKUP_ROOT,
    REPORT_DIR,
    REQUIRED_RGB_NIR_BANDS_FOR_OCM,
    TARGET_CITIES,
    get_city_inventory_csv,
    get_city_inventory_json,
    get_city_raw_root,
    get_city_report_dir,
)
from utils_ocm import (
    ensure_dir,
    load_aoi,
    safe_read_raster_info,
    save_json,
    slugify_city_name,
)


def contains_required_ocm_inputs(path: Path) -> bool:
    """
    Return True if this directory contains the minimum files required to run OCM.
    """
    return all((path / fname).exists() for fname in REQUIRED_RGB_NIR_BANDS_FOR_OCM)


def contains_any_expected_scene_files(path: Path) -> bool:
    """
    Return True if this directory contains at least one expected Sentinel-2 scene file.
    """
    return any((path / fname).exists() for fname in ALL_EXPECTED_SCENE_FILES)


def discover_scene_dirs_recursive(city_raw_root: Path) -> list[Path]:
    """
    Recursively discover scene directories.

    A directory is treated as a scene directory if it contains the minimum
    required OCM files: B03.tif, B04.tif, B08.tif.
    """
    if not city_raw_root.exists():
        return []

    scene_dirs = []
    for p in city_raw_root.rglob("*"):
        if p.is_dir() and contains_required_ocm_inputs(p):
            scene_dirs.append(p)

    # remove duplicates and sort
    scene_dirs = sorted(set(scene_dirs))
    return scene_dirs


def infer_scene_id(scene_dir: Path, city_raw_root: Path) -> str:
    """
    Use the leaf folder name as the scene id by default.
    """
    return scene_dir.name


def inspect_scene(scene_dir: Path, city_raw_root: Path) -> dict:
    """
    Inspect one discovered scene directory.
    """
    scene_id = infer_scene_id(scene_dir, city_raw_root)

    row = {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "relative_scene_dir": str(scene_dir.relative_to(city_raw_root)),
    }

    for fname in ALL_EXPECTED_SCENE_FILES:
        row[f"has_{fname.replace('.', '_')}"] = (scene_dir / fname).exists()

    row["has_required_ocm_inputs"] = all((scene_dir / fname).exists() for fname in REQUIRED_RGB_NIR_BANDS_FOR_OCM)
    row["has_all_12_bands"] = all((scene_dir / fname).exists() for fname in ALL_EXPECTED_BAND_FILES)
    row["has_scl"] = (scene_dir / OPTIONAL_SCL_FILE).exists()

    ref_candidates = ["B04.tif", "B02.tif", "B08.tif"]
    ref_info = None
    for ref_name in ref_candidates:
        ref_path = scene_dir / ref_name
        if ref_path.exists():
            try:
                ref_info = safe_read_raster_info(ref_path)
                break
            except Exception as e:
                row["reference_read_error"] = str(e)

    if ref_info is not None:
        row["reference_file"] = Path(ref_info["path"]).name
        row["crs"] = ref_info["crs"]
        row["width"] = ref_info["width"]
        row["height"] = ref_info["height"]
        row["dtype"] = ref_info["dtype"]
        row["nodata"] = ref_info["nodata"]
        row["res_x"] = ref_info["res"][0]
        row["res_y"] = ref_info["res"][1]
        row["bounds_left"] = ref_info["bounds"][0]
        row["bounds_bottom"] = ref_info["bounds"][1]
        row["bounds_right"] = ref_info["bounds"][2]
        row["bounds_top"] = ref_info["bounds"][3]
    else:
        row["reference_file"] = None
        row["crs"] = None
        row["width"] = None
        row["height"] = None
        row["dtype"] = None
        row["nodata"] = None
        row["res_x"] = None
        row["res_y"] = None
        row["bounds_left"] = None
        row["bounds_bottom"] = None
        row["bounds_right"] = None
        row["bounds_top"] = None

    return row


def process_city(city_slug: str) -> dict:
    """
    Inventory one city's raw scene directory recursively.
    """
    city_raw_root = get_city_raw_root(city_slug)
    city_report_dir = get_city_report_dir(city_slug)

    ensure_dir(city_report_dir)

    summary = {
        "city_slug": city_slug,
        "raw_backup_root": str(RAW_BACKUP_ROOT),
        "raw_city_root": str(city_raw_root),
        "aoi_found": False,
        "aoi_path": None,
        "n_scene_dirs": 0,
        "n_with_required_ocm_inputs": 0,
        "n_with_all_12_bands": 0,
        "n_with_scl": 0,
        "status": None,
        "discovery_mode": "recursive_search_for_B03_B04_B08",
    }

    try:
        aoi_path, aoi_gdf = load_aoi(AOI_DIR, city_slug)
        summary["aoi_found"] = True
        summary["aoi_path"] = str(aoi_path)
        summary["aoi_crs"] = str(aoi_gdf.crs)
        summary["aoi_bounds"] = [float(v) for v in aoi_gdf.total_bounds]
    except Exception as e:
        summary["aoi_error"] = str(e)

    if not city_raw_root.exists():
        summary["status"] = "missing_city_raw_root"
        save_json(summary, get_city_inventory_json(city_slug))
        return summary

    scene_dirs = discover_scene_dirs_recursive(city_raw_root)
    summary["n_scene_dirs"] = len(scene_dirs)

    if len(scene_dirs) == 0:
        summary["status"] = "no_scene_dirs_found"
        save_json(summary, get_city_inventory_json(city_slug))
        return summary

    rows = []
    for scene_dir in scene_dirs:
        try:
            rows.append(inspect_scene(scene_dir, city_raw_root))
        except Exception as e:
            rows.append(
                {
                    "scene_id": scene_dir.name,
                    "scene_dir": str(scene_dir),
                    "relative_scene_dir": str(scene_dir.relative_to(city_raw_root)),
                    "inspect_status": "failed",
                    "inspect_error": str(e),
                }
            )

    df = pd.DataFrame(rows)

    summary["n_with_required_ocm_inputs"] = int(df["has_required_ocm_inputs"].fillna(False).sum()) if "has_required_ocm_inputs" in df.columns else 0
    summary["n_with_all_12_bands"] = int(df["has_all_12_bands"].fillna(False).sum()) if "has_all_12_bands" in df.columns else 0
    summary["n_with_scl"] = int(df["has_scl"].fillna(False).sum()) if "has_scl" in df.columns else 0
    summary["inventory_csv"] = str(get_city_inventory_csv(city_slug))
    summary["status"] = "ok"

    ensure_dir(get_city_inventory_csv(city_slug).parent)
    df.to_csv(get_city_inventory_csv(city_slug), index=False)
    save_json(summary, get_city_inventory_json(city_slug))

    save_json(summary, city_report_dir / f"{city_slug}_scene_inventory_summary.json")
    df.to_csv(city_report_dir / f"{city_slug}_scene_inventory.csv", index=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively inventory already-downloaded hard-city Sentinel-2 scenes from raw_backup."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=TARGET_CITIES,
        help="City slugs to process. Example: recife belem sao_luis",
    )
    args = parser.parse_args()

    ensure_dir(REPORT_DIR)

    print("=" * 80)
    print("INVENTORY RAW BACKUP FOR OCM_V1")
    print("=" * 80)
    print(f"[INFO] RAW_BACKUP_ROOT : {RAW_BACKUP_ROOT}")
    print(f"[INFO] AOI_DIR         : {AOI_DIR}")
    print(f"[INFO] REPORT_DIR      : {REPORT_DIR}")
    print(f"[INFO] Cities          : {args.cities}")
    print()

    all_summaries = []

    for city in args.cities:
        city_slug = slugify_city_name(city)

        print("-" * 80)
        print(f"[CITY] {city} -> {city_slug}")

        summary = process_city(city_slug)
        all_summaries.append(summary)

        print(f"[INFO] Raw city root              : {summary['raw_city_root']}")
        print(f"[INFO] AOI found                  : {summary.get('aoi_found')}")
        print(f"[INFO] n_scene_dirs               : {summary.get('n_scene_dirs')}")
        print(f"[INFO] n_with_required_ocm_inputs : {summary.get('n_with_required_ocm_inputs')}")
        print(f"[INFO] n_with_all_12_bands        : {summary.get('n_with_all_12_bands')}")
        print(f"[INFO] n_with_scl                 : {summary.get('n_with_scl')}")
        print(f"[INFO] Status                     : {summary.get('status')}")

    global_summary = {
        "raw_backup_root": str(RAW_BACKUP_ROOT),
        "aoi_dir": str(AOI_DIR),
        "report_dir": str(REPORT_DIR),
        "cities": all_summaries,
    }

    global_json = REPORT_DIR / "inventory_raw_backup_global_summary.json"
    save_json(global_summary, global_json)

    print()
    print("=" * 80)
    print("[DONE] Inventory completed.")
    print(f"[DONE] Global summary: {global_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()