#!/usr/bin/env python3
"""
build_cloud_reduced_composite_v3_external.py

Build a cloud-reduced Sentinel-2 composite from already-downloaded scenes,
with support for keeping the raw data on an external hard disk.

What is different from the original script
-----------------------------------------
1. Raw data can live anywhere via --raw-root
   Example:
       --raw-root /media/HALLOPEAU/MySSD/raw_backup

2. Incomplete scenes are skipped automatically instead of crashing the whole city.
   A valid scene must contain all 12 input bands + SCL.

3. The rest of the V3 logic is preserved:
   - strict SCL valid mask [4, 5, 6]
   - minimum 2 contributing scenes
   - dynamic filler ranking based on currently missing pixels

Expected raw layout
-------------------
<raw_root>/<city_slug>/2022/s2_planetary_multiscene_12band/
    selected_scenes_summary.json
    <city_slug>_aoi.gpkg
    scenes/
        <scene_id>/
            B01.tif
            B02.tif
            B03.tif
            B04.tif
            B05.tif
            B06.tif
            B07.tif
            B08.tif
            B8A.tif
            B09.tif
            B11.tif
            B12.tif
            SCL.tif

Outputs
-------
<processed_root>/<city_slug>/2022/s2_planetary_multiscene_12band_v3/
    city_composite_2022.tif
    city_cloudmask_2022.tif
    city_scene_contribution_map.tif
    selected_scenes_diagnostics.json
    selected_scenes_diagnostics.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds


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
# Constants
# -----------------------------------------------------------------------------

INPUT_BANDS = [
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
SCL_BAND = "SCL"
SCENE_REF_BAND = "B02"

DEFAULT_VALID_SCL = [4, 5, 6]
DEFAULT_RESIDUAL_THRESHOLD = 0.02
DEFAULT_TARGET_RESOLUTION = 10.0
MIN_SCENES_REQUIRED = 2
REQUIRED_ASSETS = set(INPUT_BANDS + [SCL_BAND])


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DownloadedSceneInfo:
    item_id: str
    datetime_utc: str
    date: str
    tile_id: str
    cloud_cover: float
    overlap_ratio: float
    scene_dir: Path
    assets: Dict[str, Path]


@dataclass
class SceneUsability:
    item_id: str
    usable_pixels: int
    usable_ratio: float
    cloud_cover: float
    overlap_ratio: float
    date: str


@dataclass
class DynamicFillScore:
    item_id: str
    fillable_pixels: int
    fillable_ratio_of_missing: float
    global_usable_ratio: float
    overlap_ratio: float
    cloud_cover: float
    date: str


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------------------------------------------------------
# Scene loading / validation
# -----------------------------------------------------------------------------

def load_downloaded_scenes_from_summary(summary_path: Path) -> List[DownloadedSceneInfo]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = load_json(summary_path)
    downloaded = summary.get("downloaded_scenes", [])
    if not downloaded:
        raise ValueError(f"No downloaded scenes found in {summary_path}")

    scenes: List[DownloadedSceneInfo] = []
    for row in downloaded:
        assets = {k: Path(v) for k, v in row["downloaded_assets"].items()}
        scenes.append(
            DownloadedSceneInfo(
                item_id=row["item_id"],
                datetime_utc=row["datetime_utc"],
                date=row["date"],
                tile_id=row["tile_id"],
                cloud_cover=float(row["cloud_cover"]),
                overlap_ratio=float(row["overlap_ratio"]),
                scene_dir=Path(row["scene_dir"]),
                assets=assets,
            )
        )
    return scenes


def remap_scene_to_raw_root(
    scene: DownloadedSceneInfo,
    expected_city_root: Path,
) -> DownloadedSceneInfo:
    """Force all summary paths to point to the current --raw-root tree.

    This avoids problems when the JSON still contains the old internal-disk paths.
    """
    new_scene_dir = expected_city_root / "scenes" / scene.item_id
    new_assets = {band: new_scene_dir / f"{band}.tif" for band in REQUIRED_ASSETS}

    return DownloadedSceneInfo(
        item_id=scene.item_id,
        datetime_utc=scene.datetime_utc,
        date=scene.date,
        tile_id=scene.tile_id,
        cloud_cover=scene.cloud_cover,
        overlap_ratio=scene.overlap_ratio,
        scene_dir=new_scene_dir,
        assets=new_assets,
    )


def filter_valid_scenes(
    scenes: Sequence[DownloadedSceneInfo],
) -> Tuple[List[DownloadedSceneInfo], List[Dict[str, Any]]]:
    """Keep only scenes that have all required files on disk."""
    valid_scenes: List[DownloadedSceneInfo] = []
    skipped: List[Dict[str, Any]] = []

    for scene in scenes:
        missing_keys = sorted(REQUIRED_ASSETS - set(scene.assets.keys()))
        missing_files = sorted(
            band for band in REQUIRED_ASSETS if band in scene.assets and not scene.assets[band].exists()
        )

        if missing_keys or missing_files:
            skipped.append(
                {
                    "item_id": scene.item_id,
                    "reason": "incomplete_scene",
                    "missing_asset_keys": missing_keys,
                    "missing_files": missing_files,
                    "scene_dir": str(scene.scene_dir),
                }
            )
            logging.warning(
                "Skipping incomplete scene %s | missing keys=%s | missing files=%s",
                scene.item_id,
                missing_keys,
                missing_files,
            )
            continue

        valid_scenes.append(scene)

    return valid_scenes, skipped


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------

def build_target_grid_from_aoi(
    reference_raster_path: Path,
    aoi_geom_4326,
    target_resolution: float = DEFAULT_TARGET_RESOLUTION,
) -> Tuple[CRS, Affine, int, int]:
    with rasterio.open(reference_raster_path) as src:
        target_crs = src.crs
        if target_crs is None:
            raise ValueError(f"Reference raster has no CRS: {reference_raster_path}")

        ref_res_x = abs(src.transform.a)
        ref_res_y = abs(src.transform.e)

        if (
            not math.isclose(ref_res_x, target_resolution, rel_tol=0.01)
            or not math.isclose(ref_res_y, target_resolution, rel_tol=0.01)
        ):
            logging.warning(
                "Reference raster resolution is %.3f x %.3f, not exactly %.1f m.",
                ref_res_x,
                ref_res_y,
                target_resolution,
            )

        aoi_bounds_ref = transform_bounds(
            "EPSG:4326",
            target_crs,
            *aoi_geom_4326.bounds,
            densify_pts=21,
        )

        win = from_bounds(*aoi_bounds_ref, transform=src.transform)
        win = win.round_offsets().round_lengths()

        width = int(win.width)
        height = int(win.height)
        transform = src.window_transform(win)

        if width <= 0 or height <= 0:
            raise ValueError("AOI-derived target grid has non-positive width/height.")

        logging.info(
            "Target grid: CRS=%s width=%d height=%d res=(%.3f, %.3f)",
            target_crs,
            width,
            height,
            abs(transform.a),
            abs(transform.e),
        )
        return target_crs, transform, width, height


def rasterize_aoi_mask(
    aoi_geom_4326,
    target_crs: CRS,
    target_transform: Affine,
    width: int,
    height: int,
) -> np.ndarray:
    aoi_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[aoi_geom_4326], crs="EPSG:4326")
    aoi_proj = aoi_gdf.to_crs(target_crs)

    mask = rasterize(
        [(geom, 1) for geom in aoi_proj.geometry],
        out_shape=(height, width),
        transform=target_transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    mask_bool = mask.astype(bool)
    logging.info("AOI pixels: %d", int(mask_bool.sum()))
    return mask_bool


# -----------------------------------------------------------------------------
# Raster reprojection
# -----------------------------------------------------------------------------

def reproject_band_to_grid(
    src_path: Path,
    target_crs: CRS,
    target_transform: Affine,
    target_width: int,
    target_height: int,
    resampling: Resampling,
    dst_dtype: Optional[str] = None,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_array = src.read(1)
        out_dtype = dst_dtype or src.dtypes[0]
        dst = np.zeros((target_height, target_width), dtype=np.dtype(out_dtype))

        reproject(
            source=src_array,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=0,
            resampling=resampling,
        )
        return dst


def get_band_dtype(reference_band_path: Path) -> str:
    with rasterio.open(reference_band_path) as src:
        return src.dtypes[0]


# -----------------------------------------------------------------------------
# SCL logic
# -----------------------------------------------------------------------------

def build_valid_mask_from_scl(
    scl_array: np.ndarray,
    valid_scl_values: Sequence[int],
    aoi_mask: np.ndarray,
) -> np.ndarray:
    valid = np.isin(scl_array, np.array(valid_scl_values, dtype=scl_array.dtype))
    valid &= aoi_mask
    return valid


def compute_scene_usable_ratio(
    scene: DownloadedSceneInfo,
    target_crs: CRS,
    target_transform: Affine,
    width: int,
    height: int,
    aoi_mask: np.ndarray,
    valid_scl_values: Sequence[int],
) -> Tuple[SceneUsability, np.ndarray]:
    scl_reproj = reproject_band_to_grid(
        src_path=scene.assets[SCL_BAND],
        target_crs=target_crs,
        target_transform=target_transform,
        target_width=width,
        target_height=height,
        resampling=Resampling.nearest,
        dst_dtype="uint8",
    )

    valid_mask = build_valid_mask_from_scl(
        scl_array=scl_reproj,
        valid_scl_values=valid_scl_values,
        aoi_mask=aoi_mask,
    )

    aoi_pixels = int(aoi_mask.sum())
    usable_pixels = int(valid_mask.sum())
    usable_ratio = 0.0 if aoi_pixels == 0 else usable_pixels / aoi_pixels

    metrics = SceneUsability(
        item_id=scene.item_id,
        usable_pixels=usable_pixels,
        usable_ratio=float(usable_ratio),
        cloud_cover=float(scene.cloud_cover),
        overlap_ratio=float(scene.overlap_ratio),
        date=scene.date,
    )
    return metrics, valid_mask


def rank_scenes_globally(
    scenes: Sequence[DownloadedSceneInfo],
    usability: Dict[str, SceneUsability],
) -> List[DownloadedSceneInfo]:
    return sorted(
        scenes,
        key=lambda s: (
            usability[s.item_id].usable_ratio,
            s.overlap_ratio,
            -s.cloud_cover,
            s.date,
        ),
        reverse=True,
    )


def dynamic_rank_remaining_scenes(
    remaining_scenes: Sequence[DownloadedSceneInfo],
    scene_valid_masks: Dict[str, np.ndarray],
    filled_mask: np.ndarray,
    aoi_mask: np.ndarray,
    usability: Dict[str, SceneUsability],
) -> List[DynamicFillScore]:
    missing_mask = aoi_mask & (~filled_mask)
    missing_pixels = int(missing_mask.sum())

    scores: List[DynamicFillScore] = []

    for scene in remaining_scenes:
        valid_mask = scene_valid_masks[scene.item_id]
        fillable = valid_mask & missing_mask
        fillable_pixels = int(fillable.sum())
        fillable_ratio_of_missing = 0.0 if missing_pixels == 0 else fillable_pixels / missing_pixels

        scores.append(
            DynamicFillScore(
                item_id=scene.item_id,
                fillable_pixels=fillable_pixels,
                fillable_ratio_of_missing=float(fillable_ratio_of_missing),
                global_usable_ratio=usability[scene.item_id].usable_ratio,
                overlap_ratio=scene.overlap_ratio,
                cloud_cover=scene.cloud_cover,
                date=scene.date,
            )
        )

    scores.sort(
        key=lambda s: (
            s.fillable_pixels,
            s.fillable_ratio_of_missing,
            s.global_usable_ratio,
            s.overlap_ratio,
            -s.cloud_cover,
            s.date,
        ),
        reverse=True,
    )
    return scores


# -----------------------------------------------------------------------------
# Composite building
# -----------------------------------------------------------------------------

def reproject_scene_bands(
    scene: DownloadedSceneInfo,
    target_crs: CRS,
    target_transform: Affine,
    width: int,
    height: int,
    output_dtype: str,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    for band in INPUT_BANDS:
        logging.info("  Reprojecting band %s for scene %s", band, scene.item_id)
        out[band] = reproject_band_to_grid(
            src_path=scene.assets[band],
            target_crs=target_crs,
            target_transform=target_transform,
            target_width=width,
            target_height=height,
            resampling=Resampling.bilinear,
            dst_dtype=output_dtype,
        )

    return out


def build_composite_for_city(
    city_slug: str,
    aoi_geom_4326,
    scenes: Sequence[DownloadedSceneInfo],
    processed_city_root: Path,
    residual_threshold: float,
    valid_scl_values: Sequence[int],
) -> Dict[str, Any]:
    ensure_dir(processed_city_root)

    if not scenes:
        raise ValueError(f"No valid scenes available for city {city_slug}")

    reference_band_path = scenes[0].assets[SCENE_REF_BAND]
    output_dtype = get_band_dtype(reference_band_path)

    target_crs, target_transform, width, height = build_target_grid_from_aoi(
        reference_raster_path=reference_band_path,
        aoi_geom_4326=aoi_geom_4326,
        target_resolution=DEFAULT_TARGET_RESOLUTION,
    )

    aoi_mask = rasterize_aoi_mask(
        aoi_geom_4326=aoi_geom_4326,
        target_crs=target_crs,
        target_transform=target_transform,
        width=width,
        height=height,
    )

    aoi_pixels = int(aoi_mask.sum())
    if aoi_pixels == 0:
        raise ValueError(f"AOI mask for city {city_slug} contains zero pixels.")

    usability: Dict[str, SceneUsability] = {}
    scene_valid_masks: Dict[str, np.ndarray] = {}

    logging.info("Computing per-scene usability and cached valid masks...")
    for scene in scenes:
        metrics, valid_mask = compute_scene_usable_ratio(
            scene=scene,
            target_crs=target_crs,
            target_transform=target_transform,
            width=width,
            height=height,
            aoi_mask=aoi_mask,
            valid_scl_values=valid_scl_values,
        )
        usability[scene.item_id] = metrics
        scene_valid_masks[scene.item_id] = valid_mask

        logging.info(
            "Scene %s | usable_pixels=%d | usable_ratio=%.4f | overlap=%.3f | cloud=%.2f",
            scene.item_id,
            metrics.usable_pixels,
            metrics.usable_ratio,
            scene.overlap_ratio,
            scene.cloud_cover,
        )

    global_ranked = rank_scenes_globally(scenes, usability)

    logging.info("Global ranking (for base scene choice):")
    for idx, scene in enumerate(global_ranked, start=1):
        m = usability[scene.item_id]
        logging.info(
            "  [%d] %s | usable_ratio=%.4f | overlap=%.3f | cloud=%.2f | date=%s",
            idx,
            scene.item_id,
            m.usable_ratio,
            scene.overlap_ratio,
            scene.cloud_cover,
            scene.date,
        )

    composite = np.zeros((len(INPUT_BANDS), height, width), dtype=np.dtype(output_dtype))
    filled_mask = np.zeros((height, width), dtype=bool)
    contribution_map = np.zeros((height, width), dtype=np.uint16)

    scene_diagnostics: List[Dict[str, Any]] = []
    contributing_scene_ids: List[str] = []
    used_scene_order: List[str] = []

    base_scene = global_ranked[0]
    remaining_scene_dict = {scene.item_id: scene for scene in scenes if scene.item_id != base_scene.item_id}

    logging.info("Selected base scene: %s", base_scene.item_id)

    scene_order_counter = 1
    current_scene = base_scene

    while True:
        residual_before = int((aoi_mask & ~filled_mask).sum())
        residual_ratio_before = residual_before / aoi_pixels

        logging.info(
            "Applying scene order %d: %s | residual before=%d (%.4f)",
            scene_order_counter,
            current_scene.item_id,
            residual_before,
            residual_ratio_before,
        )

        valid_mask = scene_valid_masks[current_scene.item_id]
        newly_fillable = valid_mask & (~filled_mask)
        newly_fillable_pixels = int(newly_fillable.sum())

        logging.info(
            "Scene %s can newly fill %d pixels",
            current_scene.item_id,
            newly_fillable_pixels,
        )

        used_for_fill = newly_fillable_pixels > 0

        if used_for_fill:
            contributing_scene_ids.append(current_scene.item_id)
            used_scene_order.append(current_scene.item_id)

            reprojected_bands = reproject_scene_bands(
                scene=current_scene,
                target_crs=target_crs,
                target_transform=target_transform,
                width=width,
                height=height,
                output_dtype=output_dtype,
            )

            for band_idx, band_name in enumerate(INPUT_BANDS):
                composite[band_idx][newly_fillable] = reprojected_bands[band_name][newly_fillable]

            filled_mask[newly_fillable] = True
            contribution_map[newly_fillable] = scene_order_counter

        residual_after = int((aoi_mask & ~filled_mask).sum())
        residual_ratio_after = residual_after / aoi_pixels
        cumulative_filled_ratio_after = float((aoi_mask & filled_mask).sum() / aoi_pixels)

        scene_diagnostics.append(
            {
                "scene_order": scene_order_counter,
                "item_id": current_scene.item_id,
                "date": current_scene.date,
                "tile_id": current_scene.tile_id,
                "cloud_cover": current_scene.cloud_cover,
                "overlap_ratio": current_scene.overlap_ratio,
                "global_usable_ratio": usability[current_scene.item_id].usable_ratio,
                "new_pixels_filled": newly_fillable_pixels,
                "residual_pixels_before": residual_before,
                "residual_pixels_after": residual_after,
                "residual_ratio_after": residual_ratio_after,
                "cumulative_filled_ratio_after": cumulative_filled_ratio_after,
                "used_for_fill": used_for_fill,
            }
        )

        logging.info(
            "After scene %s | residual=%d (%.4f) | contributing_scenes=%d",
            current_scene.item_id,
            residual_after,
            residual_ratio_after,
            len(set(contributing_scene_ids)),
        )

        if (
            len(set(contributing_scene_ids)) >= MIN_SCENES_REQUIRED
            and residual_ratio_after <= residual_threshold
        ):
            logging.info(
                "Early stopping triggered: contributing_scenes=%d >= %d and residual_ratio=%.4f <= %.4f",
                len(set(contributing_scene_ids)),
                MIN_SCENES_REQUIRED,
                residual_ratio_after,
                residual_threshold,
            )
            break

        if not remaining_scene_dict:
            logging.info("No remaining scenes left for dynamic filling.")
            break

        remaining_scenes = list(remaining_scene_dict.values())
        dynamic_scores = dynamic_rank_remaining_scenes(
            remaining_scenes=remaining_scenes,
            scene_valid_masks=scene_valid_masks,
            filled_mask=filled_mask,
            aoi_mask=aoi_mask,
            usability=usability,
        )

        logging.info("Dynamic filler ranking for remaining scenes:")
        for idx, score in enumerate(dynamic_scores, start=1):
            logging.info(
                "  [%d] %s | fillable_pixels=%d | fillable_ratio_of_missing=%.4f | "
                "global_usable_ratio=%.4f | overlap=%.3f | cloud=%.2f",
                idx,
                score.item_id,
                score.fillable_pixels,
                score.fillable_ratio_of_missing,
                score.global_usable_ratio,
                score.overlap_ratio,
                score.cloud_cover,
            )

        next_score = dynamic_scores[0]
        if next_score.fillable_pixels <= 0:
            logging.info("Best remaining scene cannot fill any additional pixels. Stopping.")
            break

        current_scene = remaining_scene_dict.pop(next_score.item_id)
        scene_order_counter += 1

    unresolved_mask = aoi_mask & (~filled_mask)
    cloudmask = unresolved_mask.astype(np.uint8)

    composite_path = processed_city_root / "city_composite_2022.tif"
    cloudmask_path = processed_city_root / "city_cloudmask_2022.tif"
    contribution_path = processed_city_root / "city_scene_contribution_map.tif"
    diagnostics_path = processed_city_root / "selected_scenes_diagnostics.json"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "transform": target_transform,
        "crs": target_crs,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    composite_profile = profile.copy()
    composite_profile.update({"count": len(INPUT_BANDS), "dtype": output_dtype, "nodata": 0})

    with rasterio.open(composite_path, "w", **composite_profile) as dst:
        for i, band_name in enumerate(INPUT_BANDS, start=1):
            dst.write(composite[i - 1], i)
            dst.set_band_description(i, band_name)

    mask_profile = profile.copy()
    mask_profile.update({"count": 1, "dtype": "uint8", "nodata": 0})

    with rasterio.open(cloudmask_path, "w", **mask_profile) as dst:
        dst.write(cloudmask, 1)
        dst.set_band_description(1, "residual_missing_mask")

    contribution_profile = profile.copy()
    contribution_profile.update({"count": 1, "dtype": "uint16", "nodata": 0})

    with rasterio.open(contribution_path, "w", **contribution_profile) as dst:
        dst.write(contribution_map, 1)
        dst.set_band_description(1, "scene_order_contribution")

    diagnostics = {
        "city_slug": city_slug,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_grid": {
            "crs": str(target_crs),
            "width": width,
            "height": height,
            "transform": list(target_transform)[:6],
            "resolution_x": abs(target_transform.a),
            "resolution_y": abs(target_transform.e),
        },
        "parameters": {
            "input_bands": INPUT_BANDS,
            "valid_scl_values": list(valid_scl_values),
            "residual_threshold": residual_threshold,
            "target_resolution": DEFAULT_TARGET_RESOLUTION,
            "min_scenes_required": MIN_SCENES_REQUIRED,
            "filler_strategy": "dynamic_fillable_pixels",
        },
        "summary": {
            "num_input_scenes": len(scenes),
            "num_scenes_used_for_fill": len(set(contributing_scene_ids)),
            "aoi_pixels": aoi_pixels,
            "final_filled_pixels": int((aoi_mask & filled_mask).sum()),
            "final_unresolved_pixels": int(unresolved_mask.sum()),
            "final_unresolved_ratio": float(unresolved_mask.sum() / aoi_pixels),
        },
        "global_scene_ranking": [
            {
                "rank": idx,
                "item_id": scene.item_id,
                "date": scene.date,
                "tile_id": scene.tile_id,
                "cloud_cover": scene.cloud_cover,
                "overlap_ratio": scene.overlap_ratio,
                "usable_ratio": usability[scene.item_id].usable_ratio,
            }
            for idx, scene in enumerate(global_ranked, start=1)
        ],
        "scene_fill_diagnostics": scene_diagnostics,
        "used_scene_order": used_scene_order,
        "outputs": {
            "composite_path": str(composite_path),
            "cloudmask_path": str(cloudmask_path),
            "contribution_path": str(contribution_path),
        },
    }

    json_dump(diagnostics, diagnostics_path)

    summary_txt = [
        f"City slug: {city_slug}",
        f"AOI pixels: {aoi_pixels}",
        f"Final filled pixels: {int((aoi_mask & filled_mask).sum())}",
        f"Final unresolved pixels: {int(unresolved_mask.sum())}",
        f"Final unresolved ratio: {float(unresolved_mask.sum() / aoi_pixels):.4f}",
        f"Contributing scenes: {len(set(contributing_scene_ids))}",
        f"Used scene order: {used_scene_order}",
        "",
        "Global ranking:",
    ]

    for row in diagnostics["global_scene_ranking"]:
        summary_txt.append(
            f"{row['rank']}. {row['item_id']} | date={row['date']} | "
            f"usable_ratio={row['usable_ratio']:.4f} | overlap={row['overlap_ratio']:.3f} | "
            f"cloud={row['cloud_cover']:.2f}"
        )

    write_text(processed_city_root / "selected_scenes_diagnostics.txt", "\n".join(summary_txt))

    logging.info("Saved composite: %s", composite_path)
    logging.info("Saved cloud mask: %s", cloudmask_path)
    logging.info("Saved contribution map: %s", contribution_path)
    logging.info("Saved diagnostics: %s", diagnostics_path)

    return diagnostics


# -----------------------------------------------------------------------------
# City wrapper
# -----------------------------------------------------------------------------

def process_city(
    city_key: str,
    city_cfg: Dict[str, Any],
    raw_root: Path,
    processed_root: Path,
    residual_threshold: float,
    valid_scl_values: Sequence[int],
) -> Dict[str, Any]:
    city_slug = city_cfg["slug"]
    city_display = city_cfg["display_name"]

    logging.info("=" * 80)
    logging.info("Processing city composite v3: %s", city_display)
    logging.info("=" * 80)

    raw_city_root = raw_root / city_slug / "2022" / "s2_planetary_multiscene_12band"
    processed_city_root = processed_root / city_slug / "2022" / "s2_planetary_multiscene_12band_v3"

    summary_path = raw_city_root / "selected_scenes_summary.json"
    aoi_path = raw_city_root / f"{city_slug}_aoi.gpkg"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")

    logging.info("Using raw city root: %s", raw_city_root)
    logging.info("Loading AOI from: %s", aoi_path)

    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")

    aoi_gdf = aoi_gdf.to_crs(epsg=4326)
    aoi_geom_4326 = aoi_gdf.geometry.union_all()

    logging.info("Loading scenes from summary: %s", summary_path)
    scenes = load_downloaded_scenes_from_summary(summary_path)
    scenes = [remap_scene_to_raw_root(scene, raw_city_root) for scene in scenes]

    logging.info("Scenes listed in summary: %d", len(scenes))
    valid_scenes, skipped_scenes = filter_valid_scenes(scenes)
    logging.info("Valid scenes kept: %d", len(valid_scenes))
    logging.info("Incomplete scenes skipped: %d", len(skipped_scenes))

    if not valid_scenes:
        raise ValueError(
            f"No valid complete scenes remain for {city_display} under {raw_city_root}"
        )

    diagnostics = build_composite_for_city(
        city_slug=city_slug,
        aoi_geom_4326=aoi_geom_4326,
        scenes=valid_scenes,
        processed_city_root=processed_city_root,
        residual_threshold=residual_threshold,
        valid_scl_values=valid_scl_values,
    )

    diagnostics["skipped_incomplete_scenes"] = skipped_scenes
    json_dump(diagnostics, processed_city_root / "selected_scenes_diagnostics.json")

    logging.info("Finished city composite v3: %s", city_display)
    return diagnostics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cloud-reduced multi-scene Sentinel-2 composites from raw scenes stored on an external disk."
    )

    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="City key from CITY_CONFIGS, or 'all'. Example: salvador",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Base raw data root. This can be on an external SSD/HDD.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Base processed data root, usually on the internal disk.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=DEFAULT_RESIDUAL_THRESHOLD,
        help="Early-stop threshold on unresolved AOI ratio.",
    )
    parser.add_argument(
        "--valid-scl",
        nargs="+",
        type=int,
        default=DEFAULT_VALID_SCL,
        help="SCL values considered valid. Default: 4 5 6",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    args = parser.parse_args()

    if not args.raw_root.exists():
        raise FileNotFoundError(f"--raw-root does not exist: {args.raw_root}")
    if not (0.0 <= args.residual_threshold <= 1.0):
        raise ValueError("--residual-threshold must be between 0 and 1.")

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.city == "all":
        city_items = list(CITY_CONFIGS.items())
    else:
        if args.city not in CITY_CONFIGS:
            raise KeyError(
                f"Unknown city key '{args.city}'. Available: {list(CITY_CONFIGS.keys())}"
            )
        city_items = [(args.city, CITY_CONFIGS[args.city])]

    overall = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(args.raw_root),
        "processed_root": str(args.processed_root),
        "residual_threshold": args.residual_threshold,
        "valid_scl_values": args.valid_scl,
        "min_scenes_required": MIN_SCENES_REQUIRED,
        "filler_strategy": "dynamic_fillable_pixels",
        "cities": {},
    }

    for city_key, city_cfg in city_items:
        try:
            result = process_city(
                city_key=city_key,
                city_cfg=city_cfg,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                residual_threshold=args.residual_threshold,
                valid_scl_values=args.valid_scl,
            )
            overall["cities"][city_key] = {
                "status": "ok",
                "final_unresolved_ratio": result["summary"]["final_unresolved_ratio"],
                "num_input_scenes": result["summary"]["num_input_scenes"],
                "num_scenes_used_for_fill": result["summary"]["num_scenes_used_for_fill"],
                "used_scene_order": result.get("used_scene_order", []),
                "skipped_incomplete_scenes": result.get("skipped_incomplete_scenes", []),
                "output_dir": str(
                    args.processed_root
                    / city_cfg["slug"]
                    / "2022"
                    / "s2_planetary_multiscene_12band_v3"
                ),
            }
        except Exception as exc:
            logging.exception("City %s failed: %s", city_key, exc)
            overall["cities"][city_key] = {
                "status": "error",
                "error": str(exc),
            }

    overall_path = (
        args.processed_root
        / "five_cities_2022_s2_planetary_multiscene_12band_v3_composite_run_summary.json"
    )
    json_dump(overall, overall_path)

    logging.info("Composite v3 run finished.")
    logging.info("Global summary written to: %s", overall_path)


if __name__ == "__main__":
    main()