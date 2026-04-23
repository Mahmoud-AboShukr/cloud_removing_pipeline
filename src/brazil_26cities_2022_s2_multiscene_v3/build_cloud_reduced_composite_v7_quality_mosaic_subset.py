#!/usr/bin/env python3
"""
build_cloud_reduced_composite_v7_quality_mosaic_subset.py

V7 compositor for the hard cities:
- belem
- duque_de_caxias
- joao_pessoa
- maceio
- natal
- recife
- sao_luis

Method
------
V7 replaces per-band temporal median with a scene-consistent quality mosaic.

For each pixel:
1. Only scenes with strict-valid SCL are eligible
2. A per-pixel quality score is computed for each scene
3. The single best scene is selected
4. ALL 12 bands for that pixel come from the same winning scene

This avoids the spectral inconsistency of per-band medians.

Quality score
-------------
The score favors:
- strict valid pixels
- distance away from bad/cloud/shadow pixels
- lower scene cloud cover
- earlier rank among overlapping candidates

Main outputs
------------
For each city:
data/processed/<city>/2022/s2_planetary_multiscene_12band_v7/
    city_composite_2022.tif
    city_cloudmask_2022.tif
    city_scene_contribution_map.tif
    city_quality_score_max_2022.tif
    selected_scenes_diagnostics.json

Recommended rendering target
----------------------------
Use:
    city_composite_2022.tif

Example
-------
python3 src/brazil_26cities_2022_s2_multiscene_v3/build_cloud_reduced_composite_v7_quality_mosaic_subset.py \
  --city recife \
  --raw-root /media/HALLOPEAU/T7/raw_backup \
  --processed-root data/processed \
  --dilate-iters 5 \
  --log-level INFO
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
from scipy.ndimage import binary_dilation, distance_transform_edt

# -----------------------------------------------------------------------------
# Config import
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

CITY_CONFIGS_ALL: Dict[str, Dict[str, Any]] | None = None
_import_errors: List[str] = []

try:
    from cities_config import CITY_CONFIGS as _CFG  # type: ignore
    CITY_CONFIGS_ALL = _CFG
except Exception as exc:
    _import_errors.append(f"cities_config.CITY_CONFIGS failed: {exc}")

if CITY_CONFIGS_ALL is None:
    try:
        from cities_config_26 import CITY_CONFIGS as _CFG  # type: ignore
        CITY_CONFIGS_ALL = _CFG
    except Exception as exc:
        _import_errors.append(f"cities_config_26.CITY_CONFIGS failed: {exc}")

if CITY_CONFIGS_ALL is None:
    try:
        from cities_config_26 import CITY_CONFIGS_26 as _CFG  # type: ignore
        CITY_CONFIGS_ALL = _CFG
    except Exception as exc:
        _import_errors.append(f"cities_config_26.CITY_CONFIGS_26 failed: {exc}")

if CITY_CONFIGS_ALL is None:
    raise ImportError(
        "Could not import city configs.\n"
        f"Looked in: {SCRIPT_DIR}\n"
        "Errors:\n- " + "\n- ".join(_import_errors)
    )

TARGET_CITY_KEYS = [
    "belem",
    "duque_de_caxias",
    "joao_pessoa",
    "maceio",
    "natal",
    "recife",
    "sao_luis",
]

CITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    k: CITY_CONFIGS_ALL[k] for k in TARGET_CITY_KEYS if k in CITY_CONFIGS_ALL
}

missing_cfg = [k for k in TARGET_CITY_KEYS if k not in CITY_CONFIGS_ALL]
if missing_cfg:
    raise KeyError(f"Missing these target city configs: {missing_cfg}")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

INPUT_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]
SCL_BAND = "SCL"
SCENE_REF_BAND = "B02"
REQUIRED_ASSET_NAMES = INPUT_BANDS + [SCL_BAND]

STRICT_VALID_SCL = {4, 5, 6}
BAD_SCL = {0, 1, 2, 3, 7, 8, 9, 10, 11}

DEFAULT_TARGET_RESOLUTION = 10.0
DEFAULT_DILATE_ITERS = 5

RAW_COMPOSITE_NAME = "city_composite_2022.tif"
CLOUDMASK_NAME = "city_cloudmask_2022.tif"
CONTRIBUTION_NAME = "city_scene_contribution_map.tif"
QUALITY_NAME = "city_quality_score_max_2022.tif"
DIAGNOSTICS_NAME = "selected_scenes_diagnostics.json"

# score weights
DISTANCE_SCALE = 25.0
CLOUD_WEIGHT = 0.35
DISTANCE_WEIGHT = 1.00
RANK_WEIGHT = 0.15

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

# -----------------------------------------------------------------------------
# Raw-folder discovery
# -----------------------------------------------------------------------------

def find_existing_raw_city_root(raw_root: Path, city_slug: str) -> Path:
    city_root = raw_root / city_slug
    if not city_root.exists():
        raise FileNotFoundError(f"Top-level city folder not found: {city_root}")

    candidates = [
        city_root / "2022" / "s2_planetary_multiscene_12band_v3",
        city_root / "2022" / "s2_planetary_multiscene_12band",
        city_root / "2022" / "s2_planetary_multiscene",
        city_root / "s2_planetary_multiscene_12band_v3",
        city_root / "s2_planetary_multiscene_12band",
        city_root,
    ]

    for path in candidates:
        if path.exists() and path.is_dir() and (path / "selected_scenes_summary.json").exists():
            return path

    for sub in sorted(city_root.rglob("*")):
        if sub.is_dir() and (sub / "selected_scenes_summary.json").exists():
            return sub

    raise FileNotFoundError(f"Could not locate raw product folder for city '{city_slug}' under {city_root}")

def find_aoi_file(raw_city_root: Path, city_slug: str) -> Path:
    preferred = raw_city_root / f"{city_slug}_aoi.gpkg"
    if preferred.exists():
        return preferred

    gpkg_files = sorted(raw_city_root.glob("*_aoi.gpkg"))
    if gpkg_files:
        return gpkg_files[0]

    gpkg_files = sorted(raw_city_root.glob("*.gpkg"))
    if gpkg_files:
        return gpkg_files[0]

    raise FileNotFoundError(f"No AOI .gpkg found in {raw_city_root}")

# -----------------------------------------------------------------------------
# Scene loading
# -----------------------------------------------------------------------------

def _pick_first_present(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default

def _rebuild_scene_assets(scene_dir: Path) -> Dict[str, Path]:
    return {band: scene_dir / f"{band}.tif" for band in REQUIRED_ASSET_NAMES}

def load_downloaded_scenes_from_summary(summary_path: Path) -> List[DownloadedSceneInfo]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = load_json(summary_path)
    rows = summary.get("downloaded_scenes", [])
    if not rows:
        raise ValueError(f"No downloaded scenes found in {summary_path}")

    base_dir = summary_path.parent
    scenes: List[DownloadedSceneInfo] = []

    for row in rows:
        item_id = _pick_first_present(row, ["item_id"])
        if not item_id:
            continue

        scene_dir_raw = _pick_first_present(row, ["scene_dir"], default=None)
        if scene_dir_raw:
            scene_dir_candidate = Path(scene_dir_raw)
            scene_dir = base_dir / "scenes" / scene_dir_candidate.name
        else:
            scene_dir = base_dir / "scenes" / item_id

        scenes.append(
            DownloadedSceneInfo(
                item_id=item_id,
                datetime_utc=str(_pick_first_present(row, ["datetime_utc", "datetime"], "")),
                date=str(_pick_first_present(row, ["date"], "")),
                tile_id=str(_pick_first_present(row, ["tile_id", "mgrs_tile", "tile"], "")),
                cloud_cover=float(_pick_first_present(row, ["cloud_cover", "eo:cloud_cover"], 9999.0)),
                overlap_ratio=float(_pick_first_present(row, ["overlap_ratio"], 0.0)),
                scene_dir=scene_dir,
                assets=_rebuild_scene_assets(scene_dir),
            )
        )

    return scenes

def filter_valid_complete_scenes(
    scenes: Sequence[DownloadedSceneInfo],
) -> Tuple[List[DownloadedSceneInfo], List[Dict[str, Any]]]:
    valid: List[DownloadedSceneInfo] = []
    invalid_report: List[Dict[str, Any]] = []

    for scene in scenes:
        missing_files = [k for k, p in scene.assets.items() if not p.exists()]
        if missing_files:
            invalid_report.append(
                {
                    "item_id": scene.item_id,
                    "scene_dir": str(scene.scene_dir),
                    "missing_files": missing_files,
                }
            )
            logging.warning("Skipping incomplete scene %s | missing=%s", scene.item_id, missing_files)
            continue
        valid.append(scene)

    return valid, invalid_report

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
            target_crs, width, height, abs(transform.a), abs(transform.e)
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
    return mask.astype(bool)

# -----------------------------------------------------------------------------
# Raster helpers
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
# SCL masks and score maps
# -----------------------------------------------------------------------------

def make_scene_masks_and_score(
    scl_array: np.ndarray,
    aoi_mask: np.ndarray,
    dilate_iters: int,
    cloud_cover: float,
    rank_index_1based: int,
    num_scenes: int,
) -> Dict[str, np.ndarray]:
    bad_seed = np.isin(scl_array, np.array(sorted(BAD_SCL), dtype=scl_array.dtype))
    bad_dilated = binary_dilation(bad_seed, iterations=dilate_iters) if dilate_iters > 0 else bad_seed

    valid = np.isin(scl_array, np.array(sorted(STRICT_VALID_SCL), dtype=scl_array.dtype))
    valid &= aoi_mask
    valid &= ~bad_dilated

    # distance from the dilated bad mask
    # we compute distance to nearest bad pixel; larger is better
    dist = distance_transform_edt(~bad_dilated).astype(np.float32)
    dist_score = 1.0 - np.exp(-dist / DISTANCE_SCALE)

    cloud_penalty = CLOUD_WEIGHT * (float(cloud_cover) / 100.0)

    # better ranked scenes get a small bonus
    # rank 1 => bonus near RANK_WEIGHT, last rank => near 0
    if num_scenes > 1:
        rank_bonus = RANK_WEIGHT * (1.0 - (rank_index_1based - 1) / (num_scenes - 1))
    else:
        rank_bonus = RANK_WEIGHT

    score = DISTANCE_WEIGHT * dist_score + rank_bonus - cloud_penalty
    score = np.where(valid, score, -np.inf).astype(np.float32)

    return {
        "valid": valid,
        "bad_seed": bad_seed & aoi_mask,
        "bad_dilated": bad_dilated & aoi_mask,
        "distance_to_bad": dist,
        "score": score,
    }

# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------

def rank_scenes(scenes: Sequence[DownloadedSceneInfo]) -> List[DownloadedSceneInfo]:
    return sorted(
        scenes,
        key=lambda s: (
            s.overlap_ratio,
            -s.cloud_cover,
            s.date,
        ),
        reverse=True,
    )

# -----------------------------------------------------------------------------
# Quality mosaic
# -----------------------------------------------------------------------------

def build_quality_mosaic(
    ranked_scenes: Sequence[DownloadedSceneInfo],
    scene_maps: Dict[str, Dict[str, np.ndarray]],
    target_crs: CRS,
    target_transform: Affine,
    width: int,
    height: int,
    output_dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    composite : (bands, H, W)
        Winner-takes-all scene-consistent mosaic
    best_scene_index : (H, W)
        1-based scene order that won at each pixel
    best_score : (H, W)
        Max quality score at each pixel
    valid_mask : (H, W)
        Pixels where at least one valid scene existed
    """
    n_bands = len(INPUT_BANDS)
    composite = np.zeros((n_bands, height, width), dtype=np.dtype(output_dtype))

    best_score = np.full((height, width), -np.inf, dtype=np.float32)
    best_scene_index = np.zeros((height, width), dtype=np.uint16)

    # choose best scene per pixel
    for s_idx, scene in enumerate(ranked_scenes, start=1):
        score = scene_maps[scene.item_id]["score"]
        better = score > best_score
        if not np.any(better):
            continue

        best_score[better] = score[better]
        best_scene_index[better] = s_idx

    valid_mask = best_scene_index > 0

    # materialize full-spectrum pixels from winning scene
    for s_idx, scene in enumerate(ranked_scenes, start=1):
        winner_mask = best_scene_index == s_idx
        nwin = int(winner_mask.sum())
        if nwin == 0:
            continue

        logging.info("[MOSAIC] scene %s wins %d pixels", scene.item_id, nwin)

        for b_idx, band in enumerate(INPUT_BANDS):
            arr = reproject_band_to_grid(
                src_path=scene.assets[band],
                target_crs=target_crs,
                target_transform=target_transform,
                target_width=width,
                target_height=height,
                resampling=Resampling.bilinear,
                dst_dtype=output_dtype,
            )
            composite[b_idx][winner_mask] = arr[winner_mask]

    best_score_out = np.where(np.isfinite(best_score), best_score, 0.0).astype(np.float32)
    return composite, best_scene_index, best_score_out, valid_mask

# -----------------------------------------------------------------------------
# Composite building
# -----------------------------------------------------------------------------

def build_composite_for_city(
    city_slug: str,
    aoi_geom_4326,
    scenes: Sequence[DownloadedSceneInfo],
    processed_city_root: Path,
    dilate_iters: int,
) -> Dict[str, Any]:
    ensure_dir(processed_city_root)

    if not scenes:
        raise ValueError(f"No scenes available for city {city_slug}")

    ranked_scenes = rank_scenes(scenes)
    logging.info("Using %d complete scenes.", len(ranked_scenes))
    for i, s in enumerate(ranked_scenes, start=1):
        logging.info(
            "  [%d] %s | overlap=%.3f | cloud=%.2f | date=%s | tile=%s",
            i, s.item_id, s.overlap_ratio, s.cloud_cover, s.date, s.tile_id
        )

    reference_band_path = ranked_scenes[0].assets[SCENE_REF_BAND]
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
    logging.info("AOI pixels: %d", aoi_pixels)

    scene_maps: Dict[str, Dict[str, np.ndarray]] = {}
    scene_stats: List[Dict[str, Any]] = []
    n_scenes = len(ranked_scenes)

    logging.info("Computing strict valid masks and per-scene quality scores...")
    for rank_idx, scene in enumerate(ranked_scenes, start=1):
        scl = reproject_band_to_grid(
            src_path=scene.assets[SCL_BAND],
            target_crs=target_crs,
            target_transform=target_transform,
            target_width=width,
            target_height=height,
            resampling=Resampling.nearest,
            dst_dtype="uint8",
        )

        maps = make_scene_masks_and_score(
            scl_array=scl,
            aoi_mask=aoi_mask,
            dilate_iters=dilate_iters,
            cloud_cover=scene.cloud_cover,
            rank_index_1based=rank_idx,
            num_scenes=n_scenes,
        )
        scene_maps[scene.item_id] = maps

        dist_valid = maps["distance_to_bad"][maps["valid"]]
        scene_stats.append(
            {
                "item_id": scene.item_id,
                "date": scene.date,
                "tile_id": scene.tile_id,
                "cloud_cover": scene.cloud_cover,
                "overlap_ratio": scene.overlap_ratio,
                "rank_index": rank_idx,
                "valid_pixels": int(maps["valid"].sum()),
                "bad_seed_pixels": int(maps["bad_seed"].sum()),
                "bad_dilated_pixels": int(maps["bad_dilated"].sum()),
                "mean_distance_to_bad_over_valid": float(dist_valid.mean()) if dist_valid.size else 0.0,
                "median_distance_to_bad_over_valid": float(np.median(dist_valid)) if dist_valid.size else 0.0,
                "max_score_over_valid": float(np.max(maps["score"][maps["valid"]])) if np.any(maps["valid"]) else float("-inf"),
            }
        )

    logging.info("Building scene-consistent quality mosaic...")
    composite, best_scene_index, best_score, valid_mask = build_quality_mosaic(
        ranked_scenes=ranked_scenes,
        scene_maps=scene_maps,
        target_crs=target_crs,
        target_transform=target_transform,
        width=width,
        height=height,
        output_dtype=output_dtype,
    )

    unresolved_mask = aoi_mask & (~valid_mask)
    unresolved_pixels = int(unresolved_mask.sum())
    unresolved_ratio = unresolved_pixels / aoi_pixels if aoi_pixels > 0 else 0.0
    logging.info("Final unresolved pixels: %d (%.4f)", unresolved_pixels, unresolved_ratio)

    composite_path = processed_city_root / RAW_COMPOSITE_NAME
    cloudmask_path = processed_city_root / CLOUDMASK_NAME
    contribution_path = processed_city_root / CONTRIBUTION_NAME
    quality_path = processed_city_root / QUALITY_NAME
    diagnostics_path = processed_city_root / DIAGNOSTICS_NAME

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
        for i, band in enumerate(INPUT_BANDS, start=1):
            dst.write(composite[i - 1], i)
            dst.set_band_description(i, band)

    mask_profile = profile.copy()
    mask_profile.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(cloudmask_path, "w", **mask_profile) as dst:
        dst.write(unresolved_mask.astype(np.uint8), 1)
        dst.set_band_description(1, "residual_missing_mask")

    contrib_profile = profile.copy()
    contrib_profile.update({"count": 1, "dtype": "uint16", "nodata": 0})
    with rasterio.open(contribution_path, "w", **contrib_profile) as dst:
        dst.write(best_scene_index, 1)
        dst.set_band_description(1, "winning_scene_order")

    quality_profile = profile.copy()
    quality_profile.update({"count": 1, "dtype": "float32", "nodata": 0})
    with rasterio.open(quality_path, "w", **quality_profile) as dst:
        dst.write(best_score, 1)
        dst.set_band_description(1, "winning_quality_score")

    winning_counts = {
        ranked_scenes[i - 1].item_id: int((best_scene_index == i).sum())
        for i in range(1, len(ranked_scenes) + 1)
    }

    valid_scores = best_score[valid_mask]
    diagnostics = {
        "city_slug": city_slug,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "bands": INPUT_BANDS,
            "strict_valid_scl": sorted(STRICT_VALID_SCL),
            "bad_scl": sorted(BAD_SCL),
            "dilate_iters": int(dilate_iters),
            "distance_scale": float(DISTANCE_SCALE),
            "cloud_weight": float(CLOUD_WEIGHT),
            "distance_weight": float(DISTANCE_WEIGHT),
            "rank_weight": float(RANK_WEIGHT),
            "strategy": "strict_masking_quality_mosaic_scene_consistent",
        },
        "summary": {
            "num_input_scenes": len(ranked_scenes),
            "aoi_pixels": aoi_pixels,
            "valid_pixels": int(valid_mask.sum()),
            "unresolved_pixels": unresolved_pixels,
            "unresolved_ratio": unresolved_ratio,
            "winning_score_mean": float(valid_scores.mean()) if valid_scores.size else 0.0,
            "winning_score_median": float(np.median(valid_scores)) if valid_scores.size else 0.0,
            "winning_score_min": float(valid_scores.min()) if valid_scores.size else 0.0,
            "winning_score_max": float(valid_scores.max()) if valid_scores.size else 0.0,
        },
        "scene_ranking": [
            {
                "rank": i,
                "item_id": s.item_id,
                "date": s.date,
                "tile_id": s.tile_id,
                "cloud_cover": s.cloud_cover,
                "overlap_ratio": s.overlap_ratio,
            }
            for i, s in enumerate(ranked_scenes, start=1)
        ],
        "scene_mask_stats": scene_stats,
        "winning_scene_pixel_counts": winning_counts,
        "outputs": {
            "composite_path": str(composite_path),
            "cloudmask_path": str(cloudmask_path),
            "contribution_map_path": str(contribution_path),
            "quality_score_path": str(quality_path),
        },
    }

    json_dump(diagnostics, diagnostics_path)

    logging.info("Saved composite: %s", composite_path)
    logging.info("Saved cloud mask: %s", cloudmask_path)
    logging.info("Saved contribution map: %s", contribution_path)
    logging.info("Saved quality score map: %s", quality_path)
    logging.info("Saved diagnostics: %s", diagnostics_path)

    return diagnostics

# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------

def process_city(
    city_key: str,
    city_cfg: Dict[str, Any],
    raw_root: Path,
    processed_root: Path,
    dilate_iters: int,
) -> Dict[str, Any]:
    city_slug = city_cfg["slug"]
    city_display = city_cfg.get("display_name", city_slug)

    logging.info("=" * 80)
    logging.info("Processing city composite v7: %s", city_display)
    logging.info("=" * 80)

    raw_city_root = find_existing_raw_city_root(raw_root, city_slug)
    processed_city_root = processed_root / city_slug / "2022" / "s2_planetary_multiscene_12band_v7"

    summary_path = raw_city_root / "selected_scenes_summary.json"
    aoi_path = find_aoi_file(raw_city_root, city_slug)

    logging.info("Using raw city root: %s", raw_city_root)
    logging.info("Loading AOI: %s", aoi_path)

    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")

    aoi_gdf = aoi_gdf.to_crs(epsg=4326)
    aoi_geom_4326 = aoi_gdf.geometry.union_all()

    scenes_all = load_downloaded_scenes_from_summary(summary_path)
    scenes_valid, invalid_report = filter_valid_complete_scenes(scenes_all)

    logging.info("Complete scenes: %d / %d", len(scenes_valid), len(scenes_all))
    if invalid_report:
        json_dump(
            {
                "city_slug": city_slug,
                "raw_city_root": str(raw_city_root),
                "invalid_scenes": invalid_report,
            },
            processed_city_root / "skipped_incomplete_scenes.json",
        )

    if not scenes_valid:
        raise ValueError(f"No complete scenes available for city {city_slug}")

    return build_composite_for_city(
        city_slug=city_slug,
        aoi_geom_4326=aoi_geom_4326,
        scenes=scenes_valid,
        processed_city_root=processed_city_root,
        dilate_iters=dilate_iters,
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V7 quality-mosaic compositor for selected hard cities only."
    )
    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="One of: all, belem, duque_de_caxias, joao_pessoa, maceio, natal, recife, sao_luis",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Base raw root, e.g. /media/HALLOPEAU/T7/raw_backup",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Base processed root.",
    )
    parser.add_argument(
        "--dilate-iters",
        type=int,
        default=DEFAULT_DILATE_ITERS,
        help="Binary dilation iterations around bad SCL classes. Default: 5",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if not args.raw_root.exists():
        raise FileNotFoundError(f"--raw-root does not exist: {args.raw_root}")

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
        "target_city_keys": list(CITY_CONFIGS.keys()),
        "raw_root": str(args.raw_root),
        "processed_root": str(args.processed_root),
        "parameters": {
            "strict_valid_scl": sorted(STRICT_VALID_SCL),
            "bad_scl": sorted(BAD_SCL),
            "dilate_iters": int(args.dilate_iters),
            "distance_scale": float(DISTANCE_SCALE),
            "cloud_weight": float(CLOUD_WEIGHT),
            "distance_weight": float(DISTANCE_WEIGHT),
            "rank_weight": float(RANK_WEIGHT),
        },
        "cities": {},
    }

    for city_key, city_cfg in city_items:
        try:
            result = process_city(
                city_key=city_key,
                city_cfg=city_cfg,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
                dilate_iters=args.dilate_iters,
            )
            overall["cities"][city_key] = {
                "status": "ok",
                "unresolved_ratio": result["summary"]["unresolved_ratio"],
                "output_dir": str(
                    args.processed_root / city_cfg["slug"] / "2022" / "s2_planetary_multiscene_12band_v7"
                ),
            }
        except Exception as exc:
            logging.exception("City %s failed: %s", city_key, exc)
            overall["cities"][city_key] = {
                "status": "error",
                "error": str(exc),
            }

    summary_path = args.processed_root / "brazil_hardcities_2022_s2_multiscene_v7_quality_mosaic_run_summary.json"
    json_dump(overall, summary_path)
    logging.info("Run finished. Summary: %s", summary_path)

if __name__ == "__main__":
    main()