#!/usr/bin/env python3
"""
build_cloud_reduced_composite_v6_subset.py

V6 compositor for the hard cities:
- belem
- duque_de_caxias
- joao_pessoa
- maceio
- natal
- recife
- sao_luis

Method
------
1. Strict masking only
   Valid SCL classes: {4, 5, 6}
   Bad SCL classes: {0, 1, 2, 3, 7, 8, 9, 10, 11}
2. Dilation of bad SCL mask to suppress cloud edges / halos
3. Temporal median compositing across all valid observations
4. Small-hole filling only
   - fills only connected unresolved components up to a size threshold
   - large unresolved regions remain masked
5. Better diagnostics
   - per-scene valid coverage
   - per-pixel observation count summary
   - hole-size summary
   - filled-hole statistics

Outputs
-------
For each city:
data/processed/<city>/2022/s2_planetary_multiscene_12band_v6/
    city_composite_2022.tif
    city_composite_2022_smallholes_filled.tif
    city_cloudmask_2022.tif
    city_small_holes_mask_2022.tif
    city_observation_count_2022.tif
    selected_scenes_diagnostics.json

Recommended rendering target
----------------------------
Use:
    city_composite_2022_smallholes_filled.tif

Example
-------
python3 src/brazil_26cities_2022_s2_multiscene_v3/build_cloud_reduced_composite_v6_subset.py \
  --city recife \
  --raw-root /media/HALLOPEAU/T7/raw_backup \
  --processed-root data/processed \
  --dilate-iters 5 \
  --max-hole-pixels 1500 \
  --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import sys
import logging
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
from scipy.ndimage import binary_dilation, label, distance_transform_edt

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
DEFAULT_MAX_HOLE_PIXELS = 1500

RAW_COMPOSITE_NAME = "city_composite_2022.tif"
FILLED_COMPOSITE_NAME = "city_composite_2022_smallholes_filled.tif"
CLOUDMASK_NAME = "city_cloudmask_2022.tif"
SMALL_HOLES_MASK_NAME = "city_small_holes_mask_2022.tif"
OBS_COUNT_NAME = "city_observation_count_2022.tif"
DIAGNOSTICS_NAME = "selected_scenes_diagnostics.json"

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
# SCL masks
# -----------------------------------------------------------------------------

def make_strict_valid_mask(
    scl_array: np.ndarray,
    aoi_mask: np.ndarray,
    dilate_iters: int,
) -> Dict[str, np.ndarray]:
    bad_seed = np.isin(scl_array, np.array(sorted(BAD_SCL), dtype=scl_array.dtype))
    bad_dilated = binary_dilation(bad_seed, iterations=dilate_iters) if dilate_iters > 0 else bad_seed

    valid = np.isin(scl_array, np.array(sorted(STRICT_VALID_SCL), dtype=scl_array.dtype))
    valid &= aoi_mask
    valid &= ~bad_dilated

    return {
        "valid": valid,
        "bad_seed": bad_seed & aoi_mask,
        "bad_dilated": bad_dilated & aoi_mask,
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
# Median compositing
# -----------------------------------------------------------------------------

def temporal_median_composite(
    ranked_scenes: Sequence[DownloadedSceneInfo],
    scene_masks: Dict[str, Dict[str, np.ndarray]],
    target_crs: CRS,
    target_transform: Affine,
    width: int,
    height: int,
    output_dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    composite : (bands, H, W)
        Median composite, unresolved pixels remain 0
    obs_count : (H, W)
        Number of valid observations contributing per pixel
    first_scene_contrib : (H, W)
        Order index of the first ranked scene that contributed a valid obs at that pixel
        purely diagnostic
    """
    n_bands = len(INPUT_BANDS)
    n_scenes = len(ranked_scenes)

    stack = np.full((n_bands, n_scenes, height, width), np.nan, dtype=np.float32)
    obs_count = np.zeros((height, width), dtype=np.uint16)
    first_scene_contrib = np.zeros((height, width), dtype=np.uint16)
    seen_once = np.zeros((height, width), dtype=bool)

    for s_idx, scene in enumerate(ranked_scenes):
        valid = scene_masks[scene.item_id]["valid"]
        valid_n = int(valid.sum())
        logging.info("[STACK] scene %s valid pixels=%d", scene.item_id, valid_n)

        if valid_n == 0:
            continue

        for b_idx, band in enumerate(INPUT_BANDS):
            arr = reproject_band_to_grid(
                src_path=scene.assets[band],
                target_crs=target_crs,
                target_transform=target_transform,
                target_width=width,
                target_height=height,
                resampling=Resampling.bilinear,
                dst_dtype="float32",
            )
            tmp = np.full((height, width), np.nan, dtype=np.float32)
            tmp[valid] = arr[valid]
            stack[b_idx, s_idx] = tmp

        obs_count += valid.astype(np.uint16)

        newly_seen = valid & (~seen_once)
        first_scene_contrib[newly_seen] = s_idx + 1
        seen_once[newly_seen] = True

    composite_float = np.nanmedian(stack, axis=1)
    composite_float = np.where(np.isnan(composite_float), 0.0, composite_float)

    composite = np.round(composite_float).astype(np.dtype(output_dtype))
    return composite, obs_count, first_scene_contrib

# -----------------------------------------------------------------------------
# Small-hole filling only
# -----------------------------------------------------------------------------

def compute_small_holes_mask(
    unresolved_mask: np.ndarray,
    max_hole_pixels: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Identify unresolved connected components small enough to fill.
    """
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, ncomp = label(unresolved_mask, structure=structure)

    small_mask = np.zeros_like(unresolved_mask, dtype=bool)
    component_sizes: List[int] = []
    small_component_sizes: List[int] = []
    large_component_sizes: List[int] = []

    if ncomp > 0:
        counts = np.bincount(labeled.ravel())
        # counts[0] is background
        for comp_id in range(1, ncomp + 1):
            size = int(counts[comp_id])
            component_sizes.append(size)
            if size <= max_hole_pixels:
                small_mask |= (labeled == comp_id)
                small_component_sizes.append(size)
            else:
                large_component_sizes.append(size)

    diag = {
        "num_hole_components": int(ncomp),
        "max_hole_pixels_threshold": int(max_hole_pixels),
        "all_component_sizes": component_sizes,
        "small_component_sizes": small_component_sizes,
        "large_component_sizes": large_component_sizes,
        "num_small_components": len(small_component_sizes),
        "num_large_components": len(large_component_sizes),
        "small_hole_pixels": int(small_mask.sum()),
        "large_hole_pixels": int(unresolved_mask.sum() - small_mask.sum()),
    }
    return small_mask, diag

def fill_small_holes_nearest_per_band(
    composite: np.ndarray,
    valid_mask: np.ndarray,
    small_holes_mask: np.ndarray,
) -> np.ndarray:
    """
    Fill only the pixels in small_holes_mask using nearest valid pixel values.
    """
    filled = composite.copy()

    if not np.any(small_holes_mask):
        return filled

    if not np.any(valid_mask):
        logging.warning("No valid pixels exist. Small-hole fill skipped.")
        return filled

    _, indices = distance_transform_edt(~valid_mask, return_indices=True)
    rr = indices[0]
    cc = indices[1]

    for b in range(filled.shape[0]):
        band = filled[b]
        band[small_holes_mask] = band[rr[small_holes_mask], cc[small_holes_mask]]

    return filled

# -----------------------------------------------------------------------------
# Composite building
# -----------------------------------------------------------------------------

def build_composite_for_city(
    city_slug: str,
    aoi_geom_4326,
    scenes: Sequence[DownloadedSceneInfo],
    processed_city_root: Path,
    dilate_iters: int,
    max_hole_pixels: int,
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

    scene_masks: Dict[str, Dict[str, np.ndarray]] = {}
    scene_stats: List[Dict[str, Any]] = []

    logging.info("Computing strict valid masks...")
    for scene in ranked_scenes:
        scl = reproject_band_to_grid(
            src_path=scene.assets[SCL_BAND],
            target_crs=target_crs,
            target_transform=target_transform,
            target_width=width,
            target_height=height,
            resampling=Resampling.nearest,
            dst_dtype="uint8",
        )

        masks = make_strict_valid_mask(
            scl_array=scl,
            aoi_mask=aoi_mask,
            dilate_iters=dilate_iters,
        )
        scene_masks[scene.item_id] = masks

        scene_stats.append(
            {
                "item_id": scene.item_id,
                "date": scene.date,
                "tile_id": scene.tile_id,
                "cloud_cover": scene.cloud_cover,
                "overlap_ratio": scene.overlap_ratio,
                "valid_pixels": int(masks["valid"].sum()),
                "bad_seed_pixels": int(masks["bad_seed"].sum()),
                "bad_dilated_pixels": int(masks["bad_dilated"].sum()),
            }
        )

    logging.info("Building temporal median composite...")
    composite, obs_count, first_scene_contrib = temporal_median_composite(
        ranked_scenes=ranked_scenes,
        scene_masks=scene_masks,
        target_crs=target_crs,
        target_transform=target_transform,
        width=width,
        height=height,
        output_dtype=output_dtype,
    )

    valid_mask = (obs_count > 0) & aoi_mask
    unresolved_mask = aoi_mask & (~valid_mask)
    unresolved_pixels = int(unresolved_mask.sum())
    unresolved_ratio = unresolved_pixels / aoi_pixels if aoi_pixels > 0 else 0.0
    logging.info("Final unresolved pixels (strict median): %d (%.4f)", unresolved_pixels, unresolved_ratio)

    small_holes_mask, hole_diag = compute_small_holes_mask(
        unresolved_mask=unresolved_mask,
        max_hole_pixels=max_hole_pixels,
    )
    logging.info(
        "Small-hole fill eligible pixels: %d | large unresolved pixels kept masked: %d",
        int(small_holes_mask.sum()),
        int(unresolved_mask.sum() - small_holes_mask.sum()),
    )

    filled_composite = fill_small_holes_nearest_per_band(
        composite=composite,
        valid_mask=valid_mask,
        small_holes_mask=small_holes_mask,
    )

    raw_composite_path = processed_city_root / RAW_COMPOSITE_NAME
    filled_composite_path = processed_city_root / FILLED_COMPOSITE_NAME
    cloudmask_path = processed_city_root / CLOUDMASK_NAME
    small_holes_mask_path = processed_city_root / SMALL_HOLES_MASK_NAME
    obs_count_path = processed_city_root / OBS_COUNT_NAME
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

    with rasterio.open(raw_composite_path, "w", **composite_profile) as dst:
        for i, band in enumerate(INPUT_BANDS, start=1):
            dst.write(composite[i - 1], i)
            dst.set_band_description(i, band)

    with rasterio.open(filled_composite_path, "w", **composite_profile) as dst:
        for i, band in enumerate(INPUT_BANDS, start=1):
            dst.write(filled_composite[i - 1], i)
            dst.set_band_description(i, band)

    mask_profile = profile.copy()
    mask_profile.update({"count": 1, "dtype": "uint8", "nodata": 0})

    with rasterio.open(cloudmask_path, "w", **mask_profile) as dst:
        dst.write(unresolved_mask.astype(np.uint8), 1)
        dst.set_band_description(1, "residual_missing_mask")

    with rasterio.open(small_holes_mask_path, "w", **mask_profile) as dst:
        dst.write(small_holes_mask.astype(np.uint8), 1)
        dst.set_band_description(1, "small_holes_mask")

    obs_profile = profile.copy()
    obs_profile.update({"count": 1, "dtype": "uint16", "nodata": 0})
    with rasterio.open(obs_count_path, "w", **obs_profile) as dst:
        dst.write(obs_count, 1)
        dst.set_band_description(1, "valid_observation_count")

    contrib_profile = profile.copy()
    contrib_profile.update({"count": 1, "dtype": "uint16", "nodata": 0})
    contrib_path = processed_city_root / "city_scene_contribution_map.tif"
    with rasterio.open(contrib_path, "w", **contrib_profile) as dst:
        dst.write(first_scene_contrib, 1)
        dst.set_band_description(1, "first_valid_scene_order")

    obs_values = obs_count[aoi_mask]
    obs_nonzero = obs_values[obs_values > 0]

    diagnostics = {
        "city_slug": city_slug,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "bands": INPUT_BANDS,
            "strict_valid_scl": sorted(STRICT_VALID_SCL),
            "bad_scl": sorted(BAD_SCL),
            "dilate_iters": int(dilate_iters),
            "max_hole_pixels": int(max_hole_pixels),
            "strategy": "strict_masking_temporal_median_small_hole_fill_only",
        },
        "summary": {
            "num_input_scenes": len(ranked_scenes),
            "aoi_pixels": aoi_pixels,
            "valid_pixels_strict": int(valid_mask.sum()),
            "unresolved_pixels_strict": unresolved_pixels,
            "unresolved_ratio_strict": unresolved_ratio,
            "small_hole_pixels_filled": int(small_holes_mask.sum()),
            "large_unresolved_pixels_kept_masked": int(unresolved_mask.sum() - small_holes_mask.sum()),
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
        "observation_count_stats": {
            "min_over_aoi": int(obs_values.min()) if obs_values.size else 0,
            "max_over_aoi": int(obs_values.max()) if obs_values.size else 0,
            "mean_over_aoi": float(obs_values.mean()) if obs_values.size else 0.0,
            "mean_over_valid_pixels": float(obs_nonzero.mean()) if obs_nonzero.size else 0.0,
            "median_over_valid_pixels": float(np.median(obs_nonzero)) if obs_nonzero.size else 0.0,
            "pixels_with_1_obs": int((obs_count == 1).sum()),
            "pixels_with_2plus_obs": int((obs_count >= 2).sum()),
            "pixels_with_3plus_obs": int((obs_count >= 3).sum()),
            "pixels_with_4plus_obs": int((obs_count >= 4).sum()),
        },
        "hole_stats": hole_diag,
        "outputs": {
            "raw_composite_path": str(raw_composite_path),
            "filled_composite_path": str(filled_composite_path),
            "cloudmask_path": str(cloudmask_path),
            "small_holes_mask_path": str(small_holes_mask_path),
            "obs_count_path": str(obs_count_path),
            "contribution_map_path": str(contrib_path),
        },
    }

    json_dump(diagnostics, diagnostics_path)

    logging.info("Saved raw median composite: %s", raw_composite_path)
    logging.info("Saved small-hole-filled preview composite: %s", filled_composite_path)
    logging.info("Saved cloud mask: %s", cloudmask_path)
    logging.info("Saved small holes mask: %s", small_holes_mask_path)
    logging.info("Saved observation count: %s", obs_count_path)
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
    max_hole_pixels: int,
) -> Dict[str, Any]:
    city_slug = city_cfg["slug"]
    city_display = city_cfg.get("display_name", city_slug)

    logging.info("=" * 80)
    logging.info("Processing city composite v6: %s", city_display)
    logging.info("=" * 80)

    raw_city_root = find_existing_raw_city_root(raw_root, city_slug)
    processed_city_root = processed_root / city_slug / "2022" / "s2_planetary_multiscene_12band_v6"

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
        max_hole_pixels=max_hole_pixels,
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V6 compositor for selected hard cities only."
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
        "--max-hole-pixels",
        type=int,
        default=DEFAULT_MAX_HOLE_PIXELS,
        help="Maximum unresolved component size to fill. Default: 1500",
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
            "max_hole_pixels": int(args.max_hole_pixels),
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
                max_hole_pixels=args.max_hole_pixels,
            )
            overall["cities"][city_key] = {
                "status": "ok",
                "unresolved_ratio_strict": result["summary"]["unresolved_ratio_strict"],
                "small_hole_pixels_filled": result["summary"]["small_hole_pixels_filled"],
                "large_unresolved_pixels_kept_masked": result["summary"]["large_unresolved_pixels_kept_masked"],
                "output_dir": str(
                    args.processed_root / city_cfg["slug"] / "2022" / "s2_planetary_multiscene_12band_v6"
                ),
            }
        except Exception as exc:
            logging.exception("City %s failed: %s", city_key, exc)
            overall["cities"][city_key] = {
                "status": "error",
                "error": str(exc),
            }

    summary_path = args.processed_root / "brazil_hardcities_2022_s2_multiscene_v6_composite_run_summary.json"
    json_dump(overall, summary_path)
    logging.info("Run finished. Summary: %s", summary_path)

if __name__ == "__main__":
    main()