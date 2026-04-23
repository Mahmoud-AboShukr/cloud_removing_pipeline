#!/usr/bin/env python3
"""
build_cloud_reduced_composite.py

Build a cloud-reduced Sentinel-2 composite from multiple downloaded scenes.

Expected inputs
---------------
For each city, this script expects the downloader output:

data/raw/<city_slug>/2022/s2_planetary_multiscene_12band/
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
data/processed/<city_slug>/2022/s2_planetary_multiscene_12band/
    city_composite_2022.tif
    city_cloudmask_2022.tif
    city_scene_contribution_map.tif
    selected_scenes_diagnostics.json

Notes
-----
- The output composite is 12 bands only. SCL is used only for masking.
- All scenes are reprojected/aligned to a common target grid.
- The target grid is derived from the first scene's 10 m grid and clipped to the AOI.
- B01/B09/SCL and other non-10 m bands are resampled to the 10 m target grid.
- Reflectance bands use bilinear resampling.
- SCL uses nearest-neighbor resampling.
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

DEFAULT_VALID_SCL = [4, 5, 6, 7, 11]  # vegetation, bare soil, water, unclassified, snow/ice
DEFAULT_RESIDUAL_THRESHOLD = 0.02      # 2%
DEFAULT_TARGET_RESOLUTION = 10.0       # meters
SCENE_REF_BAND = "B02"                 # use 10m band as grid reference

# Known nominal native resolutions for Sentinel-2 L2A bands
NATIVE_RESOLUTION_HINTS = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B11": 20,
    "B12": 20,
    "SCL": 20,
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DownloadedSceneInfo:
    """Metadata and local paths for one downloaded scene."""
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
    """Per-scene metrics after aligning SCL to the target grid."""
    item_id: str
    usable_pixels: int
    usable_ratio: float
    cloud_cover: float
    overlap_ratio: float
    date: str


# -----------------------------------------------------------------------------
# Logging / utils
# -----------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


def ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path) -> None:
    """Write JSON with indentation."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    """Write text file."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------------------------------------------------------
# Scene discovery
# -----------------------------------------------------------------------------

def load_downloaded_scenes_from_summary(summary_path: Path) -> List[DownloadedSceneInfo]:
    """
    Read downloader summary JSON and return scenes that were downloaded successfully.
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")

    summary = load_json(summary_path)
    downloaded = summary.get("downloaded_scenes", [])

    if not downloaded:
        raise ValueError(f"No downloaded scenes found in {summary_path}")

    scenes: List[DownloadedSceneInfo] = []

    for row in downloaded:
        scene_dir = Path(row["scene_dir"])
        downloaded_assets = row["downloaded_assets"]

        assets: Dict[str, Path] = {}
        for asset_key, asset_path in downloaded_assets.items():
            assets[asset_key] = Path(asset_path)

        scenes.append(
            DownloadedSceneInfo(
                item_id=row["item_id"],
                datetime_utc=row["datetime_utc"],
                date=row["date"],
                tile_id=row["tile_id"],
                cloud_cover=float(row["cloud_cover"]),
                overlap_ratio=float(row["overlap_ratio"]),
                scene_dir=scene_dir,
                assets=assets,
            )
        )

    return scenes


def validate_scene_files(scenes: Sequence[DownloadedSceneInfo]) -> None:
    """
    Ensure every scene has all required files.
    """
    required = set(INPUT_BANDS + [SCL_BAND])

    for scene in scenes:
        missing_keys = required - set(scene.assets.keys())
        if missing_keys:
            raise FileNotFoundError(
                f"Scene {scene.item_id} is missing asset keys: {sorted(missing_keys)}"
            )

        for key in required:
            if not scene.assets[key].exists():
                raise FileNotFoundError(
                    f"Scene {scene.item_id} missing file on disk: {scene.assets[key]}"
                )


# -----------------------------------------------------------------------------
# Grid construction
# -----------------------------------------------------------------------------

def build_target_grid_from_aoi(
    reference_raster_path: Path,
    aoi_geom_4326,
    target_resolution: float = DEFAULT_TARGET_RESOLUTION,
) -> Tuple[CRS, Affine, int, int]:
    """
    Build a common target grid aligned to the first scene's 10 m grid and clipped to AOI.

    Strategy
    --------
    - Open reference 10 m band (B02) from the first scene.
    - Transform AOI bounds from EPSG:4326 into the raster CRS.
    - Build a raster window covering the AOI bounds on the reference grid.
    - Use the rounded window transform/shape as the target grid.

    Returns
    -------
    target_crs, target_transform, target_width, target_height
    """
    with rasterio.open(reference_raster_path) as src:
        target_crs = src.crs
        if target_crs is None:
            raise ValueError(f"Reference raster has no CRS: {reference_raster_path}")

        ref_res_x = abs(src.transform.a)
        ref_res_y = abs(src.transform.e)

        if not math.isclose(ref_res_x, target_resolution, rel_tol=0.01) or not math.isclose(ref_res_y, target_resolution, rel_tol=0.01):
            logging.warning(
                "Reference raster resolution is %.3f x %.3f, not exactly %.1f m. "
                "The output grid will still be aligned to the reference raster.",
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
            "Target grid built from AOI on reference raster grid: "
            "CRS=%s, width=%d, height=%d, res=(%.3f, %.3f)",
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
    """
    Rasterize AOI polygon to the target grid.
    """
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
    logging.info("AOI mask rasterized. AOI pixels: %d", int(mask_bool.sum()))
    return mask_bool


# -----------------------------------------------------------------------------
# Reprojection helpers
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
    """
    Reproject a single-band raster to the common target grid.
    """
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
    """
    Get dtype from a reference raster.
    """
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
    """
    Valid pixels are where SCL belongs to allowed classes and inside AOI.
    """
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
) -> SceneUsability:
    """
    Reproject scene SCL to target grid and compute usable AOI ratio.
    """
    scl_path = scene.assets[SCL_BAND]

    scl_reproj = reproject_band_to_grid(
        src_path=scl_path,
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

    return SceneUsability(
        item_id=scene.item_id,
        usable_pixels=usable_pixels,
        usable_ratio=float(usable_ratio),
        cloud_cover=float(scene.cloud_cover),
        overlap_ratio=float(scene.overlap_ratio),
        date=scene.date,
    )


def rank_scenes(
    scenes: Sequence[DownloadedSceneInfo],
    usability: Dict[str, SceneUsability],
) -> List[DownloadedSceneInfo]:
    """
    Rank scenes for compositing.

    Priority
    --------
    1. usable_ratio descending
    2. overlap_ratio descending
    3. cloud_cover ascending
    4. date descending
    """
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
    """
    Reproject all 12 reflectance bands to the target grid.
    """
    out: Dict[str, np.ndarray] = {}

    for band in INPUT_BANDS:
        band_path = scene.assets[band]

        logging.info("  Reprojecting band %s for scene %s", band, scene.item_id)

        out[band] = reproject_band_to_grid(
            src_path=band_path,
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
    """
    Build cloud-reduced composite for one city.
    """
    ensure_dir(processed_city_root)

    if not scenes:
        raise ValueError(f"No scenes available for city {city_slug}")

    validate_scene_files(scenes)

    # -------------------------------------------------------------------------
    # 1) Build target grid from first scene B02 and AOI
    # -------------------------------------------------------------------------
    reference_band_path = scenes[0].assets[SCENE_REF_BAND]
    output_dtype = get_band_dtype(reference_band_path)

    target_crs, target_transform, width, height = build_target_grid_from_aoi(
        reference_raster_path=reference_band_path,
        aoi_geom_4326=aoi_geom_4326,
        target_resolution=DEFAULT_TARGET_RESOLUTION,
    )

    # -------------------------------------------------------------------------
    # 2) AOI mask
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 3) Per-scene usability using SCL
    # -------------------------------------------------------------------------
    usability: Dict[str, SceneUsability] = {}

    logging.info("Computing per-scene usable AOI coverage from SCL...")
    for scene in scenes:
        metrics = compute_scene_usable_ratio(
            scene=scene,
            target_crs=target_crs,
            target_transform=target_transform,
            width=width,
            height=height,
            aoi_mask=aoi_mask,
            valid_scl_values=valid_scl_values,
        )
        usability[scene.item_id] = metrics
        logging.info(
            "Scene %s | usable_pixels=%d | usable_ratio=%.4f | overlap=%.3f | cloud=%.1f",
            scene.item_id,
            metrics.usable_pixels,
            metrics.usable_ratio,
            scene.overlap_ratio,
            scene.cloud_cover,
        )

    ranked_scenes = rank_scenes(scenes, usability)

    logging.info("Scene ranking for compositing:")
    for idx, scene in enumerate(ranked_scenes, start=1):
        m = usability[scene.item_id]
        logging.info(
            "  [%d] %s | usable_ratio=%.4f | overlap=%.3f | cloud=%.1f | date=%s",
            idx,
            scene.item_id,
            m.usable_ratio,
            scene.overlap_ratio,
            scene.cloud_cover,
            scene.date,
        )

    # -------------------------------------------------------------------------
    # 4) Composite arrays
    # -------------------------------------------------------------------------
    composite = np.zeros((len(INPUT_BANDS), height, width), dtype=np.dtype(output_dtype))
    filled_mask = np.zeros((height, width), dtype=bool)
    contribution_map = np.zeros((height, width), dtype=np.uint16)

    scene_diagnostics: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 5) Pixel-wise filling scene by scene
    # -------------------------------------------------------------------------
    for scene_rank, scene in enumerate(ranked_scenes, start=1):
        residual_before = int((aoi_mask & ~filled_mask).sum())
        residual_ratio_before = residual_before / aoi_pixels

        logging.info(
            "Applying scene %d/%d: %s | residual before=%d (%.4f)",
            scene_rank,
            len(ranked_scenes),
            scene.item_id,
            residual_before,
            residual_ratio_before,
        )

        # Reproject SCL first to know valid pixels
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

        newly_fillable = valid_mask & (~filled_mask)
        newly_fillable_pixels = int(newly_fillable.sum())

        logging.info(
            "Scene %s can newly fill %d pixels",
            scene.item_id,
            newly_fillable_pixels,
        )

        if newly_fillable_pixels == 0:
            residual_after = residual_before
            residual_ratio_after = residual_after / aoi_pixels
            scene_diagnostics.append(
                {
                    "scene_rank": scene_rank,
                    "item_id": scene.item_id,
                    "date": scene.date,
                    "tile_id": scene.tile_id,
                    "cloud_cover": scene.cloud_cover,
                    "overlap_ratio": scene.overlap_ratio,
                    "usable_ratio": usability[scene.item_id].usable_ratio,
                    "new_pixels_filled": 0,
                    "residual_pixels_before": residual_before,
                    "residual_pixels_after": residual_after,
                    "residual_ratio_after": residual_ratio_after,
                    "used_for_fill": False,
                }
            )
            continue

        reprojected_bands = reproject_scene_bands(
            scene=scene,
            target_crs=target_crs,
            target_transform=target_transform,
            width=width,
            height=height,
            output_dtype=output_dtype,
        )

        for band_idx, band_name in enumerate(INPUT_BANDS):
            composite[band_idx][newly_fillable] = reprojected_bands[band_name][newly_fillable]

        filled_mask[newly_fillable] = True
        contribution_map[newly_fillable] = scene_rank

        residual_after = int((aoi_mask & ~filled_mask).sum())
        residual_ratio_after = residual_after / aoi_pixels

        logging.info(
            "After scene %s | residual=%d (%.4f)",
            scene.item_id,
            residual_after,
            residual_ratio_after,
        )

        scene_diagnostics.append(
            {
                "scene_rank": scene_rank,
                "item_id": scene.item_id,
                "date": scene.date,
                "tile_id": scene.tile_id,
                "cloud_cover": scene.cloud_cover,
                "overlap_ratio": scene.overlap_ratio,
                "usable_ratio": usability[scene.item_id].usable_ratio,
                "new_pixels_filled": newly_fillable_pixels,
                "residual_pixels_before": residual_before,
                "residual_pixels_after": residual_after,
                "residual_ratio_after": residual_ratio_after,
                "used_for_fill": True,
            }
        )

        if residual_ratio_after <= residual_threshold:
            logging.info(
                "Early stopping triggered for %s: residual ratio %.4f <= threshold %.4f",
                city_slug,
                residual_ratio_after,
                residual_threshold,
            )
            break

    # -------------------------------------------------------------------------
    # 6) Final outputs
    # -------------------------------------------------------------------------
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
    composite_profile.update(
        {
            "count": len(INPUT_BANDS),
            "dtype": output_dtype,
            "nodata": 0,
        }
    )

    with rasterio.open(composite_path, "w", **composite_profile) as dst:
        for i, band_name in enumerate(INPUT_BANDS, start=1):
            dst.write(composite[i - 1], i)
            dst.set_band_description(i, band_name)

    logging.info("Saved composite: %s", composite_path)

    mask_profile = profile.copy()
    mask_profile.update(
        {
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
        }
    )

    with rasterio.open(cloudmask_path, "w", **mask_profile) as dst:
        dst.write(cloudmask, 1)
        dst.set_band_description(1, "residual_missing_mask")

    logging.info("Saved cloud/residual mask: %s", cloudmask_path)

    contribution_profile = profile.copy()
    contribution_profile.update(
        {
            "count": 1,
            "dtype": "uint16",
            "nodata": 0,
        }
    )

    with rasterio.open(contribution_path, "w", **contribution_profile) as dst:
        dst.write(contribution_map, 1)
        dst.set_band_description(1, "scene_rank_contribution")

    logging.info("Saved contribution map: %s", contribution_path)

    used_scene_ids = []
    for row in scene_diagnostics:
        if row["used_for_fill"] and row["new_pixels_filled"] > 0:
            used_scene_ids.append(row["item_id"])

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
        },
        "summary": {
            "num_input_scenes": len(scenes),
            "num_scenes_used_for_fill": len(set(used_scene_ids)),
            "aoi_pixels": aoi_pixels,
            "final_filled_pixels": int((aoi_mask & filled_mask).sum()),
            "final_unresolved_pixels": int(unresolved_mask.sum()),
            "final_unresolved_ratio": float(unresolved_mask.sum() / aoi_pixels),
        },
        "scene_ranking": [
            {
                "rank": idx,
                "item_id": scene.item_id,
                "date": scene.date,
                "tile_id": scene.tile_id,
                "cloud_cover": scene.cloud_cover,
                "overlap_ratio": scene.overlap_ratio,
                "usable_ratio": usability[scene.item_id].usable_ratio,
            }
            for idx, scene in enumerate(ranked_scenes, start=1)
        ],
        "scene_fill_diagnostics": scene_diagnostics,
        "outputs": {
            "composite_path": str(composite_path),
            "cloudmask_path": str(cloudmask_path),
            "contribution_path": str(contribution_path),
        },
    }

    json_dump(diagnostics, diagnostics_path)
    logging.info("Saved diagnostics JSON: %s", diagnostics_path)

    summary_txt = [
        f"City slug: {city_slug}",
        f"AOI pixels: {aoi_pixels}",
        f"Final filled pixels: {int((aoi_mask & filled_mask).sum())}",
        f"Final unresolved pixels: {int(unresolved_mask.sum())}",
        f"Final unresolved ratio: {float(unresolved_mask.sum() / aoi_pixels):.4f}",
        "",
        "Scene ranking:",
    ]
    for row in diagnostics["scene_ranking"]:
        summary_txt.append(
            f"{row['rank']}. {row['item_id']} | date={row['date']} | "
            f"usable_ratio={row['usable_ratio']:.4f} | overlap={row['overlap_ratio']:.3f} | "
            f"cloud={row['cloud_cover']:.1f}"
        )

    write_text(processed_city_root / "selected_scenes_diagnostics.txt", "\n".join(summary_txt))

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
    """
    Process one city from downloaded raw scenes to final composite.
    """
    city_slug = city_cfg["slug"]
    city_display = city_cfg["display_name"]

    logging.info("=" * 80)
    logging.info("Processing city composite: %s", city_display)
    logging.info("=" * 80)

    raw_city_root = raw_root / city_slug / "2022" / "s2_planetary_multiscene_12band"
    processed_city_root = processed_root / city_slug / "2022" / "s2_planetary_multiscene_12band"

    summary_path = raw_city_root / "selected_scenes_summary.json"
    aoi_path = raw_city_root / f"{city_slug}_aoi.gpkg"

    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")

    logging.info("Loading AOI from: %s", aoi_path)
    aoi_gdf = gpd.read_file(aoi_path)
    if aoi_gdf.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")

    aoi_gdf = aoi_gdf.to_crs(epsg=4326)
    aoi_geom_4326 = aoi_gdf.geometry.union_all()

    logging.info("Loading downloaded scene summary from: %s", summary_path)
    scenes = load_downloaded_scenes_from_summary(summary_path)
    logging.info("Found %d downloaded scenes for %s", len(scenes), city_display)

    diagnostics = build_composite_for_city(
        city_slug=city_slug,
        aoi_geom_4326=aoi_geom_4326,
        scenes=scenes,
        processed_city_root=processed_city_root,
        residual_threshold=residual_threshold,
        valid_scl_values=valid_scl_values,
    )

    logging.info("Finished city composite: %s", city_display)
    return diagnostics


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build cloud-reduced multi-scene Sentinel-2 composites for five cities."
    )

    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="City key from CITY_CONFIGS, or 'all'. Example: brasilia",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Base raw data root.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Base processed data root.",
    )
    parser.add_argument(
        "--residual-threshold",
        type=float,
        default=DEFAULT_RESIDUAL_THRESHOLD,
        help="Early-stop threshold on unresolved AOI ratio. Example: 0.02 for 2%%",
    )
    parser.add_argument(
        "--valid-scl",
        nargs="+",
        type=int,
        default=DEFAULT_VALID_SCL,
        help="SCL values considered valid.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    args = parser.parse_args()

    if not (0.0 <= args.residual_threshold <= 1.0):
        raise ValueError("--residual-threshold must be between 0 and 1.")

    return args


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    if args.city == "all":
        city_items = list(CITY_CONFIGS.items())
    else:
        if args.city not in CITY_CONFIGS:
            raise KeyError(
                f"Unknown city key '{args.city}'. "
                f"Available: {list(CITY_CONFIGS.keys())}"
            )
        city_items = [(args.city, CITY_CONFIGS[args.city])]

    overall = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "residual_threshold": args.residual_threshold,
        "valid_scl_values": args.valid_scl,
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
                "output_dir": str(
                    args.processed_root
                    / city_cfg["slug"]
                    / "2022"
                    / "s2_planetary_multiscene_12band"
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
        / "five_cities_2022_s2_planetary_multiscene_12band_composite_run_summary.json"
    )
    json_dump(overall, overall_path)

    logging.info("Composite run finished.")
    logging.info("Global summary written to: %s", overall_path)


if __name__ == "__main__":
    main()