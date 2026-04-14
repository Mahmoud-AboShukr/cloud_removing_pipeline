import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CITY_NAME = "Manaus"
CITY_COLUMN = "nm_mun"

BRAZIL_FAVELAS_PATH = (
    PROJECT_ROOT / "data" / "interim" / "polygons_clean" / "favelas_clean.gpkg"
)

RAW_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "manaus"
    / "2022"
    / "s2_planetary_multiscene"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manaus"
    / "2022"
    / "s2_planetary_multiscene"
)

CLIPPED_SCENES_DIR = PROCESSED_ROOT / "clipped_scenes"
COMPOSITE_DIR = PROCESSED_ROOT / "composite"

CLIPPED_SCENES_DIR.mkdir(parents=True, exist_ok=True)
COMPOSITE_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_SCENES_SUMMARY_PATH = RAW_INPUT_DIR / "selected_scenes_summary.json"

# 10 optical bands for the final composite
REFLECTANCE_BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]

# SCL used for masking
MASK_BAND = "SCL"

# Reference band must be 10 m
REFERENCE_BAND = "B02"

# SCL policy agreed earlier
INVALID_SCL_CLASSES = {0, 1, 3, 8, 9, 10, 11}

# Output file names
COMPOSITE_PATH = COMPOSITE_DIR / "manaus_composite_2022.tif"
CLOUDMASK_PATH = COMPOSITE_DIR / "manaus_cloudmask_2022.tif"
CONTRIBUTION_MAP_PATH = COMPOSITE_DIR / "manaus_scene_contribution_map.tif"
UPDATED_SUMMARY_PATH = COMPOSITE_DIR / "selected_scenes_summary.json"


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class SceneRecord:
    rank: int
    scene_id: str
    datetime: str
    mgrs_tile: str
    eo_cloud_cover: float
    overlap_ratio: float
    source_window: str
    filter_level: str
    scene_dir: Path
    usable_ratio: float | None = None
    valid_pixel_count: int | None = None
    invalid_pixel_count: int | None = None
    clipped_dir: Path | None = None


@dataclass
class ReferenceGrid:
    crs: Any
    transform: Any
    width: int
    height: int
    dtype: str
    nodata: float | int | None


# =============================================================================
# HELPERS
# =============================================================================
def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value


def load_city_aoi_geometry(
    favelas_path: Path,
    city_name: str,
    city_column: str,
):
    """
    Load the Brazil-wide cleaned favela polygons, filter to one city,
    and return the envelope of their union as a shapely geometry in EPSG:4326.
    """
    if not favelas_path.exists():
        raise FileNotFoundError(f"Favela polygon file not found:\n{favelas_path}")

    gdf = gpd.read_file(favelas_path)

    if gdf.empty:
        raise ValueError(f"No polygons found in {favelas_path}")

    if gdf.crs is None:
        raise ValueError(f"The file has no CRS: {favelas_path}")

    if city_column not in gdf.columns:
        raise ValueError(
            f"Column '{city_column}' not found in the polygon file.\n"
            f"Available columns are:\n{list(gdf.columns)}"
        )

    target_city = normalize_text(city_name)
    city_series = gdf[city_column].fillna("").astype(str).map(normalize_text)
    city_gdf = gdf[city_series == target_city].copy()

    if city_gdf.empty:
        unique_values = sorted(gdf[city_column].dropna().astype(str).unique())[:100]
        raise ValueError(
            f"No polygons found for city '{city_name}' in column '{city_column}'.\n"
            f"Some example values are:\n{unique_values}"
        )

    if city_gdf.crs.to_epsg() != 4326:
        city_gdf = city_gdf.to_crs(4326)

    try:
        geom = city_gdf.geometry.union_all()
    except AttributeError:
        geom = city_gdf.unary_union

    return geom.envelope


def load_selected_scenes(summary_path: Path, raw_input_dir: Path) -> list[SceneRecord]:
    """
    Load downloader summary JSON and return scene records.
    """
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Selected scenes summary not found:\n{summary_path}\n"
            "Run search_and_download_multiscene.py first."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    records: list[SceneRecord] = []
    for scene in payload.get("selected_scenes", []):
        scene_id = str(scene["id"])
        scene_dir = raw_input_dir / scene_id

        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found:\n{scene_dir}")

        records.append(
            SceneRecord(
                rank=int(scene["rank"]),
                scene_id=scene_id,
                datetime=str(scene["datetime"]),
                mgrs_tile=str(scene.get("s2:mgrs_tile", "")),
                eo_cloud_cover=float(scene["eo:cloud_cover"]),
                overlap_ratio=float(scene["overlap_ratio"]),
                source_window=str(scene.get("source_window", "")),
                filter_level=str(scene.get("filter_level", "")),
                scene_dir=scene_dir,
            )
        )

    if not records:
        raise ValueError("No selected scenes found in summary JSON.")

    return records


def ensure_scene_has_required_files(scene_dir: Path) -> None:
    required = REFLECTANCE_BANDS + [MASK_BAND, "item_metadata.json", "ranking_metadata.json"]
    missing = [name for name in required if not (scene_dir / f"{name}.tif").exists() and not name.endswith(".json")]

    missing_json = [
        name for name in ["item_metadata.json", "ranking_metadata.json"]
        if not (scene_dir / name).exists()
    ]

    missing.extend(missing_json)

    if missing:
        raise FileNotFoundError(
            f"Scene folder is missing required files:\n{scene_dir}\nMissing: {missing}"
        )


def choose_reference_grid(scene_records: list[SceneRecord]) -> ReferenceGrid:
    """
    Use the highest-ranked scene's B02 as the common 10 m reference grid.
    """
    best_scene = min(scene_records, key=lambda s: s.rank)
    reference_path = best_scene.scene_dir / f"{REFERENCE_BAND}.tif"

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference band not found:\n{reference_path}")

    with rasterio.open(reference_path) as src:
        return ReferenceGrid(
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            dtype=src.dtypes[0],
            nodata=src.nodata,
        )


def build_aoi_mask_on_reference_grid(
    aoi_geom_wgs84,
    reference_grid: ReferenceGrid,
) -> np.ndarray:
    """
    Build AOI mask on the reference grid.
    True means pixel is inside AOI.
    """
    gdf = gpd.GeoDataFrame(geometry=[aoi_geom_wgs84], crs="EPSG:4326")
    gdf_ref = gdf.to_crs(reference_grid.crs)

    geom_ref = gdf_ref.geometry.iloc[0]
    mask = geometry_mask(
        geometries=[mapping(geom_ref)],
        out_shape=(reference_grid.height, reference_grid.width),
        transform=reference_grid.transform,
        invert=True,
    )
    return mask


def get_resampling_for_band(band_name: str) -> Resampling:
    """
    Use nearest for SCL and bilinear for reflectance bands.
    """
    if band_name == MASK_BAND:
        return Resampling.nearest
    return Resampling.bilinear


def warp_to_reference_grid(
    src_path: Path,
    reference_grid: ReferenceGrid,
    band_name: str,
) -> tuple[np.ndarray, float | int | None]:
    """
    Reproject a source raster band onto the reference grid.
    """
    if band_name == MASK_BAND:
        dst_dtype = np.uint8
        dst_fill = 0
    else:
        dst_dtype = np.float32
        dst_fill = np.nan

    dst = np.full(
        (reference_grid.height, reference_grid.width),
        dst_fill,
        dtype=dst_dtype,
    )

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=reference_grid.transform,
            dst_crs=reference_grid.crs,
            dst_nodata=dst_fill if band_name != MASK_BAND else 0,
            resampling=get_resampling_for_band(band_name),
        )

        src_nodata = src.nodata

    return dst, src_nodata


def save_single_band_tif(
    output_path: Path,
    array: np.ndarray,
    reference_grid: ReferenceGrid,
    dtype: str,
    nodata: float | int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": reference_grid.height,
        "width": reference_grid.width,
        "count": 1,
        "crs": reference_grid.crs,
        "transform": reference_grid.transform,
        "dtype": dtype,
        "compress": "deflate",
    }

    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array, 1)


def compute_valid_mask_from_scl(
    scl_array: np.ndarray,
    aoi_mask: np.ndarray,
) -> np.ndarray:
    """
    Valid pixels are inside AOI and not in invalid SCL classes.
    """
    invalid = np.isin(scl_array, list(INVALID_SCL_CLASSES))
    valid = aoi_mask & (~invalid)
    return valid


def clip_and_save_scene(
    scene: SceneRecord,
    reference_grid: ReferenceGrid,
    aoi_mask: np.ndarray,
) -> SceneRecord:
    """
    Warp all bands to the reference grid, save clipped products,
    and compute per-scene AOI usable ratio from SCL.
    """
    ensure_scene_has_required_files(scene.scene_dir)

    clipped_dir = CLIPPED_SCENES_DIR / scene.scene_id
    clipped_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # First process SCL to build valid mask
    # -------------------------------------------------------------------------
    scl_path = scene.scene_dir / f"{MASK_BAND}.tif"
    scl_array, _ = warp_to_reference_grid(
        src_path=scl_path,
        reference_grid=reference_grid,
        band_name=MASK_BAND,
    )

    # Outside AOI -> 0 in saved clipped SCL
    scl_to_save = np.where(aoi_mask, scl_array, 0).astype(np.uint8)

    valid_mask = compute_valid_mask_from_scl(
        scl_array=scl_array,
        aoi_mask=aoi_mask,
    )
    valid_mask_u8 = valid_mask.astype(np.uint8)

    aoi_pixel_count = int(aoi_mask.sum())
    valid_pixel_count = int(valid_mask.sum())
    invalid_pixel_count = int(aoi_pixel_count - valid_pixel_count)

    usable_ratio = 0.0 if aoi_pixel_count == 0 else float(valid_pixel_count / aoi_pixel_count)

    save_single_band_tif(
        output_path=clipped_dir / "SCL_clip.tif",
        array=scl_to_save,
        reference_grid=reference_grid,
        dtype="uint8",
        nodata=0,
    )

    save_single_band_tif(
        output_path=clipped_dir / "valid_mask.tif",
        array=valid_mask_u8,
        reference_grid=reference_grid,
        dtype="uint8",
        nodata=0,
    )

    # -------------------------------------------------------------------------
    # Then process reflectance bands
    # -------------------------------------------------------------------------
    for band_name in REFLECTANCE_BANDS:
        band_path = scene.scene_dir / f"{band_name}.tif"
        band_array, _ = warp_to_reference_grid(
            src_path=band_path,
            reference_grid=reference_grid,
            band_name=band_name,
        )

        # Save AOI-clipped version. Outside AOI set to NaN.
        band_clipped = np.where(aoi_mask, band_array, np.nan).astype(np.float32)

        save_single_band_tif(
            output_path=clipped_dir / f"{band_name}_clip.tif",
            array=band_clipped,
            reference_grid=reference_grid,
            dtype="float32",
            nodata=np.nan,
        )

    # Update record
    scene.usable_ratio = usable_ratio
    scene.valid_pixel_count = valid_pixel_count
    scene.invalid_pixel_count = invalid_pixel_count
    scene.clipped_dir = clipped_dir

    print(
        f"[SCENE] {scene.scene_id} | "
        f"usable_ratio={usable_ratio:.4f} | "
        f"valid={valid_pixel_count} | invalid={invalid_pixel_count}"
    )

    return scene


def rerank_scenes_for_composite(scene_records: list[SceneRecord]) -> list[SceneRecord]:
    """
    Final ranking:
    1. higher AOI overlap
    2. higher AOI usable-pixel ratio from SCL
    3. lower scene-level eo:cloud_cover
    4. earlier datetime
    5. item id stable tie-breaker
    """
    missing_usable = [s.scene_id for s in scene_records if s.usable_ratio is None]
    if missing_usable:
        raise ValueError(
            f"Some scenes do not have usable_ratio computed yet: {missing_usable}"
        )

    reranked = sorted(
        scene_records,
        key=lambda s: (
            -s.overlap_ratio,
            -float(s.usable_ratio),
            s.eo_cloud_cover,
            s.datetime,
            s.scene_id,
        ),
    )

    return reranked


def load_clipped_band(scene: SceneRecord, band_name: str) -> np.ndarray:
    if scene.clipped_dir is None:
        raise ValueError(f"Scene {scene.scene_id} has no clipped_dir set.")

    path = scene.clipped_dir / f"{band_name}_clip.tif"
    if not path.exists():
        raise FileNotFoundError(f"Clipped band not found:\n{path}")

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def load_valid_mask(scene: SceneRecord) -> np.ndarray:
    if scene.clipped_dir is None:
        raise ValueError(f"Scene {scene.scene_id} has no clipped_dir set.")

    path = scene.clipped_dir / "valid_mask.tif"
    if not path.exists():
        raise FileNotFoundError(f"Valid mask not found:\n{path}")

    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8).astype(bool)


def build_composite(
    scene_records: list[SceneRecord],
    reference_grid: ReferenceGrid,
    aoi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build final composite using first-valid-pixel priority by scene rank.

    Returns:
        composite: (bands, height, width) float32
        cloudmask: (height, width) uint8
        contribution_map: (height, width) uint8
    """
    band_count = len(REFLECTANCE_BANDS)
    height = reference_grid.height
    width = reference_grid.width

    composite = np.full((band_count, height, width), np.nan, dtype=np.float32)
    contribution_map = np.zeros((height, width), dtype=np.uint8)

    unresolved = aoi_mask.copy()

    for scene_index, scene in enumerate(scene_records, start=1):
        valid_mask = load_valid_mask(scene)
        use_pixels = unresolved & valid_mask

        use_count = int(use_pixels.sum())
        print(f"[COMPOSITE] Scene rank {scene_index}: {scene.scene_id} -> fills {use_count} pixel(s).")

        if use_count == 0:
            continue

        for band_idx, band_name in enumerate(REFLECTANCE_BANDS):
            band_array = load_clipped_band(scene, band_name)
            composite[band_idx, use_pixels] = band_array[use_pixels]

        contribution_map[use_pixels] = scene_index
        unresolved[use_pixels] = False

        if not unresolved.any():
            print("[COMPOSITE] AOI fully filled before exhausting all scenes.")
            break

    # cloudmask: 1 = invalid/unresolved inside AOI, 0 = valid inside AOI, 255 outside AOI
    cloudmask = np.full((height, width), 255, dtype=np.uint8)
    cloudmask[aoi_mask] = unresolved[aoi_mask].astype(np.uint8)

    # contribution map:
    # 255 outside AOI
    # 0 unresolved inside AOI
    contribution_map_out = np.full((height, width), 255, dtype=np.uint8)
    contribution_map_out[aoi_mask] = contribution_map[aoi_mask]

    return composite, cloudmask, contribution_map_out


def save_multiband_composite(
    output_path: Path,
    composite: np.ndarray,
    reference_grid: ReferenceGrid,
) -> None:
    profile = {
        "driver": "GTiff",
        "height": reference_grid.height,
        "width": reference_grid.width,
        "count": composite.shape[0],
        "crs": reference_grid.crs,
        "transform": reference_grid.transform,
        "dtype": "float32",
        "nodata": np.nan,
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(composite)
        for idx, band_name in enumerate(REFLECTANCE_BANDS, start=1):
            dst.set_band_description(idx, band_name)


def save_summary_json(
    scene_records_original: list[SceneRecord],
    scene_records_reranked: list[SceneRecord],
    aoi_mask: np.ndarray,
    cloudmask: np.ndarray,
) -> None:
    """
    Save updated summary in final composite directory.
    """
    aoi_pixel_count = int(aoi_mask.sum())
    unresolved_count = int((cloudmask == 1).sum())
    resolved_count = int(aoi_pixel_count - unresolved_count)
    resolved_ratio = 0.0 if aoi_pixel_count == 0 else float(resolved_count / aoi_pixel_count)

    rerank_lookup = {scene.scene_id: idx for idx, scene in enumerate(scene_records_reranked, start=1)}

    payload = {
        "city_name": CITY_NAME,
        "composite_outputs": {
            "composite_path": str(COMPOSITE_PATH),
            "cloudmask_path": str(CLOUDMASK_PATH),
            "contribution_map_path": str(CONTRIBUTION_MAP_PATH),
        },
        "aoi_stats": {
            "aoi_pixel_count": aoi_pixel_count,
            "resolved_pixel_count": resolved_count,
            "unresolved_pixel_count": unresolved_count,
            "resolved_ratio": resolved_ratio,
        },
        "scenes": [],
    }

    for scene in scene_records_original:
        payload["scenes"].append(
            {
                "initial_rank_from_download_step": scene.rank,
                "final_rerank_for_composite": rerank_lookup[scene.scene_id],
                "id": scene.scene_id,
                "datetime": scene.datetime,
                "s2:mgrs_tile": scene.mgrs_tile,
                "eo:cloud_cover": scene.eo_cloud_cover,
                "overlap_ratio": scene.overlap_ratio,
                "usable_ratio": scene.usable_ratio,
                "valid_pixel_count": scene.valid_pixel_count,
                "invalid_pixel_count": scene.invalid_pixel_count,
                "source_window": scene.source_window,
                "filter_level": scene.filter_level,
                "raw_scene_dir": str(scene.scene_dir),
                "clipped_scene_dir": str(scene.clipped_dir),
            }
        )

    with open(UPDATED_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=" * 80)
    print("[STEP] Loading selected scenes...")
    print("=" * 80)
    scenes = load_selected_scenes(
        summary_path=SELECTED_SCENES_SUMMARY_PATH,
        raw_input_dir=RAW_INPUT_DIR,
    )

    print(f"[INFO] Found {len(scenes)} selected scene(s).")

    print("=" * 80)
    print("[STEP] Loading Manaus AOI...")
    print("=" * 80)
    aoi_geom_wgs84 = load_city_aoi_geometry(
        favelas_path=BRAZIL_FAVELAS_PATH,
        city_name=CITY_NAME,
        city_column=CITY_COLUMN,
    )

    print("=" * 80)
    print("[STEP] Building reference grid...")
    print("=" * 80)
    reference_grid = choose_reference_grid(scenes)
    print(f"[INFO] Reference CRS: {reference_grid.crs}")
    print(f"[INFO] Reference size: {reference_grid.width} x {reference_grid.height}")

    print("=" * 80)
    print("[STEP] Building AOI mask on reference grid...")
    print("=" * 80)
    aoi_mask = build_aoi_mask_on_reference_grid(
        aoi_geom_wgs84=aoi_geom_wgs84,
        reference_grid=reference_grid,
    )
    print(f"[INFO] AOI pixel count: {int(aoi_mask.sum())}")

    print("=" * 80)
    print("[STEP] Clipping scenes and computing SCL usable ratios...")
    print("=" * 80)
    processed_scenes: list[SceneRecord] = []
    for scene in scenes:
        processed_scene = clip_and_save_scene(
            scene=scene,
            reference_grid=reference_grid,
            aoi_mask=aoi_mask,
        )
        processed_scenes.append(processed_scene)

    print("=" * 80)
    print("[STEP] Re-ranking scenes using AOI usable-pixel ratio...")
    print("=" * 80)
    reranked_scenes = rerank_scenes_for_composite(processed_scenes)

    print("[INFO] Final composite ranking:")
    for idx, scene in enumerate(reranked_scenes, start=1):
        print(
            f"  {idx}. {scene.scene_id} | "
            f"overlap={scene.overlap_ratio:.4f} | "
            f"usable_ratio={float(scene.usable_ratio):.4f} | "
            f"cloud={scene.eo_cloud_cover:.2f} | "
            f"datetime={scene.datetime}"
        )

    print("=" * 80)
    print("[STEP] Building cloud-reduced composite...")
    print("=" * 80)
    composite, cloudmask, contribution_map = build_composite(
        scene_records=reranked_scenes,
        reference_grid=reference_grid,
        aoi_mask=aoi_mask,
    )

    unresolved_count = int((cloudmask == 1).sum())
    aoi_count = int(aoi_mask.sum())
    unresolved_ratio = 0.0 if aoi_count == 0 else float(unresolved_count / aoi_count)
    print(
        f"[INFO] Composite residual invalid pixels: {unresolved_count}/{aoi_count} "
        f"({unresolved_ratio:.4%})"
    )

    print("=" * 80)
    print("[STEP] Saving final outputs...")
    print("=" * 80)

    save_multiband_composite(
        output_path=COMPOSITE_PATH,
        composite=composite,
        reference_grid=reference_grid,
    )

    save_single_band_tif(
        output_path=CLOUDMASK_PATH,
        array=cloudmask,
        reference_grid=reference_grid,
        dtype="uint8",
        nodata=255,
    )

    save_single_band_tif(
        output_path=CONTRIBUTION_MAP_PATH,
        array=contribution_map,
        reference_grid=reference_grid,
        dtype="uint8",
        nodata=255,
    )

    save_summary_json(
        scene_records_original=processed_scenes,
        scene_records_reranked=reranked_scenes,
        aoi_mask=aoi_mask,
        cloudmask=cloudmask,
    )

    print(f"[DONE] Saved composite: {COMPOSITE_PATH}")
    print(f"[DONE] Saved cloudmask: {CLOUDMASK_PATH}")
    print(f"[DONE] Saved contribution map: {CONTRIBUTION_MAP_PATH}")
    print(f"[DONE] Saved updated summary: {UPDATED_SUMMARY_PATH}")

    print("=" * 80)
    print("[DONE] build_cloud_reduced_composite.py completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()