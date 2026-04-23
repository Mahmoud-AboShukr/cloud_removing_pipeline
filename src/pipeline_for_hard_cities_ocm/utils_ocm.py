# src/pipeline_for_hard_cities_ocm/utils_ocm.py

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject
from scipy.ndimage import binary_dilation, distance_transform_edt, label


def slugify_city_name(name: str) -> str:
    """
    Convert city names to the underscore ASCII convention used in the project.

    Examples
    --------
    "São Luís" -> "sao_luis"
    "João Pessoa" -> "joao_pessoa"
    """
    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower().strip().replace("-", " ")
    return "_".join(ascii_name.split())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_subdirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()])


def load_aoi(aoi_dir: Path, city_slug: str) -> tuple[Path, gpd.GeoDataFrame]:
    """
    Load the buffered AOI file for a city.

    Matches:
    <city_slug>_buffered_*m.gpkg
    """
    matches = sorted(aoi_dir.glob(f"{city_slug}_buffered_*m.gpkg"))
    if not matches:
        raise FileNotFoundError(f"No buffered AOI found for city '{city_slug}' in {aoi_dir}")

    aoi_path = matches[0]
    gdf = gpd.read_file(aoi_path)

    if gdf.empty:
        raise ValueError(f"AOI file is empty: {aoi_path}")
    if gdf.crs is None:
        raise ValueError(f"AOI CRS is missing: {aoi_path}")

    return aoi_path, gdf


def find_scene_dirs(city_raw_root: Path) -> list[Path]:
    """
    Find all scene directories inside the raw city root.
    """
    return list_subdirs(city_raw_root)


def safe_read_raster_info(path: Path) -> dict[str, Any]:
    """
    Read basic raster metadata without loading full data.
    """
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "crs": str(src.crs) if src.crs else None,
            "width": int(src.width),
            "height": int(src.height),
            "count": int(src.count),
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "transform": tuple(src.transform),
            "bounds": [float(v) for v in src.bounds],
            "res": [float(src.res[0]), float(src.res[1])],
        }


def read_single_band(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Read a single-band raster and return (array, metadata).
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    return arr, meta


def write_single_band(
    arr: np.ndarray,
    ref_meta: dict[str, Any],
    output_path: Path,
    dtype: str | None = None,
    nodata: float | int | None = None,
) -> None:
    """
    Write a single-band raster using reference metadata.
    """
    ensure_dir(output_path.parent)
    meta = ref_meta.copy()
    meta["count"] = 1

    if dtype is not None:
        meta["dtype"] = dtype
    if nodata is not None:
        meta["nodata"] = nodata

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(arr, 1)


def build_aoi_mask_on_raster(
    src: rasterio.io.DatasetReader,
    aoi_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """
    Rasterize AOI onto the raster grid as a boolean mask.
    True means inside AOI.
    """
    aoi_proj = aoi_gdf.to_crs(src.crs)
    geoms = [geom for geom in aoi_proj.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise ValueError("AOI became empty after reprojection to raster CRS.")

    return geometry_mask(
        geoms,
        transform=src.transform,
        out_shape=(src.height, src.width),
        invert=True,
        all_touched=False,
    )


def clip_raster_to_aoi(
    raster_path: Path,
    aoi_gdf: gpd.GeoDataFrame,
    output_path: Path,
    nodata_override: float | int | None = None,
) -> dict[str, Any]:
    """
    Clip a raster to AOI extent and save the result.
    """
    ensure_dir(output_path.parent)

    with rasterio.open(raster_path) as src:
        aoi_proj = aoi_gdf.to_crs(src.crs)
        geoms = [geom for geom in aoi_proj.geometry if geom is not None and not geom.is_empty]
        if not geoms:
            raise ValueError("AOI became empty after reprojection for clipping.")

        out_image, out_transform = rio_mask(src, geoms, crop=True, nodata=nodata_override)
        out_meta = src.meta.copy()

        out_meta.update(
            {
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
            }
        )
        if nodata_override is not None:
            out_meta["nodata"] = nodata_override

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(out_image)

    return {
        "input_raster": str(raster_path),
        "output_raster": str(output_path),
        "shape": list(out_image.shape),
        "crs": str(out_meta.get("crs")),
    }


def reproject_to_match(
    src_path: Path,
    ref_path: Path,
    dst_path: Path,
    resampling: Resampling = Resampling.nearest,
    dst_nodata: float | int | None = None,
) -> dict[str, Any]:
    """
    Reproject one raster to match another raster's grid.
    """
    ensure_dir(dst_path.parent)

    with rasterio.open(src_path) as src, rasterio.open(ref_path) as ref:
        dst_meta = ref.meta.copy()
        dst_meta.update(count=1, dtype=src.dtypes[0])

        if dst_nodata is not None:
            dst_meta["nodata"] = dst_nodata
        elif src.nodata is not None:
            dst_meta["nodata"] = src.nodata

        dst_arr = np.full((ref.height, ref.width), dst_meta.get("nodata", 0), dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=dst_meta.get("nodata", 0),
            resampling=resampling,
        )

        with rasterio.open(dst_path, "w", **dst_meta) as dst:
            dst.write(dst_arr, 1)

    return {
        "src_path": str(src_path),
        "ref_path": str(ref_path),
        "dst_path": str(dst_path),
        "dst_shape": [int(ref.height), int(ref.width)],
        "dst_crs": str(ref.crs),
    }


def scale_to_byte(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """
    Percentile stretch to uint8.
    """
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr)

    if not valid.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    vals = arr[valid]
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)

    if hi <= lo:
        out = np.zeros(arr.shape, dtype=np.uint8)
        out[valid] = np.clip(vals, 0, 255).astype(np.uint8)
        return out

    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)
    return (scaled * 255).astype(np.uint8)


def build_rgb_quicklook(
    b04_path: Path,
    b03_path: Path,
    b02_path: Path,
    output_path: Path,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> dict[str, Any]:
    """
    Build a 3-band uint8 RGB GeoTIFF quicklook from B04/B03/B02.
    """
    ensure_dir(output_path.parent)

    with rasterio.open(b04_path) as r_src, rasterio.open(b03_path) as g_src, rasterio.open(b02_path) as b_src:
        r = r_src.read(1).astype(np.float32)
        g = g_src.read(1).astype(np.float32)
        b = b_src.read(1).astype(np.float32)

        if r.shape != g.shape or r.shape != b.shape:
            raise ValueError("RGB band shapes do not match.")

        rgb = np.stack(
            [
                scale_to_byte(r, p_low, p_high),
                scale_to_byte(g, p_low, p_high),
                scale_to_byte(b, p_low, p_high),
            ],
            axis=0,
        )

        meta = r_src.meta.copy()
        meta.update(
            {
                "count": 3,
                "dtype": "uint8",
                "nodata": 0,
            }
        )

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(rgb)

    return {
        "output_path": str(output_path),
        "shape": list(rgb.shape),
    }


def largest_component_ratio(mask: np.ndarray) -> float:
    """
    Ratio of the largest connected component over all True pixels.
    """
    if mask.sum() == 0:
        return 0.0

    lbl, n = label(mask)
    if n == 0:
        return 0.0

    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    largest = counts.max() if counts.size > 1 else 0

    if largest == 0:
        return 0.0

    return float(largest / mask.sum())


def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Binary dilation wrapper.
    """
    if iterations <= 0:
        return mask.astype(bool)
    return binary_dilation(mask.astype(bool), iterations=iterations)


def fill_small_holes_with_nearest(
    data: np.ndarray,
    valid_mask: np.ndarray,
    max_hole_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill only small invalid connected components using nearest valid pixels.

    Parameters
    ----------
    data : np.ndarray
        2D band array.
    valid_mask : np.ndarray
        Boolean valid mask. True where data is valid.
    max_hole_pixels : int
        Maximum connected invalid component size eligible for filling.

    Returns
    -------
    filled_data : np.ndarray
        Output array with only small holes filled.
    small_holes_mask : np.ndarray
        Boolean mask of small holes that were filled.
    large_holes_mask : np.ndarray
        Boolean mask of invalid components that were left unresolved.
    """
    data = data.copy()
    invalid = ~valid_mask

    lbl, n = label(invalid)
    small_holes_mask = np.zeros_like(invalid, dtype=bool)
    large_holes_mask = np.zeros_like(invalid, dtype=bool)

    if n == 0:
        return data, small_holes_mask, large_holes_mask

    component_sizes = np.bincount(lbl.ravel())
    component_sizes[0] = 0

    for comp_id in range(1, n + 1):
        comp_mask = lbl == comp_id
        size = component_sizes[comp_id]
        if size <= max_hole_pixels:
            small_holes_mask |= comp_mask
        else:
            large_holes_mask |= comp_mask

    if small_holes_mask.any() and valid_mask.any():
        # nearest valid pixel indices
        _, indices = distance_transform_edt(~valid_mask, return_indices=True)
        nearest_rows = indices[0]
        nearest_cols = indices[1]
        data[small_holes_mask] = data[nearest_rows[small_holes_mask], nearest_cols[small_holes_mask]]

    return data, small_holes_mask, large_holes_mask