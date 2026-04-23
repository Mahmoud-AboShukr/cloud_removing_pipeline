# src/pipeline_for_hard_cities_ocm/06_render_ocm_v1_rgb.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from config_ocm_v1 import (
    COMPOSITES_DIR,
    LOGS_DIR,
    REPORT_DIR,
    RGB_DIR,
    RENDER_PERCENTILE_HIGH,
    RENDER_PERCENTILE_LOW,
    TARGET_CITIES,
    get_city_log_dir,
    get_city_report_dir,
)
from utils_ocm import ensure_dir, save_json, slugify_city_name


def load_rgb_bands_from_composite(
    composite_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load B04, B03, B02 from a 12-band composite using band descriptions.

    Expected band descriptions:
    B01, B02, B03, ..., B12
    """
    with rasterio.open(composite_path) as src:
        descriptions = list(src.descriptions)

        if not descriptions or all(d is None for d in descriptions):
            raise ValueError(
                f"No band descriptions found in {composite_path}. "
                "The composite writer should have set band names."
            )

        band_to_index = {}
        for idx, desc in enumerate(descriptions, start=1):
            if desc is not None:
                band_to_index[desc] = idx

        required = ["B04", "B03", "B02"]
        missing = [b for b in required if b not in band_to_index]
        if missing:
            raise ValueError(
                f"Composite missing required RGB band descriptions: {missing}"
            )

        red = src.read(band_to_index["B04"]).astype(np.float32)
        green = src.read(band_to_index["B03"]).astype(np.float32)
        blue = src.read(band_to_index["B02"]).astype(np.float32)

        meta = src.meta.copy()

    return red, green, blue, meta


def scale_to_byte(
    arr: np.ndarray,
    nodata_value: float | int | None,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """
    Percentile stretch to uint8, ignoring nodata and non-finite values.
    """
    arr = arr.astype(np.float32)

    valid = np.isfinite(arr)
    if nodata_value is not None:
        valid &= arr != nodata_value

    out = np.zeros(arr.shape, dtype=np.uint8)

    if not valid.any():
        return out

    vals = arr[valid]
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)

    if hi <= lo:
        out[valid] = np.clip(vals, 0, 255).astype(np.uint8)
        return out

    scaled = (arr - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)
    out[valid] = (scaled[valid] * 255).astype(np.uint8)

    return out


def build_rgb_array(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    nodata_value: float | int | None,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """
    Build RGB array in shape (3, H, W), uint8.
    """
    r8 = scale_to_byte(red, nodata_value, p_low, p_high)
    g8 = scale_to_byte(green, nodata_value, p_low, p_high)
    b8 = scale_to_byte(blue, nodata_value, p_low, p_high)
    rgb = np.stack([r8, g8, b8], axis=0)
    return rgb


def write_rgb_geotiff(
    rgb: np.ndarray,
    ref_meta: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Write 3-band uint8 GeoTIFF.
    """
    ensure_dir(output_path.parent)

    meta = ref_meta.copy()
    meta.update(
        {
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(rgb)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "green")
        dst.set_band_description(3, "blue")


def write_rgb_png(
    rgb: np.ndarray,
    output_path: Path,
) -> None:
    """
    Write PNG using imageio.
    """
    ensure_dir(output_path.parent)

    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(
            "imageio is required to write PNG outputs. "
            "Install it with: pip install imageio"
        ) from e

    arr = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    imageio.imwrite(output_path, arr)


def process_city(
    city_slug: str,
    use_filled: bool,
    p_low: float,
    p_high: float,
    overwrite: bool,
) -> dict[str, Any]:
    print("-" * 80)
    print(f"[CITY] {city_slug}")

    city_log_dir = get_city_log_dir(city_slug)
    city_report_dir = get_city_report_dir(city_slug)
    city_rgb_dir = RGB_DIR / city_slug

    ensure_dir(city_log_dir)
    ensure_dir(city_report_dir)
    ensure_dir(city_rgb_dir)

    suffix = "composite_smallholes_filled" if use_filled else "composite"
    composite_path = COMPOSITES_DIR / city_slug / f"{city_slug}_ocm_v1_{suffix}.tif"

    if not composite_path.exists():
        raise FileNotFoundError(
            f"Composite not found for city '{city_slug}': {composite_path}\n"
            "Run 05_build_ocm_v1_composite.py first."
        )

    geotiff_out = city_rgb_dir / f"{city_slug}_ocm_v1_rgb.tif"
    png_out = city_rgb_dir / f"{city_slug}_ocm_v1_rgb.png"

    if geotiff_out.exists() and png_out.exists() and not overwrite:
        summary = {
            "city_slug": city_slug,
            "composite_path": str(composite_path),
            "rgb_geotiff": str(geotiff_out),
            "rgb_png": str(png_out),
            "status": "exists",
            "use_filled": use_filled,
            "percentile_low": p_low,
            "percentile_high": p_high,
        }
        save_json(summary, city_log_dir / f"{city_slug}_rgb_render_summary.json")
        save_json(summary, city_report_dir / f"{city_slug}_rgb_render_summary.json")
        print(f"[SKIP] RGB already exists: {png_out}")
        return summary

    red, green, blue, meta = load_rgb_bands_from_composite(composite_path)
    nodata_value = meta.get("nodata", None)

    rgb = build_rgb_array(
        red=red,
        green=green,
        blue=blue,
        nodata_value=nodata_value,
        p_low=p_low,
        p_high=p_high,
    )

    write_rgb_geotiff(rgb, meta, geotiff_out)
    write_rgb_png(rgb, png_out)

    summary = {
        "city_slug": city_slug,
        "composite_path": str(composite_path),
        "rgb_geotiff": str(geotiff_out),
        "rgb_png": str(png_out),
        "status": "ok",
        "use_filled": use_filled,
        "percentile_low": p_low,
        "percentile_high": p_high,
        "shape_chw": [int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])],
        "shape_hwc_png": [int(rgb.shape[1]), int(rgb.shape[2]), int(rgb.shape[0])],
    }

    save_json(summary, city_log_dir / f"{city_slug}_rgb_render_summary.json")
    save_json(summary, city_report_dir / f"{city_slug}_rgb_render_summary.json")

    print(f"[OK] Composite : {composite_path}")
    print(f"[OK] RGB TIFF  : {geotiff_out}")
    print(f"[OK] RGB PNG   : {png_out}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render RGB quicklooks from OCM_V1 composites."
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=TARGET_CITIES,
        help="City slugs to process. Example: recife belem sao_luis",
    )
    parser.add_argument(
        "--use-filled",
        action="store_true",
        help="Render from the small-holes-filled composite instead of the strict composite.",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=RENDER_PERCENTILE_LOW,
        help="Lower percentile for stretching.",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=RENDER_PERCENTILE_HIGH,
        help="Upper percentile for stretching.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing RGB outputs.",
    )
    args = parser.parse_args()

    ensure_dir(RGB_DIR)
    ensure_dir(LOGS_DIR)
    ensure_dir(REPORT_DIR)

    print("=" * 80)
    print("RENDER OCM_V1 RGB")
    print("=" * 80)
    print(f"[INFO] Cities        : {args.cities}")
    print(f"[INFO] Use filled    : {args.use_filled}")
    print(f"[INFO] Percentiles   : {args.p_low} - {args.p_high}")
    print(f"[INFO] Overwrite     : {args.overwrite}")
    print()

    all_summaries = []

    for city in args.cities:
        city_slug = slugify_city_name(city)
        summary = process_city(
            city_slug=city_slug,
            use_filled=args.use_filled,
            p_low=args.p_low,
            p_high=args.p_high,
            overwrite=args.overwrite,
        )
        all_summaries.append(summary)

    global_summary = {
        "cities": all_summaries,
        "use_filled": args.use_filled,
        "percentile_low": args.p_low,
        "percentile_high": args.p_high,
        "overwrite": args.overwrite,
    }

    save_json(global_summary, LOGS_DIR / "ocm_v1_rgb_global_summary.json")
    save_json(global_summary, REPORT_DIR / "ocm_v1_rgb_global_summary.json")

    print()
    print("=" * 80)
    print("[DONE] RGB rendering finished.")
    print(f"[DONE] Global summary: {LOGS_DIR / 'ocm_v1_rgb_global_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()