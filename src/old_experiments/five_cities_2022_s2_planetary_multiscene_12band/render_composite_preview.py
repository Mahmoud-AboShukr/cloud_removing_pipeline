#!/usr/bin/env python3
"""
render_composite_preview.py

Create quick-look RGB previews from the cloud-reduced 12-band Sentinel-2 composite.

Expected input
--------------
data/processed/<city_slug>/2022/s2_planetary_multiscene_12band/
    city_composite_2022.tif

Outputs
-------
data/processed/<city_slug>/2022/s2_planetary_multiscene_12band/
    city_rgb_preview.png
    city_rgb_preview_stretched.tif   (optional)

RGB mapping
-----------
Red   = B04
Green = B03
Blue  = B02

Notes
-----
- The composite is expected to contain 12 bands in this order:
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
- The preview applies percentile stretching independently per channel.
- Pixels with value 0 across all RGB channels are treated as empty background.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

# Optional but very useful for PNG writing
import imageio.v2 as imageio

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

# 1-based band positions in the composite
COMPOSITE_BAND_INDEX = {
    "B01": 1,
    "B02": 2,
    "B03": 3,
    "B04": 4,
    "B05": 5,
    "B06": 6,
    "B07": 7,
    "B08": 8,
    "B8A": 9,
    "B09": 10,
    "B11": 11,
    "B12": 12,
}

RGB_BANDS = ("B04", "B03", "B02")


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def get_rgb_band_indices() -> Tuple[int, int, int]:
    """Return 1-based band indices for RGB."""
    return (
        COMPOSITE_BAND_INDEX["B04"],
        COMPOSITE_BAND_INDEX["B03"],
        COMPOSITE_BAND_INDEX["B02"],
    )


def read_rgb_from_composite(composite_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Read RGB bands from the 12-band composite.

    Returns
    -------
    rgb : np.ndarray
        Array of shape (3, H, W)
    profile : dict
        Raster profile copied from source
    """
    if not composite_path.exists():
        raise FileNotFoundError(f"Composite not found: {composite_path}")

    r_idx, g_idx, b_idx = get_rgb_band_indices()

    with rasterio.open(composite_path) as src:
        if src.count < 4:
            raise ValueError(
                f"Composite has only {src.count} bands; expected at least 4."
            )

        rgb = src.read([r_idx, g_idx, b_idx])
        profile = src.profile.copy()

    return rgb, profile


def compute_valid_rgb_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Valid visualization pixels = at least one RGB channel > 0.
    """
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("RGB array must have shape (3, H, W)")
    return np.any(rgb > 0, axis=0)


def stretch_channel(
    channel: np.ndarray,
    valid_mask: np.ndarray,
    p_low: float,
    p_high: float,
) -> np.ndarray:
    """
    Stretch one channel to uint8 [0, 255] using percentiles over valid pixels only.
    """
    out = np.zeros(channel.shape, dtype=np.uint8)

    valid_values = channel[valid_mask]
    if valid_values.size == 0:
        return out

    lo = np.percentile(valid_values, p_low)
    hi = np.percentile(valid_values, p_high)

    if hi <= lo:
        # fallback to min/max if percentile collapses
        lo = float(valid_values.min())
        hi = float(valid_values.max())

    if hi <= lo:
        # completely flat channel
        out[valid_mask] = 0
        return out

    scaled = (channel.astype(np.float32) - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = (scaled * 255.0).astype(np.uint8)

    out[valid_mask] = scaled[valid_mask]
    return out


def stretch_rgb(
    rgb: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    Apply independent percentile stretch to RGB array.

    Parameters
    ----------
    rgb : np.ndarray
        Shape (3, H, W)
    p_low : float
        Lower percentile
    p_high : float
        Upper percentile

    Returns
    -------
    np.ndarray
        Stretched RGB uint8 array of shape (3, H, W)
    """
    valid_mask = compute_valid_rgb_mask(rgb)
    stretched = np.zeros_like(rgb, dtype=np.uint8)

    for i in range(3):
        stretched[i] = stretch_channel(
            channel=rgb[i],
            valid_mask=valid_mask,
            p_low=p_low,
            p_high=p_high,
        )

    return stretched


def save_png(rgb_uint8: np.ndarray, output_path: Path) -> None:
    """
    Save RGB uint8 array to PNG.

    Input shape must be (3, H, W). Output PNG uses (H, W, 3).
    """
    ensure_dir(output_path.parent)

    if rgb_uint8.dtype != np.uint8:
        raise ValueError("PNG output expects uint8 array")

    rgb_hwc = np.moveaxis(rgb_uint8, 0, -1)
    imageio.imwrite(output_path, rgb_hwc)


def save_stretched_rgb_geotiff(
    rgb_uint8: np.ndarray,
    source_profile: dict,
    output_path: Path,
) -> None:
    """
    Save stretched RGB as 3-band GeoTIFF for GIS viewing.
    """
    ensure_dir(output_path.parent)

    profile = source_profile.copy()
    profile.update(
        {
            "count": 3,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgb_uint8[0], 1)
        dst.write(rgb_uint8[1], 2)
        dst.write(rgb_uint8[2], 3)

        dst.set_band_description(1, "Red_B04")
        dst.set_band_description(2, "Green_B03")
        dst.set_band_description(3, "Blue_B02")

        try:
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
            )
        except Exception:
            pass


def process_city(
    city_key: str,
    city_cfg: Dict[str, str],
    processed_root: Path,
    p_low: float,
    p_high: float,
    save_rgb_tif: bool,
) -> Dict[str, str]:
    """
    Render preview for one city.
    """
    city_slug = city_cfg["slug"]
    city_display = city_cfg["display_name"]

    logging.info("=" * 80)
    logging.info("Rendering preview for city: %s", city_display)
    logging.info("=" * 80)

    city_dir = processed_root / city_slug / "2022" / "s2_planetary_multiscene_12band"
    composite_path = city_dir / "city_composite_2022.tif"
    png_path = city_dir / "city_rgb_preview.png"
    tif_path = city_dir / "city_rgb_preview_stretched.tif"

    logging.info("Reading composite: %s", composite_path)
    rgb, profile = read_rgb_from_composite(composite_path)

    logging.info(
        "Composite RGB loaded. Shape=%s dtype=%s",
        rgb.shape,
        rgb.dtype,
    )

    valid_mask = compute_valid_rgb_mask(rgb)
    valid_pixels = int(valid_mask.sum())
    logging.info("Valid RGB pixels for stretch: %d", valid_pixels)

    logging.info(
        "Applying percentile stretch with p_low=%.2f, p_high=%.2f",
        p_low,
        p_high,
    )
    stretched = stretch_rgb(rgb, p_low=p_low, p_high=p_high)

    logging.info("Saving PNG preview: %s", png_path)
    save_png(stretched, png_path)

    if save_rgb_tif:
        logging.info("Saving stretched RGB GeoTIFF: %s", tif_path)
        save_stretched_rgb_geotiff(stretched, profile, tif_path)

    logging.info("Finished preview for city: %s", city_display)

    return {
        "city_key": city_key,
        "city_display_name": city_display,
        "city_slug": city_slug,
        "composite_path": str(composite_path),
        "png_preview_path": str(png_path),
        "rgb_tif_path": str(tif_path) if save_rgb_tif else "",
        "render_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render RGB quick-look previews from cloud-reduced composites."
    )

    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="City key from CITY_CONFIGS, or 'all'. Example: rio",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed"),
        help="Base processed data root.",
    )
    parser.add_argument(
        "--p-low",
        type=float,
        default=2.0,
        help="Lower percentile for stretch.",
    )
    parser.add_argument(
        "--p-high",
        type=float,
        default=98.0,
        help="Upper percentile for stretch.",
    )
    parser.add_argument(
        "--save-rgb-tif",
        action="store_true",
        help="Also save a stretched RGB GeoTIFF.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    args = parser.parse_args()

    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= p_low < p_high <= 100.")

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

    results = []

    for city_key, city_cfg in city_items:
        try:
            result = process_city(
                city_key=city_key,
                city_cfg=city_cfg,
                processed_root=args.processed_root,
                p_low=args.p_low,
                p_high=args.p_high,
                save_rgb_tif=args.save_rgb_tif,
            )
            results.append(result)
        except Exception as exc:
            logging.exception("City %s failed: %s", city_key, exc)

    logging.info("Preview rendering finished for %d city/cities.", len(results))


if __name__ == "__main__":
    main()