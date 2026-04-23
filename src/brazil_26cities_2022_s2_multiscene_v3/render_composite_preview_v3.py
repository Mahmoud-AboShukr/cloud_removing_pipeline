#!/usr/bin/env python3
"""
render_composite_preview_v4.py

Render RGB previews from the V4 cloud-reduced Sentinel-2 composite.

What it does
------------
- Reads:
    data/processed/<city_slug>/2022/s2_planetary_multiscene_12band_v4/city_composite_2022.tif
- Uses RGB bands:
    B04 / B03 / B02
- Applies a JOINT RGB stretch (single shared min/max for all 3 channels)
- Applies optional gamma correction
- Writes:
    preview_rgb.png
- Optionally writes:
    preview_rgb_stretched.tif

Usage examples
--------------
Single city:
python3 src/brazil_26cities_2022_s2_multiscene_v3/render_composite_preview_v4.py \
    --city rio \
    --processed-root data/processed \
    --save-rgb-tif

All cities:
python3 src/brazil_26cities_2022_s2_multiscene_v3/render_composite_preview_v4.py \
    --city all \
    --processed-root data/processed \
    --save-rgb-tif
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from PIL import Image

# -----------------------------------------------------------------------------
# Robust local import of city configs
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

CITY_CONFIGS: Dict[str, Dict[str, Any]] | None = None
_import_errors: List[str] = []

try:
    from cities_config import CITY_CONFIGS as _CFG  # type: ignore
    CITY_CONFIGS = _CFG
except Exception as exc:
    _import_errors.append(f"cities_config.CITY_CONFIGS failed: {exc}")

if CITY_CONFIGS is None:
    try:
        from cities_config_26 import CITY_CONFIGS as _CFG  # type: ignore
        CITY_CONFIGS = _CFG
    except Exception as exc:
        _import_errors.append(f"cities_config_26.CITY_CONFIGS failed: {exc}")

if CITY_CONFIGS is None:
    try:
        from cities_config_26 import CITY_CONFIGS_26 as _CFG  # type: ignore
        CITY_CONFIGS = _CFG
    except Exception as exc:
        _import_errors.append(f"cities_config_26.CITY_CONFIGS_26 failed: {exc}")

if CITY_CONFIGS is None:
    raise ImportError(
        "Could not import city configs.\n"
        f"Looked in: {SCRIPT_DIR}\n"
        "Tried:\n"
        "- cities_config.py -> CITY_CONFIGS\n"
        "- cities_config_26.py -> CITY_CONFIGS\n"
        "- cities_config_26.py -> CITY_CONFIGS_26\n"
        "Errors:\n- " + "\n- ".join(_import_errors)
    )

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

COMPOSITE_PRODUCT_DIR = "s2_planetary_multiscene_12band_v4"
COMPOSITE_FILENAME = "city_composite_2022.tif"
CLOUDMASK_FILENAME = "city_cloudmask_2022.tif"

RGB_BANDS = ("B04", "B03", "B02")

DEFAULT_PMIN = 2.0
DEFAULT_PMAX = 98.0
DEFAULT_GAMMA = 1.0

# -----------------------------------------------------------------------------
# Utilities
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


def get_city_items(city_arg: str) -> List[Tuple[str, Dict[str, Any]]]:
    if city_arg == "all":
        return list(CITY_CONFIGS.items())

    if city_arg not in CITY_CONFIGS:
        raise KeyError(
            f"Unknown city key '{city_arg}'. Available: {list(CITY_CONFIGS.keys())}"
        )
    return [(city_arg, CITY_CONFIGS[city_arg])]

# -----------------------------------------------------------------------------
# Raster helpers
# -----------------------------------------------------------------------------

def read_rgb_from_composite(composite_path: Path) -> Tuple[np.ndarray, Dict[str, Any], List[str]]:
    """
    Read B04/B03/B02 from the composite GeoTIFF using band descriptions.
    Returns:
        rgb: float32 array of shape (3, H, W)
        profile: raster profile
        descriptions: list of band descriptions
    """
    with rasterio.open(composite_path) as src:
        descriptions = list(src.descriptions)
        desc_to_index = {d: i + 1 for i, d in enumerate(descriptions) if d}

        missing = [b for b in RGB_BANDS if b not in desc_to_index]
        if missing:
            raise ValueError(
                f"Composite is missing required RGB band descriptions: {missing}. "
                f"Found descriptions: {descriptions}"
            )

        rgb = np.stack(
            [
                src.read(desc_to_index["B04"]).astype(np.float32),
                src.read(desc_to_index["B03"]).astype(np.float32),
                src.read(desc_to_index["B02"]).astype(np.float32),
            ],
            axis=0,
        )

        profile = src.profile.copy()
        return rgb, profile, descriptions


def build_valid_render_mask(rgb: np.ndarray, cloudmask_path: Path | None = None) -> np.ndarray:
    """
    Build mask of pixels to use for percentile statistics.
    Start from non-zero RGB pixels.
    Optionally exclude unresolved/cloudmask pixels if cloudmask exists.
    """
    valid = np.any(rgb > 0, axis=0)

    if cloudmask_path is not None and cloudmask_path.exists():
        with rasterio.open(cloudmask_path) as src:
            cm = src.read(1)
        valid &= (cm == 0)

    return valid


def compute_joint_stretch_limits(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    pmin: float,
    pmax: float,
) -> Tuple[float, float]:
    """
    Compute one shared lower/upper bound across all RGB values.
    """
    if not np.any(valid_mask):
        raise ValueError("No valid pixels available to compute stretch.")

    values = rgb[:, valid_mask].reshape(-1)

    lo = float(np.percentile(values, pmin))
    hi = float(np.percentile(values, pmax))

    if hi <= lo:
        raise ValueError(
            f"Invalid joint stretch limits: lo={lo}, hi={hi}. "
            "Try adjusting percentiles."
        )

    return lo, hi


def apply_joint_stretch(
    rgb: np.ndarray,
    lo: float,
    hi: float,
    gamma: float,
) -> np.ndarray:
    """
    Stretch RGB to uint8 using shared lo/hi and optional gamma.
    """
    stretched = (rgb - lo) / (hi - lo)
    stretched = np.clip(stretched, 0.0, 1.0)

    if gamma <= 0:
        raise ValueError(f"Gamma must be > 0, got {gamma}")

    if abs(gamma - 1.0) > 1e-9:
        stretched = np.power(stretched, 1.0 / gamma)

    out = np.round(stretched * 255.0).astype(np.uint8)
    return out

# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def render_city(
    city_key: str,
    city_cfg: Dict[str, Any],
    processed_root: Path,
    pmin: float,
    pmax: float,
    gamma: float,
    save_rgb_tif: bool,
) -> Dict[str, Any]:
    city_slug = city_cfg["slug"]
    city_display = city_cfg.get("display_name", city_slug)

    logging.info("=" * 80)
    logging.info("Rendering preview v4: %s", city_display)
    logging.info("=" * 80)

    city_dir = processed_root / city_slug / "2022" / COMPOSITE_PRODUCT_DIR
    composite_path = city_dir / COMPOSITE_FILENAME
    cloudmask_path = city_dir / CLOUDMASK_FILENAME

    if not composite_path.exists():
        raise FileNotFoundError(f"Composite not found: {composite_path}")

    logging.info("Reading composite: %s", composite_path)
    rgb, profile, descriptions = read_rgb_from_composite(composite_path)

    valid_mask = build_valid_render_mask(
        rgb=rgb,
        cloudmask_path=cloudmask_path if cloudmask_path.exists() else None,
    )

    lo, hi = compute_joint_stretch_limits(
        rgb=rgb,
        valid_mask=valid_mask,
        pmin=pmin,
        pmax=pmax,
    )

    logging.info(
        "Joint RGB stretch for %s | pmin=%.1f pmax=%.1f | lo=%.3f hi=%.3f | gamma=%.3f",
        city_display,
        pmin,
        pmax,
        lo,
        hi,
        gamma,
    )

    rgb_u8 = apply_joint_stretch(rgb, lo=lo, hi=hi, gamma=gamma)

    png_path = city_dir / "preview_rgb.png"
    img = np.transpose(rgb_u8, (1, 2, 0))
    Image.fromarray(img).save(png_path)

    logging.info("Saved PNG preview: %s", png_path)

    rgb_tif_path = None
    if save_rgb_tif:
        rgb_tif_path = city_dir / "preview_rgb_stretched.tif"
        rgb_profile = profile.copy()
        rgb_profile.update(
            {
                "driver": "GTiff",
                "count": 3,
                "dtype": "uint8",
                "nodata": 0,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )

        with rasterio.open(rgb_tif_path, "w", **rgb_profile) as dst:
            dst.write(rgb_u8[0], 1)
            dst.write(rgb_u8[1], 2)
            dst.write(rgb_u8[2], 3)
            dst.set_band_description(1, "R_B04")
            dst.set_band_description(2, "G_B03")
            dst.set_band_description(3, "B_B02")

        logging.info("Saved stretched RGB GeoTIFF: %s", rgb_tif_path)

    render_info = {
        "city_key": city_key,
        "city_slug": city_slug,
        "city_display_name": city_display,
        "composite_path": str(composite_path),
        "cloudmask_path": str(cloudmask_path) if cloudmask_path.exists() else None,
        "png_path": str(png_path),
        "rgb_tif_path": str(rgb_tif_path) if rgb_tif_path else None,
        "parameters": {
            "rgb_bands": list(RGB_BANDS),
            "joint_percentile_min": pmin,
            "joint_percentile_max": pmax,
            "gamma": gamma,
        },
        "stretch_limits": {
            "joint_lo": lo,
            "joint_hi": hi,
        },
        "band_descriptions_in_composite": descriptions,
        "valid_render_pixels": int(valid_mask.sum()),
        "total_pixels": int(valid_mask.size),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    json_dump(render_info, city_dir / "preview_render_info.json")
    return render_info

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RGB previews from V4 Sentinel-2 composites using joint RGB stretch."
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
        "--pmin",
        type=float,
        default=DEFAULT_PMIN,
        help="Lower percentile for joint stretch. Default: 2.0",
    )
    parser.add_argument(
        "--pmax",
        type=float,
        default=DEFAULT_PMAX,
        help="Upper percentile for joint stretch. Default: 98.0",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Gamma correction. Default: 1.0",
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

    if not (0.0 <= args.pmin < args.pmax <= 100.0):
        raise ValueError("--pmin and --pmax must satisfy 0 <= pmin < pmax <= 100")

    if args.gamma <= 0:
        raise ValueError("--gamma must be > 0")

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    city_items = get_city_items(args.city)

    overall = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "processed_root": str(args.processed_root),
        "product_dir": COMPOSITE_PRODUCT_DIR,
        "parameters": {
            "rgb_bands": list(RGB_BANDS),
            "pmin": args.pmin,
            "pmax": args.pmax,
            "gamma": args.gamma,
            "save_rgb_tif": args.save_rgb_tif,
        },
        "cities": {},
    }

    for city_key, city_cfg in city_items:
        try:
            info = render_city(
                city_key=city_key,
                city_cfg=city_cfg,
                processed_root=args.processed_root,
                pmin=args.pmin,
                pmax=args.pmax,
                gamma=args.gamma,
                save_rgb_tif=args.save_rgb_tif,
            )
            overall["cities"][city_key] = {
                "status": "ok",
                "png_path": info["png_path"],
                "rgb_tif_path": info["rgb_tif_path"],
                "joint_lo": info["stretch_limits"]["joint_lo"],
                "joint_hi": info["stretch_limits"]["joint_hi"],
            }
        except Exception as exc:
            logging.exception("Rendering failed for city %s: %s", city_key, exc)
            overall["cities"][city_key] = {
                "status": "error",
                "error": str(exc),
            }

    overall_path = args.processed_root / "brazil_26cities_2022_s2_multiscene_v4_render_run_summary.json"
    json_dump(overall, overall_path)

    logging.info("Rendering run finished.")
    logging.info("Global render summary written to: %s", overall_path)


if __name__ == "__main__":
    main()