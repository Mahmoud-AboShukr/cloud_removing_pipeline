from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import rasterio


def build_rgb(paths: Dict[str, str], city: str) -> str:
    """
    Build a 3-band RGB GeoTIFF from B04, B03, B02.
    """
    required = ["B02", "B03", "B04"]
    missing = [b for b in required if b not in paths]
    if missing:
        raise ValueError(f"Missing required bands for RGB: {missing}")

    out_dir = Path("data/processed") / city
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rgb.tif"

    with rasterio.open(paths["B04"]) as src_r:
        red = src_r.read(1)
        profile = src_r.profile.copy()

    with rasterio.open(paths["B03"]) as src_g:
        green = src_g.read(1)

    with rasterio.open(paths["B02"]) as src_b:
        blue = src_b.read(1)

    rgb = np.stack([red, green, blue], axis=0)

    profile.update(
        count=3,
        dtype=rgb.dtype,
        compress="lzw",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(rgb)

    print(f"[OK] RGB written: {out_path}")
    return str(out_path)