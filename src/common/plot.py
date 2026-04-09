from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import rasterio


def _stretch_band(band: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)

    vmin = np.percentile(valid, low)
    vmax = np.percentile(valid, high)

    if vmax <= vmin:
        return np.zeros_like(band, dtype=np.float32)

    out = (band.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def plot_rgb(rgb_path: str) -> None:
    """
    Display RGB with percentile stretch for quick visual inspection.
    """
    with rasterio.open(rgb_path) as src:
        rgb = src.read()  # shape: (3, H, W)

    rgb = np.transpose(rgb, (1, 2, 0))  # HWC

    stretched = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        stretched[:, :, i] = _stretch_band(rgb[:, :, i])

    plt.figure(figsize=(10, 10))
    plt.imshow(stretched)
    plt.title("Salvador RGB")
    plt.axis("off")
    plt.tight_layout()
    plt.show()