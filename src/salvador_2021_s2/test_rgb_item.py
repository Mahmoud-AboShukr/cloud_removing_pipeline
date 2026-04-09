import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

ITEM_ID = "S2-16D_V2_038019_20210728"
BASE = Path(f"data/raw/salvador_test/{ITEM_ID}")

def stretch(arr, low=2, high=98):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vmin = np.percentile(valid, low)
    vmax = np.percentile(valid, high)
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr.astype(np.float32) - vmin) / (vmax - vmin)
    return np.clip(out, 0, 1)

with rasterio.open(BASE / "B04.tif") as src:
    red = src.read(1)

with rasterio.open(BASE / "B03.tif") as src:
    green = src.read(1)

with rasterio.open(BASE / "B02.tif") as src:
    blue = src.read(1)

rgb = np.stack([
    stretch(red),
    stretch(green),
    stretch(blue)
], axis=-1)

plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.title(ITEM_ID)
plt.axis("off")
plt.tight_layout()
plt.show()