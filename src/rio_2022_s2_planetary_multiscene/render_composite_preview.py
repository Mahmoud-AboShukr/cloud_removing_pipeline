from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPOSITE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rio_de_janeiro"
    / "2022"
    / "s2_planetary_multiscene"
    / "composite"
)

PREVIEW_DIR = COMPOSITE_DIR / "previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

COMPOSITE_PATH = COMPOSITE_DIR / "rio_de_janeiro_composite_2022.tif"
CLOUDMASK_PATH = COMPOSITE_DIR / "rio_de_janeiro_cloudmask_2022.tif"
CONTRIBUTION_MAP_PATH = COMPOSITE_DIR / "rio_de_janeiro_scene_contribution_map.tif"

RGB_PREVIEW_PATH = PREVIEW_DIR / "rio_de_janeiro_composite_rgb_preview.png"
CLOUDMASK_PREVIEW_PATH = PREVIEW_DIR / "rio_de_janeiro_cloudmask_preview.png"
CONTRIBUTION_PREVIEW_PATH = PREVIEW_DIR / "rio_de_janeiro_scene_contribution_map_preview.png"

LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98


# =============================================================================
# HELPERS
# =============================================================================
def read_band_by_description(dataset: rasterio.io.DatasetReader, band_name: str) -> np.ndarray:
    """
    Read a band from a multiband GeoTIFF using band description.
    """
    descriptions = list(dataset.descriptions)

    if band_name not in descriptions:
        raise ValueError(
            f"Band '{band_name}' not found in dataset descriptions.\n"
            f"Available descriptions: {descriptions}"
        )

    band_index = descriptions.index(band_name) + 1
    return dataset.read(band_index).astype(np.float32)


def stretch_band(band: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Percentile stretch for visualization.
    Handles NaN safely.
    """
    valid = np.isfinite(band)
    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)

    p_low, p_high = np.percentile(band[valid], [low, high])

    if p_high <= p_low:
        return np.zeros_like(band, dtype=np.float32)

    stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    stretched[~valid] = 0
    return stretched.astype(np.float32)


def build_rgb_from_composite(composite_path: Path) -> np.ndarray:
    """
    Build RGB from composite using:
    - B04 as red
    - B03 as green
    - B02 as blue
    """
    if not composite_path.exists():
        raise FileNotFoundError(f"Composite file not found:\n{composite_path}")

    with rasterio.open(composite_path) as src:
        red = read_band_by_description(src, "B04")
        green = read_band_by_description(src, "B03")
        blue = read_band_by_description(src, "B02")

    r = stretch_band(red, LOW_PERCENTILE, HIGH_PERCENTILE)
    g = stretch_band(green, LOW_PERCENTILE, HIGH_PERCENTILE)
    b = stretch_band(blue, LOW_PERCENTILE, HIGH_PERCENTILE)

    rgb = np.dstack([r, g, b])
    return rgb


def save_rgb_preview(rgb: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def build_cloudmask_preview_array(cloudmask_path: Path) -> np.ndarray:
    """
    Convert cloudmask to a simple preview image.

    Convention from previous script:
    - 0 = valid inside AOI
    - 1 = invalid/unresolved inside AOI
    - 255 = outside AOI
    """
    if not cloudmask_path.exists():
        raise FileNotFoundError(f"Cloudmask file not found:\n{cloudmask_path}")

    with rasterio.open(cloudmask_path) as src:
        cloudmask = src.read(1)

    preview = np.ones((*cloudmask.shape, 3), dtype=np.float32)

    # valid inside AOI -> white
    preview[cloudmask == 0] = [1.0, 1.0, 1.0]

    # unresolved invalid pixels -> red
    preview[cloudmask == 1] = [1.0, 0.0, 0.0]

    # outside AOI -> black
    preview[cloudmask == 255] = [0.0, 0.0, 0.0]

    return preview


def build_contribution_preview_array(contribution_path: Path) -> np.ndarray:
    """
    Build a preview for the scene contribution map.
    Uses a discrete colormap.
    """
    if not contribution_path.exists():
        raise FileNotFoundError(f"Contribution map file not found:\n{contribution_path}")

    with rasterio.open(contribution_path) as src:
        contribution = src.read(1)

    # Mask outside AOI (255) so it shows cleanly
    masked = np.ma.masked_where(contribution == 255, contribution)
    return masked


def save_cloudmask_preview(preview_rgb: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(preview_rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_contribution_preview(contribution_masked: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(contribution_masked, interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04, label="Scene rank used")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=" * 80)
    print("[STEP] Rendering RGB preview from final composite...")
    print("=" * 80)

    rgb = build_rgb_from_composite(COMPOSITE_PATH)
    save_rgb_preview(rgb, RGB_PREVIEW_PATH)
    print(f"[DONE] Saved RGB preview: {RGB_PREVIEW_PATH}")

    print("=" * 80)
    print("[STEP] Rendering cloudmask preview...")
    print("=" * 80)

    cloudmask_preview = build_cloudmask_preview_array(CLOUDMASK_PATH)
    save_cloudmask_preview(cloudmask_preview, CLOUDMASK_PREVIEW_PATH)
    print(f"[DONE] Saved cloudmask preview: {CLOUDMASK_PREVIEW_PATH}")

    print("=" * 80)
    print("[STEP] Rendering scene contribution map preview...")
    print("=" * 80)

    contribution_preview = build_contribution_preview_array(CONTRIBUTION_MAP_PATH)
    save_contribution_preview(contribution_preview, CONTRIBUTION_PREVIEW_PATH)
    print(f"[DONE] Saved contribution map preview: {CONTRIBUTION_PREVIEW_PATH}")

    print("=" * 80)
    print("[DONE] render_composite_preview.py completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()