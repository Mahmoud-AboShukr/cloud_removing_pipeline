from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "salvador" / "2022" / "s2_planetary"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "salvador" / "2022" / "s2_planetary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1).astype(np.float32)
    return band


def stretch_band(band: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(band)
    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)

    p_low, p_high = np.percentile(band[valid], [low, high])

    if p_high <= p_low:
        return np.zeros_like(band, dtype=np.float32)

    stretched = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    return stretched.astype(np.float32)


def build_rgb(scene_dir: Path) -> np.ndarray:
    b04 = read_band(scene_dir / "B04.tif")  # red
    b03 = read_band(scene_dir / "B03.tif")  # green
    b02 = read_band(scene_dir / "B02.tif")  # blue

    r = stretch_band(b04, LOW_PERCENTILE, HIGH_PERCENTILE)
    g = stretch_band(b03, LOW_PERCENTILE, HIGH_PERCENTILE)
    b = stretch_band(b02, LOW_PERCENTILE, HIGH_PERCENTILE)

    return np.dstack([r, g, b])


def save_rgb_preview(rgb: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def is_valid_scene_dir(scene_dir: Path) -> bool:
    required = ["B02.tif", "B03.tif", "B04.tif"]
    return scene_dir.is_dir() and all((scene_dir / name).exists() for name in required)


def main() -> None:
    scene_dirs = sorted([p for p in INPUT_DIR.iterdir() if is_valid_scene_dir(p)])

    if not scene_dirs:
        raise FileNotFoundError(
            f"No valid scene folders found in {INPUT_DIR}\n"
            "Expected subfolders containing B02.tif, B03.tif, and B04.tif."
        )

    print(f"[INFO] Found {len(scene_dirs)} downloaded scene(s).")

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        print(f"[STEP] Rendering {scene_id}...")

        rgb = build_rgb(scene_dir)

        output_png = OUTPUT_DIR / f"{scene_id}_rgb_preview.png"
        save_rgb_preview(rgb, output_png)

        print(f"[DONE] Saved {output_png}")

    print("[DONE] All previews rendered.")


if __name__ == "__main__":
    main()