from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import shutil
import tempfile

import numpy as np
import requests
import rasterio
from pystac_client import Client


def _download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[SKIP] Already exists: {out_path}")
        return

    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    print(f"[OK] Downloaded: {out_path}")


def _read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return arr


def _safe_percentile(arr: np.ndarray, q: float) -> float:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0
    return float(np.percentile(valid, q))


def _stretch(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    vmin = _safe_percentile(arr, low)
    vmax = _safe_percentile(arr, high)
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def _score_preview_rgb(b02_path: Path, b03_path: Path, b04_path: Path) -> Tuple[float, Dict[str, float]]:
    """
    Heuristic scorer to avoid ocean-dominated / glint-dominated tiles.

    Returns a score where higher is better, plus diagnostics.
    """
    blue = _stretch(_read_band(b02_path))
    green = _stretch(_read_band(b03_path))
    red = _stretch(_read_band(b04_path))

    brightness = (red + green + blue) / 3.0
    whiteness = 1.0 - (
        (np.abs(red - green) + np.abs(red - blue) + np.abs(green - blue)) / 3.0
    )

    # Water / ocean tends to be relatively dark and often blue-dominant.
    water_like = (brightness < 0.22) & (blue > green) & (green >= red)

    # Bright white cloud/glint texture.
    cloud_like = (brightness > 0.72) & (whiteness > 0.82)

    # Land-like heuristic:
    # not too dark, not too white, some color variation, not classic water signature
    land_like = (
        (brightness > 0.18)
        & (brightness < 0.85)
        & (whiteness < 0.90)
        & (~water_like)
    )

    total = brightness.size
    water_ratio = float(np.sum(water_like)) / total
    cloud_ratio = float(np.sum(cloud_like)) / total
    land_ratio = float(np.sum(land_like)) / total

    # Prefer land, penalize water/clouds.
    score = land_ratio - 0.8 * water_ratio - 0.6 * cloud_ratio

    diagnostics = {
        "land_ratio": land_ratio,
        "water_ratio": water_ratio,
        "cloud_ratio": cloud_ratio,
        "score": score,
    }
    return score, diagnostics


def _download_preview_triplet(item, tmp_dir: Path) -> Dict[str, Path]:
    out = {}
    for band in ["B02", "B03", "B04"]:
        if band not in item.assets:
            raise KeyError(f"Preview band '{band}' not found for item {item.id}")
        band_path = tmp_dir / f"{band}.tif"
        _download_file(item.assets[band].href, band_path)
        out[band] = band_path
    return out


def _choose_best_item(items: List, config: Dict) -> object:
    """
    Download lightweight preview RGB for each candidate, score it, and choose the best item.
    """
    preferred_tile_ids = config.get("preferred_tile_ids", [])
    use_manual_tile_priority = bool(preferred_tile_ids)

    ranked = []

    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)

        for idx, item in enumerate(items):
            item_tmp = tmp_root / f"{idx}_{item.id.replace('/', '_')}"
            item_tmp.mkdir(parents=True, exist_ok=True)

            try:
                print(f"[PREVIEW] Testing item: {item.id}")
                preview_paths = _download_preview_triplet(item, item_tmp)
                score, diag = _score_preview_rgb(
                    preview_paths["B02"],
                    preview_paths["B03"],
                    preview_paths["B04"],
                )

                tile_bonus = 0.0
                if use_manual_tile_priority:
                    for rank, tile_id in enumerate(preferred_tile_ids):
                        if tile_id in item.id:
                            tile_bonus = 1.0 - 0.1 * rank
                            break

                final_score = score + tile_bonus

                ranked.append(
                    {
                        "item": item,
                        "score": final_score,
                        "base_score": score,
                        "tile_bonus": tile_bonus,
                        "diag": diag,
                    }
                )

                print(
                    "[PREVIEW] "
                    f"{item.id} | score={final_score:.4f} "
                    f"(base={score:.4f}, tile_bonus={tile_bonus:.2f}) | "
                    f"land={diag['land_ratio']:.3f} "
                    f"water={diag['water_ratio']:.3f} "
                    f"cloud={diag['cloud_ratio']:.3f}"
                )
            except Exception as exc:
                print(f"[WARN] Failed preview scoring for {item.id}: {exc}")

    if not ranked:
        raise RuntimeError("Could not score any candidate BDC items.")

    ranked.sort(key=lambda x: x["score"], reverse=True)

    print("\n[INFO] Top ranked candidates:")
    for row in ranked[:5]:
        print(
            f"  {row['item'].id} | score={row['score']:.4f} | "
            f"land={row['diag']['land_ratio']:.3f} "
            f"water={row['diag']['water_ratio']:.3f} "
            f"cloud={row['diag']['cloud_ratio']:.3f}"
        )

    chosen = ranked[0]["item"]
    print(f"\n[STEP] Best chosen item: {chosen.id}")
    return chosen


def download_one_item(config: Dict) -> Dict[str, str]:
    """
    Search BDC for the experiment AOI/date range, score the candidates, choose the
    best tile/date, and download the requested bands.

    Config keys expected:
    - stac_url
    - collection
    - bbox
    - datetime
    - bands
    - raw_dir
    Optional:
    - preferred_tile_ids: list[str]
    """
    stac_url = config["stac_url"]
    collection = config["collection"]
    bbox = config["bbox"]
    dt = config["datetime"]
    bands: List[str] = config["bands"]
    raw_dir = Path(config["raw_dir"])

    print("[STEP] Opening BDC STAC...")
    catalog = Client.open(stac_url)

    print("[STEP] Searching items...")
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=dt,
    )
    items = sorted(list(search.items()), key=lambda x: x.datetime or "")
    print(f"[INFO] Found {len(items)} item(s).")

    if not items:
        raise RuntimeError("No items found for the given bbox/date range.")

    print("[INFO] Matching items:")
    for i, item in enumerate(items):
        print(f"  [{i}] {item.id} | {item.datetime}")

    item = _choose_best_item(items, config)

    item_dir = raw_dir / item.id
    item_dir.mkdir(parents=True, exist_ok=True)

    band_paths: Dict[str, str] = {}
    for band in bands:
        if band not in item.assets:
            raise KeyError(
                f"Band '{band}' not found in item assets for {item.id}: "
                f"{sorted(item.assets.keys())}"
            )

        out_path = item_dir / f"{band}.tif"
        _download_file(item.assets[band].href, out_path)
        band_paths[band] = str(out_path)

    return band_paths