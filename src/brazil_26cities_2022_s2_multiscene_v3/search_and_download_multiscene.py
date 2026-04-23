#!/usr/bin/env python3
"""
search_and_download_multiscene.py

Production search + download pipeline for 26 Brazilian cities using Sentinel-2 L2A
from Microsoft Planetary Computer.

Main features
-------------
- Reads cleaned favela polygons from:
    data/interim/polygons_clean/favelas_clean.gpkg
- Builds AOI as envelope(union(city polygons))
- Searches STAC in seasonal windows first, then full-year fallback
- Filters scenes by:
    * overlap with AOI
    * eo:cloud_cover
- Deduplicates same tile + datetime
- Selects a multi-scene set
- Downloads:
    B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12, SCL
- Writes per-city scene summary JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import planetary_computer
import requests
from pystac_client import Client
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

# -----------------------------------------------------------------------------
# Robust local import of cities_config_26.py
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from cities_config_26 import CITY_CONFIGS_26
except ImportError as exc:
    raise ImportError(
        f"Could not import CITY_CONFIGS_26 from cities_config_26.py. "
        f"Expected file at: {SCRIPT_DIR / 'cities_config_26.py'}"
    ) from exc


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

DEFAULT_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
    "SCL",
]

DEFAULT_SEASONAL_WINDOWS_2022 = [
    ("2022-01-01", "2022-03-31"),
    ("2022-04-01", "2022-06-30"),
    ("2022-07-01", "2022-09-30"),
    ("2022-10-01", "2022-12-31"),
]

DOWNLOAD_TIMEOUT = 600
RETRY_SLEEP_SECONDS = 3
USER_AGENT = "brazil-26cities-s2-v3/1.0"


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class SceneCandidate:
    item_id: str
    datetime_utc: str
    date: str
    tile_id: str
    cloud_cover: float
    overlap_ratio: float
    bbox: Tuple[float, float, float, float]
    assets_available: List[str]
    stac_properties: Dict[str, Any]


@dataclass
class DownloadedScene:
    item_id: str
    datetime_utc: str
    date: str
    tile_id: str
    cloud_cover: float
    overlap_ratio: float
    scene_dir: str
    downloaded_assets: Dict[str, str]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def load_city_polygons(
    polygons_path: Path,
    city_name_in_gpkg: str,
    city_column: str,
) -> gpd.GeoDataFrame:
    if not polygons_path.exists():
        raise FileNotFoundError(f"Polygons file not found: {polygons_path}")

    logging.info("Loading cleaned polygons from: %s", polygons_path)
    gdf = gpd.read_file(polygons_path)

    if city_column not in gdf.columns:
        raise KeyError(
            f"Column '{city_column}' not found in {polygons_path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    logging.info("Filtering polygons where %s == %r", city_column, city_name_in_gpkg)
    city_gdf = gdf[gdf[city_column] == city_name_in_gpkg].copy()

    if city_gdf.empty:
        raise ValueError(
            f"No polygons found for city '{city_name_in_gpkg}' in column '{city_column}'."
        )

    city_gdf = city_gdf[city_gdf.geometry.notnull()].copy()
    city_gdf = city_gdf[~city_gdf.geometry.is_empty].copy()

    if city_gdf.empty:
        raise ValueError(f"All geometries are null/empty for city '{city_name_in_gpkg}'.")

    logging.info("Loaded %d polygons for city %s", len(city_gdf), city_name_in_gpkg)
    return city_gdf


def build_city_aoi(city_gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, BaseGeometry]:
    city_gdf_4326 = city_gdf.to_crs(epsg=4326)
    union_geom = city_gdf_4326.geometry.union_all()
    aoi_geom = union_geom.envelope

    aoi_gdf = gpd.GeoDataFrame(
        {"name": ["aoi"]},
        geometry=[aoi_geom],
        crs="EPSG:4326",
    )

    minx, miny, maxx, maxy = aoi_geom.bounds
    logging.info(
        "AOI built. Bounds: minx=%.6f miny=%.6f maxx=%.6f maxy=%.6f",
        minx, miny, maxx, maxy
    )
    return aoi_gdf, aoi_geom


def save_aoi(aoi_gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    aoi_gdf.to_file(output_path, driver="GPKG")
    logging.info("Saved AOI to: %s", output_path)


# -----------------------------------------------------------------------------
# STAC helpers
# -----------------------------------------------------------------------------

def open_stac_client() -> Client:
    return Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def parse_item_datetime(item: Any) -> datetime:
    dt = item.datetime
    if dt is None:
        dt_str = item.properties.get("datetime")
        if not dt_str:
            raise ValueError(f"Item {item.id} has no datetime.")
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    return dt


def get_tile_id(item: Any) -> str:
    props = item.properties
    for key in ("s2:mgrs_tile", "mgrs:tile", "grid:code"):
        if key in props and props[key]:
            return str(props[key])
    return "unknown_tile"


def item_bbox_geometry(item: Any) -> BaseGeometry:
    if not item.bbox:
        raise ValueError(f"Item {item.id} has no bbox.")
    return box(*item.bbox)


def compute_overlap_ratio(aoi_geom: BaseGeometry, item: Any) -> float:
    item_geom = item_bbox_geometry(item)
    inter = aoi_geom.intersection(item_geom)
    if aoi_geom.area == 0:
        return 0.0
    return float(inter.area / aoi_geom.area)


def available_assets(item: Any) -> List[str]:
    return list(item.assets.keys())


def has_required_assets(item: Any, required_assets: Sequence[str]) -> bool:
    item_assets = set(item.assets.keys())
    return all(asset in item_assets for asset in required_assets)


def signed_item(item: Any) -> Any:
    return planetary_computer.sign_inplace(item)


def get_cloud_cover(item: Any) -> float:
    return safe_float(item.properties.get("eo:cloud_cover"), default=100.0)


def scene_to_candidate(aoi_geom: BaseGeometry, item: Any) -> SceneCandidate:
    dt = parse_item_datetime(item)
    return SceneCandidate(
        item_id=item.id,
        datetime_utc=dt.isoformat(),
        date=dt.strftime("%Y-%m-%d"),
        tile_id=get_tile_id(item),
        cloud_cover=get_cloud_cover(item),
        overlap_ratio=compute_overlap_ratio(aoi_geom, item),
        bbox=tuple(float(v) for v in item.bbox),
        assets_available=available_assets(item),
        stac_properties=dict(item.properties),
    )


def deduplicate_candidates(candidates: Sequence[SceneCandidate]) -> List[SceneCandidate]:
    best_by_key: Dict[Tuple[str, str], SceneCandidate] = {}

    for cand in candidates:
        key = (cand.datetime_utc, cand.tile_id)
        existing = best_by_key.get(key)

        if existing is None:
            best_by_key[key] = cand
            continue

        replace = False
        if cand.overlap_ratio > existing.overlap_ratio:
            replace = True
        elif math.isclose(cand.overlap_ratio, existing.overlap_ratio, rel_tol=1e-9):
            if cand.cloud_cover < existing.cloud_cover:
                replace = True

        if replace:
            best_by_key[key] = cand

    out = list(best_by_key.values())
    logging.info("Deduplicated candidates: %d -> %d", len(candidates), len(out))
    return out


def sort_candidates_for_selection(candidates: Sequence[SceneCandidate]) -> List[SceneCandidate]:
    return sorted(
        candidates,
        key=lambda c: (
            c.overlap_ratio,
            -c.cloud_cover,
            c.date,
        ),
        reverse=True,
    )


def search_window(
    client: Client,
    aoi_geom: BaseGeometry,
    start_date: str,
    end_date: str,
    cloud_max: float,
    required_assets: Sequence[str],
) -> List[SceneCandidate]:
    intersects_geojson = mapping(aoi_geom)
    datetime_range = f"{start_date}/{end_date}"

    logging.info("Searching STAC window %s to %s ...", start_date, end_date)

    search = client.search(
        collections=[COLLECTION],
        intersects=intersects_geojson,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": cloud_max}},
    )

    items = list(search.items())
    logging.info("Found %d raw candidate items.", len(items))

    candidates: List[SceneCandidate] = []
    skipped_missing_assets = 0
    skipped_metadata_error = 0

    for item in items:
        if not has_required_assets(item, required_assets):
            skipped_missing_assets += 1
            continue

        try:
            cand = scene_to_candidate(aoi_geom, item)
        except Exception as exc:
            logging.warning("Skipping item %s due to metadata error: %s", item.id, exc)
            skipped_metadata_error += 1
            continue

        candidates.append(cand)

    logging.info(
        "After asset/metadata checks: kept=%d, skipped_missing_assets=%d, skipped_metadata_error=%d",
        len(candidates),
        skipped_missing_assets,
        skipped_metadata_error,
    )

    return candidates


def filter_candidates_by_overlap(
    candidates: Sequence[SceneCandidate],
    preferred_overlap: float,
    fallback_overlap: float,
) -> List[SceneCandidate]:
    preferred = [c for c in candidates if c.overlap_ratio >= preferred_overlap]
    if preferred:
        logging.info(
            "Using preferred overlap threshold %.2f (%d scenes kept).",
            preferred_overlap,
            len(preferred),
        )
        return preferred

    fallback = [c for c in candidates if c.overlap_ratio >= fallback_overlap]
    if fallback:
        logging.info(
            "Preferred overlap unavailable; using fallback %.2f (%d scenes kept).",
            fallback_overlap,
            len(fallback),
        )
        return fallback

    logging.warning(
        "No scenes reached preferred or fallback overlap thresholds. Keeping all candidates as last resort."
    )
    return list(candidates)


def collect_candidates(
    client: Client,
    aoi_geom: BaseGeometry,
    cloud_max: float,
    required_assets: Sequence[str],
    preferred_overlap: float,
    fallback_overlap: float,
    min_candidate_pool: int,
) -> Tuple[List[SceneCandidate], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "seasonal_windows": [],
        "full_year_fallback_used": False,
    }

    all_candidates: List[SceneCandidate] = []

    for start_date, end_date in DEFAULT_SEASONAL_WINDOWS_2022:
        window_candidates = search_window(
            client=client,
            aoi_geom=aoi_geom,
            start_date=start_date,
            end_date=end_date,
            cloud_max=cloud_max,
            required_assets=required_assets,
        )
        window_candidates = deduplicate_candidates(window_candidates)
        window_candidates = filter_candidates_by_overlap(
            window_candidates,
            preferred_overlap=preferred_overlap,
            fallback_overlap=fallback_overlap,
        )

        diagnostics["seasonal_windows"].append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "num_candidates_after_filter": len(window_candidates),
            }
        )

        all_candidates.extend(window_candidates)

    all_candidates = deduplicate_candidates(all_candidates)

    if len(all_candidates) < min_candidate_pool:
        diagnostics["full_year_fallback_used"] = True
        logging.info(
            "Seasonal search produced only %d candidates. Falling back to full year.",
            len(all_candidates),
        )

        full_candidates = search_window(
            client=client,
            aoi_geom=aoi_geom,
            start_date="2022-01-01",
            end_date="2022-12-31",
            cloud_max=cloud_max,
            required_assets=required_assets,
        )
        full_candidates = deduplicate_candidates(full_candidates)
        full_candidates = filter_candidates_by_overlap(
            full_candidates,
            preferred_overlap=preferred_overlap,
            fallback_overlap=fallback_overlap,
        )

        combined = {c.item_id: c for c in all_candidates}
        for cand in full_candidates:
            combined[cand.item_id] = cand
        all_candidates = list(combined.values())

    all_candidates = sort_candidates_for_selection(all_candidates)
    diagnostics["num_total_candidates"] = len(all_candidates)

    logging.info("Total candidate pool after all filtering: %d", len(all_candidates))
    return all_candidates, diagnostics


# -----------------------------------------------------------------------------
# Scene selection
# -----------------------------------------------------------------------------

def select_multiscene_set(
    candidates: Sequence[SceneCandidate],
    min_scenes: int,
    max_scenes: int,
) -> List[SceneCandidate]:
    if not candidates:
        return []

    ranked = sort_candidates_for_selection(candidates)
    selected: List[SceneCandidate] = []
    selected_dates: set[str] = set()

    base = ranked[0]
    selected.append(base)
    selected_dates.add(base.date)

    for cand in ranked[1:]:
        if len(selected) >= max_scenes:
            break
        if cand.date not in selected_dates:
            selected.append(cand)
            selected_dates.add(cand.date)

    if len(selected) < min_scenes:
        selected_ids = {c.item_id for c in selected}
        for cand in ranked[1:]:
            if len(selected) >= min_scenes:
                break
            if cand.item_id not in selected_ids:
                selected.append(cand)
                selected_ids.add(cand.item_id)

    return selected[:max_scenes]


# -----------------------------------------------------------------------------
# Downloading
# -----------------------------------------------------------------------------

def requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def stream_download(
    url: str,
    output_path: Path,
    session: requests.Session,
    max_retries: int = 3,
    chunk_size: int = 1024 * 1024,
) -> bool:
    ensure_dir(output_path.parent)

    if output_path.exists() and output_path.stat().st_size > 0:
        logging.info("Skip existing file: %s", output_path)
        return True

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            logging.info("HTTP GET attempt %d/%d for %s", attempt, max_retries, output_path.name)

            with session.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
                resp.raise_for_status()

                total_size = int(resp.headers.get("Content-Length", 0))
                if total_size > 0:
                    logging.info(
                        "Downloading %s (%.2f MB)",
                        output_path.name,
                        total_size / (1024 * 1024),
                    )
                else:
                    logging.info("Downloading %s (size unknown)", output_path.name)

                bytes_written = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)

            tmp_path.replace(output_path)

            logging.info(
                "Saved %s (%.2f MB)",
                output_path,
                bytes_written / (1024 * 1024),
            )
            return True

        except Exception as exc:
            logging.warning(
                "Download failed (%s), attempt %d/%d for %s",
                exc,
                attempt,
                max_retries,
                output_path.name,
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < max_retries:
                time.sleep(RETRY_SLEEP_SECONDS)

    return False


def fetch_item_by_id(client: Client, item_id: str) -> Any:
    collection = client.get_collection(COLLECTION)
    item = collection.get_item(item_id)
    if item is None:
        raise ValueError(f"Could not retrieve item by ID: {item_id}")
    return signed_item(item)


def download_scene_assets(
    client: Client,
    candidate: SceneCandidate,
    output_scene_dir: Path,
    assets_to_download: Sequence[str],
    session: requests.Session,
) -> Optional[DownloadedScene]:
    logging.info("Fetching signed item metadata for scene: %s", candidate.item_id)

    try:
        item = fetch_item_by_id(client, candidate.item_id)
    except Exception as exc:
        logging.error("Failed to fetch signed item %s: %s", candidate.item_id, exc)
        return None

    ensure_dir(output_scene_dir)
    downloaded_assets: Dict[str, str] = {}

    total_assets = len(assets_to_download)

    for asset_idx, asset_key in enumerate(assets_to_download, start=1):
        if asset_key not in item.assets:
            logging.error("Missing asset %s in item %s", asset_key, candidate.item_id)
            return None

        asset = item.assets[asset_key]
        href = asset.href
        if not href:
            logging.error("Asset %s in item %s has no href", asset_key, candidate.item_id)
            return None

        filename = f"{asset_key}.tif"
        output_path = output_scene_dir / filename

        logging.info(
            "Downloading scene %s | asset %d/%d | %s -> %s",
            candidate.item_id,
            asset_idx,
            total_assets,
            asset_key,
            output_path,
        )

        ok = stream_download(href, output_path, session=session)
        if not ok:
            logging.error("Failed asset download for %s / %s", candidate.item_id, asset_key)
            return None

        logging.info(
            "Finished scene %s | asset %d/%d | %s",
            candidate.item_id,
            asset_idx,
            total_assets,
            asset_key,
        )

        downloaded_assets[asset_key] = str(output_path)

    logging.info("All assets downloaded successfully for scene %s", candidate.item_id)

    return DownloadedScene(
        item_id=candidate.item_id,
        datetime_utc=candidate.datetime_utc,
        date=candidate.date,
        tile_id=candidate.tile_id,
        cloud_cover=candidate.cloud_cover,
        overlap_ratio=candidate.overlap_ratio,
        scene_dir=str(output_scene_dir),
        downloaded_assets=downloaded_assets,
    )


# -----------------------------------------------------------------------------
# City processing
# -----------------------------------------------------------------------------

def process_city(
    city_key: str,
    city_cfg: Dict[str, Any],
    polygons_path: Path,
    output_root: Path,
    client: Client,
    cloud_max: float,
    preferred_overlap: float,
    fallback_overlap: float,
    min_scenes: int,
    max_scenes: int,
    assets_to_download: Sequence[str],
) -> Dict[str, Any]:
    city_display = city_cfg["display_name"]
    city_filter_value = city_cfg["polygon_city_name"]
    city_column = city_cfg.get("polygon_city_column", "nm_mun")
    city_slug = city_cfg["slug"]

    logging.info("=" * 80)
    logging.info("Processing city: %s", city_display)
    logging.info("=" * 80)

    raw_city_root = output_root / city_slug / "2022" / "s2_planetary_multiscene_12band_v3"
    ensure_dir(raw_city_root)
    logging.info("Output directory: %s", raw_city_root)

    city_gdf = load_city_polygons(
        polygons_path=polygons_path,
        city_name_in_gpkg=city_filter_value,
        city_column=city_column,
    )

    aoi_gdf, aoi_geom = build_city_aoi(city_gdf)

    aoi_path = raw_city_root / f"{city_slug}_aoi.gpkg"
    save_aoi(aoi_gdf, aoi_path)

    candidates, diagnostics = collect_candidates(
        client=client,
        aoi_geom=aoi_geom,
        cloud_max=cloud_max,
        required_assets=assets_to_download,
        preferred_overlap=preferred_overlap,
        fallback_overlap=fallback_overlap,
        min_candidate_pool=max(min_scenes, max_scenes),
    )

    if not candidates:
        logging.warning("No valid candidates found for %s", city_display)
        summary = {
            "city_key": city_key,
            "city_display_name": city_display,
            "city_slug": city_slug,
            "status": "no_candidates",
            "aoi_path": str(aoi_path),
            "search_diagnostics": diagnostics,
            "selected_candidates": [],
            "downloaded_scenes": [],
        }
        json_dump(summary, raw_city_root / "selected_scenes_summary.json")
        return summary

    selected = select_multiscene_set(
        candidates=candidates,
        min_scenes=min_scenes,
        max_scenes=max_scenes,
    )

    logging.info("Selected %d scene(s) for %s.", len(selected), city_display)
    for i, cand in enumerate(selected, start=1):
        logging.info(
            "  [%d] %s | date=%s | tile=%s | overlap=%.3f | cloud=%.1f",
            i,
            cand.item_id,
            cand.date,
            cand.tile_id,
            cand.overlap_ratio,
            cand.cloud_cover,
        )

    session = requests_session()
    downloaded: List[DownloadedScene] = []

    selected_ids = {c.item_id for c in selected}
    ranked_all = sort_candidates_for_selection(candidates)
    backup_pool = [c for c in ranked_all if c.item_id not in selected_ids]
    backup_iter = iter(backup_pool)

    effective_targets = list(selected)

    idx = 0
    while idx < len(effective_targets):
        cand = effective_targets[idx]
        scene_dir = raw_city_root / "scenes" / cand.item_id

        logging.info(
            "Starting download for selected scene %d/%d: %s",
            idx + 1,
            len(effective_targets),
            cand.item_id,
        )

        result = download_scene_assets(
            client=client,
            candidate=cand,
            output_scene_dir=scene_dir,
            assets_to_download=assets_to_download,
            session=session,
        )

        if result is not None:
            downloaded.append(result)
            logging.info(
                "Completed selected scene %d/%d: %s",
                idx + 1,
                len(effective_targets),
                cand.item_id,
            )
            idx += 1
            continue

        logging.warning(
            "Scene %s failed to download. Attempting replacement candidate.",
            cand.item_id,
        )

        replacement = None
        current_ids = {s.item_id for s in effective_targets}
        for backup in backup_iter:
            if backup.item_id not in current_ids:
                replacement = backup
                break

        if replacement is None:
            logging.warning("No replacement candidates left for %s.", city_display)
            idx += 1
            continue

        logging.info(
            "Replacing failed scene %s with backup %s",
            cand.item_id,
            replacement.item_id,
        )
        effective_targets[idx] = replacement

    summary = {
        "city_key": city_key,
        "city_display_name": city_display,
        "city_slug": city_slug,
        "status": "ok" if downloaded else "download_failed",
        "polygons_path": str(polygons_path),
        "aoi_path": str(aoi_path),
        "search_parameters": {
            "stac_url": STAC_URL,
            "collection": COLLECTION,
            "cloud_max": cloud_max,
            "preferred_overlap": preferred_overlap,
            "fallback_overlap": fallback_overlap,
            "min_scenes": min_scenes,
            "max_scenes": max_scenes,
            "assets_to_download": list(assets_to_download),
        },
        "search_diagnostics": diagnostics,
        "selected_candidates": [asdict(c) for c in effective_targets],
        "downloaded_scenes": [asdict(d) for d in downloaded],
    }

    json_dump(summary, raw_city_root / "selected_scenes_summary.json")

    lines = [
        f"City: {city_display}",
        f"Slug: {city_slug}",
        "",
        "Selected scenes:",
    ]
    for i, cand in enumerate(effective_targets, start=1):
        lines.append(
            f"{i}. {cand.item_id} | date={cand.date} | tile={cand.tile_id} | "
            f"overlap={cand.overlap_ratio:.3f} | cloud={cand.cloud_cover:.1f}"
        )
    lines.append("")
    lines.append(f"Downloaded successfully: {len(downloaded)} scene(s)")
    write_text(raw_city_root / "selected_scenes_summary.txt", "\n".join(lines))

    logging.info(
        "Finished city %s. Downloaded %d/%d selected scenes.",
        city_display,
        len(downloaded),
        len(effective_targets),
    )

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and download multi-scene Sentinel-2 data for 26 Brazilian cities."
    )

    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="City key from CITY_CONFIGS_26, or 'all'. Example: fortaleza",
    )
    parser.add_argument(
        "--polygons-path",
        type=Path,
        default=Path("data/interim/polygons_clean/favelas_clean.gpkg"),
        help="Path to cleaned favela polygons GPKG.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/raw"),
        help="Base output root directory.",
    )
    parser.add_argument(
        "--cloud-max",
        type=float,
        default=70.0,
        help="Maximum eo:cloud_cover accepted during search.",
    )
    parser.add_argument(
        "--preferred-overlap",
        type=float,
        default=0.90,
        help="Preferred minimum overlap ratio.",
    )
    parser.add_argument(
        "--fallback-overlap",
        type=float,
        default=0.70,
        help="Fallback minimum overlap ratio.",
    )
    parser.add_argument(
        "--min-scenes",
        type=int,
        default=3,
        help="Minimum number of scenes to select.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=6,
        help="Maximum number of scenes to select.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=DEFAULT_BANDS,
        help="Assets to download. Default: 12 bands + SCL.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    args = parser.parse_args()

    if args.min_scenes < 1:
        raise ValueError("--min-scenes must be >= 1")
    if args.max_scenes < args.min_scenes:
        raise ValueError("--max-scenes must be >= --min-scenes")
    if args.fallback_overlap > args.preferred_overlap:
        raise ValueError("--fallback-overlap must be <= --preferred-overlap")

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    logging.info("Opening Planetary Computer STAC client...")
    client = open_stac_client()

    if args.city == "all":
        city_items = list(CITY_CONFIGS_26.items())
    else:
        if args.city not in CITY_CONFIGS_26:
            raise KeyError(
                f"Unknown city key '{args.city}'. "
                f"Available: {list(CITY_CONFIGS_26.keys())}"
            )
        city_items = [(args.city, CITY_CONFIGS_26[args.city])]

    overall_results: Dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stac_url": STAC_URL,
        "collection": COLLECTION,
        "cities": {},
    }

    for city_key, city_cfg in city_items:
        try:
            summary = process_city(
                city_key=city_key,
                city_cfg=city_cfg,
                polygons_path=args.polygons_path,
                output_root=args.output_root,
                client=client,
                cloud_max=args.cloud_max,
                preferred_overlap=args.preferred_overlap,
                fallback_overlap=args.fallback_overlap,
                min_scenes=args.min_scenes,
                max_scenes=args.max_scenes,
                assets_to_download=args.bands,
            )
            overall_results["cities"][city_key] = summary

        except Exception as exc:
            logging.exception("City %s failed: %s", city_key, exc)
            overall_results["cities"][city_key] = {
                "city_key": city_key,
                "status": "error",
                "error": str(exc),
            }

    overall_path = args.output_root / "brazil_26cities_2022_s2_multiscene_v3_download_run_summary.json"
    json_dump(overall_results, overall_path)

    logging.info("Run finished.")
    logging.info("Global summary written to: %s", overall_path)


if __name__ == "__main__":
    main()