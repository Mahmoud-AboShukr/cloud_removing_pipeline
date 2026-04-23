import json
import unicodedata
from pathlib import Path
from typing import Any

import geopandas as gpd
import planetary_computer
import requests
from pystac_client import Client


# =========================
# PATHS / CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLANETARY_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
DATE_RANGE = "2022-01-01/2022-12-31"

# Brazil-wide cleaned polygons
BRAZIL_FAVELAS_PATH = PROJECT_ROOT / "data" / "interim" / "polygons_clean" / "favelas_clean.gpkg"

# City selection
CITY_NAME = "Salvador"
CITY_COLUMN = "nm_mun"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "salvador" / "2022" / "s2_planetary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 3
BANDS = ["B02", "B03", "B04"]
MAX_EO_CLOUD = 50.0


def normalize_text(value: str) -> str:
    """
    Normalize text for robust matching: strip, lowercase, remove accents.
    """
    value = str(value).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value


def load_city_aoi_from_brazil_favelas(
    favelas_path: Path,
    city_name: str,
    city_column: str,
) -> dict[str, Any]:
    """
    Load the Brazil-wide cleaned favela polygons, filter them to one city,
    and return the envelope of their union as a GeoJSON geometry in EPSG:4326.
    """
    if not favelas_path.exists():
        raise FileNotFoundError(f"Favela polygon file not found:\n{favelas_path}")

    gdf = gpd.read_file(favelas_path)

    if gdf.empty:
        raise ValueError(f"No polygons found in {favelas_path}")

    if gdf.crs is None:
        raise ValueError(f"The file has no CRS: {favelas_path}")

    if city_column not in gdf.columns:
        raise ValueError(
            f"Column '{city_column}' not found in the polygon file.\n"
            f"Available columns are:\n{list(gdf.columns)}\n\n"
            f"Please update CITY_COLUMN."
        )

    target_city = normalize_text(city_name)
    city_series = gdf[city_column].fillna("").astype(str).map(normalize_text)
    city_gdf = gdf[city_series == target_city].copy()

    if city_gdf.empty:
        unique_values = sorted(gdf[city_column].dropna().astype(str).unique())[:50]
        raise ValueError(
            f"No polygons found for city '{city_name}' in column '{city_column}'.\n"
            f"Here are some example values from that column:\n{unique_values}\n\n"
            f"Please update CITY_NAME if needed."
        )

    if city_gdf.crs.to_epsg() != 4326:
        city_gdf = city_gdf.to_crs(4326)

    try:
        geom = city_gdf.geometry.union_all()
    except AttributeError:
        geom = city_gdf.unary_union

    envelope = geom.envelope
    return envelope.__geo_interface__


def search_items(aoi_geojson: dict[str, Any]) -> list:
    """
    Search Planetary Computer Sentinel-2 L2A items over the AOI and date range.
    """
    catalog = Client.open(
        PLANETARY_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[COLLECTION],
        intersects=aoi_geojson,
        datetime=DATE_RANGE,
    )

    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-2 items found for the given AOI and date range.")

    return items


def get_cloud_cover(item) -> float:
    """
    Extract eo:cloud_cover safely.
    """
    value = item.properties.get("eo:cloud_cover")
    if value is None:
        return 9999.0
    return float(value)


def rank_items(items: list) -> list:
    """
    Filter and rank items by eo:cloud_cover ascending.
    """
    filtered = [item for item in items if get_cloud_cover(item) <= MAX_EO_CLOUD]
    if not filtered:
        filtered = items

    ranked = sorted(filtered, key=get_cloud_cover)
    return ranked[:TOP_K]


def download_file(url: str, destination: Path) -> None:
    """
    Download a single file to disk.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def save_item_metadata(item, scene_dir: Path) -> None:
    """
    Save the STAC item metadata for traceability.
    """
    scene_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = scene_dir / "item_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(item.to_dict(), f, indent=2)


def download_scene(item) -> None:
    """
    Download the selected RGB bands for a scene.
    """
    scene_id = item.id
    scene_dir = OUTPUT_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    save_item_metadata(item, scene_dir)

    for band in BANDS:
        if band not in item.assets:
            print(f"[WARN] {scene_id} does not contain asset {band}")
            continue

        asset_href = item.assets[band].href
        destination = scene_dir / f"{band}.tif"

        if destination.exists():
            print(f"[SKIP] {destination} already exists")
            continue

        print(f"[DOWNLOAD] {scene_id} -> {band}")
        download_file(asset_href, destination)

    print(f"[DONE] {scene_id}")


def save_ranking_summary(items: list) -> None:
    """
    Save a small JSON summary of the selected scenes.
    """
    summary = []
    for item in items:
        summary.append(
            {
                "id": item.id,
                "datetime": item.properties.get("datetime"),
                "eo:cloud_cover": item.properties.get("eo:cloud_cover"),
                "s2:mgrs_tile": item.properties.get("s2:mgrs_tile"),
            }
        )

    with open(OUTPUT_DIR / "selected_scenes_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    print("[STEP] Loading AOI from Brazil-wide cleaned polygons...")
    print(f"[INFO] Polygon file: {BRAZIL_FAVELAS_PATH}")
    print(f"[INFO] City filter: {CITY_NAME} (column: {CITY_COLUMN})")

    aoi = load_city_aoi_from_brazil_favelas(
        favelas_path=BRAZIL_FAVELAS_PATH,
        city_name=CITY_NAME,
        city_column=CITY_COLUMN,
    )

    print("[STEP] Searching Planetary Computer STAC...")
    items = search_items(aoi)
    print(f"[INFO] Found {len(items)} candidate items.")

    print("[STEP] Ranking items by eo:cloud_cover...")
    best_items = rank_items(items)

    print("[INFO] Best scenes selected:")
    for idx, item in enumerate(best_items, start=1):
        print(
            f"  {idx}. {item.id} | "
            f"cloud={item.properties.get('eo:cloud_cover')} | "
            f"datetime={item.properties.get('datetime')} | "
            f"tile={item.properties.get('s2:mgrs_tile')}"
        )

    save_ranking_summary(best_items)

    print("[STEP] Downloading top scenes...")
    for item in best_items:
        download_scene(item)

    print("[DONE] Search and download completed.")


if __name__ == "__main__":
    main()