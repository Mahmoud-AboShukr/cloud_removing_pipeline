import json
import unicodedata
from pathlib import Path
from typing import Any

import geopandas as gpd
import planetary_computer
import requests
from pystac_client import Client
from shapely.geometry import shape


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
CITY_NAME = "Brasilia"
CITY_COLUMN = "nm_mun"

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "brasilia" / "2022" / "s2_planetary"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 3
BANDS = ["B02", "B03", "B04"]

# Ranking controls
MAX_EO_CLOUD = 50.0
MIN_AOI_OVERLAP_RATIO = 0.90  # keep only scenes covering at least 90% of AOI envelope


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


def get_acquisition_datetime(item) -> str:
    """
    Use sensing datetime to identify duplicate acquisitions.
    """
    return str(item.properties.get("datetime", ""))


def compute_aoi_overlap_ratio(item, aoi_geojson: dict[str, Any]) -> float:
    """
    Compute how much of the AOI is covered by the STAC item footprint.
    Ratio = intersection_area / aoi_area
    """
    item_geom = item.geometry
    if item_geom is None:
        return 0.0

    aoi_shape = shape(aoi_geojson)
    item_shape = shape(item_geom)

    if not item_shape.is_valid or not aoi_shape.is_valid:
        return 0.0

    if not item_shape.intersects(aoi_shape):
        return 0.0

    inter = item_shape.intersection(aoi_shape)
    if aoi_shape.area == 0:
        return 0.0

    return float(inter.area / aoi_shape.area)


def annotate_items(items: list, aoi_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Add derived ranking metadata to each item.
    """
    annotated = []
    for item in items:
        annotated.append(
            {
                "item": item,
                "cloud_cover": get_cloud_cover(item),
                "datetime": get_acquisition_datetime(item),
                "overlap_ratio": compute_aoi_overlap_ratio(item, aoi_geojson),
            }
        )
    return annotated


def filter_by_overlap_and_cloud(annotated_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep scenes with good AOI coverage and acceptable cloudiness.
    """
    filtered = [
        x for x in annotated_items
        if x["overlap_ratio"] >= MIN_AOI_OVERLAP_RATIO and x["cloud_cover"] <= MAX_EO_CLOUD
    ]

    # fallback 1: ignore cloud threshold, keep strong overlap
    if not filtered:
        filtered = [
            x for x in annotated_items
            if x["overlap_ratio"] >= MIN_AOI_OVERLAP_RATIO
        ]

    # fallback 2: if AOI coverage threshold is too strict, keep all and rank later
    if not filtered:
        filtered = annotated_items

    return filtered


def deduplicate_by_datetime(annotated_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep one best item per acquisition datetime.
    Preference:
    1. higher overlap ratio
    2. lower cloud cover
    """
    best_by_datetime: dict[str, dict[str, Any]] = {}

    for entry in annotated_items:
        dt = entry["datetime"]
        if dt not in best_by_datetime:
            best_by_datetime[dt] = entry
            continue

        current = best_by_datetime[dt]
        if (
            entry["overlap_ratio"] > current["overlap_ratio"]
            or (
                entry["overlap_ratio"] == current["overlap_ratio"]
                and entry["cloud_cover"] < current["cloud_cover"]
            )
        ):
            best_by_datetime[dt] = entry

    return list(best_by_datetime.values())


def rank_items(items: list, aoi_geojson: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Final ranking:
    1. annotate with overlap/cloud metadata
    2. filter poor overlap
    3. remove duplicate acquisitions
    4. sort by highest overlap, then lowest cloud
    """
    annotated = annotate_items(items, aoi_geojson)
    filtered = filter_by_overlap_and_cloud(annotated)
    deduped = deduplicate_by_datetime(filtered)

    ranked = sorted(
        deduped,
        key=lambda x: (-x["overlap_ratio"], x["cloud_cover"], x["datetime"])
    )

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


def save_item_metadata(item, scene_dir: Path, extra: dict[str, Any]) -> None:
    """
    Save the STAC item metadata and ranking metadata for traceability.
    """
    scene_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "item_metadata.json", "w", encoding="utf-8") as f:
        json.dump(item.to_dict(), f, indent=2)

    with open(scene_dir / "ranking_metadata.json", "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2)


def download_scene(entry: dict[str, Any]) -> None:
    """
    Download the selected RGB bands for a ranked scene.
    """
    item = entry["item"]
    scene_id = item.id
    scene_dir = OUTPUT_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    save_item_metadata(
        item,
        scene_dir,
        extra={
            "datetime": entry["datetime"],
            "cloud_cover": entry["cloud_cover"],
            "overlap_ratio": entry["overlap_ratio"],
        },
    )

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


def save_ranking_summary(ranked_entries: list[dict[str, Any]]) -> None:
    """
    Save a JSON summary of the selected scenes.
    """
    summary = []
    for entry in ranked_entries:
        item = entry["item"]
        summary.append(
            {
                "id": item.id,
                "datetime": entry["datetime"],
                "eo:cloud_cover": entry["cloud_cover"],
                "overlap_ratio": entry["overlap_ratio"],
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

    print("[STEP] Ranking items by AOI overlap and eo:cloud_cover...")
    ranked_entries = rank_items(items, aoi)

    print("[INFO] Best scenes selected:")
    for idx, entry in enumerate(ranked_entries, start=1):
        item = entry["item"]
        print(
            f"  {idx}. {item.id} | "
            f"overlap={entry['overlap_ratio']:.3f} | "
            f"cloud={entry['cloud_cover']:.3f} | "
            f"datetime={entry['datetime']} | "
            f"tile={item.properties.get('s2:mgrs_tile')}"
        )

    save_ranking_summary(ranked_entries)

    print("[STEP] Downloading top scenes...")
    for entry in ranked_entries:
        download_scene(entry)

    print("[DONE] Search and download completed.")


if __name__ == "__main__":
    main()