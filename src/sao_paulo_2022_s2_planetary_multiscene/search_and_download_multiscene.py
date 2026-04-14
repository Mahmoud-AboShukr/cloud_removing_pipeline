import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import planetary_computer
import requests
from pystac_client import Client
from shapely.geometry import shape


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLANETARY_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

BRAZIL_FAVELAS_PATH = (
    PROJECT_ROOT / "data" / "interim" / "polygons_clean" / "favelas_clean.gpkg"
)

CITY_NAME = "Sao Paulo"
CITY_COLUMN = "nm_mun"

RAW_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sao_paulo"
    / "2022"
    / "s2_planetary_multiscene"
)
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_RANGE_ALL_2022 = "2022-01-01/2022-12-31"

# Seasonal-first search windows.
# Brasília typically has a clearer dry season, so we start with mid-year windows first.
SEASONAL_WINDOWS = [
    ("2022-05-01", "2022-09-30"),
    ("2022-04-01", "2022-08-31"),
    ("2022-06-01", "2022-10-31"),
]

MIN_SCENES_PER_CITY = 3
MAX_SCENES_PER_CITY = 8

# 10 optical reflectance bands + SCL
BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "SCL",
]

PREFERRED_MIN_AOI_OVERLAP_RATIO = 0.90
FALLBACK_MIN_AOI_OVERLAP_RATIO = 0.70

PREFERRED_MAX_EO_CLOUD = 60.0
FALLBACK_MAX_EO_CLOUD = 70.0

REQUEST_TIMEOUT_SECONDS = 180
CHUNK_SIZE_BYTES = 1024 * 1024


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class RankedItem:
    item: Any
    datetime: str
    mgrs_tile: str
    cloud_cover: float
    overlap_ratio: float
    source_window: str
    filter_level: str


# =============================================================================
# HELPERS
# =============================================================================
def normalize_text(value: str) -> str:
    """
    Normalize text for robust matching:
    strip, lowercase, remove accents.
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
    Load Brazil-wide cleaned favela polygons, filter them to one city,
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
            f"Available columns are:\n{list(gdf.columns)}"
        )

    target_city = normalize_text(city_name)
    city_series = gdf[city_column].fillna("").astype(str).map(normalize_text)
    city_gdf = gdf[city_series == target_city].copy()

    if city_gdf.empty:
        unique_values = sorted(gdf[city_column].dropna().astype(str).unique())[:100]
        raise ValueError(
            f"No polygons found for city '{city_name}' in column '{city_column}'.\n"
            f"Some example values are:\n{unique_values}"
        )

    if city_gdf.crs.to_epsg() != 4326:
        city_gdf = city_gdf.to_crs(4326)

    try:
        geom = city_gdf.geometry.union_all()
    except AttributeError:
        geom = city_gdf.unary_union

    envelope = geom.envelope
    return envelope.__geo_interface__


def open_catalog() -> Client:
    """
    Open the Planetary Computer STAC catalog.
    """
    return Client.open(
        PLANETARY_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )


def search_items(
    catalog: Client,
    aoi_geojson: dict[str, Any],
    date_range: str,
) -> list:
    """
    Search Sentinel-2 L2A items over AOI and date range.
    """
    search = catalog.search(
        collections=[COLLECTION],
        intersects=aoi_geojson,
        datetime=date_range,
    )
    return list(search.items())


def get_cloud_cover(item: Any) -> float:
    """
    Extract eo:cloud_cover safely.
    """
    value = item.properties.get("eo:cloud_cover")
    if value is None:
        return 9999.0
    return float(value)


def get_acquisition_datetime(item: Any) -> str:
    """
    Extract sensing datetime for deduplication and ranking.
    """
    return str(item.properties.get("datetime", ""))


def get_mgrs_tile(item: Any) -> str:
    """
    Extract MGRS tile safely.
    """
    return str(item.properties.get("s2:mgrs_tile", ""))


def compute_aoi_overlap_ratio(item: Any, aoi_geojson: dict[str, Any]) -> float:
    """
    Compute AOI overlap ratio:
    area(item footprint ∩ AOI) / area(AOI)
    """
    item_geom = item.geometry
    if item_geom is None:
        return 0.0

    item_shape = shape(item_geom)
    aoi_shape = shape(aoi_geojson)

    if not item_shape.is_valid or not aoi_shape.is_valid:
        return 0.0

    if not item_shape.intersects(aoi_shape):
        return 0.0

    aoi_area = aoi_shape.area
    if aoi_area == 0:
        return 0.0

    overlap = item_shape.intersection(aoi_shape).area / aoi_area
    return float(overlap)


def annotate_items(
    items: list,
    aoi_geojson: dict[str, Any],
    source_window: str,
) -> list[RankedItem]:
    """
    Add ranking metadata to each STAC item.
    """
    annotated: list[RankedItem] = []

    for item in items:
        annotated.append(
            RankedItem(
                item=item,
                datetime=get_acquisition_datetime(item),
                mgrs_tile=get_mgrs_tile(item),
                cloud_cover=get_cloud_cover(item),
                overlap_ratio=compute_aoi_overlap_ratio(item, aoi_geojson),
                source_window=source_window,
                filter_level="unfiltered",
            )
        )

    return annotated


def filter_candidates(
    entries: list[RankedItem],
    min_overlap_ratio: float,
    max_cloud_cover: float,
    filter_level: str,
) -> list[RankedItem]:
    """
    Filter by overlap and scene-level cloud cover.
    """
    filtered = [
        RankedItem(
            item=e.item,
            datetime=e.datetime,
            mgrs_tile=e.mgrs_tile,
            cloud_cover=e.cloud_cover,
            overlap_ratio=e.overlap_ratio,
            source_window=e.source_window,
            filter_level=filter_level,
        )
        for e in entries
        if e.overlap_ratio >= min_overlap_ratio and e.cloud_cover <= max_cloud_cover
    ]
    return filtered


def deduplicate_by_datetime_and_tile(entries: list[RankedItem]) -> list[RankedItem]:
    """
    Deduplicate using:
    - sensing datetime
    - MGRS tile

    Keep the best processing instance using:
    1. higher overlap ratio
    2. lower cloud cover
    3. lexicographically smaller item id as stable tie-breaker
    """
    best: dict[tuple[str, str], RankedItem] = {}

    for entry in entries:
        key = (entry.datetime, entry.mgrs_tile)

        if key not in best:
            best[key] = entry
            continue

        current = best[key]
        current_id = str(current.item.id)
        new_id = str(entry.item.id)

        better = (
            (entry.overlap_ratio > current.overlap_ratio)
            or (
                entry.overlap_ratio == current.overlap_ratio
                and entry.cloud_cover < current.cloud_cover
            )
            or (
                entry.overlap_ratio == current.overlap_ratio
                and entry.cloud_cover == current.cloud_cover
                and new_id < current_id
            )
        )

        if better:
            best[key] = entry

    return list(best.values())


def sort_ranked_entries(entries: list[RankedItem]) -> list[RankedItem]:
    """
    Current pre-download ranking:
    1. highest AOI overlap
    2. lowest scene cloud cover
    3. earliest acquisition datetime
    4. item id for stable output
    """
    return sorted(
        entries,
        key=lambda e: (
            -e.overlap_ratio,
            e.cloud_cover,
            e.datetime,
            str(e.item.id),
        ),
    )


def select_candidate_pool(
    seasonal_entries: list[RankedItem],
    all_year_entries: list[RankedItem],
) -> list[RankedItem]:
    """
    Apply progressive relaxation policy:

    1. seasonal + preferred overlap + preferred cloud
    2. seasonal + fallback overlap + preferred cloud
    3. seasonal + fallback overlap + fallback cloud
    4. all-year + preferred overlap + preferred cloud
    5. all-year + fallback overlap + preferred cloud
    6. all-year + fallback overlap + fallback cloud

    Stop at the first level that yields at least MIN_SCENES_PER_CITY.
    """
    filter_configs = [
        (
            seasonal_entries,
            PREFERRED_MIN_AOI_OVERLAP_RATIO,
            PREFERRED_MAX_EO_CLOUD,
            "seasonal_preferred_overlap_preferred_cloud",
        ),
        (
            seasonal_entries,
            FALLBACK_MIN_AOI_OVERLAP_RATIO,
            PREFERRED_MAX_EO_CLOUD,
            "seasonal_fallback_overlap_preferred_cloud",
        ),
        (
            seasonal_entries,
            FALLBACK_MIN_AOI_OVERLAP_RATIO,
            FALLBACK_MAX_EO_CLOUD,
            "seasonal_fallback_overlap_fallback_cloud",
        ),
        (
            all_year_entries,
            PREFERRED_MIN_AOI_OVERLAP_RATIO,
            PREFERRED_MAX_EO_CLOUD,
            "all_year_preferred_overlap_preferred_cloud",
        ),
        (
            all_year_entries,
            FALLBACK_MIN_AOI_OVERLAP_RATIO,
            PREFERRED_MAX_EO_CLOUD,
            "all_year_fallback_overlap_preferred_cloud",
        ),
        (
            all_year_entries,
            FALLBACK_MIN_AOI_OVERLAP_RATIO,
            FALLBACK_MAX_EO_CLOUD,
            "all_year_fallback_overlap_fallback_cloud",
        ),
    ]

    for base_entries, min_overlap, max_cloud, level_name in filter_configs:
        filtered = filter_candidates(
            entries=base_entries,
            min_overlap_ratio=min_overlap,
            max_cloud_cover=max_cloud,
            filter_level=level_name,
        )
        deduped = deduplicate_by_datetime_and_tile(filtered)
        ranked = sort_ranked_entries(deduped)

        print(
            f"[INFO] Filter level '{level_name}' -> "
            f"{len(filtered)} filtered, {len(deduped)} deduplicated."
        )

        if len(ranked) >= MIN_SCENES_PER_CITY:
            return ranked[:MAX_SCENES_PER_CITY]

    # Final fallback: deduplicate and sort everything from all-year search
    # even if it does not satisfy preferred thresholds.
    deduped_all = deduplicate_by_datetime_and_tile(all_year_entries)
    ranked_all = sort_ranked_entries(deduped_all)

    print(
        "[WARN] Could not satisfy minimum scene count under configured filters. "
        "Using best available all-year deduplicated items."
    )

    return ranked_all[:MAX_SCENES_PER_CITY]


def download_file(url: str, destination: Path) -> None:
    """
    Download a single file to disk.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                if chunk:
                    f.write(chunk)


def save_scene_metadata(entry: RankedItem, scene_dir: Path) -> None:
    """
    Save full STAC item metadata and ranking metadata.
    """
    scene_dir.mkdir(parents=True, exist_ok=True)

    with open(scene_dir / "item_metadata.json", "w", encoding="utf-8") as f:
        json.dump(entry.item.to_dict(), f, indent=2)

    ranking_metadata = {
        "city_name": CITY_NAME,
        "datetime": entry.datetime,
        "s2:mgrs_tile": entry.mgrs_tile,
        "eo:cloud_cover": entry.cloud_cover,
        "overlap_ratio": entry.overlap_ratio,
        "source_window": entry.source_window,
        "filter_level": entry.filter_level,
        "downloaded_assets": BANDS,
    }

    with open(scene_dir / "ranking_metadata.json", "w", encoding="utf-8") as f:
        json.dump(ranking_metadata, f, indent=2)


def download_scene(entry: RankedItem) -> None:
    """
    Download all required assets for one ranked scene.
    """
    item = entry.item
    scene_id = str(item.id)
    scene_dir = RAW_OUTPUT_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    save_scene_metadata(entry, scene_dir)

    for asset_name in BANDS:
        if asset_name not in item.assets:
            print(f"[WARN] {scene_id} missing asset '{asset_name}'.")
            continue

        destination = scene_dir / f"{asset_name}.tif"
        if destination.exists():
            print(f"[SKIP] Already exists: {destination}")
            continue

        asset_href = item.assets[asset_name].href
        print(f"[DOWNLOAD] {scene_id} -> {asset_name}")
        download_file(asset_href, destination)

    print("[DONE] Sao Paulo multiscene search and download completed.")
    


def save_selected_scenes_summary(
    selected_entries: list[RankedItem],
    seasonal_counts: dict[str, int],
    all_year_count: int,
) -> None:
    """
    Save summary JSON for the selected scenes and search context.
    """
    summary = {
        "city_name": CITY_NAME,
        "city_column": CITY_COLUMN,
        "collection": COLLECTION,
        "date_range_all_2022": DATE_RANGE_ALL_2022,
        "seasonal_windows": SEASONAL_WINDOWS,
        "min_scenes_per_city": MIN_SCENES_PER_CITY,
        "max_scenes_per_city": MAX_SCENES_PER_CITY,
        "bands": BANDS,
        "preferred_min_overlap_ratio": PREFERRED_MIN_AOI_OVERLAP_RATIO,
        "fallback_min_overlap_ratio": FALLBACK_MIN_AOI_OVERLAP_RATIO,
        "preferred_max_eo_cloud": PREFERRED_MAX_EO_CLOUD,
        "fallback_max_eo_cloud": FALLBACK_MAX_EO_CLOUD,
        "search_counts": {
            "seasonal_window_candidates": seasonal_counts,
            "all_year_candidates": all_year_count,
        },
        "selected_scenes": [],
    }

    for idx, entry in enumerate(selected_entries, start=1):
        summary["selected_scenes"].append(
            {
                "rank": idx,
                "id": str(entry.item.id),
                "datetime": entry.datetime,
                "s2:mgrs_tile": entry.mgrs_tile,
                "eo:cloud_cover": entry.cloud_cover,
                "overlap_ratio": entry.overlap_ratio,
                "source_window": entry.source_window,
                "filter_level": entry.filter_level,
            }
        )

    with open(
        RAW_OUTPUT_DIR / "selected_scenes_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    print("=" * 80)
    print("[STEP] Loading city AOI from Brazil-wide cleaned polygons...")
    print("=" * 80)
    print(f"[INFO] Polygon file: {BRAZIL_FAVELAS_PATH}")
    print(f"[INFO] City filter: {CITY_NAME} (column: {CITY_COLUMN})")

    aoi_geojson = load_city_aoi_from_brazil_favelas(
        favelas_path=BRAZIL_FAVELAS_PATH,
        city_name=CITY_NAME,
        city_column=CITY_COLUMN,
    )

    print("=" * 80)
    print("[STEP] Opening Planetary Computer STAC catalog...")
    print("=" * 80)
    catalog = open_catalog()

    # -------------------------------------------------------------------------
    # Search seasonal windows first
    # -------------------------------------------------------------------------
    seasonal_entries: list[RankedItem] = []
    seasonal_counts: dict[str, int] = {}

    print("=" * 80)
    print("[STEP] Searching seasonal candidate windows...")
    print("=" * 80)

    for start_date, end_date in SEASONAL_WINDOWS:
        date_range = f"{start_date}/{end_date}"
        items = search_items(
            catalog=catalog,
            aoi_geojson=aoi_geojson,
            date_range=date_range,
        )
        seasonal_counts[date_range] = len(items)

        print(f"[INFO] Window {date_range}: {len(items)} candidate item(s).")

        seasonal_entries.extend(
            annotate_items(
                items=items,
                aoi_geojson=aoi_geojson,
                source_window=date_range,
            )
        )

    # -------------------------------------------------------------------------
    # Search full year as fallback / backup pool
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[STEP] Searching full-year candidate pool...")
    print("=" * 80)

    all_year_items = search_items(
        catalog=catalog,
        aoi_geojson=aoi_geojson,
        date_range=DATE_RANGE_ALL_2022,
    )
    all_year_count = len(all_year_items)
    print(f"[INFO] Full-year 2022: {all_year_count} candidate item(s).")

    all_year_entries = annotate_items(
        items=all_year_items,
        aoi_geojson=aoi_geojson,
        source_window=DATE_RANGE_ALL_2022,
    )

    # -------------------------------------------------------------------------
    # Select final candidate pool
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[STEP] Selecting best multiscene candidate pool...")
    print("=" * 80)

    selected_entries = select_candidate_pool(
        seasonal_entries=seasonal_entries,
        all_year_entries=all_year_entries,
    )

    if len(selected_entries) < MIN_SCENES_PER_CITY:
        raise RuntimeError(
            f"Only {len(selected_entries)} suitable scene(s) found for {CITY_NAME}. "
            f"Minimum required is {MIN_SCENES_PER_CITY}."
        )

    print("[INFO] Selected scenes:")
    for idx, entry in enumerate(selected_entries, start=1):
        print(
            f"  {idx}. {entry.item.id} | "
            f"overlap={entry.overlap_ratio:.3f} | "
            f"cloud={entry.cloud_cover:.2f} | "
            f"datetime={entry.datetime} | "
            f"tile={entry.mgrs_tile} | "
            f"window={entry.source_window} | "
            f"filter={entry.filter_level}"
        )

    save_selected_scenes_summary(
        selected_entries=selected_entries,
        seasonal_counts=seasonal_counts,
        all_year_count=all_year_count,
    )

    # -------------------------------------------------------------------------
    # Download
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("[STEP] Downloading selected scenes...")
    print("=" * 80)

    for entry in selected_entries:
        try:
            download_scene(entry)
        except Exception as exc:
            print(f"[ERROR] Failed downloading {entry.item.id}: {exc}")

    print("=" * 80)
    print("[DONE] Brasília multiscene search and download completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()