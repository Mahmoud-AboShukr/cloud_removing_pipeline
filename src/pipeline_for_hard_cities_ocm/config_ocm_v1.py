# src/pipeline_for_hard_cities_ocm/config_ocm_v1.py

from __future__ import annotations

from pathlib import Path


# =============================================================================
# ROOT PATHS
# =============================================================================

# Input root containing the already-downloaded Sentinel-2 scenes.
# Expected structure (default):
# /media/HALLOPEAU/T7/raw_backup/<city>/2022/s2_planetary_multiscene_12band_v3/<scene_id>/
RAW_BACKUP_ROOT = Path("/media/HALLOPEAU/T7/raw_backup")

# Output root for the new OCM_V1 experiment.
OUTPUT_ROOT = Path("/media/HALLOPEAU/T7/ocm_v1_outputs")

# Buffered AOIs previously created in the repo.
AOI_DIR = Path("data/interim/aoi_buffered_hard_cities")

# Optional repo-side reports for logs and summaries.
REPORT_DIR = Path("reports/pipeline_for_hard_cities_ocm")


# =============================================================================
# TARGET CITIES
# =============================================================================

TARGET_CITIES = [
    "belem",
    "duque_de_caxias",
    "joao_pessoa",
    "maceio",
    "natal",
    "recife",
    "sao_luis",
]


# =============================================================================
# EXPECTED RAW DIRECTORY STRUCTURE
# =============================================================================

RAW_YEAR_SUBDIR = "2022"
RAW_EXPERIMENT_SUBDIR = "s2_planetary_multiscene_12band_v3"

# Expected per-scene files
REQUIRED_RGB_NIR_BANDS_FOR_OCM = ["B03.tif", "B04.tif", "B08.tif"]
OPTIONAL_SCL_FILE = "SCL.tif"

ALL_EXPECTED_BAND_FILES = [
    "B01.tif",
    "B02.tif",
    "B03.tif",
    "B04.tif",
    "B05.tif",
    "B06.tif",
    "B07.tif",
    "B08.tif",
    "B8A.tif",
    "B09.tif",
    "B11.tif",
    "B12.tif",
]

ALL_EXPECTED_SCENE_FILES = ALL_EXPECTED_BAND_FILES + [OPTIONAL_SCL_FILE]


# =============================================================================
# OCM_V1 COMPOSITING DEFAULTS
# =============================================================================

# OmniCloudMask class semantics:
# 0 = clear
# 1 = thick cloud
# 2 = thin cloud
# 3 = cloud shadow
OCM_CLEAR_VALUE = 0
OCM_BAD_VALUES = [1, 2, 3]

# Cloud-edge safety dilation for later compositing.
DEFAULT_DILATE_ITERS = 3

# Small-hole filling threshold for later compositing.
DEFAULT_MAX_HOLE_PIXELS = 1500

# Default nodata values.
DEFAULT_MASK_NODATA = 255
DEFAULT_COMPOSITE_NODATA = 0


# =============================================================================
# RENDERING DEFAULTS
# =============================================================================

RGB_BAND_MAP = {
    "red": "B04",
    "green": "B03",
    "blue": "B02",
}

RENDER_PERCENTILE_LOW = 2.0
RENDER_PERCENTILE_HIGH = 98.0


# =============================================================================
# OUTPUT SUBDIRECTORIES
# =============================================================================

INVENTORY_DIR = OUTPUT_ROOT / "inventory"
OCM_MASKS_DIR = OUTPUT_ROOT / "ocm_masks"
GRID_SPECS_DIR = OUTPUT_ROOT / "grid_specs"
ALIGNED_DIR = OUTPUT_ROOT / "aligned"
COMPOSITES_DIR = OUTPUT_ROOT / "composites"
RGB_DIR = OUTPUT_ROOT / "rgb"
QC_DIR = OUTPUT_ROOT / "qc"
LOGS_DIR = OUTPUT_ROOT / "logs"


# =============================================================================
# HELPERS
# =============================================================================

def get_city_raw_root(city_slug: str) -> Path:
    """
    Return the raw scene root for a city.

    Example:
    /media/HALLOPEAU/T7/raw_backup/recife/2022/s2_planetary_multiscene_12band_v3
    """
    return RAW_BACKUP_ROOT / city_slug / RAW_YEAR_SUBDIR / RAW_EXPERIMENT_SUBDIR


def get_city_inventory_csv(city_slug: str) -> Path:
    return INVENTORY_DIR / f"{city_slug}_scene_inventory.csv"


def get_city_inventory_json(city_slug: str) -> Path:
    return LOGS_DIR / city_slug / f"{city_slug}_scene_inventory_summary.json"


def get_city_ocm_mask_dir(city_slug: str) -> Path:
    return OCM_MASKS_DIR / city_slug


def get_city_log_dir(city_slug: str) -> Path:
    return LOGS_DIR / city_slug


def get_city_report_dir(city_slug: str) -> Path:
    return REPORT_DIR / city_slug