# Cloud Removal Pipeline for Sentinel-2 (Brazil – Favela Dataset Project)

## Overview

This repository contains a series of experiments and pipelines developed to produce **cloud-free Sentinel-2 composites** over Brazilian cities, as part of a broader project aiming to build a **deep-learning-ready dataset for favela detection and segmentation**.

The work explores multiple strategies for cloud removal, ranging from **SCL-based compositing** to **OmniCloudMask (OCM)-based approaches**, with a focus on robustness in challenging tropical environments.

---

## Project Objective

The goal is to generate **high-quality, cloud-reduced Sentinel-2 imagery** aligned with refined favela labels, enabling:

* Semantic segmentation of informal settlements
* Large-scale dataset creation across Brazil
* Robust training data under real-world atmospheric conditions

---

## Repository Structure

```
.
├── src/
│   ├── experiment_03_brazil_26cities_scl/      # Main SCL-based multi-scene pipeline (26 cities)
│   ├── experiment_04_hard_cities_ocm/          # OCM-based pipeline for difficult cases
│   └── method_development/                    # Early experiments and prototypes
│
├── logs/                                      # Execution logs and batch summaries
├── reports/                                   # Experiment summaries and diagnostics
│
├── main.py                                    # Entry point (optional orchestration)
├── requirements.txt
├── LICENSE
└── README.md
```

⚠️ Note:

* The `data/` directory is excluded from version control (large geospatial files).
* Only code, logs, and experiment metadata are tracked.

---

## Methods Overview

### 1. SCL-Based Multi-Scene Compositing

Located in:

```
src/experiment_03_brazil_26cities_scl/
```

This method:

* Uses Sentinel-2 **Scene Classification Layer (SCL)**
* Filters clouds and shadows at the pixel level
* Combines multiple scenes (3–6 per city)
* Applies **iterative filling strategies** to reduce cloud coverage

#### Key Features

* Multi-scene compositing
* Dynamic pixel filling (V3 → V7 improvements)
* AOI-aware ranking of scenes

#### Limitations

* Struggles in **high-cloud tropical regions**
* Residual artifacts may persist
* Dependent on SCL accuracy

---

### 2. OCM-Based Pipeline (OmniCloudMask)

Located in:

```
src/experiment_04_hard_cities_ocm/
```

This method:

* Uses **OmniCloudMask (OCM)** instead of SCL
* Generates more accurate cloud and shadow masks
* Applies reprojection + grid alignment
* Builds composite using cleaner masks

#### Key Features

* Better cloud detection in difficult conditions
* Improved results for cities like:

  * Belém
  * Recife
  * São Luís
  * Maceió

#### Limitations

* Additional preprocessing steps
* More computationally expensive
* Requires pre-downloaded scenes

---

### 3. Method Development (Early Experiments)

Located in:

```
src/method_development/
```

Contains:

* Single-city experiments
* Early multi-scene prototypes
* Different compositing strategies
* Testing pipelines (S2, S1, etc.)

These are preserved for:

* Reproducibility
* Method comparison
* Research documentation

---

## Results Summary

* **SCL pipeline** performs well in moderate cloud conditions
* **OCM pipeline** significantly improves results in high-cloud regions
* Combined approach allows full coverage of **26 major Brazilian cities**

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run SCL pipeline (example)

```bash
cd src/experiment_03_brazil_26cities_scl

python batch_run_all_cities.py
```

---

### 3. Run OCM pipeline (example)

```bash
cd src/experiment_04_hard_cities_ocm

python 01_inventory_raw_backup.py
python 02_generate_ocm_masks.py
python 03_prepare_city_grid.py
python 04_reproject_city_scenes.py
python 05_build_ocm_v1_composite.py
python 06_render_ocm_v1_rgb.py
```

---

## Logs and Reports

* `logs/`: execution logs per city and batch runs
* `reports/`: global summaries, diagnostics, and evaluation outputs

These files help:

* Debug pipeline runs
* Compare methods
* Track performance across cities

---

## Key Challenges

* Persistent cloud coverage in tropical regions
* Limitations of SCL cloud detection
* Temporal mismatch between scenes
* Trade-off between coverage and visual quality

---

## Future Work

* Hybrid SCL + OCM approaches
* Learning-based cloud removal
* Integration with Sentinel-1 data
* Automated quality scoring for composites

---

## License

This project is released under the terms of the LICENSE file included in this repository.

---

## Author

Mahmoud Abo Shukr
---
