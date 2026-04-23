from pathlib import Path

import requests
from pystac_client import Client

STAC_URL = "https://data.inpe.br/bdc/stac/v1/"
COLLECTION = "S2-16D-2"

# Change this to test another tile/date
ITEM_ID = "S2-16D_V2_038019_20210728"

BANDS = ["B02", "B03", "B04"]

OUTPUT_DIR = Path(f"data/raw/salvador_test/{ITEM_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, path: Path) -> None:
    if path.exists():
        print(f"[SKIP] {path}")
        return

    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[OK] {path}")


def main() -> None:
    catalog = Client.open(STAC_URL)

    search = catalog.search(
        collections=[COLLECTION],
        ids=[ITEM_ID],
    )
    items = list(search.items())

    if not items:
        raise RuntimeError(f"Item not found: {ITEM_ID}")

    item = items[0]
    print(f"[INFO] Downloading {item.id}")

    for band in BANDS:
        if band not in item.assets:
            raise KeyError(f"Band '{band}' not found. Available: {sorted(item.assets.keys())}")

        url = item.assets[band].href
        out_path = OUTPUT_DIR / f"{band}.tif"
        download_file(url, out_path)

    print("[DONE]")


if __name__ == "__main__":
    main()