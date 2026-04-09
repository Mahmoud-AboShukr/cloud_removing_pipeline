from src.download.bdc import download_one_item
from src.preprocess.rgb import build_rgb
from src.visualize.plot import plot_rgb
from experiments.salvador.config import SALVADOR_CONFIG


def main() -> None:
    print("[STEP 1] Downloading BDC data...")
    paths = download_one_item(SALVADOR_CONFIG)

    print("[STEP 2] Building RGB...")
    rgb_path = build_rgb(paths, city=SALVADOR_CONFIG["name"])

    print("[STEP 3] Plotting RGB...")
    plot_rgb(rgb_path)

    print("[DONE] Salvador experiment completed.")


if __name__ == "__main__":
    main()