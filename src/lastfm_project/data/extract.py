from pathlib import Path
import zipfile


ZIP_PATH = Path("data/raw/hetrec2011-lastfm-2k.zip")
EXTRACT_DIR = Path("data/extracted")


def extract_dataset() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    print(f"Dataset extraído em: {EXTRACT_DIR}")


if __name__ == "__main__":
    extract_dataset()