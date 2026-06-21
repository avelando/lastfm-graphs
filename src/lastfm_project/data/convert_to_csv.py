from pathlib import Path
import pandas as pd


EXTRACTED_DIR = Path("data/extracted")
CSV_DIR = Path("data/interim/csv")


FILES_TO_CONVERT = {
    "artists.dat": "artists.csv",
    "tags.dat": "tags.csv",
    "user_artists.dat": "user_artists.csv",
    "user_friends.dat": "user_friends.csv",
    "user_taggedartists.dat": "user_taggedartists.csv",
    "user_taggedartists-timestamps.dat": "user_taggedartists_timestamps.csv",
}


def convert_dat_to_csv(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, sep="\t", encoding="latin-1")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Convertido: {input_path.name} -> {output_path.name} | {len(df)} linhas")


def convert_all() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    for input_name, output_name in FILES_TO_CONVERT.items():
        input_path = EXTRACTED_DIR / input_name
        output_path = CSV_DIR / output_name

        if not input_path.exists():
            print(f"Arquivo não encontrado: {input_path}")
            continue

        convert_dat_to_csv(input_path, output_path)


if __name__ == "__main__":
    convert_all()