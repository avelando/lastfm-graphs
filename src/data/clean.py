from pathlib import Path
import pandas as pd


CSV_DIR = Path("data/interim/csv")
PROCESSED_DIR = Path("data/processed")


def clean_artists() -> None:
    df = pd.read_csv(CSV_DIR / "artists.csv")

    df = df.rename(columns={
        "id": "artistID",
        "name": "artist_name"
    })

    df = df[["artistID", "artist_name"]]
    df = df.dropna(subset=["artistID", "artist_name"])
    df = df.drop_duplicates(subset=["artistID"])
    df["artistID"] = df["artistID"].astype(int)
    df["artist_name"] = df["artist_name"].astype(str).str.strip()

    df.to_csv(PROCESSED_DIR / "artists_clean.csv", index=False, encoding="utf-8-sig")


def clean_tags() -> None:
    df = pd.read_csv(CSV_DIR / "tags.csv")

    df = df.rename(columns={
        "tagValue": "tag_value"
    })

    df = df[["tagID", "tag_value"]]
    df = df.dropna(subset=["tagID", "tag_value"])
    df = df.drop_duplicates(subset=["tagID"])
    df["tagID"] = df["tagID"].astype(int)
    df["tag_value"] = df["tag_value"].astype(str).str.strip().str.lower()

    df.to_csv(PROCESSED_DIR / "tags_clean.csv", index=False, encoding="utf-8-sig")


def clean_user_artists() -> None:
    df = pd.read_csv(CSV_DIR / "user_artists.csv")

    df = df[["userID", "artistID", "weight"]]
    df = df.dropna(subset=["userID", "artistID", "weight"])
    df = df.drop_duplicates(subset=["userID", "artistID"])
    df["userID"] = df["userID"].astype(int)
    df["artistID"] = df["artistID"].astype(int)
    df["weight"] = df["weight"].astype(float)
    df = df[df["weight"] > 0]

    df.to_csv(PROCESSED_DIR / "user_artists_clean.csv", index=False, encoding="utf-8-sig")


def clean_user_friends() -> None:
    df = pd.read_csv(CSV_DIR / "user_friends.csv")

    df = df[["userID", "friendID"]]
    df = df.dropna(subset=["userID", "friendID"])
    df["userID"] = df["userID"].astype(int)
    df["friendID"] = df["friendID"].astype(int)
    df = df[df["userID"] != df["friendID"]]
    df[["userID", "friendID"]] = pd.DataFrame(
        df[["userID", "friendID"]].apply(lambda row: sorted(row), axis=1).tolist(),
        index=df.index
    )
    df = df.drop_duplicates(subset=["userID", "friendID"])

    df.to_csv(PROCESSED_DIR / "user_friends_clean.csv", index=False, encoding="utf-8-sig")


def clean_artist_tags() -> None:
    tagged = pd.read_csv(CSV_DIR / "user_taggedartists.csv")
    tags = pd.read_csv(PROCESSED_DIR / "tags_clean.csv")
    artists = pd.read_csv(PROCESSED_DIR / "artists_clean.csv")

    tagged = tagged[["artistID", "tagID"]]
    tagged = tagged.dropna(subset=["artistID", "tagID"])
    tagged["artistID"] = tagged["artistID"].astype(int)
    tagged["tagID"] = tagged["tagID"].astype(int)

    artist_tags = (
        tagged
        .groupby(["artistID", "tagID"])
        .size()
        .reset_index(name="tag_weight")
    )

    artist_tags = artist_tags.merge(artists[["artistID"]], on="artistID", how="inner")
    artist_tags = artist_tags.merge(tags, on="tagID", how="inner")

    artist_tags = artist_tags[["artistID", "tagID", "tag_value", "tag_weight"]]
    artist_tags = artist_tags.drop_duplicates(subset=["artistID", "tagID"])
    artist_tags["tag_weight"] = artist_tags["tag_weight"].astype(float)

    artist_tags.to_csv(PROCESSED_DIR / "artist_tags_clean.csv", index=False, encoding="utf-8-sig")


def clean_all() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_artists()
    clean_tags()
    clean_user_artists()
    clean_user_friends()
    clean_artist_tags()

    print("Arquivos limpos gerados em data/processed")


if __name__ == "__main__":
    clean_all()