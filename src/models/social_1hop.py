from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.metrics import evaluate_recommendations


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/models")


def build_index(values: pd.Series) -> tuple:
    unique = sorted(values.unique())
    return unique, {v: i for i, v in enumerate(unique)}


def build_R_train(train_df: pd.DataFrame, user_idx: dict, artist_idx: dict) -> csr_matrix:
    rows = train_df["userID"].map(user_idx).values
    cols = train_df["artistID"].map(artist_idx).values
    vals = np.log1p(train_df["weight"].values)
    return csr_matrix((vals, (rows, cols)), shape=(len(user_idx), len(artist_idx)))


def build_A_social(friends_df: pd.DataFrame, user_idx: dict) -> csr_matrix:
    valid = friends_df[
        friends_df["userID"].isin(user_idx) & friends_df["friendID"].isin(user_idx)
    ]
    rows = valid["userID"].map(user_idx).values
    cols = valid["friendID"].map(user_idx).values
    n = len(user_idx)
    # grafo não-direcionado: adiciona ambas as direções
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    vals = np.ones(len(all_rows), dtype=np.float32)
    return csr_matrix((vals, (all_rows, all_cols)), shape=(n, n))


def compute_scores(A_social: csr_matrix, R_train: csr_matrix,
                   normalized: bool = False) -> np.ndarray:
    if normalized:
        degree = np.array(A_social.sum(axis=1)).ravel()
        degree[degree == 0] = 1
        D_inv = csr_matrix((1.0 / degree, (np.arange(len(degree)), np.arange(len(degree)))),
                           shape=A_social.shape)
        scores = D_inv.dot(A_social).dot(R_train)
    else:
        scores = A_social.dot(R_train)
    if hasattr(scores, "toarray"):
        return scores.toarray()
    return np.asarray(scores)


def get_train_items(train_df: pd.DataFrame) -> dict:
    return train_df.groupby("userID")["artistID"].apply(set).to_dict()


def build_recommendations(scores: np.ndarray, users: list, artists: list,
                          train_items: dict, k: int = 10) -> dict:
    artists_arr = np.array(artists)
    recommendations = {}
    for i, user_id in enumerate(users):
        seen = train_items.get(user_id, set())
        user_scores = scores[i].copy()

        seen_idx = [j for j, a in enumerate(artists_arr) if a in seen]
        user_scores[seen_idx] = -np.inf

        top_idx = np.argsort(user_scores)[::-1][:k]
        recommendations[user_id] = artists_arr[top_idx].tolist()

    return recommendations


def save_results(val_metrics: pd.DataFrame, test_metrics: pd.DataFrame,
                 variant: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    val_metrics.to_csv(
        REPORT_DIR / f"social_1hop_{variant}_val.csv", index=False, encoding="utf-8-sig"
    )
    test_metrics.to_csv(
        REPORT_DIR / f"social_1hop_{variant}_test.csv", index=False, encoding="utf-8-sig"
    )

    def summary(df, split):
        means = df.drop(columns="userID").mean()
        print(f"  [{split}] " + "  ".join(f"{c}={v:.4f}" for c, v in means.items()))

    print(f"\n=== Social 1-hop {variant} ===")
    summary(val_metrics, "val")
    summary(test_metrics, "test")


def run_social_1hop() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")
    friends_df = pd.read_csv(PROCESSED_DIR / "user_friends_clean.csv")

    print("Construindo índices e matrizes...")
    users, user_idx = build_index(train_df["userID"])
    artists, artist_idx = build_index(train_df["artistID"])

    R_train = build_R_train(train_df, user_idx, artist_idx)
    A_social = build_A_social(friends_df, user_idx)
    train_items = get_train_items(train_df)

    for variant, normalized in [("bruto", False), ("normalizado", True)]:
        print(f"\nCalculando scores Social 1-hop {variant}...")
        scores = compute_scores(A_social, R_train, normalized=normalized)

        recs = build_recommendations(scores, users, artists, train_items, k=10)

        val_metrics = evaluate_recommendations(recs, val_df, k=10)
        test_metrics = evaluate_recommendations(recs, test_df, k=10)

        save_results(val_metrics, test_metrics, variant)

    print("\nResultados salvos em reports/models/")


if __name__ == "__main__":
    run_social_1hop()
