from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.models.metrics import evaluate_recommendations, filter_warm_items


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


def build_community_scores(R_train: csr_matrix, users: list,
                           community_map: dict) -> np.ndarray:
    n_users = len(users)
    n_artists = R_train.shape[1]
    scores = np.zeros((n_users, n_artists), dtype=np.float64)

    communities = {}
    for i, user_id in enumerate(users):
        c = community_map.get(user_id)
        if c is not None:
            communities.setdefault(c, []).append(i)

    for community_indices in communities.values():
        community_sum = np.array(R_train[community_indices].sum(axis=0)).ravel()
        for i in community_indices:
            scores[i] = community_sum

    return scores


def get_train_items(train_df: pd.DataFrame) -> dict:
    return train_df.groupby("userID")["artistID"].apply(set).to_dict()


def build_recommendations(scores: np.ndarray, users: list, artists: list,
                          train_items: dict, k: int = 10) -> dict:
    artists_arr = np.array(artists)
    recommendations = {}
    for i, user_id in enumerate(users):
        user_scores = scores[i].copy()
        seen = train_items.get(user_id, set())
        seen_idx = [j for j, a in enumerate(artists_arr) if a in seen]
        user_scores[seen_idx] = -np.inf
        if not np.any(np.isfinite(user_scores)) or np.nanmax(user_scores) <= 0:
            recommendations[user_id] = []
            continue
        top_idx = np.argsort(user_scores)[::-1][:k]
        recommendations[user_id] = artists_arr[top_idx].tolist()
    return recommendations


def run_social_community() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")
    metrics_df = pd.read_csv(PROCESSED_DIR / "user_network_metrics.csv")

    train_artists = set(train_df["artistID"].unique())
    val_df = filter_warm_items(val_df, train_artists)
    test_df = filter_warm_items(test_df, train_artists)

    community_map = dict(zip(metrics_df["userID"], metrics_df["community"]))

    users, user_idx = build_index(train_df["userID"])
    artists, artist_idx = build_index(train_df["artistID"])
    R_train = build_R_train(train_df, user_idx, artist_idx)
    train_items = get_train_items(train_df)

    print("Calculando scores Social Comunidade (Louvain)...")
    scores = build_community_scores(R_train, users, community_map)

    recs = build_recommendations(scores, users, artists, train_items)
    val_m = evaluate_recommendations(recs, val_df)
    test_m = evaluate_recommendations(recs, test_df)

    val_m.to_csv(REPORT_DIR / "social_community_val.csv", index=False, encoding="utf-8-sig")
    test_m.to_csv(REPORT_DIR / "social_community_test.csv", index=False, encoding="utf-8-sig")

    def fmt(df, split):
        means = df.drop(columns="userID").mean()
        return f"  [{split}] " + "  ".join(f"{c}={v:.4f}" for c, v in means.items())

    print("\n=== Social Comunidade ===")
    print(fmt(val_m, "val"))
    print(fmt(test_m, "test"))
    print("\nResultados salvos em reports/models/")


if __name__ == "__main__":
    run_social_community()
