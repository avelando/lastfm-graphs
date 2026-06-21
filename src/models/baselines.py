from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.models.metrics import evaluate_recommendations


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/models")
K_EVAL = 10
K_NEIGHBORS = 50


def build_index(values: pd.Series) -> tuple:
    unique = sorted(values.unique())
    return unique, {v: i for i, v in enumerate(unique)}


def build_R_train(train_df: pd.DataFrame, user_idx: dict, artist_idx: dict) -> csr_matrix:
    rows = train_df["userID"].map(user_idx).values
    cols = train_df["artistID"].map(artist_idx).values
    vals = np.log1p(train_df["weight"].values)
    return csr_matrix((vals, (rows, cols)), shape=(len(user_idx), len(artist_idx)))


def get_train_items(train_df: pd.DataFrame) -> dict:
    return train_df.groupby("userID")["artistID"].apply(set).to_dict()


def build_recommendations(scores: np.ndarray, users: list, artists: list,
                          train_items: dict, k: int = K_EVAL) -> dict:
    artists_arr = np.array(artists)
    recommendations = {}
    for i, user_id in enumerate(users):
        user_scores = scores[i].copy()
        seen = train_items.get(user_id, set())
        seen_idx = [j for j, a in enumerate(artists_arr) if a in seen]
        user_scores[seen_idx] = -np.inf
        top_idx = np.argsort(user_scores)[::-1][:k]
        recommendations[user_id] = artists_arr[top_idx].tolist()
    return recommendations


# --- Popularity ---

def popularity_scores(train_df: pd.DataFrame, artist_idx: dict) -> np.ndarray:
    pop = train_df.groupby("artistID")["weight"].sum()
    scores = np.zeros(len(artist_idx))
    for artist_id, w in pop.items():
        if artist_id in artist_idx:
            scores[artist_idx[artist_id]] = np.log1p(w)
    return scores


def run_popularity(train_df, val_df, test_df, users, artists, user_idx, artist_idx, train_items):
    print("Popularity...")
    pop = popularity_scores(train_df, artist_idx)
    scores = np.tile(pop, (len(users), 1))

    recs = build_recommendations(scores, users, artists, train_items)
    val_m = evaluate_recommendations(recs, val_df)
    test_m = evaluate_recommendations(recs, test_df)
    return val_m, test_m


# --- UserKNN ---

def truncate_top_k(sim_matrix: np.ndarray, k: int) -> np.ndarray:
    truncated = np.zeros_like(sim_matrix)
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[::-1][:k]
        truncated[i, top_k] = sim_matrix[i, top_k]
    return truncated


def run_userknn(R_train, val_df, test_df, users, artists, train_items, k=K_NEIGHBORS):
    print(f"UserKNN (k={k})...")
    R_norm = normalize(R_train, norm="l2")
    user_sim = (R_norm @ R_norm.T).toarray()
    user_sim_k = truncate_top_k(user_sim, k)

    scores = user_sim_k @ R_train.toarray()

    recs = build_recommendations(scores, users, artists, train_items)
    val_m = evaluate_recommendations(recs, val_df)
    test_m = evaluate_recommendations(recs, test_df)
    return val_m, test_m


# --- ItemKNN ---

def run_itemknn(R_train, val_df, test_df, users, artists, train_items, k=K_NEIGHBORS):
    print(f"ItemKNN (k={k})...")
    R_norm_items = normalize(R_train.T, norm="l2")
    item_sim = (R_norm_items @ R_norm_items.T).toarray()

    # Truncar top-k por item (zerando similares fora do top-k)
    for j in range(item_sim.shape[0]):
        row = item_sim[j].copy()
        row[j] = -np.inf
        top_k_idx = np.argsort(row)[::-1][:k]
        mask = np.zeros(len(row), dtype=bool)
        mask[top_k_idx] = True
        item_sim[j, ~mask] = 0.0

    scores = R_train.toarray() @ item_sim

    recs = build_recommendations(scores, users, artists, train_items)
    val_m = evaluate_recommendations(recs, val_df)
    test_m = evaluate_recommendations(recs, test_df)
    return val_m, test_m


# --- Runner ---

def print_summary(model_name: str, val_m: pd.DataFrame, test_m: pd.DataFrame) -> None:
    def fmt(df, split):
        means = df.drop(columns="userID").mean()
        return f"  [{split}] " + "  ".join(f"{c}={v:.4f}" for c, v in means.items())
    print(f"\n=== {model_name} ===")
    print(fmt(val_m, "val"))
    print(fmt(test_m, "test"))


def run_baselines() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")

    users, user_idx = build_index(train_df["userID"])
    artists, artist_idx = build_index(train_df["artistID"])
    R_train = build_R_train(train_df, user_idx, artist_idx)
    train_items = get_train_items(train_df)

    results = {}

    val_m, test_m = run_popularity(train_df, val_df, test_df, users, artists, user_idx, artist_idx, train_items)
    print_summary("Popularity", val_m, test_m)
    results["popularity"] = (val_m, test_m)

    val_m, test_m = run_userknn(R_train, val_df, test_df, users, artists, train_items)
    print_summary("UserKNN", val_m, test_m)
    results["userknn"] = (val_m, test_m)

    val_m, test_m = run_itemknn(R_train, val_df, test_df, users, artists, train_items)
    print_summary("ItemKNN", val_m, test_m)
    results["itemknn"] = (val_m, test_m)

    for model_name, (val_m, test_m) in results.items():
        val_m.to_csv(REPORT_DIR / f"{model_name}_val.csv", index=False, encoding="utf-8-sig")
        test_m.to_csv(REPORT_DIR / f"{model_name}_test.csv", index=False, encoding="utf-8-sig")

    print("\nResultados salvos em reports/models/")


if __name__ == "__main__":
    run_baselines()
