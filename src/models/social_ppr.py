from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from src.models.metrics import evaluate_recommendations, filter_warm_items


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/models")
ALPHA_CANDIDATES = [0.15, 0.25, 0.50, 0.85]
N_ITER = 30


def build_index(values: pd.Series) -> tuple:
    unique = sorted(values.unique())
    return unique, {v: i for i, v in enumerate(unique)}


def build_R_train(train_df: pd.DataFrame, user_idx: dict, artist_idx: dict) -> csr_matrix:
    rows = train_df["userID"].map(user_idx).values
    cols = train_df["artistID"].map(artist_idx).values
    vals = np.log1p(train_df["weight"].values)
    return csr_matrix((vals, (rows, cols)), shape=(len(user_idx), len(artist_idx)))


def build_A_norm(friends_df: pd.DataFrame, user_idx: dict) -> csr_matrix:
    valid = friends_df[
        friends_df["userID"].isin(user_idx) & friends_df["friendID"].isin(user_idx)
    ]
    n = len(user_idx)
    rows = valid["userID"].map(user_idx).values
    cols = valid["friendID"].map(user_idx).values
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    vals = np.ones(len(all_rows), dtype=np.float32)
    A = csr_matrix((vals, (all_rows, all_cols)), shape=(n, n))
    return normalize(A, norm="l1")


def compute_ppr_matrix(A_norm: csr_matrix, alpha: float,
                       n_iter: int = N_ITER) -> np.ndarray:
    n = A_norm.shape[0]
    # PPR: π = α·I + (1-α)·A_norm·π  →  iteração de potência
    # Equivale a propagar scores de cada nó pela rede
    PPR = alpha * np.eye(n, dtype=np.float32)
    for _ in range(n_iter):
        PPR = alpha * np.eye(n, dtype=np.float32) + (1 - alpha) * A_norm.toarray() @ PPR
    return PPR


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


def run_social_ppr() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")
    friends_df = pd.read_csv(PROCESSED_DIR / "user_friends_clean.csv")

    train_artists = set(train_df["artistID"].unique())
    val_df = filter_warm_items(val_df, train_artists)
    test_df = filter_warm_items(test_df, train_artists)

    users, user_idx = build_index(train_df["userID"])
    artists, artist_idx = build_index(train_df["artistID"])
    R_train = build_R_train(train_df, user_idx, artist_idx)
    A_norm = build_A_norm(friends_df, user_idx)
    train_items = get_train_items(train_df)

    best_alpha = None
    best_val_ndcg = -1.0
    val_results = {}

    for alpha in ALPHA_CANDIDATES:
        print(f"Calculando PPR com alpha={alpha} ({N_ITER} iterações)...")
        PPR = compute_ppr_matrix(A_norm, alpha)
        scores = PPR @ R_train.toarray()

        recs = build_recommendations(scores, users, artists, train_items)
        val_m = evaluate_recommendations(recs, val_df)
        mean_ndcg = val_m["ndcg@10"].mean()
        val_results[alpha] = (val_m, mean_ndcg)

        print(f"  alpha={alpha}  val ndcg@10={mean_ndcg:.4f}")

        if mean_ndcg > best_val_ndcg:
            best_val_ndcg = mean_ndcg
            best_alpha = alpha
            best_scores = scores

    print(f"\nMelhor alpha (validação): {best_alpha}")

    best_recs = build_recommendations(best_scores, users, artists, train_items)
    val_m_best = val_results[best_alpha][0]
    test_m_best = evaluate_recommendations(best_recs, test_df)

    val_m_best.to_csv(REPORT_DIR / "social_ppr_val.csv", index=False, encoding="utf-8-sig")
    test_m_best.to_csv(REPORT_DIR / "social_ppr_test.csv", index=False, encoding="utf-8-sig")

    alpha_summary = pd.DataFrame([
        {"alpha": a, "val_ndcg@10": ndcg}
        for a, (_, ndcg) in val_results.items()
    ])
    alpha_summary.to_csv(REPORT_DIR / "social_ppr_alpha_search.csv",
                         index=False, encoding="utf-8-sig")

    def fmt(df, split):
        means = df.drop(columns="userID").mean()
        return f"  [{split}] " + "  ".join(f"{c}={v:.4f}" for c, v in means.items())

    print(f"\n=== Social PPR (alpha={best_alpha}) ===")
    print(fmt(val_m_best, "val"))
    print(fmt(test_m_best, "test"))
    print("\nResultados salvos em reports/models/")


if __name__ == "__main__":
    run_social_ppr()
