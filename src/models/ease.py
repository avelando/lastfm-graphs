from pathlib import Path
import numpy as np
import pandas as pd
from src.models.baselines import (
    build_index, build_R_train, get_train_items, build_recommendations,
)
from src.models.metrics import evaluate_recommendations, filter_warm_items


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/models")
LAMBDA_CANDIDATES = [50, 100, 200, 500, 1000, 2000]


def compute_ease(R_dense: np.ndarray, lambda_: float = 500.0) -> np.ndarray:
    """EASE closed-form via Woodbury identity (efficient when n_users << n_items).

    Woodbury: P = (X^T X + λI)^{-1} = (1/λ)(I - A)
    where A = X^T (X X^T + λI)^{-1} X
    B_ij = A_ij / (1 - A_jj)  for i≠j;  B_ii = 0
    scores = X @ B
    """
    n_users, _ = R_dense.shape
    G_user = R_dense @ R_dense.T + lambda_ * np.eye(n_users)  # n_users × n_users
    inv_G = np.linalg.inv(G_user)
    A = R_dense.T @ inv_G @ R_dense                           # n_items × n_items
    denom = 1.0 - np.diag(A)
    B = A / denom[np.newaxis, :]
    np.fill_diagonal(B, 0.0)
    return R_dense @ B


def run_ease() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Carregando dados...")
    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val_data.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")

    train_artists = set(train_df["artistID"].unique())
    val_df = filter_warm_items(val_df, train_artists)
    test_df = filter_warm_items(test_df, train_artists)

    users, user_idx = build_index(train_df["userID"])
    artists, artist_idx = build_index(train_df["artistID"])
    R_train = build_R_train(train_df, user_idx, artist_idx)
    R_dense = R_train.toarray()
    train_items = get_train_items(train_df)

    print("Tunando lambda para EASE...")
    best_lambda, best_ndcg = LAMBDA_CANDIDATES[0], -1.0
    for lam in LAMBDA_CANDIDATES:
        print(f"  lambda={lam}  ...", end="", flush=True)
        scores = compute_ease(R_dense, lam)
        recs = build_recommendations(scores, users, artists, train_items)
        ndcg = evaluate_recommendations(recs, val_df)["ndcg@10"].mean()
        print(f"  val ndcg@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg, best_lambda = ndcg, lam
    print(f"  -> melhor lambda={best_lambda}")

    print("Calculando scores finais com melhor lambda...")
    scores = compute_ease(R_dense, best_lambda)
    recs = build_recommendations(scores, users, artists, train_items)
    val_m = evaluate_recommendations(recs, val_df)
    test_m = evaluate_recommendations(recs, test_df)

    val_m.to_csv(REPORT_DIR / "ease_val.csv", index=False, encoding="utf-8-sig")
    test_m.to_csv(REPORT_DIR / "ease_test.csv", index=False, encoding="utf-8-sig")

    def fmt(df, split):
        means = df.drop(columns="userID").mean()
        return f"  [{split}] " + "  ".join(f"{c}={v:.4f}" for c, v in means.items())

    print(f"\n=== EASE (lambda={best_lambda}) ===")
    print(fmt(val_m, "val"))
    print(fmt(test_m, "test"))
    print("\nResultados salvos em reports/models/")


if __name__ == "__main__":
    run_ease()
