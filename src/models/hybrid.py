from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from src.models.baselines import (
    build_index, build_R_train, get_train_items,
    popularity_scores, truncate_top_k, K_NEIGHBORS,
)
from src.models.social_ppr import build_A_norm, compute_ppr_matrix
from src.models.metrics import evaluate_recommendations, filter_warm_items


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/models")
PPR_ALPHA = 0.5


def minmax_per_user(scores: np.ndarray) -> np.ndarray:
    result = np.zeros_like(scores, dtype=np.float64)
    for i in range(scores.shape[0]):
        row = scores[i]
        rmin, rmax = row.min(), row.max()
        if rmax > rmin:
            result[i] = (row - rmin) / (rmax - rmin)
    return result


def build_recommendations(scores: np.ndarray, users: list, artists: list,
                          train_items: dict, k: int = 10) -> dict:
    artists_arr = np.array(artists)
    seen_idx_map = {
        user_id: [j for j, a in enumerate(artists_arr) if a in train_items.get(user_id, set())]
        for user_id in users
    }
    recommendations = {}
    for i, user_id in enumerate(users):
        user_scores = scores[i].copy()
        user_scores[seen_idx_map[user_id]] = -np.inf
        if not np.any(np.isfinite(user_scores)) or np.nanmax(user_scores) <= 0:
            recommendations[user_id] = []
            continue
        top_idx = np.argsort(user_scores)[::-1][:k]
        recommendations[user_id] = artists_arr[top_idx].tolist()
    return recommendations


def weight_grid_2(step: float = 0.1):
    weights = np.round(np.arange(0, 1 + step, step), 10)
    return [(a, round(1 - a, 10)) for a in weights]


def weight_grid_3(step: float = 0.1):
    weights = np.round(np.arange(0, 1 + step, step), 10)
    combos = []
    for a in weights:
        for b in weights:
            g = round(1 - a - b, 10)
            if 0 <= g <= 1:
                combos.append((a, b, g))
    return combos


def search_weights(score_matrices: list, val_df: pd.DataFrame,
                   users: list, artists: list, train_items: dict,
                   grid: list) -> tuple:
    best_weights = None
    best_ndcg = -1.0

    for weights in grid:
        combined = sum(w * s for w, s in zip(weights, score_matrices))
        recs = build_recommendations(combined, users, artists, train_items)
        val_m = evaluate_recommendations(recs, val_df)
        ndcg = val_m["ndcg@10"].mean()
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_weights = weights

    return best_weights, best_ndcg


def run_hybrid() -> None:
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
    train_items = get_train_items(train_df)
    n_users = len(users)

    print("Computando scores dos modelos base...")

    R_norm = normalize(R_train, norm="l2")
    user_sim = (R_norm @ R_norm.T).toarray()
    user_sim_k = truncate_top_k(user_sim, K_NEIGHBORS)
    s_userknn = user_sim_k @ R_train.toarray()

    pop = popularity_scores(train_df, artist_idx)
    s_pop = np.tile(pop, (n_users, 1))

    print(f"Computando PPR (alpha={PPR_ALPHA})...")
    A_norm = build_A_norm(friends_df, user_idx)
    PPR = compute_ppr_matrix(A_norm, PPR_ALPHA)
    s_ppr = PPR @ R_train.toarray()

    print("Normalizando scores por usuário (min-max)...")
    s_knn_n = minmax_per_user(s_userknn)
    s_pop_n = minmax_per_user(s_pop)
    s_ppr_n = minmax_per_user(s_ppr)

    results = {}

    # --- UserKNN + Popularity ---
    print("\nBusca de pesos: UserKNN + Popularity...")
    grid_2 = weight_grid_2()
    best_w, best_ndcg = search_weights(
        [s_knn_n, s_pop_n], val_df, users, artists, train_items, grid_2
    )
    print(f"  Melhor: w_knn={best_w[0]:.1f}  w_pop={best_w[1]:.1f}  val ndcg@10={best_ndcg:.4f}")

    combined = best_w[0] * s_knn_n + best_w[1] * s_pop_n
    recs = build_recommendations(combined, users, artists, train_items)
    val_m = evaluate_recommendations(recs, val_df)
    test_m = evaluate_recommendations(recs, test_df)
    results["knn_pop"] = (val_m, test_m, {"alpha_knn": best_w[0], "alpha_pop": best_w[1]})

    # --- UserKNN + Popularity + Social PPR ---
    print("\nBusca de pesos: UserKNN + Popularity + Social PPR...")
    grid_3 = weight_grid_3()
    best_w3, best_ndcg3 = search_weights(
        [s_knn_n, s_pop_n, s_ppr_n], val_df, users, artists, train_items, grid_3
    )
    print(f"  Melhor: w_knn={best_w3[0]:.1f}  w_pop={best_w3[1]:.1f}  w_ppr={best_w3[2]:.1f}  val ndcg@10={best_ndcg3:.4f}")

    combined3 = best_w3[0] * s_knn_n + best_w3[1] * s_pop_n + best_w3[2] * s_ppr_n
    recs3 = build_recommendations(combined3, users, artists, train_items)
    val_m3 = evaluate_recommendations(recs3, val_df)
    test_m3 = evaluate_recommendations(recs3, test_df)
    results["knn_pop_ppr"] = (val_m3, test_m3, {
        "alpha_knn": best_w3[0], "alpha_pop": best_w3[1], "alpha_ppr": best_w3[2]
    })

    def fmt(df, split):
        means = df.drop(columns="userID").mean()
        return f"  [{split}] " + "  ".join(f"{c}={v:.4f}" for c, v in means.items())

    for model_name, (val_m, test_m, weights) in results.items():
        val_m.to_csv(REPORT_DIR / f"{model_name}_val.csv", index=False, encoding="utf-8-sig")
        test_m.to_csv(REPORT_DIR / f"{model_name}_test.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== {model_name} | pesos: {weights} ===")
        print(fmt(val_m, "val"))
        print(fmt(test_m, "test"))

    print("\nResultados salvos em reports/models/")


if __name__ == "__main__":
    run_hybrid()
