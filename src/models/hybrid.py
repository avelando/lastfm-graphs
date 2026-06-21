from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from src.models.baselines import (
    build_index, build_R_train, get_train_items,
    popularity_scores, truncate_top_k, build_recommendations as bl_build_recommendations,
    K_SEARCH_USER, K_SEARCH_ITEM,
)
from src.models.social_ppr import build_A_norm, compute_ppr_matrix, ALPHA_CANDIDATES
from src.models.metrics import evaluate_recommendations, filter_warm_items


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/models")


def mask_and_normalize(scores: np.ndarray, users: list, artists: list,
                       train_items: dict) -> np.ndarray:
    """Mask seen items with -inf THEN min-max normalize only unseen candidates per user.

    Correct order (Avelar): raw scores → remove seen → normalize unseen → top-k
    Previous buggy order was: raw scores → normalize all → remove seen → top-k
    """
    artist_to_idx = {a: j for j, a in enumerate(artists)}
    result = scores.copy().astype(np.float64)
    for i, user_id in enumerate(users):
        seen = train_items.get(user_id, set())
        seen_idx = [artist_to_idx[a] for a in seen if a in artist_to_idx]
        if seen_idx:
            result[i, seen_idx] = -np.inf
        finite = np.isfinite(result[i])
        if np.any(finite):
            vals = result[i, finite]
            rmin, rmax = vals.min(), vals.max()
            result[i, finite] = (vals - rmin) / (rmax - rmin) if rmax > rmin else 0.0
    return result


def compute_rrf_scores(scores: np.ndarray, c: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion from pre-masked scores.

    Input: scores where seen items = -inf, unseen = normalized [0, 1].
    Output: RRF scores where seen items = -inf, unseen = 1/(c + rank).
    Rank computed per user descending (rank 1 = highest score).
    """
    finite_mask = np.isfinite(scores)
    # -inf → +inf after negation → sorted last → gets highest rank numbers
    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1) + 1  # 1-indexed
    rrf = 1.0 / (c + ranks.astype(np.float64))
    rrf[~finite_mask] = -np.inf
    return rrf


def build_recommendations(scores: np.ndarray, users: list, artists: list,
                          k: int = 10) -> dict:
    """Top-k from pre-masked, pre-normalized scores (-inf marks seen items)."""
    artists_arr = np.array(artists)
    recommendations = {}
    for i, user_id in enumerate(users):
        user_scores = scores[i]
        finite = np.isfinite(user_scores)
        if not np.any(finite):
            recommendations[user_id] = []
            continue
        top_idx = np.argsort(user_scores)[::-1][:k]
        valid = [idx for idx in top_idx if np.isfinite(user_scores[idx])][:k]
        recommendations[user_id] = artists_arr[valid].tolist()
    return recommendations


def weight_grid_2(step: float = 0.05):
    weights = np.round(np.arange(0, 1 + step, step), 10)
    return [(a, round(1 - a, 10)) for a in weights]


def weight_grid_3(step: float = 0.05):
    weights = np.round(np.arange(0, 1 + step, step), 10)
    combos = []
    for a in weights:
        for b in weights:
            g = round(1 - a - b, 10)
            if 0 <= g <= 1:
                combos.append((a, b, g))
    return combos


def search_weights(score_matrices: list, val_df: pd.DataFrame,
                   users: list, artists: list, grid: list) -> tuple:
    """Grid search on pre-masked, pre-normalized score matrices."""
    best_weights, best_ndcg = None, -1.0
    for weights in grid:
        combined = sum(w * s for w, s in zip(weights, score_matrices))
        recs = build_recommendations(combined, users, artists)
        ndcg = evaluate_recommendations(recs, val_df)["ndcg@10"].mean()
        if ndcg > best_ndcg:
            best_ndcg, best_weights = ndcg, weights
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
    R_dense = R_train.toarray()

    # --- Tune UserKNN k ---
    print("Tunando k para UserKNN...")
    R_norm = normalize(R_train, norm="l2")
    user_sim = (R_norm @ R_norm.T).toarray()
    best_k_uknn, best_ndcg_uknn = K_SEARCH_USER[0], -1.0
    for k in K_SEARCH_USER:
        sim_k = truncate_top_k(user_sim, k)
        sc = sim_k @ R_dense
        recs = bl_build_recommendations(sc, users, artists, train_items)
        ndcg = evaluate_recommendations(recs, val_df)["ndcg@10"].mean()
        print(f"  k={k}  val ndcg@10={ndcg:.4f}")
        if ndcg > best_ndcg_uknn:
            best_ndcg_uknn, best_k_uknn = ndcg, k
    print(f"  -> melhor k={best_k_uknn}")
    s_userknn = truncate_top_k(user_sim, best_k_uknn) @ R_dense

    # --- Tune ItemKNN k ---
    print("Tunando k para ItemKNN...")
    R_norm_items = normalize(R_train.T, norm="l2")
    item_sim = (R_norm_items @ R_norm_items.T).toarray()
    best_k_iknn, best_ndcg_iknn = K_SEARCH_ITEM[0], -1.0
    for k in K_SEARCH_ITEM:
        sim_k = truncate_top_k(item_sim, k)
        sc = R_dense @ sim_k
        recs = bl_build_recommendations(sc, users, artists, train_items)
        ndcg = evaluate_recommendations(recs, val_df)["ndcg@10"].mean()
        print(f"  k={k}  val ndcg@10={ndcg:.4f}")
        if ndcg > best_ndcg_iknn:
            best_ndcg_iknn, best_k_iknn = ndcg, k
    print(f"  -> melhor k={best_k_iknn}")
    s_itemknn = R_dense @ truncate_top_k(item_sim, best_k_iknn)

    pop = popularity_scores(train_df, artist_idx)
    s_pop = np.tile(pop, (n_users, 1))

    # --- Tune PPR alpha ---
    print("Tunando alpha do PPR...")
    A_norm = build_A_norm(friends_df, user_idx)
    best_alpha, best_alpha_ndcg = ALPHA_CANDIDATES[0], -1.0
    for alpha in ALPHA_CANDIDATES:
        PPR_try = compute_ppr_matrix(A_norm, alpha)
        sc = PPR_try @ R_dense
        recs = bl_build_recommendations(sc, users, artists, train_items)
        ndcg = evaluate_recommendations(recs, val_df)["ndcg@10"].mean()
        print(f"  alpha={alpha}  val ndcg@10={ndcg:.4f}")
        if ndcg > best_alpha_ndcg:
            best_alpha_ndcg, best_alpha = ndcg, alpha
    print(f"  -> melhor alpha={best_alpha}")
    s_ppr = compute_ppr_matrix(A_norm, best_alpha) @ R_dense

    # --- Mask seen items THEN normalize unseen per user (corrected order) ---
    print("Mascarando itens vistos e normalizando por usuário...")
    s_knn_n  = mask_and_normalize(s_userknn, users, artists, train_items)
    s_iknn_n = mask_and_normalize(s_itemknn, users, artists, train_items)
    s_pop_n  = mask_and_normalize(s_pop,     users, artists, train_items)
    s_ppr_n  = mask_and_normalize(s_ppr,     users, artists, train_items)

    # --- RRF scores (ranking is preserved after mask+normalize) ---
    r_knn  = compute_rrf_scores(s_knn_n)
    r_iknn = compute_rrf_scores(s_iknn_n)
    r_ppr  = compute_rrf_scores(s_ppr_n)

    results = {}

    def _save(name, combined, weights):
        recs = build_recommendations(combined, users, artists)
        results[name] = (
            evaluate_recommendations(recs, val_df),
            evaluate_recommendations(recs, test_df),
            weights,
        )

    # ======== Score-based hybrids ========

    print("\nBusca de pesos: UserKNN + Popularity...")
    w, _ = search_weights([s_knn_n, s_pop_n], val_df, users, artists, weight_grid_2())
    print(f"  Melhor: w_uknn={w[0]:.2f}  w_pop={w[1]:.2f}")
    _save("knn_pop", w[0]*s_knn_n + w[1]*s_pop_n, {"w_uknn": w[0], "w_pop": w[1]})

    print("\nBusca de pesos: UserKNN + Popularity + PPR...")
    w3, _ = search_weights([s_knn_n, s_pop_n, s_ppr_n], val_df, users, artists, weight_grid_3())
    print(f"  Melhor: w_uknn={w3[0]:.2f}  w_pop={w3[1]:.2f}  w_ppr={w3[2]:.2f}")
    _save("knn_pop_ppr", w3[0]*s_knn_n + w3[1]*s_pop_n + w3[2]*s_ppr_n,
          {"w_uknn": w3[0], "w_pop": w3[1], "w_ppr": w3[2]})

    print("\nBusca de pesos: UserKNN + ItemKNN...")
    w_ui, _ = search_weights([s_knn_n, s_iknn_n], val_df, users, artists, weight_grid_2())
    print(f"  Melhor: w_uknn={w_ui[0]:.2f}  w_iknn={w_ui[1]:.2f}")
    _save("userknn_itemknn", w_ui[0]*s_knn_n + w_ui[1]*s_iknn_n,
          {"w_uknn": w_ui[0], "w_iknn": w_ui[1]})

    print("\nBusca de pesos: UserKNN + ItemKNN + PPR...")
    w_uip, _ = search_weights([s_knn_n, s_iknn_n, s_ppr_n], val_df, users, artists, weight_grid_3())
    print(f"  Melhor: w_uknn={w_uip[0]:.2f}  w_iknn={w_uip[1]:.2f}  w_ppr={w_uip[2]:.2f}")
    _save("userknn_itemknn_ppr", w_uip[0]*s_knn_n + w_uip[1]*s_iknn_n + w_uip[2]*s_ppr_n,
          {"w_uknn": w_uip[0], "w_iknn": w_uip[1], "w_ppr": w_uip[2]})

    print("\nBusca de pesos: ItemKNN + PPR...")
    w_ip, _ = search_weights([s_iknn_n, s_ppr_n], val_df, users, artists, weight_grid_2())
    print(f"  Melhor: w_iknn={w_ip[0]:.2f}  w_ppr={w_ip[1]:.2f}")
    _save("itemknn_ppr", w_ip[0]*s_iknn_n + w_ip[1]*s_ppr_n,
          {"w_iknn": w_ip[0], "w_ppr": w_ip[1]})

    # ======== RRF hybrids ========

    print("\nBusca de pesos: RRF UserKNN + PPR...")
    w_rrf_kp, _ = search_weights([r_knn, r_ppr], val_df, users, artists, weight_grid_2())
    print(f"  Melhor: w_uknn={w_rrf_kp[0]:.2f}  w_ppr={w_rrf_kp[1]:.2f}")
    _save("rrf_userknn_ppr", w_rrf_kp[0]*r_knn + w_rrf_kp[1]*r_ppr,
          {"w_uknn": w_rrf_kp[0], "w_ppr": w_rrf_kp[1]})

    print("\nBusca de pesos: RRF ItemKNN + PPR...")
    w_rrf_ip, _ = search_weights([r_iknn, r_ppr], val_df, users, artists, weight_grid_2())
    print(f"  Melhor: w_iknn={w_rrf_ip[0]:.2f}  w_ppr={w_rrf_ip[1]:.2f}")
    _save("rrf_itemknn_ppr", w_rrf_ip[0]*r_iknn + w_rrf_ip[1]*r_ppr,
          {"w_iknn": w_rrf_ip[0], "w_ppr": w_rrf_ip[1]})

    print("\nBusca de pesos: RRF UserKNN + ItemKNN + PPR...")
    w_rrf_all, _ = search_weights([r_knn, r_iknn, r_ppr], val_df, users, artists, weight_grid_3())
    print(f"  Melhor: w_uknn={w_rrf_all[0]:.2f}  w_iknn={w_rrf_all[1]:.2f}  w_ppr={w_rrf_all[2]:.2f}")
    _save("rrf_userknn_itemknn_ppr",
          w_rrf_all[0]*r_knn + w_rrf_all[1]*r_iknn + w_rrf_all[2]*r_ppr,
          {"w_uknn": w_rrf_all[0], "w_iknn": w_rrf_all[1], "w_ppr": w_rrf_all[2]})

    # --- Salvar e imprimir ---
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
