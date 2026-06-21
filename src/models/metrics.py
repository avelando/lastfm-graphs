import numpy as np
import pandas as pd


def ndcg_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(n_ideal))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(recommended: list, relevant: set, k: int = 10) -> float:
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_recommendations(
    recommendations: dict,
    ground_truth: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    relevant_map = ground_truth.groupby("userID")["artistID"].apply(set).to_dict()

    rows = []
    for user_id, ranked_items in recommendations.items():
        relevant = relevant_map.get(user_id, set())
        if not relevant:
            continue
        rows.append({
            "userID": user_id,
            f"ndcg@{k}": ndcg_at_k(ranked_items, relevant, k),
            f"precision@{k}": precision_at_k(ranked_items, relevant, k),
            f"recall@{k}": recall_at_k(ranked_items, relevant, k),
            f"f1@{k}": f1_at_k(ranked_items, relevant, k),
        })

    return pd.DataFrame(rows)
