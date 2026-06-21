from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/analysis")
N_PERMUTATIONS = 1000
RANDOM_SEED = 42


def build_matrices(user_artists: pd.DataFrame) -> tuple:
    users = sorted(user_artists["userID"].unique())
    artists = sorted(user_artists["artistID"].unique())
    user_idx = {u: i for i, u in enumerate(users)}
    artist_idx = {a: i for i, a in enumerate(artists)}

    row_idx = user_artists["userID"].map(user_idx).values
    col_idx = user_artists["artistID"].map(artist_idx).values
    log_weights = np.log1p(user_artists["weight"].values)

    mat = csr_matrix((log_weights, (row_idx, col_idx)), shape=(len(users), len(artists)))
    mat_binary = (mat > 0).astype(np.float32)
    mat_norm = normalize(mat, norm="l2").astype(np.float32)

    return users, user_idx, mat_binary, mat_norm


def pair_similarities(mi: np.ndarray, mj: np.ndarray,
                      mat_binary: csr_matrix, mat_norm: csr_matrix,
                      sizes: np.ndarray) -> tuple:
    intersections = np.array(mat_binary[mi].multiply(mat_binary[mj]).sum(axis=1)).ravel()
    unions = sizes[mi] + sizes[mj] - intersections
    jaccard = np.where(unions > 0, intersections / unions, 0.0)
    cosine = np.array(mat_norm[mi].multiply(mat_norm[mj]).sum(axis=1)).ravel()
    return jaccard.astype(np.float64), cosine.astype(np.float64)


def permutation_test(mi: np.ndarray, mj: np.ndarray,
                     mat_binary: csr_matrix, mat_norm: csr_matrix,
                     sizes: np.ndarray, obs_j_mean: float, obs_c_mean: float,
                     n_permutations: int, rng: np.random.Generator) -> tuple:
    perm_j_means = np.empty(n_permutations)
    perm_c_means = np.empty(n_permutations)

    for k in range(n_permutations):
        mj_perm = rng.permutation(mj)
        pj, pc = pair_similarities(mi, mj_perm, mat_binary, mat_norm, sizes)
        perm_j_means[k] = pj.mean()
        perm_c_means[k] = pc.mean()

        if (k + 1) % 100 == 0:
            print(f"  permutação {k + 1}/{n_permutations}")

    p_jaccard = (perm_j_means >= obs_j_mean).mean()
    p_cosine = (perm_c_means >= obs_c_mean).mean()

    return perm_j_means, perm_c_means, p_jaccard, p_cosine


def save_report(friend_df: pd.DataFrame, random_df: pd.DataFrame,
                perm_j_means: np.ndarray, perm_c_means: np.ndarray,
                p_jaccard: float, p_cosine: float) -> None:
    fj, fc = friend_df["jaccard"], friend_df["cosine"]
    rj, rc = random_df["jaccard"], random_df["cosine"]

    lines = [
        "# Teste de Homofilia Musical — Last.fm\n",
        f"Pares de amigos analisados: {len(friend_df)}  ",
        f"Pares aleatórios (baseline): {len(random_df)}  ",
        f"Permutações: {len(perm_j_means)}  ",
        "",
        "## Jaccard (overlap de artistas)",
        f"| Grupo | Média | Mediana | Std |",
        f"|---|---|---|---|",
        f"| Amigos | {fj.mean():.4f} | {fj.median():.4f} | {fj.std():.4f} |",
        f"| Aleatório | {rj.mean():.4f} | {rj.median():.4f} | {rj.std():.4f} |",
        f"| Permutação (média) | {perm_j_means.mean():.4f} | — | {perm_j_means.std():.4f} |",
        f"\n**p-valor (permutação):** {p_jaccard:.4f}",
        "",
        "## Cosseno com log1p(weight)",
        f"| Grupo | Média | Mediana | Std |",
        f"|---|---|---|---|",
        f"| Amigos | {fc.mean():.4f} | {fc.median():.4f} | {fc.std():.4f} |",
        f"| Aleatório | {rc.mean():.4f} | {rc.median():.4f} | {rc.std():.4f} |",
        f"| Permutação (média) | {perm_c_means.mean():.4f} | — | {perm_c_means.std():.4f} |",
        f"\n**p-valor (permutação):** {p_cosine:.4f}",
        "",
        "## Notas metodológicas",
        "- Jaccard usa conjunto binário de artistas (ignora volume de escuta).",
        "- Cosseno usa log1p(weight) para reduzir o efeito de volume absoluto.",
        "- Teste de permutação: shuffling da coluna friendID preserva a distribuição de grau do lado esquerdo.",
        "- Baseline aleatório: pares uniformemente amostrados entre todos os usuários com dados de escuta.",
        "- Controle por grau/atividade disponível em homophily_pairs.csv (colunas degree_a, degree_b, n_artists_a, n_artists_b).",
    ]

    report_path = REPORT_DIR / "homophily_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório salvo em: {report_path}")


def run_homophily(n_permutations: int = N_PERMUTATIONS) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("Carregando dados...")
    user_artists = pd.read_csv(PROCESSED_DIR / "user_artists_clean.csv")
    user_friends = pd.read_csv(PROCESSED_DIR / "user_friends_clean.csv")

    print("Construindo matrizes...")
    users, user_idx, mat_binary, mat_norm = build_matrices(user_artists)
    sizes = np.array(mat_binary.sum(axis=1)).ravel()

    n_artists_map = user_artists.groupby("userID")["artistID"].count().to_dict()
    degree_map = (
        pd.concat([user_friends["userID"], user_friends["friendID"]])
        .value_counts()
        .to_dict()
    )

    valid_users = set(user_idx.keys())
    friends_valid = user_friends[
        user_friends["userID"].isin(valid_users) & user_friends["friendID"].isin(valid_users)
    ].reset_index(drop=True)

    mi = friends_valid["userID"].map(user_idx).values
    mj = friends_valid["friendID"].map(user_idx).values

    print(f"Calculando similaridade para {len(mi)} pares de amigos...")
    obs_j, obs_c = pair_similarities(mi, mj, mat_binary, mat_norm, sizes)

    friend_df = pd.DataFrame({
        "userID_a": friends_valid["userID"].values,
        "userID_b": friends_valid["friendID"].values,
        "pair_type": "friend",
        "degree_a": friends_valid["userID"].map(degree_map).values,
        "degree_b": friends_valid["friendID"].map(degree_map).values,
        "n_artists_a": friends_valid["userID"].map(n_artists_map).values,
        "n_artists_b": friends_valid["friendID"].map(n_artists_map).values,
        "jaccard": obs_j,
        "cosine": obs_c,
    })

    print("Calculando baseline aleatório...")
    n_pairs = len(mi)
    all_idx = np.arange(len(users))
    ri = rng.choice(all_idx, size=n_pairs, replace=True)
    rj = rng.choice(all_idx, size=n_pairs, replace=True)
    rand_j, rand_c = pair_similarities(ri, rj, mat_binary, mat_norm, sizes)

    users_arr = np.array(users)
    random_df = pd.DataFrame({
        "userID_a": users_arr[ri],
        "userID_b": users_arr[rj],
        "pair_type": "random",
        "degree_a": pd.Series(users_arr[ri]).map(degree_map).values,
        "degree_b": pd.Series(users_arr[rj]).map(degree_map).values,
        "n_artists_a": pd.Series(users_arr[ri]).map(n_artists_map).values,
        "n_artists_b": pd.Series(users_arr[rj]).map(n_artists_map).values,
        "jaccard": rand_j,
        "cosine": rand_c,
    })

    all_pairs = pd.concat([friend_df, random_df], ignore_index=True)
    pairs_path = REPORT_DIR / "homophily_pairs.csv"
    all_pairs.to_csv(pairs_path, index=False, encoding="utf-8-sig")
    print(f"Pares salvos em: {pairs_path}")

    print(f"Executando teste de permutação ({n_permutations} permutações)...")
    perm_j_means, perm_c_means, p_jaccard, p_cosine = permutation_test(
        mi, mj, mat_binary, mat_norm, sizes,
        obs_j.mean(), obs_c.mean(),
        n_permutations, rng
    )

    perm_df = pd.DataFrame({"perm_jaccard_mean": perm_j_means, "perm_cosine_mean": perm_c_means})
    perm_df.to_csv(REPORT_DIR / "homophily_permutations.csv", index=False, encoding="utf-8-sig")

    save_report(friend_df, random_df, perm_j_means, perm_c_means, p_jaccard, p_cosine)

    print("\n=== Resultado ===")
    print(f"Jaccard  — amigos: {obs_j.mean():.4f}  aleatório: {rand_j.mean():.4f}  p={p_jaccard:.4f}")
    print(f"Cosseno  — amigos: {obs_c.mean():.4f}  aleatório: {rand_c.mean():.4f}  p={p_cosine:.4f}")


if __name__ == "__main__":
    run_homophily()
