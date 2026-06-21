from pathlib import Path
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports/analysis")
MIN_ARTISTS = 5
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42


def split_user_interactions(
    user_artists: pd.DataFrame,
    min_artists: int = MIN_ARTISTS,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple:
    rng = np.random.default_rng(seed)

    user_counts = user_artists.groupby("userID")["artistID"].count()
    valid_users = user_counts[user_counts >= min_artists].index
    df = user_artists[user_artists["userID"].isin(valid_users)].copy()

    train_rows, val_rows, test_rows = [], [], []

    for user_id, group in df.groupby("userID"):
        items = group.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
        n = len(items)

        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            continue

        train_rows.append(items.iloc[:n_train])
        val_rows.append(items.iloc[n_train:n_train + n_val])
        test_rows.append(items.iloc[n_train + n_val:])

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    return train_df, val_df, test_df


def save_split_report(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      test_df: pd.DataFrame, n_removed: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_users = train_df["userID"].nunique()
    lines = [
        "# Split Treino/Validação/Teste\n",
        f"Usuários com menos de {MIN_ARTISTS} artistas removidos: {n_removed}  ",
        f"Usuários no split final: {n_users}  ",
        f"Proporção: {TRAIN_RATIO:.0%} treino / {VAL_RATIO:.0%} validação / {1 - TRAIN_RATIO - VAL_RATIO:.0%} teste  ",
        f"Seed: {RANDOM_SEED}  ",
        "",
        "| Conjunto | Interações | Usuários | Artistas únicos |",
        "|---|---|---|---|",
        f"| Treino | {len(train_df)} | {train_df['userID'].nunique()} | {train_df['artistID'].nunique()} |",
        f"| Validação | {len(val_df)} | {val_df['userID'].nunique()} | {val_df['artistID'].nunique()} |",
        f"| Teste | {len(test_df)} | {test_df['userID'].nunique()} | {test_df['artistID'].nunique()} |",
        "",
        "## Distribuição de itens por usuário",
        "| Conjunto | Média | Mediana | Min | Max |",
        "|---|---|---|---|---|",
    ]

    for name, df in [("Treino", train_df), ("Validação", val_df), ("Teste", test_df)]:
        counts = df.groupby("userID")["artistID"].count()
        lines.append(
            f"| {name} | {counts.mean():.1f} | {counts.median():.1f} "
            f"| {counts.min()} | {counts.max()} |"
        )

    lines += [
        "",
        "## Notas",
        f"- Split estratificado por usuário (não global).",
        f"- Para usuários com poucos artistas: n_test = max(1, floor(n × {1 - TRAIN_RATIO - VAL_RATIO:.1f})), n_val = max(1, floor(n × {VAL_RATIO:.1f})).",
        "- Pesos salvos como originais (weight). Aplicar log1p nos modelos conforme necessário.",
        "- Grafo de amizade (user_friends_clean.csv) não é dividido — usado completo como A_social.",
        "- Validação: escolher hiperparâmetros e pesos dos híbridos.",
        "- Teste: avaliação final única. Não usar para seleção de modelos.",
    ]

    report_path = REPORT_DIR / "split_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Relatório salvo em: {report_path}")


def generate_split() -> None:
    print("Carregando user_artists_clean.csv...")
    user_artists = pd.read_csv(PROCESSED_DIR / "user_artists_clean.csv")

    total_users = user_artists["userID"].nunique()
    print(f"Usuários no dataset original: {total_users}")

    train_df, val_df, test_df = split_user_interactions(user_artists)

    n_removed = total_users - train_df["userID"].nunique()

    train_df.to_csv(PROCESSED_DIR / "train_data.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(PROCESSED_DIR / "val_data.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(PROCESSED_DIR / "test_data.csv", index=False, encoding="utf-8-sig")

    print(f"Usuários removidos (< {MIN_ARTISTS} artistas): {n_removed}")
    print(f"Treino:    {len(train_df)} interações, {train_df['userID'].nunique()} usuários")
    print(f"Validação: {len(val_df)} interações, {val_df['userID'].nunique()} usuários")
    print(f"Teste:     {len(test_df)} interações, {test_df['userID'].nunique()} usuários")

    save_split_report(train_df, val_df, test_df, n_removed)


if __name__ == "__main__":
    generate_split()
