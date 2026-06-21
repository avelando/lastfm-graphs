from pathlib import Path
import pandas as pd

REPORT_DIR = Path("reports/models")

MODEL_ORDER = [
    ("popularity",               "Popularidade"),
    ("social_community",         "Social Comunidade (Louvain)"),
    ("social_1hop_bruto",        "Social 1-hop"),
    ("social_ppr",               "Social PPR"),
    ("itemknn",                  "ItemKNN"),
    ("userknn",                  "UserKNN"),
    ("ease",                     "EASE"),
    ("knn_pop",                  "UserKNN + Popularidade"),
    ("knn_pop_ppr",              "UserKNN + PPR"),
    ("itemknn_ppr",              "ItemKNN + PPR"),
    ("userknn_itemknn",          "UserKNN + ItemKNN"),
    ("userknn_itemknn_ppr",      "UserKNN + ItemKNN + PPR"),
    ("rrf_userknn_ppr",          "RRF: UserKNN + PPR"),
    ("rrf_itemknn_ppr",          "RRF: ItemKNN + PPR"),
    ("rrf_userknn_itemknn_ppr",  "RRF: UserKNN + ItemKNN + PPR"),
]

METRICS = ["ndcg@10", "precision@10", "recall@10", "f1@10"]


def load_split(model_key: str, split: str) -> pd.Series | None:
    path = REPORT_DIR / f"{model_key}_{split}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[METRICS].mean()


def run_results_table() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows_val, rows_test = [], []

    for model_key, model_label in MODEL_ORDER:
        val_m = load_split(model_key, "val")
        test_m = load_split(model_key, "test")

        if val_m is None and test_m is None:
            print(f"  [aviso] sem resultados para {model_key}, pulando")
            continue

        if val_m is not None:
            rows_val.append({"modelo": model_label, **val_m.to_dict()})
        if test_m is not None:
            rows_test.append({"modelo": model_label, **test_m.to_dict()})

    def fmt_table(rows: list, split: str) -> str:
        df = pd.DataFrame(rows)
        header = f"## Resultados — {split.capitalize()}\n\n"
        col_names = ["Modelo"] + [m.upper() for m in METRICS]
        lines = ["| " + " | ".join(col_names) + " |",
                 "|" + "|".join(["---"] * len(col_names)) + "|"]
        for _, row in df.iterrows():
            cells = [row["modelo"]] + [f"{row[m]:.4f}" for m in METRICS]
            lines.append("| " + " | ".join(cells) + " |")
        return header + "\n".join(lines)

    report_lines = [
        "# Tabela de Resultados — Last.fm HetRec2011 (warm-start)\n",
        fmt_table(rows_val, "validação"),
        "",
        fmt_table(rows_test, "teste"),
    ]
    report_path = REPORT_DIR / "results_table.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Tabela salva em: {report_path}")

    pd.DataFrame(rows_test).to_csv(
        REPORT_DIR / "results_table_test.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rows_val).to_csv(
        REPORT_DIR / "results_table_val.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== Resultados (teste) ===")
    for row in rows_test:
        metrics_str = "  ".join(f"{m}={row[m]:.4f}" for m in METRICS)
        print(f"  {row['modelo']:35s}  {metrics_str}")


if __name__ == "__main__":
    run_results_table()
