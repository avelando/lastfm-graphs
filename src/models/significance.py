from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPORT_DIR = Path("reports/models")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return diff.mean() / diff.std() if diff.std() > 0 else 0.0


def run_significance(model_a: str = "userknn", model_b: str = "knn_pop_ppr",
                     split: str = "test") -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(REPORT_DIR / f"{model_a}_{split}.csv")
    df_b = pd.read_csv(REPORT_DIR / f"{model_b}_{split}.csv")

    merged = df_a.merge(df_b, on="userID", suffixes=("_a", "_b"))
    if merged.empty:
        print("Nenhum usuário em comum entre os dois modelos.")
        return

    ndcg_a = merged["ndcg@10_a"].values
    ndcg_b = merged["ndcg@10_b"].values

    stat, p = wilcoxon(ndcg_b, ndcg_a, alternative="greater")
    d = cohens_d(ndcg_b, ndcg_a)

    lines = [
        f"# Teste de Significância: {model_b} vs {model_a} ({split})\n",
        f"Usuários comparados: {len(merged)}",
        f"",
        f"| Modelo | NDCG@10 médio | Mediana |",
        f"|---|---|---|",
        f"| {model_a} | {ndcg_a.mean():.4f} | {np.median(ndcg_a):.4f} |",
        f"| {model_b} | {ndcg_b.mean():.4f} | {np.median(ndcg_b):.4f} |",
        f"",
        f"## Wilcoxon signed-rank (one-sided: {model_b} > {model_a})",
        f"- Estatística W: {stat:.0f}",
        f"- p-valor: {'< 0.001' if p < 0.001 else f'{p:.4f}'}",
        f"- Cohen's d: {d:.4f}",
        f"",
        f"## Interpretação",
    ]

    if p < 0.05:
        lines.append(f"A melhoria de {model_b} sobre {model_a} é estatisticamente significativa (p={p:.4f}).")
    else:
        lines.append(f"A melhoria de {model_b} sobre {model_a} NÃO é estatisticamente significativa (p={p:.4f}).")

    if abs(d) < 0.2:
        lines.append("Tamanho de efeito: pequeno (|d| < 0.2)")
    elif abs(d) < 0.5:
        lines.append("Tamanho de efeito: médio (0.2 ≤ |d| < 0.5)")
    else:
        lines.append("Tamanho de efeito: grande (|d| ≥ 0.5)")

    report_path = REPORT_DIR / f"significance_{model_b}_vs_{model_a}_{split}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n=== Significância: {model_b} vs {model_a} ({split}) ===")
    print(f"  Usuários: {len(merged)}")
    print(f"  {model_a} NDCG@10: {ndcg_a.mean():.4f}")
    print(f"  {model_b} NDCG@10: {ndcg_b.mean():.4f}")
    print(f"  Wilcoxon W={stat:.0f}  p={'< 0.001' if p < 0.001 else f'{p:.4f}'}")
    print(f"  Cohen's d: {d:.4f}")
    print(f"  Relatório salvo em: {report_path}")


if __name__ == "__main__":
    run_significance()
