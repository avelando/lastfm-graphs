# Teste de Significância: knn_pop_ppr vs userknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| userknn | 0.1687 | 0.1208 |
| knn_pop_ppr | 0.1722 | 0.1208 |

## Wilcoxon signed-rank (one-sided: knn_pop_ppr > userknn)
- Estatística W: 169722
- p-valor: 0.0051
- Cohen's d: 0.0617

## Interpretação
A melhoria de knn_pop_ppr sobre userknn é estatisticamente significativa (p=0.0051).
Tamanho de efeito: pequeno (|d| < 0.2)