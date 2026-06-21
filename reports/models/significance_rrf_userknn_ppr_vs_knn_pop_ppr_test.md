# Teste de Significância: rrf_userknn_ppr vs knn_pop_ppr (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| knn_pop_ppr | 0.1722 | 0.1208 |
| rrf_userknn_ppr | 0.1688 | 0.1208 |

## Wilcoxon signed-rank (one-sided: rrf_userknn_ppr > knn_pop_ppr)
- Estatística W: 95318
- p-valor: 0.9948
- Cohen's d: -0.0677

## Interpretação
A melhoria de rrf_userknn_ppr sobre knn_pop_ppr NÃO é estatisticamente significativa (p=0.9948).
Tamanho de efeito: pequeno (|d| < 0.2)