# Teste de Significância: userknn_itemknn_ppr vs userknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| userknn | 0.1687 | 0.1208 |
| userknn_itemknn_ppr | 0.1879 | 0.1312 |

## Wilcoxon signed-rank (one-sided: userknn_itemknn_ppr > userknn)
- Estatística W: 316554
- p-valor: < 0.001
- Cohen's d: 0.2011

## Interpretação
A melhoria de userknn_itemknn_ppr sobre userknn é estatisticamente significativa (p=0.0000).
Tamanho de efeito: médio (0.2 ≤ |d| < 0.5)