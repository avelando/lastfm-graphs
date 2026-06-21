# Teste de Significância: rrf_userknn_itemknn_ppr vs userknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| userknn | 0.1687 | 0.1208 |
| rrf_userknn_itemknn_ppr | 0.1848 | 0.1312 |

## Wilcoxon signed-rank (one-sided: rrf_userknn_itemknn_ppr > userknn)
- Estatística W: 334415
- p-valor: < 0.001
- Cohen's d: 0.1597

## Interpretação
A melhoria de rrf_userknn_itemknn_ppr sobre userknn é estatisticamente significativa (p=0.0000).
Tamanho de efeito: pequeno (|d| < 0.2)