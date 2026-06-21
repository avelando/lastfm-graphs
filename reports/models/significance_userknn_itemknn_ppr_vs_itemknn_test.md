# Teste de Significância: userknn_itemknn_ppr vs itemknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| itemknn | 0.1819 | 0.1266 |
| userknn_itemknn_ppr | 0.1879 | 0.1312 |

## Wilcoxon signed-rank (one-sided: userknn_itemknn_ppr > itemknn)
- Estatística W: 223026
- p-valor: 0.0066
- Cohen's d: 0.0750

## Interpretação
A melhoria de userknn_itemknn_ppr sobre itemknn é estatisticamente significativa (p=0.0066).
Tamanho de efeito: pequeno (|d| < 0.2)