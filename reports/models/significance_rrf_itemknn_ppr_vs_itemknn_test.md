# Teste de Significância: rrf_itemknn_ppr vs itemknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| itemknn | 0.1819 | 0.1266 |
| rrf_itemknn_ppr | 0.1785 | 0.1266 |

## Wilcoxon signed-rank (one-sided: rrf_itemknn_ppr > itemknn)
- Estatística W: 157503
- p-valor: 0.9823
- Cohen's d: -0.0427

## Interpretação
A melhoria de rrf_itemknn_ppr sobre itemknn NÃO é estatisticamente significativa (p=0.9823).
Tamanho de efeito: pequeno (|d| < 0.2)