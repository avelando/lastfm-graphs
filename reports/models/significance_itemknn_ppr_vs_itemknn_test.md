# Teste de Significância: itemknn_ppr vs itemknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| itemknn | 0.1819 | 0.1266 |
| itemknn_ppr | 0.1841 | 0.1312 |

## Wilcoxon signed-rank (one-sided: itemknn_ppr > itemknn)
- Estatística W: 119538
- p-valor: 0.4741
- Cohen's d: 0.0368

## Interpretação
A melhoria de itemknn_ppr sobre itemknn NÃO é estatisticamente significativa (p=0.4741).
Tamanho de efeito: pequeno (|d| < 0.2)