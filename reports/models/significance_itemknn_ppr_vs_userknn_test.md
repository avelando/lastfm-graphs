# Teste de Significância: itemknn_ppr vs userknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| userknn | 0.1687 | 0.1208 |
| itemknn_ppr | 0.1841 | 0.1312 |

## Wilcoxon signed-rank (one-sided: itemknn_ppr > userknn)
- Estatística W: 345786
- p-valor: < 0.001
- Cohen's d: 0.1258

## Interpretação
A melhoria de itemknn_ppr sobre userknn é estatisticamente significativa (p=0.0000).
Tamanho de efeito: pequeno (|d| < 0.2)