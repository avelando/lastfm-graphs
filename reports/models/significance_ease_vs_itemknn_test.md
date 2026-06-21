# Teste de Significância: ease vs itemknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| itemknn | 0.1819 | 0.1266 |
| ease | 0.1976 | 0.1461 |

## Wilcoxon signed-rank (one-sided: ease > itemknn)
- Estatística W: 359242
- p-valor: < 0.001
- Cohen's d: 0.1304

## Interpretação
A melhoria de ease sobre itemknn é estatisticamente significativa (p=0.0000).
Tamanho de efeito: pequeno (|d| < 0.2)