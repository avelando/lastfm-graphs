# Teste de Significância: itemknn vs userknn (test)

Usuários comparados: 1870

| Modelo | NDCG@10 médio | Mediana |
|---|---|---|
| userknn | 0.1687 | 0.1208 |
| itemknn | 0.1819 | 0.1266 |

## Wilcoxon signed-rank (one-sided: itemknn > userknn)
- Estatística W: 344790
- p-valor: < 0.001
- Cohen's d: 0.1111

## Interpretação
A melhoria de itemknn sobre userknn é estatisticamente significativa (p=0.0000).
Tamanho de efeito: pequeno (|d| < 0.2)