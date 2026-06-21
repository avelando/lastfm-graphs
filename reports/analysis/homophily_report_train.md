# Teste de Homofilia Musical — Last.fm (dados de treino (80%))

Pares de amigos analisados: 12609  
Pares aleatórios (baseline): 12609  
Permutações: 1000  

## Jaccard (overlap de artistas)
| Grupo | Média | Mediana | Std |
|---|---|---|---|
| Amigos | 0.0804 | 0.0667 | 0.0665 |
| Aleatório | 0.0188 | 0.0000 | 0.0318 |
| Permutação (média) | 0.0421 | — | 0.0004 |

**Teste de permutação — p-valor:** < 0.001  
**Mann-Whitney U:** 130470332, p < 0.001  
**Cliff's Delta:** 0.6413  

## Cosseno com log1p(weight)
| Grupo | Média | Mediana | Std |
|---|---|---|---|
| Amigos | 0.1604 | 0.1401 | 0.1239 |
| Aleatório | 0.0374 | 0.0000 | 0.0627 |
| Permutação (média) | 0.0824 | — | 0.0007 |

**Teste de permutação — p-valor:** < 0.001  
**Mann-Whitney U:** 131322128, p < 0.001  
**Cliff's Delta:** 0.6520  

## Notas metodológicas
- Jaccard usa conjunto binário de artistas (ignora volume de escuta).
- Cosseno usa log1p(weight) para reduzir o efeito de volume absoluto.
- Baseline aleatório: pares amostrados sem reposição, excluindo self-pairs, pares de amigos e duplicatas.
- Teste de permutação: embaralha um dos extremos das arestas de amizade e recomputa a média de similaridade.
  Não deve ser interpretado como controle completo por grau (ver limpeza de user_friends_clean.csv).
  p-valor calculado com correção de Northrop: (extremos + 1) / (1000 + 1).
- Mann-Whitney U testa se as distribuições diferem; Cliff's Delta mede tamanho de efeito (-1 a 1).
- Controle por grau/atividade disponível em homophily_pairs.csv (colunas degree_a, degree_b, n_artists_a, n_artists_b).