# Teste de Homofilia Musical — Last.fm

Pares de amigos analisados: 12717  
Pares aleatórios (baseline): 12717  
Permutações: 1000  

## Jaccard (overlap de artistas)
| Grupo | Média | Mediana | Std |
|---|---|---|---|
| Amigos | 0.1026 | 0.0870 | 0.0839 |
| Aleatório | 0.0234 | 0.0101 | 0.0402 |
| Permutação (média) | 0.0532 | — | 0.0005 |

**Teste de permutação — p-valor:** < 0.001  
**Mann-Whitney U:** 134569257, p < 0.001  
**Cliff's Delta:** 0.6642  

## Cosseno com log1p(weight)
| Grupo | Média | Mediana | Std |
|---|---|---|---|
| Amigos | 0.1999 | 0.1774 | 0.1487 |
| Aleatório | 0.0461 | 0.0177 | 0.0769 |
| Permutação (média) | 0.1023 | — | 0.0008 |

**Teste de permutação — p-valor:** < 0.001  
**Mann-Whitney U:** 135468652, p < 0.001  
**Cliff's Delta:** 0.6753  

## Notas metodológicas
- Jaccard usa conjunto binário de artistas (ignora volume de escuta).
- Cosseno usa log1p(weight) para reduzir o efeito de volume absoluto.
- Baseline aleatório: pares amostrados sem reposição, excluindo self-pairs, pares de amigos e duplicatas.
- Teste de permutação: embaralha um dos extremos das arestas de amizade e recomputa a média de similaridade.
  Não deve ser interpretado como controle completo por grau (ver limpeza de user_friends_clean.csv).
  p-valor calculado com correção de Northrop: (extremos + 1) / (1000 + 1).
- Mann-Whitney U testa se as distribuições diferem; Cliff's Delta mede tamanho de efeito (-1 a 1).
- Controle por grau/atividade disponível em homophily_pairs.csv (colunas degree_a, degree_b, n_artists_a, n_artists_b).
- Análise realizada com histórico completo de escuta. Versão com dados de treino: a executar após split treino/teste.