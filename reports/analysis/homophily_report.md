# Teste de Homofilia Musical — Last.fm

Pares de amigos analisados: 12717  
Pares aleatórios (baseline): 12717  
Permutações: 1000  

## Jaccard (overlap de artistas)
| Grupo | Média | Mediana | Std |
|---|---|---|---|
| Amigos | 0.1026 | 0.0870 | 0.0839 |
| Aleatório | 0.0247 | 0.0101 | 0.0470 |
| Permutação (média) | 0.0532 | — | 0.0005 |

**p-valor (permutação):** 0.0000

## Cosseno com log1p(weight)
| Grupo | Média | Mediana | Std |
|---|---|---|---|
| Amigos | 0.1999 | 0.1774 | 0.1487 |
| Aleatório | 0.0482 | 0.0179 | 0.0824 |
| Permutação (média) | 0.1022 | — | 0.0008 |

**p-valor (permutação):** 0.0000

## Notas metodológicas
- Jaccard usa conjunto binário de artistas (ignora volume de escuta).
- Cosseno usa log1p(weight) para reduzir o efeito de volume absoluto.
- Teste de permutação: shuffling da coluna friendID preserva a distribuição de grau do lado esquerdo.
- Baseline aleatório: pares uniformemente amostrados entre todos os usuários com dados de escuta.
- Controle por grau/atividade disponível em homophily_pairs.csv (colunas degree_a, degree_b, n_artists_a, n_artists_b).