# Split Treino/Validação/Teste

Usuários com menos de 5 artistas removidos: 15  
Usuários no split final: 1877  
Proporção: 80% treino / 10% validação / 10% teste  
Seed: 42  

| Conjunto | Interações | Usuários | Artistas únicos |
|---|---|---|---|
| Treino | 74246 | 1877 | 15347 |
| Validação | 9279 | 1877 | 4022 |
| Teste | 9276 | 1877 | 3978 |

## Distribuição de itens por usuário
| Conjunto | Média | Mediana | Min | Max |
|---|---|---|---|---|
| Treino | 39.6 | 40.0 | 3 | 40 |
| Validação | 4.9 | 5.0 | 1 | 5 |
| Teste | 4.9 | 5.0 | 1 | 5 |

## Notas
- Split estratificado por usuário (não global).
- Para usuários com poucos artistas: n_test = max(1, floor(n × 0.1)), n_val = max(1, floor(n × 0.1)).
- Pesos salvos como originais (weight). Aplicar log1p nos modelos conforme necessário.
- Grafo de amizade (user_friends_clean.csv) não é dividido — usado completo como A_social.
- Validação: escolher hiperparâmetros e pesos dos híbridos.
- Teste: avaliação final única. Não usar para seleção de modelos.