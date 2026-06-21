# Split Treino/Validação/Teste

Usuários com menos de 5 artistas removidos: 15  
Usuários no split final: 1877  
Proporção: 80% treino / 10% validação / 10% teste  
Seed: 42  

| Conjunto | Interações | Usuários | Artistas únicos |
|---|---|---|---|
| Treino | 76106 | 1877 | 15590 |
| Validação | 9263 | 1877 | 3984 |
| Teste | 7432 | 1877 | 3423 |

## Distribuição de itens por usuário
| Conjunto | Média | Mediana | Min | Max |
|---|---|---|---|---|
| Treino | 40.5 | 41.0 | 3 | 41 |
| Validação | 4.9 | 5.0 | 1 | 5 |
| Teste | 4.0 | 4.0 | 1 | 4 |

## Notas
- Split estratificado por usuário (não global).
- Para usuários com poucos artistas: n_test = max(1, floor(n × 0.1)), n_val = max(1, floor(n × 0.1)).
- Pesos salvos como originais (weight). Aplicar log1p nos modelos conforme necessário.
- Grafo de amizade (user_friends_clean.csv) não é dividido — usado completo como A_social.
- Validação: escolher hiperparâmetros e pesos dos híbridos.
- Teste: avaliação final única. Não usar para seleção de modelos.