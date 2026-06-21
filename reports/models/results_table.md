# Tabela de Resultados — Last.fm HetRec2011 (warm-start)

## Resultados — Validação

| Modelo | NDCG@10 | PRECISION@10 | RECALL@10 | F1@10 |
|---|---|---|---|---|
| Popularidade | 0.0545 | 0.0312 | 0.0686 | 0.0425 |
| Social Comunidade (Louvain) | 0.1038 | 0.0516 | 0.1149 | 0.0704 |
| Social 1-hop | 0.1251 | 0.0616 | 0.1363 | 0.0841 |
| Social PPR | 0.1301 | 0.0638 | 0.1419 | 0.0871 |
| ItemKNN | 0.1816 | 0.0898 | 0.2004 | 0.1228 |
| UserKNN | 0.1705 | 0.0853 | 0.1918 | 0.1169 |
| EASE | 0.1971 | 0.0990 | 0.2221 | 0.1355 |
| UserKNN + Popularidade | 0.1707 | 0.0855 | 0.1921 | 0.1171 |
| UserKNN + PPR | 0.1749 | 0.0881 | 0.1970 | 0.1205 |
| ItemKNN + PPR | 0.1826 | 0.0910 | 0.2038 | 0.1245 |
| UserKNN + ItemKNN | 0.1865 | 0.0926 | 0.2092 | 0.1269 |
| UserKNN + ItemKNN + PPR | 0.1880 | 0.0935 | 0.2103 | 0.1280 |
| RRF: UserKNN + PPR | 0.1723 | 0.0876 | 0.1964 | 0.1200 |
| RRF: ItemKNN + PPR | 0.1818 | 0.0912 | 0.2045 | 0.1248 |
| RRF: UserKNN + ItemKNN + PPR | 0.1873 | 0.0929 | 0.2088 | 0.1273 |

## Resultados — Teste

| Modelo | NDCG@10 | PRECISION@10 | RECALL@10 | F1@10 |
|---|---|---|---|---|
| Popularidade | 0.0528 | 0.0307 | 0.0673 | 0.0418 |
| Social Comunidade (Louvain) | 0.1045 | 0.0536 | 0.1168 | 0.0730 |
| Social 1-hop | 0.1193 | 0.0602 | 0.1322 | 0.0820 |
| Social PPR | 0.1251 | 0.0634 | 0.1403 | 0.0866 |
| ItemKNN | 0.1819 | 0.0911 | 0.2021 | 0.1244 |
| UserKNN | 0.1687 | 0.0856 | 0.1909 | 0.1171 |
| EASE | 0.1976 | 0.0985 | 0.2192 | 0.1346 |
| UserKNN + Popularidade | 0.1685 | 0.0856 | 0.1907 | 0.1170 |
| UserKNN + PPR | 0.1722 | 0.0879 | 0.1956 | 0.1201 |
| ItemKNN + PPR | 0.1841 | 0.0916 | 0.2048 | 0.1253 |
| UserKNN + ItemKNN | 0.1859 | 0.0938 | 0.2098 | 0.1283 |
| UserKNN + ItemKNN + PPR | 0.1879 | 0.0943 | 0.2108 | 0.1290 |
| RRF: UserKNN + PPR | 0.1688 | 0.0867 | 0.1941 | 0.1187 |
| RRF: ItemKNN + PPR | 0.1785 | 0.0906 | 0.2024 | 0.1240 |
| RRF: UserKNN + ItemKNN + PPR | 0.1848 | 0.0934 | 0.2082 | 0.1277 |