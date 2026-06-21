# Relatório dos grafos Last.fm

## Resumo geral

| graph                 |   nodes |   edges |    density |   average_degree |   median_degree |   minimum_degree |   maximum_degree |   average_weighted_degree |   median_weighted_degree |   minimum_weighted_degree |   maximum_weighted_degree |   communities |   modularity |
|:----------------------|--------:|--------:|-----------:|-----------------:|----------------:|-----------------:|-----------------:|--------------------------:|-------------------------:|--------------------------:|--------------------------:|--------------:|-------------:|
| user_user_friendship  |    1892 |   12717 | 0.00710889 |          13.4429 |               6 |                1 |              119 |                   13.4429 |                        6 |                         1 |             119           |            34 |     0.458481 |
| user_artist_bipartite |   19524 |   92834 | 0.0004871  |           9.5097 |               1 |                1 |              611 |                 7087.07   |                      439 |                         1 |               2.39314e+06 |            32 |     0.586644 |
| artist_tag_bipartite  |   21851 |  108437 | 0.00045424 |           9.9251 |               3 |                1 |             2248 |                   16.9275 |                        3 |                         1 |            7459           |           206 |     0.523534 |
| heterogeneous_lastfm  |   29242 |  213988 | 0.00050052 |          14.6357 |               3 |                1 |             2248 |                 4745.34   |                      139 |                         1 |               2.39407e+06 |            23 |     0.591673 |

## Componentes conectados

| graph                 |   connected_components |   largest_component_nodes |   largest_component_percent |   smallest_component_nodes |   average_component_size |   median_component_size |
|:----------------------|-----------------------:|--------------------------:|----------------------------:|---------------------------:|-------------------------:|------------------------:|
| user_user_friendship  |                     20 |                      1843 |                     97.4101 |                          2 |                   94.6   |                       2 |
| user_artist_bipartite |                      8 |                     19507 |                     99.9129 |                          2 |                 2440.5   |                       2 |
| artist_tag_bipartite  |                    169 |                     21431 |                     98.0779 |                          2 |                  129.296 |                       2 |
| heterogeneous_lastfm  |                      1 |                     29242 |                    100      |                      29242 |                29242     |                   29242 |

## Comunidades Louvain

| graph                 |   communities |   modularity |   largest_community_nodes |   smallest_community_nodes |   average_community_size |   median_community_size |
|:----------------------|--------------:|-------------:|--------------------------:|---------------------------:|-------------------------:|------------------------:|
| user_user_friendship  |            34 |     0.458481 |                       668 |                          2 |                  55.6471 |                       3 |
| user_artist_bipartite |            32 |     0.586644 |                      4132 |                          2 |                 610.125  |                      96 |
| artist_tag_bipartite  |           206 |     0.523534 |                      3517 |                          2 |                 106.073  |                       2 |
| heterogeneous_lastfm  |            23 |     0.591673 |                      5530 |                          9 |                1271.39   |                     610 |
