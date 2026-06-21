from pathlib import Path
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities


GRAPH_CSV_DIR = Path("data/graphs/csv")
REPORT_DIR = Path("reports/graphs")


def load_user_graph() -> nx.Graph:
    nodes_path = GRAPH_CSV_DIR / "user_user_friendship" / "nodes.csv"
    edges_path = GRAPH_CSV_DIR / "user_user_friendship" / "edges.csv"

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    graph = nx.Graph()

    for _, row in nodes.iterrows():
        graph.add_node(row["node_id"], node_type=row["node_type"])

    for _, row in edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=float(row["weight"]))

    return graph


def compute_community_map(graph: nx.Graph) -> dict:
    communities = louvain_communities(graph, weight="weight", seed=42)
    community_map = {}
    for community_id, members in enumerate(communities):
        for node in members:
            community_map[node] = community_id
    return community_map


def compute_component_map(graph: nx.Graph) -> dict:
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_map = {}
    for component_id, members in enumerate(components):
        for node in members:
            component_map[node] = component_id
    return component_map


def generate_user_metrics() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    graph = load_user_graph()

    degree = dict(graph.degree())
    clustering = nx.clustering(graph)
    pagerank = nx.pagerank(graph, weight="weight")
    community_map = compute_community_map(graph)
    component_map = compute_component_map(graph)

    rows = []
    for node_id in graph.nodes():
        user_id = int(node_id.split("_")[1])
        rows.append({
            "userID": user_id,
            "degree": degree[node_id],
            "clustering": round(clustering[node_id], 6),
            "pagerank": round(pagerank[node_id], 8),
            "community": community_map[node_id],
            "component": component_map[node_id],
        })

    df = pd.DataFrame(rows).sort_values("userID").reset_index(drop=True)
    output_path = REPORT_DIR / "user_metrics.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"user_metrics.csv gerado: {len(df)} usuários")
    print(f"Comunidades Louvain: {df['community'].nunique()}")
    print(f"Componentes: {df['component'].nunique()} (maior: {(df['component'] == 0).sum()} nós)")
    print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    generate_user_metrics()
