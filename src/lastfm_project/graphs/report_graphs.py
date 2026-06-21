from pathlib import Path
from statistics import mean, median
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity


GRAPH_CSV_DIR = Path("data/graphs/csv")
REPORT_DIR = Path("reports/graphs")

GRAPH_NAMES = [
    "user_user_friendship",
    "user_artist_bipartite",
    "artist_tag_bipartite",
    "heterogeneous_lastfm",
]


def load_graph_from_csv(graph_name: str) -> nx.Graph:
    nodes_path = GRAPH_CSV_DIR / graph_name / "nodes.csv"
    edges_path = GRAPH_CSV_DIR / graph_name / "edges.csv"

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)

    graph = nx.Graph()

    for _, row in nodes.iterrows():
        node_id = row["node_id"]
        attrs = {
            column: row[column]
            for column in nodes.columns
            if column != "node_id" and pd.notna(row[column])
        }
        graph.add_node(node_id, **attrs)

    for _, row in edges.iterrows():
        source = row["source"]
        target = row["target"]
        attrs = {
            column: row[column]
            for column in edges.columns
            if column not in ["source", "target"] and pd.notna(row[column])
        }
        graph.add_edge(source, target, **attrs)

    return graph


def get_validation_summary(graph: nx.Graph, graph_name: str) -> dict:
    edges = list(graph.edges(data=True))

    normalized_edges = [
        tuple(sorted([source, target])) + (attrs.get("edge_type", ""),)
        for source, target, attrs in edges
    ]

    return {
        "graph": graph_name,
        "duplicated_nodes": graph.number_of_nodes() - len(set(graph.nodes())),
        "duplicated_edges": len(normalized_edges) - len(set(normalized_edges)),
        "self_loops": nx.number_of_selfloops(graph),
        "isolated_nodes": nx.number_of_isolates(graph),
        "nodes_without_type": sum(
            1
            for _, attrs in graph.nodes(data=True)
            if "node_type" not in attrs or pd.isna(attrs.get("node_type"))
        ),
    }


def get_node_type_summary(graph: nx.Graph, graph_name: str) -> list:
    counts = {}

    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "undefined")
        counts[node_type] = counts.get(node_type, 0) + 1

    return [
        {
            "graph": graph_name,
            "node_type": node_type,
            "count": count,
        }
        for node_type, count in counts.items()
    ]


def get_edge_type_summary(graph: nx.Graph, graph_name: str) -> list:
    counts = {}

    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get("edge_type", "undefined")
        counts[edge_type] = counts.get(edge_type, 0) + 1

    return [
        {
            "graph": graph_name,
            "edge_type": edge_type,
            "count": count,
        }
        for edge_type, count in counts.items()
    ]


def get_top_degree_nodes(graph: nx.Graph, graph_name: str, top_n: int = 20) -> list:
    rows = []

    for node_id, degree in sorted(graph.degree(), key=lambda item: item[1], reverse=True)[:top_n]:
        attrs = graph.nodes[node_id]

        rows.append({
            "graph": graph_name,
            "node_id": node_id,
            "node_type": attrs.get("node_type", ""),
            "label": attrs.get("label", ""),
            "degree": degree,
        })

    return rows


def get_top_weighted_degree_nodes(graph: nx.Graph, graph_name: str, top_n: int = 20) -> list:
    rows = []

    for node_id, degree in sorted(graph.degree(weight="weight"), key=lambda item: item[1], reverse=True)[:top_n]:
        attrs = graph.nodes[node_id]

        rows.append({
            "graph": graph_name,
            "node_id": node_id,
            "node_type": attrs.get("node_type", ""),
            "label": attrs.get("label", ""),
            "weighted_degree": degree,
        })

    return rows


def get_component_summary(graph: nx.Graph, graph_name: str) -> dict:
    components = list(nx.connected_components(graph))
    component_sizes = [len(component) for component in components]
    largest_component = max(component_sizes) if component_sizes else 0

    return {
        "graph": graph_name,
        "connected_components": len(components),
        "largest_component_nodes": largest_component,
        "largest_component_percent": round((largest_component / graph.number_of_nodes()) * 100, 4),
        "smallest_component_nodes": min(component_sizes) if component_sizes else 0,
        "average_component_size": round(mean(component_sizes), 4) if component_sizes else 0,
        "median_component_size": round(median(component_sizes), 4) if component_sizes else 0,
    }


def get_community_summary(graph: nx.Graph, graph_name: str) -> tuple:
    communities = louvain_communities(graph, weight="weight", seed=42)
    community_rows = []

    for community_id, community in enumerate(communities, start=1):
        for node_id in community:
            attrs = graph.nodes[node_id]

            community_rows.append({
                "graph": graph_name,
                "community_id": community_id,
                "node_id": node_id,
                "node_type": attrs.get("node_type", ""),
                "label": attrs.get("label", ""),
            })

    summary = {
        "graph": graph_name,
        "communities": len(communities),
        "modularity": round(modularity(graph, communities, weight="weight"), 6),
        "largest_community_nodes": max(len(community) for community in communities) if communities else 0,
        "smallest_community_nodes": min(len(community) for community in communities) if communities else 0,
        "average_community_size": round(mean([len(community) for community in communities]), 4) if communities else 0,
        "median_community_size": round(median([len(community) for community in communities]), 4) if communities else 0,
    }

    return summary, community_rows


def get_graph_summary(graph: nx.Graph, graph_name: str, community_summary: dict) -> dict:
    degrees = [degree for _, degree in graph.degree()]
    weighted_degrees = [degree for _, degree in graph.degree(weight="weight")]

    return {
        "graph": graph_name,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 8),
        "average_degree": round(mean(degrees), 4) if degrees else 0,
        "median_degree": round(median(degrees), 4) if degrees else 0,
        "minimum_degree": min(degrees) if degrees else 0,
        "maximum_degree": max(degrees) if degrees else 0,
        "average_weighted_degree": round(mean(weighted_degrees), 4) if weighted_degrees else 0,
        "median_weighted_degree": round(median(weighted_degrees), 4) if weighted_degrees else 0,
        "minimum_weighted_degree": min(weighted_degrees) if weighted_degrees else 0,
        "maximum_weighted_degree": max(weighted_degrees) if weighted_degrees else 0,
        "communities": community_summary["communities"],
        "modularity": community_summary["modularity"],
    }


def save_markdown_report(graph_summaries: list, component_summaries: list, community_summaries: list) -> None:
    summary = pd.DataFrame(graph_summaries)
    components = pd.DataFrame(component_summaries)
    communities = pd.DataFrame(community_summaries)

    report_path = REPORT_DIR / "graph_report.md"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("# Relatório dos grafos Last.fm\n\n")

        file.write("## Resumo geral\n\n")
        file.write(summary.to_markdown(index=False))
        file.write("\n\n")

        file.write("## Componentes conectados\n\n")
        file.write(components.to_markdown(index=False))
        file.write("\n\n")

        file.write("## Comunidades Louvain\n\n")
        file.write(communities.to_markdown(index=False))
        file.write("\n")


def generate_graph_reports() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "communities").mkdir(parents=True, exist_ok=True)

    graph_summaries = []
    validation_summaries = []
    node_type_summaries = []
    edge_type_summaries = []
    top_degree_nodes = []
    top_weighted_degree_nodes = []
    component_summaries = []
    community_summaries = []

    for graph_name in GRAPH_NAMES:
        graph = load_graph_from_csv(graph_name)

        community_summary, community_rows = get_community_summary(graph, graph_name)

        graph_summaries.append(get_graph_summary(graph, graph_name, community_summary))
        validation_summaries.append(get_validation_summary(graph, graph_name))
        node_type_summaries.extend(get_node_type_summary(graph, graph_name))
        edge_type_summaries.extend(get_edge_type_summary(graph, graph_name))
        top_degree_nodes.extend(get_top_degree_nodes(graph, graph_name))
        top_weighted_degree_nodes.extend(get_top_weighted_degree_nodes(graph, graph_name))
        component_summaries.append(get_component_summary(graph, graph_name))
        community_summaries.append(community_summary)

        pd.DataFrame(community_rows).to_csv(
            REPORT_DIR / "communities" / f"{graph_name}_communities.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"Relatório gerado para: {graph_name}")

    pd.DataFrame(graph_summaries).to_csv(REPORT_DIR / "graph_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validation_summaries).to_csv(REPORT_DIR / "graph_validation.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(node_type_summaries).to_csv(REPORT_DIR / "node_type_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(edge_type_summaries).to_csv(REPORT_DIR / "edge_type_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top_degree_nodes).to_csv(REPORT_DIR / "top_degree_nodes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(top_weighted_degree_nodes).to_csv(REPORT_DIR / "top_weighted_degree_nodes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(component_summaries).to_csv(REPORT_DIR / "component_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(community_summaries).to_csv(REPORT_DIR / "community_summary.csv", index=False, encoding="utf-8-sig")

    save_markdown_report(graph_summaries, component_summaries, community_summaries)

    print(f"Relatórios salvos em: {REPORT_DIR}")


if __name__ == "__main__":
    generate_graph_reports()