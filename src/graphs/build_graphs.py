from pathlib import Path
import pickle
import pandas as pd
import networkx as nx


PROCESSED_DIR = Path("data/processed")
GRAPH_DIR = Path("data/graphs")


def get_artist_labels() -> dict:
    artists = pd.read_csv(PROCESSED_DIR / "artists_clean.csv")
    return dict(zip(artists["artistID"].astype(int), artists["artist_name"].astype(str)))


def get_tag_labels() -> dict:
    tags = pd.read_csv(PROCESSED_DIR / "tags_clean.csv")
    return dict(zip(tags["tagID"].astype(int), tags["tag_value"].astype(str)))


def add_artist_node(graph: nx.Graph, artist_id: int, artist_labels: dict) -> None:
    graph.add_node(
        f"artist_{artist_id}",
        node_type="artist",
        label=artist_labels.get(artist_id, "")
    )


def add_tag_node(graph: nx.Graph, tag_id: int, tag_labels: dict) -> None:
    graph.add_node(
        f"tag_{tag_id}",
        node_type="tag",
        label=tag_labels.get(tag_id, "")
    )


def build_user_user_graph() -> nx.Graph:
    friends = pd.read_csv(PROCESSED_DIR / "user_friends_clean.csv")

    graph = nx.Graph()

    for _, row in friends.iterrows():
        user_id = f"user_{int(row['userID'])}"
        friend_id = f"user_{int(row['friendID'])}"

        graph.add_node(user_id, node_type="user")
        graph.add_node(friend_id, node_type="user")
        graph.add_edge(user_id, friend_id, edge_type="friendship", weight=1.0)

    return graph


def build_user_artist_graph() -> nx.Graph:
    user_artists = pd.read_csv(PROCESSED_DIR / "user_artists_clean.csv")
    artist_labels = get_artist_labels()

    graph = nx.Graph()

    for _, row in user_artists.iterrows():
        user_id = int(row["userID"])
        artist_id = int(row["artistID"])

        graph.add_node(f"user_{user_id}", node_type="user")
        add_artist_node(graph, artist_id, artist_labels)

        graph.add_edge(
            f"user_{user_id}",
            f"artist_{artist_id}",
            edge_type="listened",
            weight=float(row["weight"])
        )

    return graph


def build_artist_tag_graph() -> nx.Graph:
    artist_tags = pd.read_csv(PROCESSED_DIR / "artist_tags_clean.csv")
    artist_labels = get_artist_labels()
    tag_labels = get_tag_labels()

    graph = nx.Graph()

    for _, row in artist_tags.iterrows():
        artist_id = int(row["artistID"])
        tag_id = int(row["tagID"])

        add_artist_node(graph, artist_id, artist_labels)
        add_tag_node(graph, tag_id, tag_labels)

        graph.add_edge(
            f"artist_{artist_id}",
            f"tag_{tag_id}",
            edge_type="has_tag",
            weight=float(row["tag_weight"])
        )

    return graph


def build_heterogeneous_graph() -> nx.Graph:
    graph = nx.Graph()

    friends = pd.read_csv(PROCESSED_DIR / "user_friends_clean.csv")
    user_artists = pd.read_csv(PROCESSED_DIR / "user_artists_clean.csv")
    artist_tags = pd.read_csv(PROCESSED_DIR / "artist_tags_clean.csv")

    artist_labels = get_artist_labels()
    tag_labels = get_tag_labels()

    for _, row in friends.iterrows():
        user_id = int(row["userID"])
        friend_id = int(row["friendID"])

        graph.add_node(f"user_{user_id}", node_type="user")
        graph.add_node(f"user_{friend_id}", node_type="user")

        graph.add_edge(
            f"user_{user_id}",
            f"user_{friend_id}",
            edge_type="friendship",
            weight=1.0
        )

    for _, row in user_artists.iterrows():
        user_id = int(row["userID"])
        artist_id = int(row["artistID"])

        graph.add_node(f"user_{user_id}", node_type="user")
        add_artist_node(graph, artist_id, artist_labels)

        graph.add_edge(
            f"user_{user_id}",
            f"artist_{artist_id}",
            edge_type="listened",
            weight=float(row["weight"])
        )

    for _, row in artist_tags.iterrows():
        artist_id = int(row["artistID"])
        tag_id = int(row["tagID"])

        add_artist_node(graph, artist_id, artist_labels)
        add_tag_node(graph, tag_id, tag_labels)

        graph.add_edge(
            f"artist_{artist_id}",
            f"tag_{tag_id}",
            edge_type="has_tag",
            weight=float(row["tag_weight"])
        )

    return graph


def save_graph_csv(graph: nx.Graph, graph_name: str) -> None:
    csv_dir = GRAPH_DIR / "csv" / graph_name
    csv_dir.mkdir(parents=True, exist_ok=True)

    nodes = []

    for node_id, attrs in graph.nodes(data=True):
        row = {"node_id": node_id}
        row.update(attrs)
        nodes.append(row)

    edges = []

    for source, target, attrs in graph.edges(data=True):
        row = {
            "source": source,
            "target": target
        }
        row.update(attrs)
        edges.append(row)

    pd.DataFrame(nodes).to_csv(
        csv_dir / "nodes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(edges).to_csv(
        csv_dir / "edges.csv",
        index=False,
        encoding="utf-8-sig"
    )


def save_graph(graph: nx.Graph, graph_name: str) -> None:
    graphml_dir = GRAPH_DIR / "graphml"
    pkl_dir = GRAPH_DIR / "pkl"
    gpickle_dir = GRAPH_DIR / "gpickle"

    graphml_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    gpickle_dir.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(
        graph,
        graphml_dir / f"{graph_name}.graphml"
    )

    with open(pkl_dir / f"{graph_name}.pkl", "wb") as file:
        pickle.dump(graph, file)

    with open(gpickle_dir / f"{graph_name}.gpickle", "wb") as file:
        pickle.dump(graph, file)

    save_graph_csv(graph, graph_name)

    print(f"Grafo salvo: {graph_name}")
    print(f"Nós: {graph.number_of_nodes()}")
    print(f"Arestas: {graph.number_of_edges()}")


def build_all_graphs() -> None:
    save_graph(build_user_user_graph(), "user_user_friendship")
    save_graph(build_user_artist_graph(), "user_artist_bipartite")
    save_graph(build_artist_tag_graph(), "artist_tag_bipartite")
    save_graph(build_heterogeneous_graph(), "heterogeneous_lastfm")


if __name__ == "__main__":
    build_all_graphs()