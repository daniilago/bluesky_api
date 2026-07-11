"""Construção da rede e detecção de comunidades."""

from __future__ import annotations

import re
from collections import defaultdict

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "em", "um", "uma", "os", "as", "dos", "das",
    "para", "com", "por", "que", "na", "no", "se", "é", "ao", "mais", "como",
    "the", "and", "to", "of", "in", "is", "it", "for", "on", "this", "that",
    "rt", "via", "https", "http", "www",
}


def _author_terms(texts: list[str], query: str = "") -> set[str]:
    """Extrai termos relevantes do perfil textual, ignorando o termo de busca."""
    query_words = {w.lower() for w in re.findall(r"\w+", query)}
    terms: set[str] = set()
    for text in texts:
        for word in re.findall(r"\w+", text.lower()):
            if len(word) > 2 and word not in STOPWORDS and word not in query_words:
                terms.add(word)
    return terms


def build_graph(data: dict, content_similarity: float = 0.08) -> nx.Graph:
    """
    Constrói grafo não-direcionado a partir de:
    - arestas de follow (source -> target)
    - similaridade de conteúdo entre autores (Jaccard >= limiar)
    """
    g = nx.Graph()
    authors = data.get("authors") or list(dict.fromkeys(p["author"] for p in data["posts"]))
    query = data.get("query", "")

    posts_by_author: dict[str, list[str]] = defaultdict(list)
    for post in data["posts"]:
        if post["author"] in authors:
            posts_by_author[post["author"]].append(post["text"])

    for author in authors:
        g.add_node(author)

    for edge in data.get("follows", []):
        if edge["source"] in authors and edge["target"] in authors:
            g.add_edge(edge["source"], edge["target"], weight=2, type="follow")

    terms_by_author = {author: _author_terms(texts, query) for author, texts in posts_by_author.items()}
    author_list = list(terms_by_author.keys())
    for i, a in enumerate(author_list):
        for b in author_list[i + 1 :]:
            terms_a = terms_by_author[a]
            terms_b = terms_by_author[b]
            if not terms_a or not terms_b:
                continue

            intersection = terms_a & terms_b
            union = terms_a | terms_b
            jaccard = len(intersection) / len(union)
            if jaccard < content_similarity:
                continue

            weight = 1 + jaccard
            if g.has_edge(a, b):
                g[a][b]["weight"] += weight
                g[a][b]["type"] = "mixed"
            else:
                g.add_edge(a, b, weight=weight, type="content")

    return g


def _text_cluster_communities(data: dict, n_clusters: int | None = None) -> dict[str, int]:
    """Agrupa usuários por similaridade textual quando a rede social é esparsa."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    authors = data.get("authors") or list(dict.fromkeys(p["author"] for p in data["posts"]))
    profiles: dict[str, str] = defaultdict(str)
    for post in data["posts"]:
        if post["author"] in authors:
            profiles[post["author"]] += " " + (post["text"] or "")

    author_list = [a for a in authors if profiles.get(a, "").strip()]
    if len(author_list) < 2:
        return {a: 0 for a in author_list}

    k = n_clusters or max(2, min(10, len(author_list) // 6))
    vectorizer = TfidfVectorizer(max_features=300, stop_words="english")
    matrix = vectorizer.fit_transform([profiles[a] for a in author_list])
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)
    return {author: int(label) for author, label in zip(author_list, labels)}


def detect_communities(g: nx.Graph, data: dict | None = None) -> dict[str, int]:
    """
    Detecta comunidades com Louvain na rede social.
    """
    n_nodes = g.number_of_nodes()
    if n_nodes == 0:
        return {}

    n_edges = g.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) // 2
    density = n_edges / max_edges if max_edges else 0

    # Grafo quase completo: Louvain retorna 1 comunidade 
    if density > 0.85 and data:
        return _text_cluster_communities(data)

    if n_edges == 0:
        if data:
            return _text_cluster_communities(data)
        return {node: 0 for node in g.nodes}

    partition = community_louvain.best_partition(g, weight="weight")

    # Muitos isolados: Louvain cria 1 comunidade por nó 
    sizes = defaultdict(int)
    for comm in partition.values():
        sizes[comm] += 1
    singleton_ratio = sum(1 for size in sizes.values() if size == 1) / len(sizes)
    if singleton_ratio > 0.6 and data:
        return _text_cluster_communities(data)

    return partition


def compute_metrics(g: nx.Graph, communities: dict[str, int]) -> pd.DataFrame:
    """Calcula métricas básicas por nó."""
    degree = dict(g.degree())
    rows = []
    for node in g.nodes:
        rows.append(
            {
                "usuario": node,
                "comunidade": communities.get(node, -1),
                "grau": degree.get(node, 0),
            }
        )
    return pd.DataFrame(rows)


def community_summary(g: nx.Graph, communities: dict[str, int]) -> pd.DataFrame:
    """Resumo estatístico por comunidade."""
    comm_nodes: dict[int, list[str]] = defaultdict(list)
    for node, comm in communities.items():
        comm_nodes[comm].append(node)

    rows = []
    for comm_id, nodes in sorted(comm_nodes.items()):
        subgraph = g.subgraph(nodes)
        rows.append(
            {
                "comunidade": comm_id,
                "n_usuarios": len(nodes),
                "n_arestas": subgraph.number_of_edges(),
                "densidade": nx.density(subgraph) if len(nodes) > 1 else 0,
                "usuarios": ", ".join(nodes[:3]) + (f" (+{len(nodes) - 3} outros)" if len(nodes) > 3 else ""),
            }
        )
    return pd.DataFrame(rows)


def plot_network(
    g: nx.Graph,
    communities: dict[str, int],
    output_path: str = "output/rede_comunidades.png",
) -> str:
    """Visualiza a rede com cores por comunidade."""
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(g, seed=42, k=1.5)

    colors = [communities.get(node, -1) for node in g.nodes]
    nx.draw_networkx_nodes(g, pos, node_color=colors, cmap=plt.cm.Set3, node_size=400, alpha=0.9)
    nx.draw_networkx_edges(g, pos, alpha=0.3, width=1)
    nx.draw_networkx_labels(g, pos, font_size=7, font_family="sans-serif")

    plt.title("Rede de usuários Bluesky — Comunidades detectadas (Louvain)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
