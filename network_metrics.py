"""
- Distribuição de graus (histograma + curva de frequência acumulada)
- Coeficiente de clustering global e por nó
- Centralidade de grau e eigenvector
- Visualização da rede com tamanho dos nós proporcional à centralidade
"""

from __future__ import annotations

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# Cálculo de métricas

def compute_basic_properties(g: nx.Graph) -> dict:
    """Número de vértices, arestas, densidade, diâmetro (se conexo), etc."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = nx.density(g)
    components = list(nx.connected_components(g))
    largest_cc = max(components, key=len)
    g_lcc = g.subgraph(largest_cc)

    avg_clustering = nx.average_clustering(g)
    avg_degree = (2 * m / n) if n > 0 else 0

    try:
        diameter = nx.diameter(g_lcc)
        avg_path = nx.average_shortest_path_length(g_lcc)
    except Exception:
        diameter = None
        avg_path = None

    return {
        "n_vertices": n,
        "n_arestas": m,
        "densidade": round(density, 4),
        "n_componentes": len(components),
        "tamanho_maior_componente": len(largest_cc),
        "grau_medio": round(avg_degree, 3),
        "coef_clustering_medio": round(avg_clustering, 4),
        "diametro_lcc": diameter,
        "caminho_medio_lcc": round(avg_path, 3) if avg_path else None,
    }


def compute_centralities(g: nx.Graph) -> pd.DataFrame:
    """Calcula centralidades de grau e eigenvector para todos os nós."""
    degree_cent = nx.degree_centrality(g)

    # eigenvector pode não convergir em grafos desconexos — tratar exceção
    try:
        eigen_cent = nx.eigenvector_centrality(g, max_iter=500, weight="weight")
    except nx.PowerIterationFailedConvergence:
        # fallback: calcular na maior componente conexa
        lcc_nodes = max(nx.connected_components(g), key=len)
        g_lcc = g.subgraph(lcc_nodes)
        eigen_lcc = nx.eigenvector_centrality(g_lcc, max_iter=500, weight="weight")
        eigen_cent = {node: eigen_lcc.get(node, 0.0) for node in g.nodes}

    clustering = nx.clustering(g)

    rows = []
    for node in g.nodes:
        rows.append(
            {
                "usuario": node,
                "grau": g.degree(node),
                "centralidade_grau": round(degree_cent[node], 4),
                "centralidade_eigenvector": round(eigen_cent.get(node, 0.0), 4),
                "coef_clustering": round(clustering.get(node, 0.0), 4),
            }
        )
    df = pd.DataFrame(rows).sort_values("centralidade_eigenvector", ascending=False)
    return df.reset_index(drop=True)


# Visualizações 

def plot_degree_distribution(
    g: nx.Graph,
    output_path: str = "output/distribuicao_graus.png",
) -> str:
    """Histograma de distribuição de graus + CDF acumulada."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    degrees = [d for _, d in g.degree()]
    degree_count = Counter(degrees)
    deg_vals = sorted(degree_count.keys())
    freq = [degree_count[d] for d in deg_vals]
    total = sum(freq)
    cdf = np.cumsum(freq) / total

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histograma
    axes[0].bar(deg_vals, freq, color="#4472C4", alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Distribuição de Graus")
    axes[0].set_xlabel("Grau")
    axes[0].set_ylabel("Frequência")
    mean_deg = np.mean(degrees)
    axes[0].axvline(mean_deg, color="#e74c3c", linestyle="--", linewidth=1.5,
                    label=f"Média = {mean_deg:.2f}")
    axes[0].legend()

    # CDF
    axes[1].step(deg_vals, cdf, where="post", color="#2ecc71", linewidth=2)
    axes[1].set_title("CDF dos Graus")
    axes[1].set_xlabel("Grau")
    axes[1].set_ylabel("P(X ≤ grau)")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Propriedades da Rede — Distribuição de Graus", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_network_centrality(
    g: nx.Graph,
    communities: dict[str, int],
    centrality_df: pd.DataFrame,
    mode: str = "eigenvector",
    output_path: str = "output/rede_centralidade.png",
) -> str:
    """
    Visualiza a rede com:
    - Cor dos nós = comunidade
    - Tamanho dos nós proporcional à centralidade escolhida
    - Espessura das arestas proporcional ao peso
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    col = f"centralidade_{mode}"
    cent_map = dict(zip(centrality_df["usuario"], centrality_df[col]))

    # normaliza tamanhos: min=100, max=1200
    values = np.array([cent_map.get(n, 0.0) for n in g.nodes])
    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        sizes = 100 + (values - v_min) / (v_max - v_min) * 1100
    else:
        sizes = np.full(len(values), 300)

    colors = [communities.get(n, 0) for n in g.nodes]
    pos = nx.spring_layout(g, seed=42, k=1.5)

    # espessura das arestas
    edge_weights = [g[u][v].get("weight", 1) for u, v in g.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + (w / max_w) * 2.5 for w in edge_weights]

    fig, ax = plt.subplots(figsize=(14, 9))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors,
                           cmap=plt.cm.Set3, node_size=sizes, alpha=0.92)
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.35,
                           width=edge_widths, edge_color="#555555")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=6.5, font_family="sans-serif")

    label_name = "Eigenvector" if mode == "eigenvector" else "Grau"
    ax.set_title(
        f"Rede Bluesky — Tamanho dos nós proporcional à centralidade de {label_name}",
        fontsize=12, fontweight="bold",
    )
    ax.axis("off")

    # legenda de tamanho
    for size_val, label in [(100, "baixa"), (600, "média"), (1200, "alta")]:
        ax.scatter([], [], s=size_val, c="gray", alpha=0.6, label=f"Centralidade {label}")
    ax.legend(scatterpoints=1, frameon=True, loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
