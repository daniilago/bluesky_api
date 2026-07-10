"""Pipeline completo integrando as quatro atividades:
  Atividade 1 — Métricas de rede + centralidade
  Atividade 3 — Análise de sentimento por comunidade
  Atividade 4 — Detecção de comunidades + recomendação
"""

from __future__ import annotations

import json

from network_analysis import (
    build_graph,
    community_summary,
    compute_metrics,
    detect_communities,
    plot_network,
)
from network_metrics import (
    compute_basic_properties,
    compute_centralities,
    plot_degree_distribution,
    plot_network_centrality,
)
from recommendation import build_user_profiles, recommend_posts, recommend_users
from sentiment_analysis import (
    analyze_posts,
    community_sentiment_summary,
    plot_sentiment_distribution,
    plot_sentiment_stacked,
)
from text_mining import plot_wordcloud, texts_by_community, top_terms_by_community


def run_analysis(data: dict) -> dict:
    print(f"\n{'='*55}")
    print("  ANÁLISE DE REDES SOCIAIS — BLUESKY (integrado)")
    print(f"{'='*55}\n")

    # ── Etapa 1: Grafo ─────────────────────────────────────────
    print("[1/7] Construindo grafo de usuários...")
    graph = build_graph(data)
    print(f"      Nós: {graph.number_of_nodes()}, Arestas: {graph.number_of_edges()}")

    # ── Etapa 2: Comunidades ──────────────────────────────────
    print("[2/7] Detectando comunidades (Louvain)...")
    communities = detect_communities(graph, data)
    n_communities = len(set(communities.values()))
    print(f"      Comunidades encontradas: {n_communities}")

    metrics_df = compute_metrics(graph, communities)
    comm_summary_df = community_summary(graph, communities)

    # ── Etapa 3: Visualização base ────────────────────────────
    print("[3/7] Visualização da rede (por comunidade)...")
    network_img = plot_network(graph, communities, output_path="output/rede_comunidades.png")

    # ── Etapa 4: Métricas de rede (Atividade 1) ───────────────
    print("[4/7] Métricas de rede e centralidade (Atividade 1)...")
    basic_props = compute_basic_properties(graph)
    centrality_df = compute_centralities(graph)
    degree_dist_img = plot_degree_distribution(graph, output_path="output/distribuicao_graus.png")
    centrality_img = plot_network_centrality(
        graph, communities, centrality_df,
        mode="eigenvector",
        output_path="output/rede_centralidade_eigen.png",
    )
    print(f"      Grau médio: {basic_props['grau_medio']}, Clustering médio: {basic_props['coef_clustering_medio']}")
    print(f"      Nó mais central (eigenvector): {centrality_df.iloc[0]['usuario']}")

    # ── Etapa 5: Text mining ──────────────────────────────────
    print("[5/7] Text mining por comunidade...")
    grouped = texts_by_community(data, communities)
    top_terms = top_terms_by_community(grouped)
    wordcloud_images = []
    for comm_id in sorted(grouped.keys()):
        wc = plot_wordcloud(grouped, comm_id, output_dir="output")
        if wc:
            wordcloud_images.append(wc)

    # ── Etapa 6: Sentimento (Atividade 3) ────────────────────
    print("[6/7] Análise de sentimento por comunidade (Atividade 3)...")
    sentiment_df = analyze_posts(data, communities)
    sentiment_summary_df = community_sentiment_summary(sentiment_df)
    sent_dist_img = plot_sentiment_distribution(
        sentiment_df, output_path="output/sentimento_distribuicao.png"
    )
    sent_stacked_img = plot_sentiment_stacked(
        sentiment_summary_df, output_path="output/sentimento_empilhado.png"
    )
    print(f"      Sentimento médio geral: {sentiment_df['compound'].mean():+.3f}")

    # ── Etapa 7: Recomendações ────────────────────────────────
    print("[7/7] Recomendações baseadas em comunidades...")
    profiles = build_user_profiles(data)
    sample_users = list(profiles.keys())[:3]
    recommendations = []
    for user in sample_users:
        recommendations.append(
            {
                "usuario": user,
                "comunidade": communities.get(user, -1),
                "usuarios_similares": recommend_users(user, profiles, communities),
                "posts": recommend_posts(user, data, communities, profiles),
            }
        )

    print(f"\n{'='*55}")
    print("  Pipeline concluído. Gerando relatório...")
    print(f"{'='*55}\n")

    return {
        "data": data,
        "graph": graph,
        "communities": communities,
        "metrics_df": metrics_df,
        "comm_summary_df": comm_summary_df,
        "basic_props": basic_props,
        "centrality_df": centrality_df,
        "top_terms": top_terms,
        "sentiment_df": sentiment_df,
        "sentiment_summary_df": sentiment_summary_df,
        "recommendations": recommendations,
        "imgs": {
            "network": network_img,
            "centrality": centrality_img,
            "degree_dist": degree_dist_img,
            "sent_dist": sent_dist_img,
            "sent_stacked": sent_stacked_img,
            "wordclouds": wordcloud_images,
        },
    }


if __name__ == "__main__":
    with open("data/collected_data.json", encoding="utf-8") as f:
        data = json.load(f)
    results = run_analysis(data)
    print("Resultados prontos. Chamando gerador de relatório...")
