"""
Análise de Comunidades e Recomendação no Bluesky
=================================================
Pipeline completo para a atividade de Análise de Redes Sociais.

Uso:
    python main.py --query python            # coleta (requer .env)
    python main.py --query "machine learning" --max-posts 50
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from collector import collect_network, save_data
from network_analysis import (
    build_graph,
    community_summary,
    compute_metrics,
    detect_communities,
    plot_network,
)
from recommendation import build_user_profiles, recommend_posts, recommend_users
from report_generator import generate_report
from text_mining import plot_wordcloud, texts_by_community, top_terms_by_community


def run_analysis(data: dict) -> str:
    print(f"\n{'='*50}")
    print("  ANÁLISE DE REDES SOCIAIS — BLUESKY")
    print(f"{'='*50}\n")

    # Etapa 1: Construção da rede
    print("[1/5] Construindo grafo de usuários...")
    graph = build_graph(data)
    print(f"      Nós: {graph.number_of_nodes()}, Arestas: {graph.number_of_edges()}")

    # Etapa 2: Detecção de comunidades
    print("[2/5] Detectando comunidades (Louvain)...")
    communities = detect_communities(graph, data)
    n_communities = len(set(communities.values()))
    print(f"      Comunidades encontradas: {n_communities}")

    metrics_df = compute_metrics(graph, communities)
    comm_summary_df = community_summary(graph, communities)

    # Etapa 3: Visualização
    print("[3/5] Gerando visualização da rede...")
    network_img = plot_network(graph, communities)

    # Etapa 4: Text mining por comunidade
    print("[4/5] Análise de texto por comunidade...")
    grouped = texts_by_community(data, communities)
    top_terms = top_terms_by_community(grouped)

    wordcloud_images = []
    for comm_id in sorted(grouped.keys()):
        wc = plot_wordcloud(grouped, comm_id)
        if wc:
            wordcloud_images.append(wc)

    # Etapa 5: Recomendações
    print("[5/5] Gerando recomendações...")
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

    # Relatório PDF
    print("\nGerando relatório PDF...")
    report_path = generate_report(
        data=data,
        metrics_df=metrics_df,
        comm_summary_df=comm_summary_df,
        top_terms=top_terms,
        recommendations=recommendations,
        network_image=network_img,
        wordcloud_images=wordcloud_images,
    )

    print(f"\n{'='*50}")
    print(f"  Relatório salvo em: {report_path}")
    print(f"  Visualização da rede: {network_img}")
    print(f"{'='*50}\n")

    # Resumo no terminal
    print("Resumo das comunidades:")
    print(comm_summary_df.to_string(index=False))
    print("\nTermos por comunidade:")
    for comm_id, terms in sorted(top_terms.items()):
        print(f"  Comunidade {comm_id}: {', '.join(t for t, _ in terms)}")

    return report_path


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Análise de comunidades no Bluesky")
    parser.add_argument("--query", default="python", help="Termo de busca no Bluesky")
    parser.add_argument("--max-posts", type=int, default=80, help="Máximo de posts")
    parser.add_argument("--max-users", type=int, default=30, help="Máximo de usuários")
    args = parser.parse_args()

    try:
        print(f"Coletando dados do Bluesky (termo: '{args.query}')...")
        print("Isso pode levar alguns minutos (rate limiting da API).\n")
        data = collect_network(
            query=args.query,
            max_posts=args.max_posts,
            max_users=args.max_users,
        )
        save_data(data)
        print(f"Dados salvos em data/collected_data.json")
    except ValueError as e:
        print(f"\nErro: {e}")
        print("\nDica: crie um arquivo .env com suas credenciais:")
        print("  BLUESKY_HANDLE=seuusuario.bsky.social")
        print("  BLUESKY_PASSWORD=sua-senha-de-app")
        sys.exit(1)

    run_analysis(data)


if __name__ == "__main__":
    main()
