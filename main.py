from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from collector import collect_network, collect_network_multi, save_data
from run_pipeline import run_analysis


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Análise de comunidades no Bluesky",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # --query aceita um único valor (retrocompatível)
    parser.add_argument(
        "--query",
        default=None,
        help="Termo único de busca no Bluesky",
    )
    # --queries aceita vários valores: --queries "a" "b" "c"
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        metavar="QUERY",
        help="Um ou mais termos de busca (ex: --queries 'silent hill' 'resident evil')",
    )
    parser.add_argument("--max-posts", type=int, default=80,
                        help="Máximo de posts por query (default: 80)")
    parser.add_argument("--max-users", type=int, default=50,
                        help="Máximo de usuários na rede (default: 50)")
    args = parser.parse_args()

    # resolve lista de queries
    if args.queries:
        queries = args.queries
    elif args.query:
        queries = [args.query]
    else:
        queries = ["python"]

    try:
        if len(queries) == 1:
            print(f"Coletando dados do Bluesky (termo: '{queries[0]}')...")
            print("Isso pode levar alguns minutos (rate limiting da API).\n")
            data = collect_network(
                query=queries[0],
                max_posts=args.max_posts,
                max_users=args.max_users,
            )
        else:
            print(f"Coletando dados do Bluesky ({len(queries)} queries):")
            for q in queries:
                print(f"  • {q}")
            print(f"\nMáx. {args.max_posts} posts/query, {args.max_users} usuários totais.")
            print("Isso pode levar vários minutos (rate limiting da API).\n")
            data = collect_network_multi(
                queries=queries,
                max_posts_per_query=args.max_posts,
                max_users=args.max_users,
            )

        path = save_data(data)
        print(f"\nDados salvos em {path}")
        print(f"  Posts coletados : {len(data['posts'])}")
        print(f"  Autores únicos  : {len(data['authors'])}")
        print(f"  Relações follow : {len(data['follows'])}\n")

    except ValueError as e:
        print(f"\nErro: {e}")
        print("\nDica: crie um arquivo .env com suas credenciais:")
        print("  BLUESKY_HANDLE=seuusuario.bsky.social")
        print("  BLUESKY_PASSWORD=sua-senha-de-app")
        sys.exit(1)

    run_analysis(data)


if __name__ == "__main__":
    main()