"""Coleta de dados da API do Bluesky (AT Protocol)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from atproto import Client

DATA_DIR = Path("data")


def _get_client() -> Client:
    handle = os.getenv("BLUESKY_HANDLE")
    password = os.getenv("BLUESKY_PASSWORD")
    if not handle or not password:
        raise ValueError(
            "Credenciais não encontradas. Defina BLUESKY_HANDLE e BLUESKY_PASSWORD "
            "no arquivo .env (use uma senha de app do Bluesky)."
        )
    client = Client()
    client.login(handle, password)
    return client


def search_posts(client: Client, query: str, max_posts: int = 100) -> list[dict]:
    """Busca posts por termo/hashtag com paginação."""
    posts: list[dict] = []
    cursor: str | None = None

    while len(posts) < max_posts:
        params: dict = {"q": query, "limit": min(100, max_posts - len(posts))}
        if cursor:
            params["cursor"] = cursor

        response = client.app.bsky.feed.search_posts(params)
        for item in response.posts:
            posts.append(
                {
                    "uri": item.uri,
                    "author": item.author.handle,
                    "text": item.record.text or "",
                    "likes": item.like_count or 0,
                    "reposts": item.repost_count or 0,
                }
            )

        cursor = response.cursor
        if not cursor:
            break
        time.sleep(0.5)

    return posts[:max_posts]


def get_follows(client: Client, actor: str, max_follows: int = 50) -> list[str]:
    """Lista contas que o usuário segue."""
    follows: list[str] = []
    cursor: str | None = None

    while len(follows) < max_follows:
        params: dict = {"actor": actor, "limit": min(100, max_follows - len(follows))}
        if cursor:
            params["cursor"] = cursor

        response = client.app.bsky.graph.get_follows(params)
        for follow in response.follows:
            follows.append(follow.handle)

        cursor = response.cursor
        if not cursor:
            break
        time.sleep(0.5)

    return follows[:max_follows]


def collect_network(
    query: str = "python",
    max_posts: int = 80,
    max_users: int = 30,
    max_follows_per_user: int = 40,
) -> dict:
    """
    Coleta posts e relações de follow entre autores encontrados.

    Metodologia:
    1. Buscar posts por termo
    2. Extrair autores únicos
    3. Coletar relações de follow entre autores da amostra
    """
    client = _get_client()
    posts = search_posts(client, query, max_posts)

    authors = list(dict.fromkeys(p["author"] for p in posts))[:max_users]
    author_set = set(authors)

    follows: list[dict] = []
    for author in authors:
        targets = get_follows(client, author, max_follows_per_user)
        for target in targets:
            if target in author_set and target != author:
                follows.append({"source": author, "target": target})
        time.sleep(0.3)

    return {
        "query": query,
        "posts": posts,
        "follows": follows,
        "authors": authors,
    }



def save_data(data: dict, path: Path | None = None) -> Path:
    path = path or DATA_DIR / "collected_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
