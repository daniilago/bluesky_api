"""Recomendação simples baseada em comunidades e similaridade de conteúdo."""

from __future__ import annotations

from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_user_profiles(data: dict) -> dict[str, str]:
    """Concatena todos os posts de cada usuário em um perfil textual."""
    profiles: dict[str, list[str]] = defaultdict(list)
    for post in data["posts"]:
        profiles[post["author"]].append(post["text"])
    return {user: " ".join(texts) for user, texts in profiles.items()}


def recommend_users(
    target_user: str,
    profiles: dict[str, str],
    communities: dict[str, int],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """
    Recomenda usuários similares dentro da mesma comunidade.
    Usa similaridade de cosseno entre perfis TF-IDF.
    """
    if target_user not in profiles or target_user not in communities:
        return []

    target_comm = communities[target_user]
    candidates = [
        u for u in profiles if u != target_user and communities.get(u) == target_comm
    ]
    if not candidates:
        return []

    all_users = [target_user] + candidates
    texts = [profiles[u] for u in all_users]

    vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = sorted(zip(candidates, sim), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def recommend_posts(
    target_user: str,
    data: dict,
    communities: dict[str, int],
    profiles: dict[str, str],
    top_n: int = 3,
) -> list[dict]:
    """
    Recomenda posts populares de usuários similares na mesma comunidade.
    """
    similar_users = [u for u, _ in recommend_users(target_user, profiles, communities, top_n=5)]
    if not similar_users:
        return []

    recommended = []
    for post in data["posts"]:
        if post["author"] in similar_users:
            recommended.append(
                {
                    "autor": post["author"],
                    "texto": post["text"][:120] + ("..." if len(post["text"]) > 120 else ""),
                    "likes": post.get("likes", 0),
                    "score": post.get("likes", 0) + post.get("reposts", 0),
                }
            )

    recommended.sort(key=lambda x: x["score"], reverse=True)
    return recommended[:top_n]
