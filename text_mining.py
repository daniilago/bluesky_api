"""Análise de texto por comunidade (text mining básico)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

STOPWORDS_PT = {
    "a", "o", "e", "de", "da", "do", "em", "um", "uma", "os", "as", "dos", "das",
    "para", "com", "por", "que", "na", "no", "se", "é", "ao", "mais", "como",
    "the", "and", "to", "of", "in", "is", "it", "for", "on", "this", "that",
    "rt", "via", "https", "http", "www",
}


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s#@]", " ", text)
    return text


def texts_by_community(data: dict, communities: dict[str, int]) -> dict[int, list[str]]:
    """Agrupa textos dos posts por comunidade do autor."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for post in data["posts"]:
        author = post["author"]
        if author in communities:
            grouped[communities[author]].append(_clean_text(post["text"]))
    return grouped


def top_terms_by_community(
    grouped: dict[int, list[str]], top_n: int = 10
) -> dict[int, list[tuple[str, int]]]:
    """Extrai termos mais frequentes por comunidade."""
    result: dict[int, list[tuple[str, int]]] = {}
    for comm_id, texts in grouped.items():
        words: Counter[str] = Counter()
        for text in texts:
            for word in text.split():
                if len(word) > 2 and word not in STOPWORDS_PT and not word.startswith("@"):
                    words[word] += 1
        result[comm_id] = words.most_common(top_n)
    return result


def tfidf_keywords(grouped: dict[int, list[str]], top_n: int = 8) -> dict[int, list[str]]:
    """Palavras-chave distintivas por comunidade via TF-IDF."""
    keywords: dict[int, list[str]] = {}
    for comm_id, texts in grouped.items():
        if not texts:
            keywords[comm_id] = []
            continue
        vectorizer = CountVectorizer(
            max_features=200,
            stop_words=list(STOPWORDS_PT),
            token_pattern=r"(?u)\b\w{3,}\b",
        )
        try:
            matrix = vectorizer.fit_transform(texts)
            scores = matrix.sum(axis=0).A1
            terms = vectorizer.get_feature_names_out()
            ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
            keywords[comm_id] = [t for t, _ in ranked[:top_n]]
        except ValueError:
            keywords[comm_id] = []
    return keywords


def plot_wordcloud(
    grouped: dict[int, list[str]],
    comm_id: int,
    output_dir: str = "output",
) -> str | None:
    """Gera nuvem de palavras para uma comunidade."""
    import os

    texts = grouped.get(comm_id, [])
    if not texts:
        return None

    os.makedirs(output_dir, exist_ok=True)
    text = " ".join(texts)
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=STOPWORDS_PT,
        colormap="viridis",
    ).generate(text)

    path = f"{output_dir}/wordcloud_comunidade_{comm_id}.png"
    wc.to_file(path)
    return path
