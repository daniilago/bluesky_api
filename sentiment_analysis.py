"""Análise de sentimento por post e por comunidade (Atividade 3 — Text Mining).

Usa VADER (Valence Aware Dictionary and sEntiment Reasoner), adequado para
textos curtos de redes sociais. Como ~75% dos posts coletados estão em inglês,
VADER oferece boa cobertura; posts em português são analisados com menor precisão,
mas o sinal geral por comunidade permanece interpretável.

Escala de compound: [-1, -0.05) negativo | [-0.05, 0.05] neutro | (0.05, 1] positivo
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def _label(compound: float) -> str:
    if compound >= 0.05:
        return "positivo"
    if compound <= -0.05:
        return "negativo"
    return "neutro"


def analyze_posts(data: dict, communities: dict[str, int]) -> pd.DataFrame:
    """Retorna DataFrame com sentimento por post."""
    rows = []
    for post in data["posts"]:
        sc = _analyzer.polarity_scores(post["text"])
        rows.append(
            {
                "autor": post["author"],
                "comunidade": communities.get(post["author"], -1),
                "compound": sc["compound"],
                "pos": sc["pos"],
                "neu": sc["neu"],
                "neg": sc["neg"],
                "rotulo": _label(sc["compound"]),
                "likes": post.get("likes", 0),
                "texto": post["text"][:80],
            }
        )
    return pd.DataFrame(rows)


def community_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas de sentimento agregadas por comunidade."""
    rows = []
    for comm_id, grp in df.groupby("comunidade"):
        compound_vals = grp["compound"]
        counts = grp["rotulo"].value_counts()
        rows.append(
            {
                "comunidade": int(comm_id),
                "n_posts": len(grp),
                "media_compound": round(compound_vals.mean(), 3),
                "mediana_compound": round(compound_vals.median(), 3),
                "desvio_pad": round(compound_vals.std(), 3),
                "n_positivos": int(counts.get("positivo", 0)),
                "n_neutros": int(counts.get("neutro", 0)),
                "n_negativos": int(counts.get("negativo", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("comunidade").reset_index(drop=True)


def plot_sentiment_distribution(
    df: pd.DataFrame,
    output_path: str = "output/sentimento_distribuicao.png",
) -> str:
    """Box-plot de compound por comunidade + histograma geral."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box-plot por comunidade
    communities_sorted = sorted(df["comunidade"].unique())
    data_by_comm = [df[df["comunidade"] == c]["compound"].values for c in communities_sorted]
    bp = axes[0].boxplot(data_by_comm, tick_labels=[f"Com. {c}" for c in communities_sorted], patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities_sorted)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Distribuição de sentimento por comunidade")
    axes[0].set_ylabel("Compound VADER")
    axes[0].set_xlabel("Comunidade")

    # Histograma geral com distribuição dos rótulos
    label_colors = {"positivo": "#2ecc71", "neutro": "#95a5a6", "negativo": "#e74c3c"}
    for rotulo, color in label_colors.items():
        subset = df[df["rotulo"] == rotulo]["compound"]
        axes[1].hist(subset, bins=15, alpha=0.7, label=rotulo, color=color)
    axes[1].set_title("Distribuição geral de sentimento (compound)")
    axes[1].set_xlabel("Compound VADER")
    axes[1].set_ylabel("Frequência")
    axes[1].legend()
    axes[1].axvline(0, color="black", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_sentiment_stacked(
    summary: pd.DataFrame,
    output_path: str = "output/sentimento_empilhado.png",
) -> str:
    """Gráfico de barras empilhadas: proporção pos/neu/neg por comunidade."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    comms = summary["comunidade"].astype(str)
    totals = summary["n_posts"]
    pos_pct = summary["n_positivos"] / totals * 100
    neu_pct = summary["n_neutros"] / totals * 100
    neg_pct = summary["n_negativos"] / totals * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(comms))
    ax.bar(x, pos_pct, label="Positivo", color="#2ecc71")
    ax.bar(x, neu_pct, bottom=pos_pct, label="Neutro", color="#95a5a6")
    ax.bar(x, neg_pct, bottom=pos_pct + neu_pct, label="Negativo", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Com. {c}" for c in comms])
    ax.set_ylabel("Proporção (%)")
    ax.set_title("Composição de sentimento por comunidade", pad=30)
    ax.legend(loc="upper right")
    # anotação da média compound
    for i, (_, row) in enumerate(summary.iterrows()):
        ax.text(i, 103, f"μ={row['media_compound']:+.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
