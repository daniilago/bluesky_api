"""Geração do relatório PDF com resultados da análise."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH, _ = A4
USABLE_WIDTH = PAGE_WIDTH - 4 * cm


def _escape(text) -> str:
    s = str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return _escape(value)


def _table_from_df(df, styles, col_widths=None):
    """Tabela com quebra de linha automática e largura limitada à página."""
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        parent=cell_style,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )

    n_cols = len(df.columns)
    if col_widths is None:
        col_widths = [USABLE_WIDTH / n_cols] * n_cols

    header = [Paragraph(_escape(col), header_style) for col in df.columns]
    rows = [
        [Paragraph(_format_cell(val), cell_style) for val in row]
        for row in df.values.tolist()
    ]
    table = Table([header, *rows], colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2FA")]),
            ]
        )
    )
    return table


def _community_summary_table(df, styles):
    """Tabela de resumo com colunas proporcionais à página A4."""
    display_df = df.copy()
    if "densidade" in display_df.columns:
        display_df["densidade"] = display_df["densidade"].map(lambda x: f"{float(x):.3f}")
    if "usuarios" in display_df.columns:
        display_df = display_df.rename(columns={"usuarios": "exemplos_usuarios"})

    col_widths = [1.6 * cm, 2.2 * cm, 2.2 * cm, 2.4 * cm, USABLE_WIDTH - 8.4 * cm]
    return _table_from_df(display_df, styles, col_widths=col_widths)


def _metrics_table(df, styles):
    col_widths = [USABLE_WIDTH - 4.8 * cm, 2.4 * cm, 2.4 * cm]
    return _table_from_df(df, styles, col_widths=col_widths)


def generate_report(
    data: dict,
    metrics_df,
    comm_summary_df,
    top_terms: dict,
    recommendations: list[dict],
    network_image: str,
    wordcloud_images: list[str],
    output_path: str = "output/relatorio_analise_redes.pdf",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=20)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=16, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)

    story = []

    story.append(Paragraph("Análise de Redes Sociais — Bluesky", title_style))
    story.append(
        Paragraph(
            f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>"
            f"<b>Termo de busca:</b> {data.get('query', 'N/A')}<br/>"
            f"<b>Posts coletados:</b> {len(data['posts'])}<br/>"
            f"<b>Usuários na amostra:</b> {len(data.get('authors', []))}<br/>"
            f"<b>Relações de follow:</b> {len(data.get('follows', []))}",
            body,
        )
    )

    # 1. Metodologia
    story.append(Paragraph("1. Metodologia", h2))
    story.append(
        Paragraph(
            "Esta análise segue a metodologia de análise de redes sociais em quatro etapas: "
            "<b>(1) Coleta de dados</b> via API do Bluesky (AT Protocol), buscando posts por "
            "termo e relações de follow entre autores; <b>(2) Construção da rede</b>, onde nós "
            "representam usuários e arestas representam follows e co-participação temática; "
            "<b>(3) Detecção de comunidades</b> pelo algoritmo de Louvain, que identifica grupos "
            "densamente conectados; <b>(4) Análise complementar</b> com text mining por comunidade "
            "e recomendação de usuários/posts similares dentro do mesmo grupo.",
            body,
        )
    )

    # 2. Rede e comunidades
    story.append(Paragraph("2. Rede e Comunidades Detectadas", h2))
    if os.path.exists(network_image):
        story.append(Image(network_image, width=16 * cm, height=10 * cm))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Resumo por comunidade:", body))
    story.append(_community_summary_table(comm_summary_df, styles))

    # 3. Métricas
    story.append(PageBreak())
    story.append(Paragraph("3. Métricas dos Usuários", h2))
    story.append(_metrics_table(metrics_df.head(20), styles))

    # 4. Text mining
    story.append(Paragraph("4. Análise de Texto por Comunidade", h2))
    story.append(
        Paragraph(
            "Comparação do vocabulário utilizado em cada comunidade — conectando com a "
            "metodologia de text mining da atividade anterior.",
            body,
        )
    )
    for comm_id, terms in sorted(top_terms.items()):
        term_str = ", ".join(f"{t} ({c})" for t, c in terms)
        story.append(Paragraph(f"<b>Comunidade {comm_id}:</b> {term_str}", body))

    for wc_path in wordcloud_images:
        if wc_path and os.path.exists(wc_path):
            story.append(Spacer(1, 0.3 * cm))
            story.append(Image(wc_path, width=14 * cm, height=7 * cm))

    # 5. Recomendações
    story.append(PageBreak())
    story.append(Paragraph("5. Recomendações Baseadas em Comunidades", h2))
    story.append(
        Paragraph(
            "Usuários similares são identificados por similaridade de cosseno (TF-IDF) "
            "entre perfis textuais, restritos à mesma comunidade. Posts recomendados são "
            "aqueles com maior engajamento de usuários similares.",
            body,
        )
    )
    for rec in recommendations:
        story.append(
            Paragraph(
                f"<b>Usuário alvo:</b> {rec['usuario']} (Comunidade {rec['comunidade']})",
                body,
            )
        )
        if rec["usuarios_similares"]:
            sims = ", ".join(f"{u} ({s:.2f})" for u, s in rec["usuarios_similares"])
            story.append(Paragraph(f"Usuários similares: {sims}", body))
        if rec["posts"]:
            story.append(Paragraph("Posts recomendados:", body))
            for p in rec["posts"]:
                story.append(
                    Paragraph(
                        f"• [{p['autor']}] {p['texto']} (likes: {p['likes']})",
                        body,
                    )
                )
        story.append(Spacer(1, 0.3 * cm))

    # 6. Conclusão
    story.append(Paragraph("6. Considerações Finais", h2))
    story.append(
        Paragraph(
            "A detecção de comunidades revelou grupos de usuários com padrões distintos de "
            "conexão e vocabulário. A integração entre análise de rede e text mining permite "
            "comparar o comportamento discursivo de cada grupo. O sistema de recomendação "
            "demonstra como informações de comunidade podem direcionar sugestões de conteúdo "
            "e conexões a usuários com interesses similares.",
            body,
        )
    )

    doc.build(story)
    return output_path
