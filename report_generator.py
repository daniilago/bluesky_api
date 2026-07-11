"""Geração do relatório PDF com resultados da análise integrada."""

from __future__ import annotations

import os
from datetime import datetime

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
    cell_style = ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_LEFT,
    )
    header_style = ParagraphStyle(
        "TableHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold",
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
    table.setStyle(TableStyle([
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
    ]))
    return table


def _simple_table(rows_data, styles, col_widths=None):
    cell_style = ParagraphStyle(
        "TableCell2", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_LEFT,
    )
    header_style = ParagraphStyle(
        "TableHeader2", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold",
    )
    header = [Paragraph(_escape(str(c)), header_style) for c in rows_data[0]]
    rows = [
        [Paragraph(_escape(str(v)), cell_style) for v in row]
        for row in rows_data[1:]
    ]
    n_cols = len(rows_data[0])
    if col_widths is None:
        col_widths = [USABLE_WIDTH / n_cols] * n_cols
    table = Table([header, *rows], colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
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
    ]))
    return table


def generate_report(
    data: dict,
    metrics_df,
    comm_summary_df,
    top_terms: dict,
    recommendations: list[dict],
    network_image: str,
    wordcloud_images: list[str],
    basic_props: dict | None = None,
    centrality_df=None,
    sentiment_summary_df=None,
    imgs: dict | None = None,
    output_path: str = "output/relatorio_analise_redes.pdf",
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imgs = imgs or {}

    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)

    story = []

    # Capa 
    story.append(Paragraph("Análise de Redes Sociais — Bluesky", title_style))
    queries_str = ", ".join(data.get("queries", [data.get("query", "N/A")]))
    story.append(Paragraph(
        f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>"
        f"<b>Queries:</b> {_escape(queries_str)}<br/>"
        f"<b>Posts coletados:</b> {len(data['posts'])}<br/>"
        f"<b>Usuários na amostra:</b> {len(data.get('authors', []))}<br/>"
        f"<b>Relações de follow:</b> {len(data.get('follows', []))}",
        body,
    ))

    # 1. Metodologia 
    story.append(Paragraph("1. Metodologia", h2))
    story.append(Paragraph(
        "Pipeline integrado com quatro etapas: "
        "<b>(1) Coleta</b> via API do Bluesky (AT Protocol) com suporte a múltiplas queries; "
        "<b>(2) Construção da rede</b> com arestas de follow e co-participação temática (Jaccard); "
        "<b>(3) Detecção de comunidades</b> pelo algoritmo de Louvain (fallback: K-Means textual); "
        "<b>(4) Análises derivadas:</b> centralidade, text mining, sentimento e recomendação.",
        body,
    ))

    # 2. Propriedades da Rede 
    story.append(Paragraph("2. Propriedades da Rede", h2))

    if basic_props:
        story.append(Paragraph("Estatísticas básicas:", h3))
        props_rows = [
            ["Propriedade", "Valor"],
            ["Número de vértices", str(basic_props.get("n_vertices", "—"))],
            ["Número de arestas", str(basic_props.get("n_arestas", "—"))],
            ["Densidade", f"{basic_props.get('densidade', 0):.4f}"],
            ["Componentes conexas", str(basic_props.get("n_componentes", "—"))],
            ["Tamanho da maior componente", str(basic_props.get("tamanho_maior_componente", "—"))],
            ["Grau médio", f"{basic_props.get('grau_medio', 0):.3f}"],
            ["Coef. clustering médio", f"{basic_props.get('coef_clustering_medio', 0):.4f}"],
            ["Diâmetro (maior comp.)", str(basic_props.get("diametro_lcc", "—"))],
            ["Caminho médio (maior comp.)", str(basic_props.get("caminho_medio_lcc", "—"))],
        ]
        story.append(_simple_table(props_rows, styles,
                                   col_widths=[USABLE_WIDTH * 0.65, USABLE_WIDTH * 0.35]))
        story.append(Spacer(1, 0.3 * cm))

    deg_img = imgs.get("degree_dist")
    if deg_img and os.path.exists(deg_img):
        story.append(Paragraph("Distribuição de graus:", h3))
        story.append(Image(deg_img, width=16 * cm, height=7 * cm))
        story.append(Spacer(1, 0.3 * cm))

    if centrality_df is not None and len(centrality_df) > 0:
        story.append(Paragraph("Centralidade dos vértices (top 15):", h3))
        top15 = centrality_df.head(15)[
            ["usuario", "grau", "centralidade_grau", "centralidade_eigenvector", "coef_clustering"]
        ].copy()
        col_w = [USABLE_WIDTH*0.42, USABLE_WIDTH*0.1,
                 USABLE_WIDTH*0.16, USABLE_WIDTH*0.16, USABLE_WIDTH*0.16]
        story.append(_table_from_df(top15, styles, col_widths=col_w))
        story.append(Spacer(1, 0.3 * cm))

    cent_img = imgs.get("centrality")
    if cent_img and os.path.exists(cent_img):
        story.append(Paragraph("Rede com tamanho proporcional à centralidade eigenvector:", h3))
        story.append(Image(cent_img, width=16 * cm, height=10 * cm))

    # 3. Comunidades 
    story.append(PageBreak())
    story.append(Paragraph("3. Rede e Comunidades Detectadas", h2))

    if os.path.exists(network_image):
        story.append(Image(network_image, width=16 * cm, height=10 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Resumo por comunidade:", h3))
    display_df = comm_summary_df.copy()
    if "densidade" in display_df.columns:
        display_df["densidade"] = display_df["densidade"].map(lambda x: f"{float(x):.3f}")
    if "usuarios" in display_df.columns:
        display_df = display_df.rename(columns={"usuarios": "exemplos_usuarios"})
    col_w = [1.6*cm, 2.0*cm, 2.0*cm, 2.2*cm, USABLE_WIDTH - 7.8*cm]
    story.append(_table_from_df(display_df, styles, col_widths=col_w))

    # 4. Text Mining 
    story.append(PageBreak())
    story.append(Paragraph("4. Text Mining por Comunidade", h2))
    story.append(Paragraph(
        "Termos mais frequentes por comunidade após remoção de stopwords e do termo de busca.",
        body,
    ))
    for comm_id, terms in sorted(top_terms.items()):
        term_str = ", ".join(f"{t} ({c})" for t, c in terms)
        story.append(Paragraph(f"<b>Comunidade {comm_id}:</b> {_escape(term_str)}", body))

    for wc_path in wordcloud_images:
        if wc_path and os.path.exists(wc_path):
            story.append(Spacer(1, 0.3 * cm))
            story.append(Image(wc_path, width=14 * cm, height=7 * cm))

    # 5. Sentimento 
    story.append(PageBreak())
    story.append(Paragraph("5. Análise de Sentimento por Comunidade", h2))
    story.append(Paragraph(
        "Sentimento calculado com VADER (compound: −1 negativo → +1 positivo). "
        "Limiar: compound ≥ 0,05 = positivo; ≤ −0,05 = negativo; entre = neutro.",
        body,
    ))

    sent_dist = imgs.get("sent_dist")
    if sent_dist and os.path.exists(sent_dist):
        story.append(Image(sent_dist, width=16 * cm, height=7 * cm))
        story.append(Spacer(1, 0.3 * cm))

    sent_stack = imgs.get("sent_stacked")
    if sent_stack and os.path.exists(sent_stack):
        story.append(Image(sent_stack, width=14 * cm, height=7 * cm))
        story.append(Spacer(1, 0.3 * cm))

    if sentiment_summary_df is not None and len(sentiment_summary_df) > 0:
        story.append(Paragraph("Estatísticas por comunidade:", h3))
        sent_df = sentiment_summary_df.copy()
        for col in ["media_compound", "mediana_compound", "desvio_pad"]:
            if col in sent_df.columns:
                sent_df[col] = sent_df[col].map(lambda x: f"{x:+.3f}")
        col_w = [1.8*cm, 1.5*cm, 2.2*cm, 2.4*cm, 2.0*cm, 1.6*cm, 1.6*cm, 1.6*cm]
        story.append(_table_from_df(sent_df, styles, col_widths=col_w))

    # 6. Recomendações 
    story.append(PageBreak())
    story.append(Paragraph("6. Recomendações Baseadas em Comunidades", h2))
    story.append(Paragraph(
        "Usuários similares identificados por similaridade de cosseno (TF-IDF) "
        "entre perfis textuais, restritos à mesma comunidade. "
        "Posts recomendados são os de maior engajamento de usuários similares.",
        body,
    ))
    for rec in recommendations:
        story.append(Paragraph(
            f"<b>Usuário alvo:</b> {_escape(rec['usuario'])} (Comunidade {rec['comunidade']})",
            body,
        ))
        if rec["usuarios_similares"]:
            sims = ", ".join(f"{_escape(u)} ({s:.2f})" for u, s in rec["usuarios_similares"])
            story.append(Paragraph(f"Usuários similares: {sims}", body))
        if rec["posts"]:
            story.append(Paragraph("Posts recomendados:", body))
            for p in rec["posts"]:
                story.append(Paragraph(
                    f"• [{_escape(p['autor'])}] {_escape(p['texto'])} (likes: {p['likes']})",
                    body,
                ))
        story.append(Spacer(1, 0.3 * cm))

    # 7. Considerações Finais 
    story.append(Paragraph("7. Considerações Finais", h2))
    story.append(Paragraph(
        "A detecção de comunidades revelou grupos de usuários com padrões distintos de "
        "conexão e vocabulário. As métricas de centralidade identificam os nós mais "
        "influentes estruturalmente na rede. A análise de sentimento adiciona a dimensão "
        "emocional do discurso por comunidade, complementando o text mining. "
        "O sistema de recomendação demonstra como informações de comunidade podem "
        "direcionar sugestões de conteúdo a usuários com interesses similares.",
        body,
    ))

    doc.build(story)
    return output_path