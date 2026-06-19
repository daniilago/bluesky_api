# Análise de Comunidades no Bluesky

Atividade de **Análise de Redes Sociais** — detecção de comunidades e recomendação de conteúdo usando a API do Bluesky.

## O que faz

1. **Coleta** posts e relações de follow via API (AT Protocol)
2. **Constrói** uma rede de usuários (follows + co-participação temática)
3. **Detecta comunidades** com o algoritmo de Louvain
4. **Analisa textos** de cada comunidade (text mining)
5. **Recomenda** usuários e posts similares dentro da mesma comunidade
6. **Gera relatório PDF** com todos os resultados

## Instalação

```bash
# Com uv (recomendado)
uv sync

# Ou com pip
pip install -e .
```

## Configuração

A API do Bluesky exige autenticação. Crie uma **senha de app** em:
**Configurações → Segurança → Senhas de app**

Copie o exemplo e preencha:

```bash
cp .env.example .env
# Edite .env com seu handle e senha de app
```

## Uso

```bash
# Coleta do Bluesky
python main.py --query python

# Personalizar coleta
python main.py --query "machine learning" --max-posts 50 --max-users 20
```

## Saídas

Tudo é salvo em `output/`:

| Arquivo | Descrição |
|---------|-----------|
| `relatorio_analise_redes.pdf` | Relatório completo |
| `rede_comunidades.png` | Visualização da rede |
| `wordcloud_comunidade_*.png` | Nuvens de palavras por comunidade |

Dados coletados ficam em `data/collected_data.json`.

## Estrutura do código

```
main.py              → Orquestra o pipeline
collector.py         → Coleta de dados da API Bluesky
network_analysis.py  → Grafo + Louvain + métricas
text_mining.py       → Análise de texto por comunidade
recommendation.py    → Recomendação por similaridade
report_generator.py  → Geração do PDF
```

## Metodologia

A análise segue quatro etapas da metodologia de redes sociais:

- **Coleta**: API `app.bsky.feed.searchPosts` + `app.bsky.graph.getFollows`
- **Rede**: nós = usuários, arestas = follows e co-ocorrência temática
- **Comunidades**: algoritmo de Louvain (`python-louvain`)
- **Complemento**: TF-IDF + similaridade de cosseno para recomendação
