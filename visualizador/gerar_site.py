"""
Gerador do Site / Dashboard de Benchmarking Científico do Traveling Thief Problem (TTP).

Características:
- Design sóbrio, acadêmico e focado em dados (Data-First).
- Suporte a Modo Claro e Modo Noturno (com bordas suaves e sem contraste excessivo).
- Heat Grid (Matriz de Desempenho) com:
  - Cores especiais e inconfundíveis para Top 1 (Ouro 🥇 - Amarelo Ouro), Top 2 (Prata 🥈 - Cinza Prata) e Top 3 (Bronze 🥉 - Cobre Terracota).
  - Gradiente em azul cobalto técnico exclusivamente para os quadrantes do 4º lugar em diante.
- Aba dedicada e separada para a Tabela de Dados Brutos.
- Ordenação robusta por valor numérico exato (via atributo data-val), imune a formatações pt-BR com vírgula e ponto.
- Suporte a múltiplos datasets:
  - 'equilibradas' (PADRÃO): Instâncias com Receita Máx / Custo Mín >= 0.5 (34 instâncias).
  - 'todas': Todas as 550 instâncias (incluindo R extremo).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_raiz = Path(__file__).resolve().parent.parent
_pasta_results = _raiz / "results"
_pasta_visualizador = _raiz / "visualizador"


def ler_csv(caminho: Path):
    if not caminho.exists():
        return []
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        primeira_linha = f.readline()
        delimitador = ";" if ";" in primeira_linha else ","
        f.seek(0)
        return list(csv.DictReader(f, delimiter=delimitador))


def gerar_html_dashboard(
    modo: str = "equilibradas",
    csv_bruto_path: str = None,
    csv_norm_path: str = None,
    csv_filtro_path: str = None,
):
    _pasta_visualizador.mkdir(parents=True, exist_ok=True)

    # Dataset de instâncias equilibradas
    arq_eq_bruto = _pasta_results / "comparativo_estatisticas_equilibradas.csv"
    arq_eq_norm = _pasta_results / "comparativo_estatisticas_equilibradas_normalizadas.csv"
    if not arq_eq_bruto.exists():
        arq_eq_bruto = _pasta_results / "comparativo_estatisticas_ttp1_ttp2.csv"
    if not arq_eq_norm.exists():
        arq_eq_norm = _pasta_results / "comparativo_estatisticas_normalizadas.csv"

    # Dataset completo (todas)
    arq_todas_bruto = _pasta_results / "comparativo_estatisticas_ttp1_ttp2.csv"
    arq_todas_norm = _pasta_results / "comparativo_estatisticas_normalizadas.csv"

    # Triagem econômica
    arq_filtro = _pasta_results / "triagem_alto_prejuizo.csv"
    if not arq_filtro.exists():
        arq_filtro = _pasta_results / "filtro_instancias_lucrativas.csv"

    if csv_bruto_path:
        arq_eq_bruto = Path(csv_bruto_path)
    if csv_norm_path:
        arq_eq_norm = Path(csv_norm_path)
    if csv_filtro_path:
        arq_filtro = Path(csv_filtro_path)

    dados_eq_bruto = ler_csv(arq_eq_bruto)
    dados_eq_norm = ler_csv(arq_eq_norm)
    dados_todas_bruto = ler_csv(arq_todas_bruto)
    dados_todas_norm = ler_csv(arq_todas_norm)
    dados_filtro = ler_csv(arq_filtro)

    json_eq_bruto = json.dumps(dados_eq_bruto, ensure_ascii=False)
    json_eq_norm = json.dumps(dados_eq_norm, ensure_ascii=False)
    json_todas_bruto = json.dumps(dados_todas_bruto, ensure_ascii=False)
    json_todas_norm = json.dumps(dados_todas_norm, ensure_ascii=False)
    json_filtro = json.dumps(dados_filtro, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark TTP - Painel Analítico</title>
  <!-- Chart.js CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #1e3a8a;
      --primary-hover: #172554;
      --table-header: #f1f5f9;
      --table-hover: #f8fafc;
      --radius: 6px;
    }}

    body.dark {{
      --bg: #0b0f19;
      --surface: #111827;
      --border: rgba(255, 255, 255, 0.08);
      --border-dark: rgba(255, 255, 255, 0.12);
      --text: #f9fafb;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #60a5fa;
      --table-header: #161e2e;
      --table-hover: #1a2234;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}

    body {{
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.45;
      padding: 16px 20px;
      transition: background-color 0.15s, color 0.15s;
    }}

    .container {{
      max-width: 1480px;
      margin: 0 auto;
    }}

    /* Cabeçalho */
    header {{
      background: var(--surface);
      border: 1px solid var(--border-dark);
      border-radius: var(--radius);
      padding: 14px 18px;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}

    .header-info h1 {{
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.01em;
    }}

    .header-info p {{
      color: var(--text-muted);
      font-size: 0.82rem;
      margin-top: 2px;
    }}

    .header-controls {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}

    .control-label {{
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
    }}

    select, button {{
      font-size: 0.82rem;
      border: 1px solid var(--border-dark);
      background: var(--surface);
      color: var(--text);
      border-radius: var(--radius);
      padding: 6px 10px;
      outline: none;
    }}

    select:focus, button:focus {{
      border-color: var(--primary);
    }}

    button {{
      cursor: pointer;
      font-weight: 600;
    }}

    button:hover {{
      background: var(--table-header);
    }}

    button.btn-primary {{
      background: var(--primary);
      color: #ffffff;
      border-color: var(--primary);
    }}

    button.btn-primary:hover {{
      background: var(--primary-hover);
    }}

    /* Resumo Numérico (KPIs) */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}

    .kpi-box {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 10px 14px;
    }}

    .kpi-box-label {{
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    .kpi-box-value {{
      font-size: 1.25rem;
      font-weight: 700;
      margin: 3px 0 1px;
      color: var(--text);
      font-variant-numeric: tabular-nums;
    }}

    .kpi-box-desc {{
      font-size: 0.74rem;
      color: var(--text-muted);
    }}

    /* Navegação */
    .tabs-bar {{
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--border-dark);
      margin-bottom: 14px;
    }}

    .tab-button {{
      padding: 8px 14px;
      background: transparent;
      border: 1px solid transparent;
      border-bottom: 2px solid transparent;
      color: var(--text-muted);
      font-size: 0.85rem;
      font-weight: 600;
      border-radius: 0;
      margin-bottom: -1px;
    }}

    .tab-button:hover {{
      color: var(--text);
      background: transparent;
    }}

    .tab-button.active {{
      color: var(--primary);
      border-bottom: 2px solid var(--primary);
      background: transparent;
    }}

    .tab-view {{
      display: none;
    }}
    .tab-view.active {{
      display: block;
    }}

    /* Barra de Filtros Interna */
    .toolbar {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 8px 12px;
      margin-bottom: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .toolbar-left {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex: 1;
      min-width: 250px;
    }}

    .toolbar-left input {{
      width: 100%;
      max-width: 320px;
      padding: 6px 10px;
      border: 1px solid var(--border-dark);
      border-radius: var(--radius);
      font-size: 0.82rem;
      background: var(--surface);
      color: var(--text);
      outline: none;
    }}

    .toolbar-left input:focus {{
      border-color: var(--primary);
    }}

    .model-toggles {{
      display: flex;
      border: 1px solid var(--border-dark);
      border-radius: var(--radius);
      overflow: hidden;
    }}

    .model-btn {{
      border: none;
      border-radius: 0;
      padding: 6px 14px;
      font-size: 0.78rem;
      background: var(--surface);
      color: var(--text-muted);
      cursor: pointer;
    }}

    .model-btn.active {{
      background: var(--primary);
      color: #ffffff;
      font-weight: 700;
    }}

    /* Legenda da Heat Grid */
    .heat-legend {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 0.74rem;
      color: var(--text-muted);
      flex-wrap: wrap;
    }}

    .legend-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}

    .legend-box {{
      width: 13px;
      height: 13px;
      border-radius: 2px;
      display: inline-block;
    }}

    /* Ouro: Amarelo Dourado Vivo */
    .legend-box.gold {{
      background-color: #fef08a;
      border: 1px solid #ca8a04;
    }}
    body.dark .legend-box.gold {{
      background-color: #713f12;
      border: 1px solid #eab308;
    }}

    /* Prata: Cinza Metálico / Slate */
    .legend-box.silver {{
      background-color: #e2e8f0;
      border: 1px solid #94a3b8;
    }}
    body.dark .legend-box.silver {{
      background-color: #334155;
      border: 1px solid #64748b;
    }}

    /* Bronze: Cobre Terracota Avermelhado */
    .legend-box.bronze {{
      background-color: #fed7aa;
      border: 1px solid #ea580c;
    }}
    body.dark .legend-box.bronze {{
      background-color: #7c2d12;
      border: 1px solid #f97316;
    }}

    .legend-scale {{
      display: flex;
      height: 12px;
      width: 80px;
      border: 1px solid var(--border-dark);
      border-radius: 2px;
      background: linear-gradient(90deg, rgba(30, 64, 175, 0.05), rgba(30, 64, 175, 0.75));
    }}
    body.dark .legend-scale {{
      background: linear-gradient(90deg, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.75));
    }}

    /* Tabelas */
    .table-card {{
      background: var(--surface);
      border: 1px solid var(--border-dark);
      border-radius: var(--radius);
      overflow-x: auto;
      margin-bottom: 18px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 0.81rem;
    }}

    thead th {{
      background: var(--table-header);
      color: var(--text);
      font-weight: 700;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-dark);
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }}

    thead th:hover {{
      background: var(--border);
    }}

    tbody td {{
      padding: 6px 10px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}

    tbody tr:hover {{
      background: var(--table-hover);
    }}

    td.num {{
      text-align: right;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.8rem;
    }}

    /* ====================================================
       CORES ESPECIAIS PARA OS QUADRANTES TOP 1, 2 E 3
       Ouro (Amarelo) | Prata (Cinza) | Bronze (Cobre Avermelhado)
       Completamente distintos e sem bordas duras no escuro
       ==================================================== */

    /* TOP 1 - OURO: AMARELO DOURADO VIVO */
    td.cell-gold {{
      background-color: #fef08a !important;
      color: #713f12 !important;
      font-weight: 700 !important;
    }}
    body.dark td.cell-gold {{
      background-color: #713f12 !important;
      color: #fef08a !important;
      font-weight: 700 !important;
    }}

    /* TOP 2 - PRATA: SLATE METÁLICO NEUTRO */
    td.cell-silver {{
      background-color: #e2e8f0 !important;
      color: #1e293b !important;
      font-weight: 700 !important;
    }}
    body.dark td.cell-silver {{
      background-color: #334155 !important;
      color: #f8fafc !important;
      font-weight: 700 !important;
    }}

    /* TOP 3 - BRONZE: COBRE TERRACOTA AVERMELHADO */
    td.cell-bronze {{
      background-color: #fed7aa !important;
      color: #7c2d12 !important;
      font-weight: 700 !important;
    }}
    body.dark td.cell-bronze {{
      background-color: #7c2d12 !important;
      color: #ffedd5 !important;
      font-weight: 700 !important;
    }}

    .medal-icon {{
      font-size: 0.75rem;
      margin-left: 3px;
      vertical-align: middle;
    }}

    /* Gráficos */
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}

    .chart-box {{
      background: var(--surface);
      border: 1px solid var(--border-dark);
      border-radius: var(--radius);
      padding: 12px 14px;
    }}

    .chart-box h3 {{
      font-size: 0.88rem;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 10px;
    }}

    .chart-canvas-wrap {{
      position: relative;
      height: 360px;
      width: 100%;
    }}

    .badge-status {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 0.7rem;
      font-weight: 700;
    }}
    .badge-ok {{
      background: #dcfce7;
      color: #166534;
      border: 1px solid #86efac;
    }}
    body.dark .badge-ok {{
      background: #052e16;
      color: #86efac;
      border-color: #166534;
    }}
    .badge-prejuizo {{
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fca5a5;
    }}
    body.dark .badge-prejuizo {{
      background: #450a0a;
      color: #fca5a5;
      border-color: #991b1b;
    }}
  </style>
</head>
<body>
  <div class="container">
    <!-- Cabeçalho -->
    <header>
      <div class="header-info">
        <h1>Traveling Thief Problem (TTP) - Relatório Experimental</h1>
        <p id="headerDesc">Benchmark estatístico e comparativo analítico de algoritmos (TTP1 e TTP2)</p>
      </div>
      <div class="header-controls">
        <span class="control-label">Dataset:</span>
        <select id="datasetSelect" onchange="selecionarDataset(this.value)">
          <option value="equilibradas" {'selected' if modo == 'equilibradas' else ''}>Instâncias Equilibradas (Receita/Custo ≥ 0.5) - 34 instâncias</option>
          <option value="todas" {'selected' if modo == 'todas' else ''}>Todas as Instâncias (550 instâncias)</option>
        </select>
        <button id="btnTheme" onclick="toggleTheme()">🌙 Modo Noturno</button>
        <button class="btn-primary" onclick="exportarTabelaAtivaCSV()">Exportar CSV</button>
      </div>
    </header>

    <!-- Resumo dos Indicadores -->
    <div class="kpi-row">
      <div class="kpi-box">
        <div class="kpi-box-label">Solucionadores</div>
        <div class="kpi-box-value" id="kpiSolvers">30</div>
        <div class="kpi-box-desc">Algoritmos registrados avaliados</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-box-label">Líder TTP1 (Maior Mediana G)</div>
        <div class="kpi-box-value" id="kpiLeaderTTP1" style="color: var(--primary);">cosolver</div>
        <div class="kpi-box-desc">Melhor resultado no modelo de aluguel</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-box-label">Líder TTP2 (Maior Mediana G)</div>
        <div class="kpi-box-value" id="kpiLeaderTTP2" style="color: var(--primary);">queimada_adj</div>
        <div class="kpi-box-desc">Melhor resultado sob taxa temporal</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-box-label">Instâncias Ativas</div>
        <div class="kpi-box-value" id="kpiInstancias">34</div>
        <div class="kpi-box-desc" id="kpiInstanciasDesc">Filtradas por critério de equilíbrio</div>
      </div>
    </div>

    <!-- Navegação de Abas -->
    <div class="tabs-bar">
      <button class="tab-button active" onclick="trocarAba('tab-heatgrid', this)">Matriz de Desempenho (Heat Grid)</button>
      <button class="tab-button" onclick="trocarAba('tab-bruta', this)">Tabela de Dados Brutos</button>
      <button class="tab-button" onclick="trocarAba('tab-graficos', this)">Gráficos de Comparação</button>
      <button class="tab-button" onclick="trocarAba('tab-triagem', this)">Triagem das Instâncias</button>
    </div>

    <!-- ABA 1: HEAT GRID -->
    <div id="tab-heatgrid" class="tab-view active">
      <div class="toolbar">
        <div class="toolbar-left">
          <input type="text" id="buscaHeat" placeholder="Buscar algoritmo na matriz..." onkeyup="filtrarTabela('tableHeat', this.value)">
          <div class="model-toggles">
            <button class="model-btn active" id="btnHeatTTP1" onclick="definirModeloHeat('TTP1')">TTP1</button>
            <button class="model-btn" id="btnHeatTTP2" onclick="definirModeloHeat('TTP2')">TTP2</button>
          </div>
        </div>
        <div class="heat-legend">
          <div class="legend-item"><span class="legend-box gold"></span> <b>🥇 1º Ouro</b></div>
          <div class="legend-item"><span class="legend-box silver"></span> <b>🥈 2º Prata</b></div>
          <div class="legend-item"><span class="legend-box bronze"></span> <b>🥉 3º Bronze</b></div>
          <div class="legend-item">
            <span>Gradiente (4º em diante):</span>
            <span class="legend-scale"></span>
            <span>0.00 → 1.00</span>
          </div>
        </div>
      </div>

      <div class="table-card">
        <table id="tableHeat">
          <thead>
            <tr>
              <th onclick="ordenarTabela('tableHeat', 0)">#</th>
              <th onclick="ordenarTabela('tableHeat', 1)">Solucionador</th>
              <th onclick="ordenarTabela('tableHeat', 2)" style="text-align:right;">Média G</th>
              <th onclick="ordenarTabela('tableHeat', 3)" style="text-align:right;">Mediana G</th>
              <th onclick="ordenarTabela('tableHeat', 4)" style="text-align:right;">Estabilidade (Std)</th>
              <th onclick="ordenarTabela('tableHeat', 5)" style="text-align:right;">Mínimo G</th>
              <th onclick="ordenarTabela('tableHeat', 6)" style="text-align:right;">Q1 (25%)</th>
              <th onclick="ordenarTabela('tableHeat', 7)" style="text-align:right;">Q3 (75%)</th>
              <th onclick="ordenarTabela('tableHeat', 8)" style="text-align:right;">Máximo G</th>
              <th onclick="ordenarTabela('tableHeat', 9)" style="text-align:right;">Velocidade CPU</th>
            </tr>
          </thead>
          <tbody id="tbodyHeat"></tbody>
        </table>
      </div>
    </div>

    <!-- ABA 2: TABELA DE DADOS BRUTOS CONSOLIDADOS (EM SUA PRÓPRIA ABA) -->
    <div id="tab-bruta" class="tab-view">
      <div class="toolbar">
        <div class="toolbar-left">
          <input type="text" id="buscaRaw" placeholder="Buscar algoritmo na tabela..." onkeyup="filtrarTabela('tableRaw', this.value)">
          <div class="model-toggles">
            <button class="model-btn active" id="btnRawTTP1" onclick="definirModeloRaw('TTP1')">TTP1</button>
            <button class="model-btn" id="btnRawTTP2" onclick="definirModeloRaw('TTP2')">TTP2</button>
          </div>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          Valores absolutos: lucro líquido G, dispersão e tempos de computação
        </div>
      </div>

      <div class="table-card">
        <table id="tableRaw">
          <thead>
            <tr>
              <th onclick="ordenarTabela('tableRaw', 0)">Solucionador</th>
              <th onclick="ordenarTabela('tableRaw', 1)" style="text-align:right;">Média (G)</th>
              <th onclick="ordenarTabela('tableRaw', 2)" style="text-align:right;">Mediana (G)</th>
              <th onclick="ordenarTabela('tableRaw', 3)" style="text-align:right;">Desvio Padrão</th>
              <th onclick="ordenarTabela('tableRaw', 4)" style="text-align:right;">Q1 (25%)</th>
              <th onclick="ordenarTabela('tableRaw', 5)" style="text-align:right;">Q3 (75%)</th>
              <th onclick="ordenarTabela('tableRaw', 6)" style="text-align:right;">Mínimo</th>
              <th onclick="ordenarTabela('tableRaw', 7)" style="text-align:right;">Máximo</th>
              <th onclick="ordenarTabela('tableRaw', 8)" style="text-align:right;">Tempo CPU (ms)</th>
            </tr>
          </thead>
          <tbody id="tbodyRaw"></tbody>
        </table>
      </div>
    </div>

    <!-- ABA 3: GRÁFICOS ANALÍTICOS -->
    <div id="tab-graficos" class="tab-view">
      <div class="toolbar">
        <div class="model-toggles">
          <button class="model-btn active" id="btnGrafTTP1" onclick="definirModeloGraf('TTP1')">TTP1</button>
          <button class="model-btn" id="btnGrafTTP2" onclick="definirModeloGraf('TTP2')">TTP2</button>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          Dispersão e ranking comparativo
        </div>
      </div>

      <div class="charts-grid">
        <div class="chart-box">
          <h3>Ranking de Lucro Líquido (G): Mediana e Média</h3>
          <div class="chart-canvas-wrap">
            <canvas id="chartRanking"></canvas>
          </div>
        </div>
        <div class="chart-box">
          <h3>Trade-off: Tempo de Computação (ms) vs. Mediana de G</h3>
          <div class="chart-canvas-wrap">
            <canvas id="chartTradeoff"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- ABA 4: TRIAGEM ECONÔMICA -->
    <div id="tab-triagem" class="tab-view">
      <div class="toolbar">
        <div class="toolbar-left">
          <input type="text" id="buscaTriagem" placeholder="Buscar instância..." onkeyup="filtrarTabela('tableTriagem', this.value)">
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          Razão = Receita Mochila Fracionária / Custo Transporte Mínimo (MST / v_max * R)
        </div>
      </div>

      <div class="table-card">
        <table id="tableTriagem">
          <thead>
            <tr>
              <th onclick="ordenarTabela('tableTriagem', 0)">Instância</th>
              <th onclick="ordenarTabela('tableTriagem', 1)" style="text-align:right;">Cidades</th>
              <th onclick="ordenarTabela('tableTriagem', 2)" style="text-align:right;">Itens</th>
              <th onclick="ordenarTabela('tableTriagem', 3)" style="text-align:right;">Aluguel (R)</th>
              <th onclick="ordenarTabela('tableTriagem', 4)" style="text-align:right;">Receita Mochila</th>
              <th onclick="ordenarTabela('tableTriagem', 5)" style="text-align:right;">Custo Min Aluguel</th>
              <th onclick="ordenarTabela('tableTriagem', 6)" style="text-align:right;">Razão (Rec / Custo)</th>
              <th onclick="ordenarTabela('tableTriagem', 7)" style="text-align:center;">Classificação</th>
            </tr>
          </thead>
          <tbody id="tbodyTriagem"></tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    const DATASETS = {{
      'equilibradas': {{
        'nome': 'Instâncias Equilibradas (Receita/Custo ≥ 0.5)',
        'instancias': 34,
        'brutos': {json_eq_bruto},
        'norm': {json_eq_norm}
      }},
      'todas': {{
        'nome': 'Todas as Instâncias da Bateria (550 instâncias)',
        'instancias': 550,
        'brutos': {json_todas_bruto},
        'norm': {json_todas_norm}
      }}
    }};

    const DADOS_TRIAGEM = {json_filtro};

    let datasetAtivo = '{modo if modo in ["equilibradas", "todas"] else "equilibradas"}';
    let DADOS_BRUTOS = DATASETS[datasetAtivo].brutos;
    let DADOS_NORM = DATASETS[datasetAtivo].norm;

    let modeloHeat = 'TTP1';
    let modeloRaw = 'TTP1';
    let modeloGraf = 'TTP1';

    let chartRankInstance = null;
    let chartTradeoffInstance = null;
    let sortDirections = {{}};

    document.addEventListener('DOMContentLoaded', () => {{
      const savedTheme = localStorage.getItem('ttp_theme');
      if (savedTheme === 'dark') {{
        document.body.classList.add('dark');
        document.getElementById('btnTheme').innerText = '☀️ Modo Claro';
      }}

      atualizarTudo();
      renderTabelaTriagem();
    }});

    function toggleTheme() {{
      document.body.classList.toggle('dark');
      const isDark = document.body.classList.contains('dark');
      document.getElementById('btnTheme').innerText = isDark ? '☀️ Modo Claro' : '🌙 Modo Noturno';
      localStorage.setItem('ttp_theme', isDark ? 'dark' : 'light');
      renderHeatGrid(modeloHeat);
      renderGraficos(modeloGraf);
    }}

    function selecionarDataset(chave) {{
      if (!DATASETS[chave]) return;
      datasetAtivo = chave;
      DADOS_BRUTOS = DATASETS[chave].brutos;
      DADOS_NORM = DATASETS[chave].norm;

      document.getElementById('headerDesc').innerText = chave === 'equilibradas'
        ? 'Avaliação de benchmark em instâncias equilibradas (Receita/Custo ≥ 0.5) - 34 instâncias'
        : 'Avaliação de benchmark completa em todas as 550 instâncias (inclusive com R extremo)';
      document.getElementById('kpiInstancias').innerText = DATASETS[chave].instancias;
      document.getElementById('kpiInstanciasDesc').innerText = chave === 'equilibradas'
        ? 'Filtradas por critério de equilíbrio'
        : 'Total de instâncias da bateria';

      atualizarTudo();
    }}

    function definirModeloHeat(mod) {{
      modeloHeat = mod;
      document.getElementById('btnHeatTTP1').classList.toggle('active', mod === 'TTP1');
      document.getElementById('btnHeatTTP2').classList.toggle('active', mod === 'TTP2');
      renderHeatGrid(mod);
    }}

    function definirModeloRaw(mod) {{
      modeloRaw = mod;
      document.getElementById('btnRawTTP1').classList.toggle('active', mod === 'TTP1');
      document.getElementById('btnRawTTP2').classList.toggle('active', mod === 'TTP2');
      renderTabelaRaw(mod);
    }}

    function definirModeloGraf(mod) {{
      modeloGraf = mod;
      document.getElementById('btnGrafTTP1').classList.toggle('active', mod === 'TTP1');
      document.getElementById('btnGrafTTP2').classList.toggle('active', mod === 'TTP2');
      renderGraficos(mod);
    }}

    function atualizarTudo() {{
      atualizarKPIs();
      renderHeatGrid(modeloHeat);
      renderTabelaRaw(modeloRaw);
      renderGraficos(modeloGraf);
    }}

    function atualizarKPIs() {{
      const solversSet = new Set(DADOS_BRUTOS.map(r => r.solver));
      document.getElementById('kpiSolvers').innerText = solversSet.size || '30';

      const ttp1Rows = DADOS_BRUTOS.filter(r => r.modelo === 'TTP1');
      if (ttp1Rows.length > 0) {{
        ttp1Rows.sort((a,b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));
        document.getElementById('kpiLeaderTTP1').innerText = ttp1Rows[0].solver;
      }}

      const ttp2Rows = DADOS_BRUTOS.filter(r => r.modelo === 'TTP2');
      if (ttp2Rows.length > 0) {{
        ttp2Rows.sort((a,b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));
        document.getElementById('kpiLeaderTTP2').innerText = ttp2Rows[0].solver;
      }}
    }}

    function trocarAba(tabId, btn) {{
      document.querySelectorAll('.tab-view').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btn.classList.add('active');
      if (tabId === 'tab-graficos') {{
        setTimeout(() => renderGraficos(modeloGraf), 50);
      }}
    }}

    // ==========================================
    // HEAT GRID COM CORES ESPECIAIS PARA TOP 1-3
    // E GRADIENTE PARA OS DEMAIS
    // ==========================================
    const COLUNAS_HEAT = [
      'media_G', 'mediana_G', 'std_G', 'min_G', 'q1_G', 'q3_G', 'max_G', 'media_tempo_cpu_ms'
    ];

    function calcularTop3PorColuna(linhas) {{
      const top3 = {{}};
      COLUNAS_HEAT.forEach(col => {{
        const vals = Array.from(new Set(linhas.map(r => parseFloat(r[col])))).sort((a, b) => b - a);
        top3[col] = {{
          ouro: vals[0],
          prata: vals[1],
          bronze: vals[2]
        }};
      }});
      return top3;
    }}

    function renderHeatGrid(modelo) {{
      const tbody = document.getElementById('tbodyHeat');
      tbody.innerHTML = '';
      const filtrados = DADOS_NORM.filter(r => r.modelo === modelo);
      filtrados.sort((a, b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));

      const top3 = calcularTop3PorColuna(filtrados);
      const isDarkTheme = document.body.classList.contains('dark');

      filtrados.forEach((r, idx) => {{
        const tr = document.createElement('tr');
        let htmlCols = `
          <td data-val="${{idx + 1}}" style="color: var(--text-muted); font-size: 0.75rem;">${{idx + 1}}</td>
          <td><b>${{r.solver}}</b></td>
        `;

        COLUNAS_HEAT.forEach(col => {{
          const v = parseFloat(r[col] || 0.0);
          const t = top3[col];

          // QUADRANTES TOP 1, 2 E 3: CORES ESPECIAIS (NÃO FAZEM PARTE DO GRADIENTE)
          if (v === t.ouro) {{
            htmlCols += `
              <td class="num cell-gold" data-val="${{v}}">
                ${{v.toFixed(4)}}<span class="medal-icon">🥇</span>
              </td>
            `;
          }} else if (v === t.prata) {{
            htmlCols += `
              <td class="num cell-silver" data-val="${{v}}">
                ${{v.toFixed(4)}}<span class="medal-icon">🥈</span>
              </td>
            `;
          }} else if (v === t.bronze) {{
            htmlCols += `
              <td class="num cell-bronze" data-val="${{v}}">
                ${{v.toFixed(4)}}<span class="medal-icon">🥉</span>
              </td>
            `;
          }} else {{
            // DEMAIS QUADRANTES: GRADIENTE DA COR ESPECÍFICA (AZUL COBALTO SÓBRIO)
            const alpha = Math.max(0.04, Math.min(0.75, 0.04 + v * 0.70));
            const bg = isDarkTheme
              ? `rgba(59, 130, 246, ${{alpha.toFixed(2)}})`
              : `rgba(30, 64, 175, ${{alpha.toFixed(2)}})`;
            const textColor = (v >= 0.55 || (isDarkTheme && v >= 0.38)) ? '#ffffff' : (isDarkTheme ? '#e2e8f0' : '#0f172a');

            htmlCols += `
              <td class="num" data-val="${{v}}" style="background-color: ${{bg}}; color: ${{textColor}};">
                ${{v.toFixed(4)}}
              </td>
            `;
          }}
        }});

        tr.innerHTML = htmlCols;
        tbody.appendChild(tr);
      }});
    }}

    // ==========================================
    // TABELA DE DADOS BRUTOS (COM DATA-VAL EXATO)
    // ==========================================
    function renderTabelaRaw(modelo) {{
      const tbody = document.getElementById('tbodyRaw');
      tbody.innerHTML = '';
      const filtrados = DADOS_BRUTOS.filter(r => r.modelo === modelo);
      filtrados.sort((a, b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));

      filtrados.forEach((r) => {{
        const tr = document.createElement('tr');
        const gMed = parseFloat(r.media_G);
        const gMedian = parseFloat(r.mediana_G);
        const gStd = parseFloat(r.std_G);
        const gQ1 = parseFloat(r.q1_G);
        const gQ3 = parseFloat(r.q3_G);
        const gMin = parseFloat(r.min_G);
        const gMax = parseFloat(r.max_G);
        const cpuTime = parseFloat(r.media_tempo_cpu_ms);

        tr.innerHTML = `
          <td><b>${{r.solver}}</b></td>
          <td class="num" data-val="${{gMed}}">${{gMed.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gMedian}}"><b>${{gMedian.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</b></td>
          <td class="num" data-val="${{gStd}}">${{gStd.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gQ1}}">${{gQ1.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gQ3}}">${{gQ3.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gMin}}">${{gMin.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gMax}}">${{gMax.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{cpuTime}}">${{cpuTime.toFixed(2)}} ms</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // ==========================================
    // TABELA DE TRIAGEM
    // ==========================================
    function renderTabelaTriagem() {{
      const tbody = document.getElementById('tbodyTriagem');
      tbody.innerHTML = '';
      DADOS_TRIAGEM.forEach((r) => {{
        const tr = document.createElement('tr');
        const ratio = parseFloat(r.razao_receita_custo || 0.0);
        const cidades = parseInt(r.cidades);
        const itens = parseInt(r.itens);
        const aluguel = parseFloat(r.taxa_aluguel_R);
        const receita = parseFloat(r.valor_mochila_divisivel);
        const custo = parseFloat(r.custo_min_aluguel);

        const ehEquilibrada = (r.classificacao === 'EQUILIBRADA') || (r.potencialmente_lucrativa === 'SIM') || (ratio >= 0.5);
        const badgeClass = ehEquilibrada ? 'badge-ok' : 'badge-prejuizo';
        const labelStatus = ehEquilibrada ? 'EQUILIBRADA' : 'ALTO PREJUÍZO';

        tr.innerHTML = `
          <td><b>${{r.arquivo || r.arquivo_origem}}</b></td>
          <td class="num" data-val="${{cidades}}">${{cidades}}</td>
          <td class="num" data-val="${{itens}}">${{itens}}</td>
          <td class="num" data-val="${{aluguel}}">${{aluguel.toFixed(2)}}</td>
          <td class="num" data-val="${{receita}}">${{receita.toLocaleString('pt-BR', {{maximumFractionDigits: 0}})}}</td>
          <td class="num" data-val="${{custo}}">${{custo.toLocaleString('pt-BR', {{maximumFractionDigits: 0}})}}</td>
          <td class="num" data-val="${{ratio}}"><b>${{ratio.toFixed(3)}}</b></td>
          <td style="text-align:center;"><span class="badge-status ${{badgeClass}}">${{labelStatus}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // ==========================================
    // GRÁFICOS
    // ==========================================
    function renderGraficos(modelo) {{
      const ctxRank = document.getElementById('chartRanking').getContext('2d');
      const ctxTrade = document.getElementById('chartTradeoff').getContext('2d');

      const isDark = document.body.classList.contains('dark');
      const textColor = isDark ? '#f9fafb' : '#0f172a';
      const mutedColor = isDark ? '#94a3b8' : '#64748b';
      const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : '#f1f5f9';

      const dados = DADOS_BRUTOS.filter(r => r.modelo === modelo);
      dados.sort((a, b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));

      const labels = dados.map(r => r.solver);
      const medianas = dados.map(r => parseFloat(r.mediana_G));
      const medias = dados.map(r => parseFloat(r.media_G));

      if (chartRankInstance) chartRankInstance.destroy();
      chartRankInstance = new Chart(ctxRank, {{
        type: 'bar',
        data: {{
          labels: labels,
          datasets: [
            {{
              label: 'Mediana de G',
              data: medianas,
              backgroundColor: isDark ? '#3b82f6' : '#1e3a8a',
              borderRadius: 2,
            }},
            {{
              label: 'Média de G',
              data: medias,
              backgroundColor: isDark ? '#6b7280' : '#64748b',
              borderRadius: 2,
            }}
          ]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ position: 'top', labels: {{ color: textColor, font: {{ size: 11 }} }} }}
          }},
          scales: {{
            x: {{ grid: {{ color: gridColor }}, ticks: {{ color: mutedColor }} }},
            y: {{ grid: {{ display: false }}, ticks: {{ color: textColor, font: {{ size: 10 }} }} }}
          }}
        }}
      }});

      if (chartTradeoffInstance) chartTradeoffInstance.destroy();
      const scatterPoints = dados.map(r => ({{
        x: parseFloat(r.media_tempo_cpu_ms),
        y: parseFloat(r.mediana_G),
        solver: r.solver
      }}));

      chartTradeoffInstance = new Chart(ctxTrade, {{
        type: 'scatter',
        data: {{
          datasets: [{{
            label: 'Solucionadores (X: Tempo CPU ms, Y: Mediana G)',
            data: scatterPoints,
            backgroundColor: isDark ? '#60a5fa' : '#1e3a8a',
            borderColor: isDark ? '#93c5fd' : '#0f172a',
            pointRadius: 5,
            pointHoverRadius: 8
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            tooltip: {{
              callbacks: {{
                label: function(ctx) {{
                  const pt = ctx.raw;
                  return `${{pt.solver}}: ${{pt.y.toLocaleString('pt-BR')}} pts | ${{pt.x.toFixed(1)}} ms`;
                }}
              }}
            }},
            legend: {{ position: 'top', labels: {{ color: textColor }} }}
          }},
          scales: {{
            x: {{
              title: {{ display: true, text: 'Tempo Médio de CPU (ms)', color: mutedColor, font: {{ weight: 'bold' }} }},
              grid: {{ color: gridColor }},
              ticks: {{ color: mutedColor }}
            }},
            y: {{
              title: {{ display: true, text: 'Mediana do Lucro G', color: mutedColor, font: {{ weight: 'bold' }} }},
              grid: {{ color: gridColor }},
              ticks: {{ color: mutedColor }}
            }}
          }}
        }}
      }});
    }}

    // ==========================================
    // ORDENAÇÃO ROBUSTA BASEADA EM DATA-VAL EXATO
    // ==========================================
    function extrairValorParaOrdenacao(cell) {{
      if (cell.hasAttribute('data-val')) {{
        const n = parseFloat(cell.getAttribute('data-val'));
        if (!isNaN(n)) return n;
      }}
      let txt = cell.innerText.replace(/[🥇🥈🥉\\s]/g, '').trim();
      // Trata números formatados em pt-BR com vírgula e pontos
      if (txt.includes(',')) {{
        txt = txt.replace(/\\./g, '').replace(',', '.');
      }}
      let match = txt.match(/^-?\\d+(\\.\\d+)?/);
      if (match) {{
        let n = parseFloat(match[0]);
        if (!isNaN(n)) return n;
      }}
      return cell.innerText.trim().toLowerCase();
    }}

    function ordenarTabela(tableId, colIdx) {{
      const table = document.getElementById(tableId);
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));

      const sortKey = `${{tableId}}_${{colIdx}}`;
      sortDirections[sortKey] = !sortDirections[sortKey];
      const asc = sortDirections[sortKey];

      rows.sort((a, b) => {{
        const valA = extrairValorParaOrdenacao(a.cells[colIdx]);
        const valB = extrairValorParaOrdenacao(b.cells[colIdx]);

        if (typeof valA === 'number' && typeof valB === 'number') {{
          return asc ? valA - valB : valB - valA;
        }}
        return asc
          ? String(valA).localeCompare(String(valB))
          : String(valB).localeCompare(String(valA));
      }});

      // Atualiza indicadores visuais nos headers
      const ths = table.querySelectorAll('thead th');
      ths.forEach((th, idx) => {{
        th.innerText = th.innerText.replace(/ [▲▼]$/, '');
        if (idx === colIdx) {{
          th.innerText += asc ? ' ▲' : ' ▼';
        }}
      }});

      rows.forEach(r => tbody.appendChild(r));
    }}

    function filtrarTabela(tableId, query) {{
      const q = query.toLowerCase();
      const rows = document.querySelectorAll(`#${{tableId}} tbody tr`);
      rows.forEach(tr => {{
        tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}

    function exportarTabelaAtivaCSV() {{
      const activeTab = document.querySelector('.tab-view.active');
      const table = activeTab.querySelector('table');
      if (!table) return alert('Nenhuma tabela ativa na tela.');

      let csv = [];
      for (let row of table.rows) {{
        let cols = Array.from(row.cells).map(c => `"${{c.innerText.replace(/"/g, '""')}}"`);
        csv.push(cols.join(';'));
      }}
      const blob = new Blob([csv.join('\\n')], {{ type: 'text/csv;charset=utf-8;' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'export_ttp_benchmark.csv';
      link.click();
    }}
  </script>
</body>
</html>
"""
    caminho_index = _pasta_visualizador / "index.html"
    with open(caminho_index, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Dashboard web gerado com sucesso em:\n   {caminho_index}")
    print(f"   Modo Ativo Padrão: {modo.upper()}")
    return caminho_index


def main():
    parser = argparse.ArgumentParser(
        description="Gerador do Dashboard Web Interativo dos Resultados do TTP (Data-First)."
    )
    parser.add_argument(
        "--modo",
        choices=["equilibradas", "todas"],
        default="equilibradas",
        help="Modo padrão de exibição do site: 'equilibradas' (padrão: Receita/Custo >= 0.5) ou 'todas' (completo)",
    )
    parser.add_argument(
        "--csv-bruto",
        type=str,
        default=None,
        help="Caminho customizado do CSV de estatísticas brutas",
    )
    parser.add_argument(
        "--csv-norm",
        type=str,
        default=None,
        help="Caminho customizado do CSV de estatísticas normalizadas",
    )
    parser.add_argument(
        "--csv-filtro",
        type=str,
        default=None,
        help="Caminho customizado do CSV de triagem teórica",
    )

    args = parser.parse_args()
    gerar_html_dashboard(
        modo=args.modo,
        csv_bruto_path=args.csv_bruto,
        csv_norm_path=args.csv_norm,
        csv_filtro_path=args.csv_filtro,
    )


if __name__ == "__main__":
    main()
