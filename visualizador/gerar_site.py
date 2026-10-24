"""
Gerador do Site / Dashboard de Benchmarking Científico do Traveling Thief Problem (TTP).

Recursos Universais:
- Suporte a QUALQUER arquivo CSV:
  * CSV de Grid Search da Queimada (agrupa por config_id, calcula métricas e exibe ranking).
  * CSV de Bateria Detalhada de Testes (agrupa por solver/modelo e consolida estatísticas).
  * CSV de Estatísticas Consolidadas.
  * CSV Genérico Livre (tabela dinâmica interativa).
- Botão "📂 Carregar CSV" na interface web + suporte a Arrastar e Soltar (Drag & Drop) de arquivos CSV.
- Argumento de linha de comando: python visualizador/gerar_site.py --csv <caminho_csv>.
- Heat Grid com pódios 🥇 Ouro, 🥈 Prata, 🥉 Bronze e gradiente azul cobalto.
- Tabela de Dados Brutos, Gráficos comparativos (Radar, Dispersão, Barras) e Triagem de Instâncias.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_raiz = Path(__file__).resolve().parent.parent
_pasta_results = _raiz / "results"
_pasta_visualizador = _raiz / "visualizador"


def ler_csv_linhas(caminho: Path) -> List[Dict[str, Any]]:
    if not caminho.exists() or caminho.stat().st_size == 0:
        return []
    with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
        primeira_linha = f.readline()
        delimitador = ";" if ";" in primeira_linha else ("\t" if "\t" in primeira_linha else ",")
        f.seek(0)
        return list(csv.DictReader(f, delimiter=delimitador))


def calcular_metricas_lista(valores: List[float]) -> Dict[str, float]:
    if not valores:
        return {"media": 0.0, "std": 0.0, "min": 0.0, "q1": 0.0, "mediana": 0.0, "q3": 0.0, "max": 0.0}
    vals = sorted(valores)
    n = len(vals)
    media = sum(vals) / n
    std = statistics.stdev(vals) if n > 1 else 0.0
    v_min = vals[0]
    v_max = vals[-1]
    if n >= 4:
        q = statistics.quantiles(vals, n=4, method="inclusive")
        q1, med, q3 = q[0], q[1], q[2]
    elif n == 1:
        q1 = med = q3 = vals[0]
    else:
        med = statistics.median(vals)
        q1, q3 = vals[0], vals[-1]
    return {"media": media, "std": std, "min": v_min, "q1": q1, "mediana": med, "q3": q3, "max": v_max}


def normalizar_metricas(dados_brutos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not dados_brutos:
        return []
    por_modelo = defaultdict(list)
    for row in dados_brutos:
        por_modelo[row.get("modelo", "TTP1")].append(row)

    colunas_max = ["media_G", "mediana_G", "min_G", "q1_G", "q3_G", "max_G"]
    colunas_min = ["std_G", "media_tempo_cpu_ms"]

    resultado_norm = []
    for mod, linhas in por_modelo.items():
        min_max = {}
        for col in colunas_max + colunas_min:
            vals = [float(r.get(col, 0.0)) for r in linhas if col in r]
            if vals:
                min_max[col] = (min(vals), max(vals))
            else:
                min_max[col] = (0.0, 1.0)

        for r in linhas:
            item_norm = {"modelo": mod, "solver": r.get("solver", "")}
            for col in colunas_max:
                v = float(r.get(col, 0.0))
                v_min, v_max = min_max[col]
                diff = v_max - v_min
                score = (v - v_min) / diff if diff > 1e-9 else 1.0
                item_norm[col] = f"{max(0.0, min(1.0, score)):.4f}"

            for col in colunas_min:
                v = float(r.get(col, 0.0))
                v_min, v_max = min_max[col]
                diff = v_max - v_min
                score = (v_max - v) / diff if diff > 1e-9 else 1.0
                item_norm[col] = f"{max(0.0, min(1.0, score)):.4f}"

            resultado_norm.append(item_norm)

    return resultado_norm


def processar_qualquer_csv_para_dataset(caminho_csv: Path) -> Tuple[str, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    linhas = ler_csv_linhas(caminho_csv)
    if not linhas:
        return caminho_csv.name, 0, [], []

    chaves_lower = {k.lower(): k for k in linhas[0].keys()}

    # CASO 1: CSV de Grid Search da Queimada
    if "config_id" in chaves_lower and any(k in chaves_lower for k in ("lucro_g", "lucro")):
        k_lucro = chaves_lower.get("lucro_g") or chaves_lower.get("lucro")
        k_cfg = chaves_lower["config_id"]
        k_inst = chaves_lower.get("instancia", "instancia")
        k_cpu = chaves_lower.get("tempo_cpu_ms") or chaves_lower.get("tempo_ms")

        configs_agrupadas = defaultdict(lambda: {"lucros": [], "cpus": [], "meta": {}})
        instancias_vistas = set()

        for r in linhas:
            try:
                c_id = int(r[k_cfg])
                g = float(r[k_lucro])
                inst = r.get(k_inst, "")
                if inst:
                    instancias_vistas.add(inst)
                configs_agrupadas[c_id]["lucros"].append(g)
                if k_cpu and r.get(k_cpu):
                    configs_agrupadas[c_id]["cpus"].append(float(r[k_cpu]))
                if not configs_agrupadas[c_id]["meta"]:
                    configs_agrupadas[c_id]["meta"] = r
            except (ValueError, KeyError):
                continue

        dados_brutos = []
        for c_id, d in configs_agrupadas.items():
            st = calcular_metricas_lista(d["lucros"])
            meta = d["meta"]
            metodo = meta.get("metodo_construcao", "queimada")
            buracos = "SIM" if str(meta.get("otimizar_buracos", "")).lower() in ("true", "1") else "NÃO"
            opt2 = "SIM" if str(meta.get("refinamento_rota", "")).lower() in ("true", "1") else "NÃO"
            ref = "SIM" if str(meta.get("refinamento", "")).lower() in ("true", "1") else "NÃO"
            afin = "SIM" if str(meta.get("afinidade_termica", "")).lower() in ("true", "1") else "NÃO"

            nome_solver = f"Config #{c_id:04d} ({metodo} | buraco:{buracos} | 2opt:{opt2} | afin:{afin} | ref:{ref})"
            cpu_med = (sum(d["cpus"]) / len(d["cpus"])) if d["cpus"] else 1.0

            dados_brutos.append({
                "modelo": "TTP1",
                "solver": nome_solver,
                "media_G": f"{st['media']:.4f}",
                "std_G": f"{st['std']:.4f}",
                "min_G": f"{st['min']:.4f}",
                "q1_G": f"{st['q1']:.4f}",
                "mediana_G": f"{st['mediana']:.4f}",
                "q3_G": f"{st['q3']:.4f}",
                "max_G": f"{st['max']:.4f}",
                "media_tempo_cpu_ms": f"{cpu_med:.2f}",
                "media_tempo_viagem": f"{float(meta.get('tempo_viagem', 0.0)):.4f}",
                "media_peso_acumulado": f"{float(meta.get('peso', 0.0)):.4f}",
                "media_pct_ocupacao_mochila": "50.0",
                "media_qtd_itens": f"{meta.get('qtd_itens', 0)}",
            })

        dados_norm = normalizar_metricas(dados_brutos)
        nome_dataset = f"Grid Search Queimada ({caminho_csv.name}) - {len(configs_agrupadas)} configs"
        return nome_dataset, len(instancias_vistas) or len(linhas), dados_brutos, dados_norm

    # CASO 2: CSV Detalhado de Bateria de Testes (run a run)
    if any(k in chaves_lower for k in ("arquivo_instancia", "instancia")) and any(k in chaves_lower for k in ("solucionador", "solver")) and any(k in chaves_lower for k in ("lucro_liquido_g", "lucro_g")):
        k_solv = chaves_lower.get("solucionador") or chaves_lower.get("solver")
        k_lucro = chaves_lower.get("lucro_liquido_g") or chaves_lower.get("lucro_g")
        k_mod = chaves_lower.get("modelo_ttp") or chaves_lower.get("modelo")
        k_cpu = chaves_lower.get("tempo_computacional_seg") or chaves_lower.get("tempo_cpu_ms")
        k_inst = chaves_lower.get("arquivo_instancia") or chaves_lower.get("instancia")

        agrupado = defaultdict(lambda: {"lucros": [], "cpus": []})
        instancias_set = set()

        for r in linhas:
            try:
                mod = r.get(k_mod, "TTP1") if k_mod else "TTP1"
                solv = r[k_solv]
                g = float(r[k_lucro])
                inst = r.get(k_inst, "")
                if inst:
                    instancias_set.add(inst)

                agrupado[(mod, solv)]["lucros"].append(g)
                if k_cpu and r.get(k_cpu):
                    val_cpu = float(r[k_cpu])
                    if "seg" in k_cpu.lower():
                        val_cpu *= 1000.0
                    agrupado[(mod, solv)]["cpus"].append(val_cpu)
            except (ValueError, KeyError):
                continue

        dados_brutos = []
        for (mod, solv), d in agrupado.items():
            st = calcular_metricas_lista(d["lucros"])
            cpu_med = (sum(d["cpus"]) / len(d["cpus"])) if d["cpus"] else 1.0
            dados_brutos.append({
                "modelo": mod,
                "solver": solv,
                "media_G": f"{st['media']:.4f}",
                "std_G": f"{st['std']:.4f}",
                "min_G": f"{st['min']:.4f}",
                "q1_G": f"{st['q1']:.4f}",
                "mediana_G": f"{st['mediana']:.4f}",
                "q3_G": f"{st['q3']:.4f}",
                "max_G": f"{st['max']:.4f}",
                "media_tempo_cpu_ms": f"{cpu_med:.2f}",
                "media_tempo_viagem": "0.0",
                "media_peso_acumulado": "0.0",
                "media_pct_ocupacao_mochila": "0.0",
                "media_qtd_itens": "0",
            })

        dados_norm = normalizar_metricas(dados_brutos)
        nome_dataset = f"Bateria Consolidada ({caminho_csv.name}) - {len(agrupado)} solvers"
        return nome_dataset, len(instancias_set) or len(linhas), dados_brutos, dados_norm

    # CASO 3: CSV já em formato de Estatísticas Consolidadas
    if any(k in chaves_lower for k in ("solver", "solucionador")) and any(k in chaves_lower for k in ("media_g", "media")):
        k_solv = chaves_lower.get("solver") or chaves_lower.get("solucionador")
        k_mod = chaves_lower.get("modelo") or chaves_lower.get("modelo_ttp")
        dados_brutos = []
        for r in linhas:
            item = dict(r)
            item["solver"] = r[k_solv]
            item["modelo"] = r.get(k_mod, "TTP1") if k_mod else "TTP1"
            dados_brutos.append(item)
        dados_norm = normalizar_metricas(dados_brutos)
        return f"Estatísticas ({caminho_csv.name})", len(linhas), dados_brutos, dados_norm

    return caminho_csv.name, len(linhas), [], []


def gerar_html_dashboard(
    modo: str = "equilibradas",
    csv_bruto_path: Optional[str] = None,
    csv_norm_path: Optional[str] = None,
    csv_filtro_path: Optional[str] = None,
    csv_qualquer_path: Optional[str] = None,
):
    _pasta_visualizador.mkdir(parents=True, exist_ok=True)

    # Dataset 1: Equilibradas
    arq_eq_bruto = _pasta_results / "comparativo_estatisticas_equilibradas.csv"
    arq_eq_norm = _pasta_results / "comparativo_estatisticas_equilibradas_normalizadas.csv"
    if not arq_eq_bruto.exists():
        arq_eq_bruto = _pasta_results / "comparativo_estatisticas_ttp1_ttp2.csv"
    if not arq_eq_norm.exists():
        arq_eq_norm = _pasta_results / "comparativo_estatisticas_normalizadas.csv"

    # Dataset 2: Todas as instâncias
    arq_todas_bruto = _pasta_results / "comparativo_estatisticas_ttp1_ttp2.csv"
    arq_todas_norm = _pasta_results / "comparativo_estatisticas_normalizadas.csv"

    # Dataset 3: Triagem econômica
    arq_filtro = _pasta_results / "triagem_alto_prejuizo.csv"
    if not arq_filtro.exists():
        arq_filtro = _pasta_results / "filtro_instancias_lucrativas.csv"

    if csv_bruto_path:
        arq_eq_bruto = Path(csv_bruto_path)
    if csv_norm_path:
        arq_eq_norm = Path(csv_norm_path)
    if csv_filtro_path:
        arq_filtro = Path(csv_filtro_path)

    dados_eq_bruto = ler_csv_linhas(arq_eq_bruto)
    dados_eq_norm = ler_csv_linhas(arq_eq_norm)
    dados_todas_bruto = ler_csv_linhas(arq_todas_bruto)
    dados_todas_norm = ler_csv_linhas(arq_todas_norm)
    dados_filtro = ler_csv_linhas(arq_filtro)

    datasets_dict: Dict[str, Any] = {
        "equilibradas": {
            "nome": "Instâncias Equilibradas (Receita/Custo ≥ 0.5) - 34 instâncias",
            "instancias": 34,
            "brutos": dados_eq_bruto,
            "norm": dados_eq_norm,
        },
        "todas": {
            "nome": "Todas as Instâncias da Bateria (550 instâncias)",
            "instancias": 550,
            "brutos": dados_todas_bruto,
            "norm": dados_todas_norm,
        },
    }

    # Se o usuário passou qualquer CSV pela CLI
    modo_selecionado = modo
    if csv_qualquer_path:
        p_csv = Path(csv_qualquer_path)
        nome_ds, n_inst, d_bruto, d_norm = processar_qualquer_csv_para_dataset(p_csv)
        if d_bruto:
            datasets_dict["custom"] = {
                "nome": nome_ds,
                "instancias": n_inst,
                "brutos": d_bruto,
                "norm": d_norm,
            }
            modo_selecionado = "custom"
            print(f"📊 CSV customizado incorporado com sucesso como dataset principal:\n   {nome_ds}")

    json_datasets = json.dumps(datasets_dict, ensure_ascii=False)
    json_filtro = json.dumps(dados_filtro, ensure_ascii=False)

    opt_custom = ""
    if "custom" in datasets_dict:
        opt_custom = f'<option value="custom" selected>{datasets_dict["custom"]["nome"]}</option>'

    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Benchmark TTP - Painel Analítico Universal</title>
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
      max-width: 1540px;
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

    select, button, .btn-label {{
      font-size: 0.82rem;
      border: 1px solid var(--border-dark);
      background: var(--surface);
      color: var(--text);
      border-radius: var(--radius);
      padding: 6px 12px;
      outline: none;
      cursor: pointer;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    select:focus, button:focus {{
      border-color: var(--primary);
    }}

    button:hover, .btn-label:hover {{
      background: var(--table-header);
    }}

    .btn-primary {{
      background: var(--primary);
      color: #ffffff;
      border: none;
    }}
    .btn-primary:hover {{
      background: var(--primary-hover);
    }}

    .btn-upload {{
      background: #0284c7;
      color: #ffffff;
      border: none;
    }}
    .btn-upload:hover {{
      background: #0369a1;
    }}

    /* Notificação / Alerta */
    #bannerNotificacao {{
      display: none;
      padding: 10px 14px;
      border-radius: var(--radius);
      font-size: 0.84rem;
      margin-bottom: 12px;
      font-weight: 600;
      border: 1px solid #7dd3fc;
      background: #f0f9ff;
      color: #0369a1;
      justify-content: space-between;
      align-items: center;
    }}
    body.dark #bannerNotificacao {{
      background: #0c4a6e;
      color: #e0f2fe;
      border-color: #0284c7;
    }}

    /* Drop Overlay */
    #dropOverlay {{
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(30, 58, 138, 0.88);
      z-index: 99999;
      justify-content: center;
      align-items: center;
      color: #ffffff;
      font-size: 1.6rem;
      font-weight: 700;
      pointer-events: none;
    }}

    /* KPIs */
    .kpi-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}

    .kpi-box {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 14px;
    }}

    .kpi-box-label {{
      font-size: 0.73rem;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
    }}

    .kpi-box-value {{
      font-size: 1.25rem;
      font-weight: 700;
      margin: 4px 0 2px 0;
      color: var(--text);
    }}

    .kpi-box-desc {{
      font-size: 0.75rem;
      color: var(--text-muted);
    }}

    /* Abas */
    .tabs-bar {{
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--border-dark);
      margin-bottom: 14px;
      overflow-x: auto;
    }}

    .tab-button {{
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      padding: 8px 14px;
      font-size: 0.84rem;
      font-weight: 600;
      color: var(--text-muted);
      cursor: pointer;
      white-space: nowrap;
    }}

    .tab-button:hover {{
      color: var(--text);
      background: transparent;
    }}

    .tab-button.active {{
      color: var(--primary);
      border-bottom-color: var(--primary);
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

    /* Ouro: Amarelo Dourado */
    .legend-box.gold {{
      background-color: #fef08a;
      border: 1px solid #ca8a04;
    }}
    body.dark .legend-box.gold {{
      background-color: #713f12;
      border: 1px solid #eab308;
    }}

    /* Prata: Cinza Slate */
    .legend-box.silver {{
      background-color: #e2e8f0;
      border: 1px solid #94a3b8;
    }}
    body.dark .legend-box.silver {{
      background-color: #334155;
      border: 1px solid #64748b;
    }}

    /* Bronze: Cobre Terracota */
    .legend-box.bronze {{
      background-color: #fed7aa;
      border: 1px solid #c2410c;
    }}
    body.dark .legend-box.bronze {{
      background-color: #7c2d12;
      border: 1px solid #ea580c;
    }}

    .legend-scale {{
      width: 60px;
      height: 10px;
      border-radius: 2px;
      background: linear-gradient(to right, rgba(30, 64, 175, 0.08), rgba(30, 64, 175, 0.75));
      display: inline-block;
      border: 1px solid var(--border);
    }}
    body.dark .legend-scale {{
      background: linear-gradient(to right, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.75));
    }}

    /* Tabelas */
    .table-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow-x: auto;
      margin-bottom: 20px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.81rem;
      text-align: left;
    }}

    th {{
      background-color: var(--table-header);
      color: var(--text);
      font-weight: 700;
      padding: 9px 11px;
      border-bottom: 1px solid var(--border-dark);
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }}

    th:hover {{
      background-color: var(--border);
    }}

    td {{
      padding: 7px 11px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}

    tr:hover td {{
      background-color: var(--table-hover);
    }}

    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SFMono-Regular", Consolas, monospace;
      font-size: 0.8rem;
    }}

    /* TOP 1 - OURO */
    td.cell-gold {{
      background-color: #fef08a !important;
      color: #854d0e !important;
      font-weight: 700 !important;
    }}
    body.dark td.cell-gold {{
      background-color: #713f12 !important;
      color: #fef08a !important;
      font-weight: 700 !important;
    }}

    /* TOP 2 - PRATA */
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

    /* TOP 3 - BRONZE */
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
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
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
  <!-- Overlay para Drag and Drop de qualquer CSV -->
  <div id="dropOverlay">📂 Solte seu arquivo CSV aqui para visualizá-lo instantaneamente!</div>

  <div class="container">
    <!-- Cabeçalho -->
    <header>
      <div class="header-info">
        <h1>Traveling Thief Problem (TTP) - Relatório Experimental Universal</h1>
        <p id="headerDesc">Benchmark analítico com suporte a carregamento dinâmico de qualquer arquivo CSV</p>
      </div>
      <div class="header-controls">
        <span class="control-label">Dataset:</span>
        <select id="datasetSelect" onchange="selecionarDataset(this.value)">
          <option value="equilibradas" {"selected" if modo_selecionado == "equilibradas" else ""}>Instâncias Equilibradas (Receita/Custo ≥ 0.5) - 34 instâncias</option>
          <option value="todas" {"selected" if modo_selecionado == "todas" else ""}>Todas as Instâncias (550 instâncias)</option>
          {opt_custom}
        </select>
        
        <!-- Botão Universal de Upload de CSV -->
        <label class="btn-label btn-upload" title="Carregar qualquer arquivo CSV do computador">
          <span>📂 Carregar CSV...</span>
          <input type="file" id="csvFileInput" accept=".csv,.txt" style="display:none;" onchange="processarCSVUpload(event)">
        </label>

        <button id="btnTheme" onclick="toggleTheme()">🌙 Modo Noturno</button>
        <button class="btn-primary" onclick="exportarTabelaAtivaCSV()">Exportar CSV</button>
      </div>
    </header>

    <!-- Banner de Notificação de Arquivo Carregado -->
    <div id="bannerNotificacao">
      <span id="bannerMsg">Arquivo CSV carregado com sucesso!</span>
      <span style="cursor:pointer;" onclick="fecharBanner()">✖</span>
    </div>

    <!-- Resumo dos Indicadores (KPIs) -->
    <div class="kpi-row">
      <div class="kpi-box">
        <div class="kpi-box-label">Solucionadores / Configs</div>
        <div class="kpi-box-value" id="kpiSolvers">30</div>
        <div class="kpi-box-desc">Algoritmos ou variantes avaliadas</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-box-label">Líder TTP1 (Maior Mediana G)</div>
        <div class="kpi-box-value" id="kpiLeaderTTP1" style="color: var(--primary);">cosolver</div>
        <div class="kpi-box-desc">Melhor resultado sob modelo de aluguel</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-box-label">Líder TTP2 (Maior Mediana G)</div>
        <div class="kpi-box-value" id="kpiLeaderTTP2" style="color: var(--primary);">queimada_adj</div>
        <div class="kpi-box-desc">Melhor resultado sob taxa temporal</div>
      </div>
      <div class="kpi-box">
        <div class="kpi-box-label">Instâncias Ativas</div>
        <div class="kpi-box-value" id="kpiInstancias">34</div>
        <div class="kpi-box-desc" id="kpiInstanciasDesc">Amostras avaliadas no dataset ativo</div>
      </div>
    </div>

    <!-- Navegação de Abas -->
    <div class="tabs-bar">
      <button class="tab-button active" onclick="trocarAba('tab-heatgrid', this)">Matriz de Desempenho (Heat Grid)</button>
      <button class="tab-button" onclick="trocarAba('tab-bruta', this)">Tabela de Dados Brutos</button>
      <button class="tab-button" onclick="trocarAba('tab-graficos', this)">Gráficos de Comparação</button>
      <button class="tab-button" onclick="trocarAba('tab-triagem', this)">Triagem das Instâncias</button>
      <button class="tab-button" id="tabBtnGenerica" style="display:none;" onclick="trocarAba('tab-generica', this)">CSV Livre Carregado</button>
    </div>

    <!-- ABA 1: HEAT GRID -->
    <div id="tab-heatgrid" class="tab-view active">
      <div class="toolbar">
        <div class="toolbar-left">
          <input type="text" id="buscaHeat" placeholder="Buscar algoritmo/configuração na matriz..." onkeyup="filtrarTabela('tableHeat', this.value)">
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
              <th onclick="ordenarTabela('tableHeat', 1)">Solucionador / Configuração</th>
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

    <!-- ABA 2: TABELA DE DADOS BRUTOS CONSOLIDADOS -->
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
              <th onclick="ordenarTabela('tableRaw', 0)">Solucionador / Configuração</th>
              <th onclick="ordenarTabela('tableRaw', 1)" style="text-align:right;">Média (G)</th>
              <th onclick="ordenarTabela('tableRaw', 2)" style="text-align:right;">Mediana (G)</th>
              <th onclick="ordenarTabela('tableRaw', 3)" style="text-align:right;">Desvio Padrão</th>
              <th onclick="ordenarTabela('tableRaw', 4)" style="text-align:right;">Mínimo (G)</th>
              <th onclick="ordenarTabela('tableRaw', 5)" style="text-align:right;">Q1 (25%)</th>
              <th onclick="ordenarTabela('tableRaw', 6)" style="text-align:right;">Q3 (75%)</th>
              <th onclick="ordenarTabela('tableRaw', 7)" style="text-align:right;">Máximo (G)</th>
              <th onclick="ordenarTabela('tableRaw', 8)" style="text-align:right;">Tempo CPU Médio (ms)</th>
              <th onclick="ordenarTabela('tableRaw', 9)" style="text-align:right;">Tempo Viagem</th>
              <th onclick="ordenarTabela('tableRaw', 10)" style="text-align:right;">Peso Mochila (kg)</th>
              <th onclick="ordenarTabela('tableRaw', 11)" style="text-align:right;">% Ocupação</th>
              <th onclick="ordenarTabela('tableRaw', 12)" style="text-align:right;">Itens Coletados</th>
            </tr>
          </thead>
          <tbody id="tbodyRaw"></tbody>
        </table>
      </div>
    </div>

    <!-- ABA 3: GRÁFICOS -->
    <div id="tab-graficos" class="tab-view">
      <div class="toolbar">
        <div class="toolbar-left">
          <div class="model-toggles">
            <button class="model-btn active" id="btnGrafTTP1" onclick="definirModeloGraf('TTP1')">TTP1</button>
            <button class="model-btn" id="btnGrafTTP2" onclick="definirModeloGraf('TTP2')">TTP2</button>
          </div>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          Visualização gráfica do trade-off entre rentabilidade e esforço computacional
        </div>
      </div>

      <div class="charts-grid">
        <div class="chart-box">
          <h3>Top 10 Solucionadores - Mediana de Lucro Líquido (G)</h3>
          <div class="chart-canvas-wrap">
            <canvas id="chartBarLucro"></canvas>
          </div>
        </div>
        <div class="chart-box">
          <h3>Trade-off: Mediana de Lucro vs Tempo de CPU (ms)</h3>
          <div class="chart-canvas-wrap">
            <canvas id="chartScatter"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- ABA 4: TRIAGEM DE INSTÂNCIAS -->
    <div id="tab-triagem" class="tab-view">
      <div class="toolbar">
        <div class="toolbar-left">
          <input type="text" id="buscaTriagem" placeholder="Buscar por arquivo de instância..." onkeyup="filtrarTabela('tableTriagem', this.value)">
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          Diagnóstico teórico: Razão entre Receita Total da Mochila e Custo Mínimo de Viagem
        </div>
      </div>

      <div class="table-card">
        <table id="tableTriagem">
          <thead>
            <tr>
              <th onclick="ordenarTabela('tableTriagem', 0)">Arquivo de Instância</th>
              <th onclick="ordenarTabela('tableTriagem', 1)" style="text-align:right;">Cidades</th>
              <th onclick="ordenarTabela('tableTriagem', 2)" style="text-align:right;">Itens</th>
              <th onclick="ordenarTabela('tableTriagem', 3)" style="text-align:right;">Taxa Aluguel (R)</th>
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

    <!-- ABA 5: CSV LIVRE / GENÉRICO -->
    <div id="tab-generica" class="tab-view">
      <div class="toolbar">
        <div class="toolbar-left">
          <input type="text" id="buscaGenerica" placeholder="Buscar em todas as colunas..." onkeyup="filtrarTabelaGenerica(this.value)">
          <span id="infoGenerica" style="font-size:0.8rem; color:var(--text-muted);"></span>
        </div>
        <button class="btn-primary" onclick="exportarTabelaAtivaCSV()">Exportar Este CSV</button>
      </div>
      <div class="table-card">
        <table id="tableGenerica">
          <thead id="theadGenerica"></thead>
          <tbody id="tbodyGenerica"></tbody>
        </table>
      </div>
    </div>

  </div>

  <script>
    const DATASETS = {json_datasets};
    const DADOS_TRIAGEM = {json_filtro};

    let datasetAtivo = '{modo_selecionado}';
    let DADOS_BRUTOS = DATASETS[datasetAtivo] ? DATASETS[datasetAtivo].brutos : [];
    let DADOS_NORM = DATASETS[datasetAtivo] ? DATASETS[datasetAtivo].norm : [];

    let modeloHeat = 'TTP1';
    let modeloRaw = 'TTP1';
    let modeloGraf = 'TTP1';

    let chartBarInstance = null;
    let chartScatterInstance = null;
    const sortDirections = {{}};

    // ==========================================
    // SISTEMA DE ARRASTAR E SOLTAR (DRAG & DROP)
    // ==========================================
    window.addEventListener('dragover', (e) => {{
      e.preventDefault();
      document.getElementById('dropOverlay').style.display = 'flex';
    }});

    window.addEventListener('dragleave', (e) => {{
      if (e.clientX === 0 && e.clientY === 0) {{
        document.getElementById('dropOverlay').style.display = 'none';
      }}
    }});

    window.addEventListener('drop', (e) => {{
      e.preventDefault();
      document.getElementById('dropOverlay').style.display = 'none';
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {{
        processarArquivoCSV(e.dataTransfer.files[0]);
      }}
    }});

    function processarCSVUpload(event) {{
      const file = event.target.files[0];
      if (file) {{
        processarArquivoCSV(file);
      }}
    }}

    function fecharBanner() {{
      document.getElementById('bannerNotificacao').style.display = 'none';
    }}

    function exibirBanner(msg, isSuccess = true) {{
      const b = document.getElementById('bannerNotificacao');
      document.getElementById('bannerMsg').innerText = msg;
      b.style.display = 'flex';
      b.style.borderColor = isSuccess ? '#7dd3fc' : '#fca5a5';
      b.style.background = isSuccess ? '#f0f9ff' : '#fef2f2';
      b.style.color = isSuccess ? '#0369a1' : '#991b1b';
    }}

    // ==========================================
    // PROCESSADOR UNIVERSAL DE CSV NO NAVEGADOR
    // ==========================================
    function processarArquivoCSV(file) {{
      const reader = new FileReader();
      reader.onload = function(e) {{
        const text = e.target.result;
        interpretarQualquerCSV(file.name, text);
      }};
      reader.readAsText(file, 'utf-8');
    }}

    function interpretarQualquerCSV(nomeArquivo, texto) {{
      const linhasRaw = texto.split(/\\r?\\n/).filter(l => l.trim().length > 0);
      if (linhasRaw.length < 2) return alert('O arquivo CSV está vazio ou incompleto.');

      const primeira = linhasRaw[0];
      const delim = primeira.includes(';') ? ';' : (primeira.includes('\\t') ? '\\t' : ',');

      function parseLine(line, delimiter) {{
        const res = [];
        let cur = '';
        let inQuotes = false;
        for (let i = 0; i < line.length; i++) {{
          const c = line[i];
          if (c === '"') {{
            inQuotes = !inQuotes;
          }} else if (c === delimiter && !inQuotes) {{
            res.push(cur.trim());
            cur = '';
          }} else {{
            cur += c;
          }}
        }}
        res.push(cur.trim());
        return res.map(v => v.replace(/^"|"$/g, '').replace(/""/g, '"').trim());
      }}

      const headers = parseLine(linhasRaw[0], delim);
      const headersLower = headers.map(h => h.toLowerCase());

      const registros = [];
      for (let i = 1; i < linhasRaw.length; i++) {{
        const cols = parseLine(linhasRaw[i], delim);
        if (cols.length === headers.length) {{
          const obj = {{}};
          headers.forEach((h, idx) => {{ obj[h] = cols[idx]; }});
          registros.push(obj);
        }}
      }}

      if (registros.length === 0) return alert('Nenhum registro legível encontrado.');

      // Detecção de Tipo
      const hasConfigId = headersLower.includes('config_id');
      const hasLucroG = headersLower.some(h => h === 'lucro_g' || h === 'lucro_liquido_g' || h === 'lucro');
      const hasSolver = headersLower.some(h => h === 'solver' || h === 'solucionador');
      const hasMediaG = headersLower.some(h => h === 'media_g' || h === 'media');

      // CASO A: Grid Search da Queimada
      if (hasConfigId && hasLucroG) {{
        const kLucro = headers[headersLower.findIndex(h => h === 'lucro_g' || h === 'lucro')];
        const kCfg = headers[headersLower.indexOf('config_id')];
        const kInst = headers[headersLower.findIndex(h => h === 'instancia')] || '';
        const kCpu = headers[headersLower.findIndex(h => h === 'tempo_cpu_ms' || h === 'tempo_ms')] || '';

        const agrupado = {{}};
        const instanciasSet = new Set();

        registros.forEach(r => {{
          const cid = parseInt(r[kCfg]);
          const g = parseFloat(r[kLucro]);
          if (isNaN(cid) || isNaN(g)) return;

          if (kInst && r[kInst]) instanciasSet.add(r[kInst]);
          if (!agrupado[cid]) {{
            agrupado[cid] = {{ lucros: [], cpus: [], meta: r }};
          }}
          agrupado[cid].lucros.push(g);
          if (kCpu && r[kCpu]) agrupado[cid].cpus.push(parseFloat(r[kCpu]));
        }});

        const brutos = [];
        Object.keys(agrupado).forEach(cid => {{
          const d = agrupado[cid];
          const st = calcularEstatisticasArray(d.lucros);
          const meta = d.meta;
          const met = meta.metodo_construcao || 'queimada';
          const bur = (meta.otimizar_buracos === 'True' || meta.otimizar_buracos === '1') ? 'SIM' : 'NÃO';
          const r2 = (meta.refinamento_rota === 'True' || meta.refinamento_rota === '1') ? 'SIM' : 'NÃO';
          const ref = (meta.refinamento === 'True' || meta.refinamento === '1') ? 'SIM' : 'NÃO';
          const afin = (meta.afinidade_termica === 'True' || meta.afinidade_termica === '1') ? 'SIM' : 'NÃO';

          const sNome = `Config #${{String(cid).padStart(4, '0')}} (${{met}} | buraco:${{bur}} | 2opt:${{r2}} | afin:${{afin}} | ref:${{ref}})`;
          const cpuMed = d.cpus.length > 0 ? (d.cpus.reduce((a,b)=>a+b,0)/d.cpus.length) : 1.0;

          brutos.push({{
            modelo: 'TTP1',
            solver: sNome,
            media_G: st.media.toFixed(4),
            std_G: st.std.toFixed(4),
            min_G: st.min.toFixed(4),
            q1_G: st.q1.toFixed(4),
            mediana_G: st.mediana.toFixed(4),
            q3_G: st.q3.toFixed(4),
            max_G: st.max.toFixed(4),
            media_tempo_cpu_ms: cpuMed.toFixed(2),
            media_tempo_viagem: parseFloat(meta.tempo_viagem || 0).toFixed(4),
            media_peso_acumulado: parseFloat(meta.peso || 0).toFixed(4),
            media_pct_ocupacao_mochila: '50.0',
            media_qtd_itens: meta.qtd_itens || '0'
          }});
        }});

        const norm = normalizarArray(brutos);
        const dsKey = 'upload_' + Date.now();
        DATASETS[dsKey] = {{
          nome: `Grid Search (${{nomeArquivo}}) - ${{brutos.length}} configs`,
          instancias: instanciasSet.size || registros.length,
          brutos: brutos,
          norm: norm
        }};

        adicionarOpcaoDataset(dsKey, DATASETS[dsKey].nome);
        selecionarDataset(dsKey);
        trocarAba('tab-heatgrid', document.querySelector('.tab-button'));
        exibirBanner(`✨ Sucesso! CSV de Grid Search (${{registros.length.toLocaleString()}} execuções consolidadas em ${{brutos.length}} configurações em ${{instanciasSet.size}} instâncias) carregado no dashboard!`);
        return;
      }}

      // CASO B: Bateria Detalhada
      if (hasSolver && hasLucroG && !hasMediaG) {{
        const kSolv = headers[headersLower.findIndex(h => h === 'solucionador' || h === 'solver')];
        const kLucro = headers[headersLower.findIndex(h => h === 'lucro_liquido_g' || h === 'lucro_g')];
        const kMod = headers[headersLower.findIndex(h => h === 'modelo_ttp' || h === 'modelo')] || '';
        const kCpu = headers[headersLower.findIndex(h => h === 'tempo_computacional_seg' || h === 'tempo_cpu_ms')] || '';
        const kInst = headers[headersLower.findIndex(h => h === 'arquivo_instancia' || h === 'instancia')] || '';

        const agrupado = {{}};
        const instSet = new Set();

        registros.forEach(r => {{
          const mod = kMod ? (r[kMod] || 'TTP1') : 'TTP1';
          const solv = r[kSolv];
          const g = parseFloat(r[kLucro]);
          if (!solv || isNaN(g)) return;

          if (kInst && r[kInst]) instSet.add(r[kInst]);
          const key = `${{mod}}||${{solv}}`;
          if (!agrupado[key]) agrupado[key] = {{ mod, solv, lucros: [], cpus: [] }};
          agrupado[key].lucros.push(g);
          if (kCpu && r[kCpu]) {{
            let c = parseFloat(r[kCpu]);
            if (kCpu.toLowerCase().includes('seg')) c *= 1000.0;
            agrupado[key].cpus.push(c);
          }}
        }});

        const brutos = [];
        Object.values(agrupado).forEach(d => {{
          const st = calcularEstatisticasArray(d.lucros);
          const cpuMed = d.cpus.length > 0 ? (d.cpus.reduce((a,b)=>a+b,0)/d.cpus.length) : 1.0;
          brutos.push({{
            modelo: d.mod,
            solver: d.solv,
            media_G: st.media.toFixed(4),
            std_G: st.std.toFixed(4),
            min_G: st.min.toFixed(4),
            q1_G: st.q1.toFixed(4),
            mediana_G: st.mediana.toFixed(4),
            q3_G: st.q3.toFixed(4),
            max_G: st.max.toFixed(4),
            media_tempo_cpu_ms: cpuMed.toFixed(2),
            media_tempo_viagem: '0.0',
            media_peso_acumulado: '0.0',
            media_pct_ocupacao_mochila: '0.0',
            media_qtd_itens: '0'
          }});
        }});

        const norm = normalizarArray(brutos);
        const dsKey = 'upload_' + Date.now();
        DATASETS[dsKey] = {{
          nome: `Bateria (${{nomeArquivo}}) - ${{Object.keys(agrupado).length}} solvers`,
          instancias: instSet.size || registros.length,
          brutos: brutos,
          norm: norm
        }};

        adicionarOpcaoDataset(dsKey, DATASETS[dsKey].nome);
        selecionarDataset(dsKey);
        trocarAba('tab-heatgrid', document.querySelector('.tab-button'));
        exibirBanner(`✨ Bateria de testes (${{registros.length.toLocaleString()}} soluções) consolidada com sucesso!`);
        return;
      }}

      // CASO C: Estatísticas Consolidadas
      if (hasSolver && hasMediaG) {{
        const kSolv = headers[headersLower.findIndex(h => h === 'solver' || h === 'solucionador')];
        const kMod = headers[headersLower.findIndex(h => h === 'modelo' || h === 'modelo_ttp')] || '';
        const brutos = registros.map(r => ({{
          ...r,
          solver: r[kSolv],
          modelo: kMod ? (r[kMod] || 'TTP1') : 'TTP1'
        }}));

        const norm = normalizarArray(brutos);
        const dsKey = 'upload_' + Date.now();
        DATASETS[dsKey] = {{
          nome: `Estatísticas (${{nomeArquivo}})`,
          instancias: registros.length,
          brutos: brutos,
          norm: norm
        }};

        adicionarOpcaoDataset(dsKey, DATASETS[dsKey].nome);
        selecionarDataset(dsKey);
        trocarAba('tab-heatgrid', document.querySelector('.tab-button'));
        exibirBanner(`✨ Tabela de estatísticas com ${{brutos.length}} solucionadores carregada com sucesso!`);
        return;
      }}

      // CASO D: CSV Genérico / Livre
      renderTabelaGenericaCSV(nomeArquivo, headers, registros);
    }}

    function adicionarOpcaoDataset(val, texto) {{
      const sel = document.getElementById('datasetSelect');
      const opt = document.createElement('option');
      opt.value = val;
      opt.innerText = texto;
      opt.selected = true;
      sel.appendChild(opt);
    }}

    function calcularEstatisticasArray(vals) {{
      if (!vals || vals.length === 0) return {{ media:0, std:0, min:0, q1:0, mediana:0, q3:0, max:0 }};
      const arr = [...vals].sort((a,b)=>a-b);
      const n = arr.length;
      const media = arr.reduce((a,b)=>a+b,0)/n;
      let std = 0;
      if (n > 1) {{
        const v = arr.reduce((acc,x)=>acc + Math.pow(x - media, 2), 0)/(n - 1);
        std = Math.sqrt(v);
      }}
      const min = arr[0];
      const max = arr[n - 1];
      let q1, med, q3;
      if (n === 1) {{
        q1 = med = q3 = arr[0];
      }} else {{
        med = n % 2 === 1 ? arr[Math.floor(n/2)] : (arr[n/2 - 1] + arr[n/2])/2;
        q1 = arr[Math.floor(n * 0.25)];
        q3 = arr[Math.floor(n * 0.75)];
      }}
      return {{ media, std, min, q1, mediana: med, q3, max }};
    }}

    function normalizarArray(linhas) {{
      const porMod = {{}};
      linhas.forEach(r => {{
        const m = r.modelo || 'TTP1';
        if (!porMod[m]) porMod[m] = [];
        porMod[m].push(r);
      }});

      const colMax = ['media_G', 'mediana_G', 'min_G', 'q1_G', 'q3_G', 'max_G'];
      const colMin = ['std_G', 'media_tempo_cpu_ms'];
      const res = [];

      Object.keys(porMod).forEach(m => {{
        const arr = porMod[m];
        const minMax = {{}};
        [...colMax, ...colMin].forEach(col => {{
          const vals = arr.map(x => parseFloat(x[col] || 0));
          minMax[col] = {{ min: Math.min(...vals), max: Math.max(...vals) }};
        }});

        arr.forEach(r => {{
          const item = {{ modelo: m, solver: r.solver }};
          colMax.forEach(col => {{
            const v = parseFloat(r[col] || 0);
            const mm = minMax[col];
            const diff = mm.max - mm.min;
            item[col] = (diff > 1e-9 ? Math.max(0, Math.min(1, (v - mm.min)/diff)) : 1.0).toFixed(4);
          }});
          colMin.forEach(col => {{
            const v = parseFloat(r[col] || 0);
            const mm = minMax[col];
            const diff = mm.max - mm.min;
            item[col] = (diff > 1e-9 ? Math.max(0, Math.min(1, (mm.max - v)/diff)) : 1.0).toFixed(4);
          }});
          res.push(item);
        }});
      }});
      return res;
    }}

    function renderTabelaGenericaCSV(nomeArquivo, headers, registros) {{
      const thead = document.getElementById('theadGenerica');
      const tbody = document.getElementById('tbodyGenerica');
      thead.innerHTML = '';
      tbody.innerHTML = '';

      const trHead = document.createElement('tr');
      headers.forEach((h, idx) => {{
        const th = document.createElement('th');
        th.innerText = h;
        th.onclick = () => ordenarTabela('tableGenerica', idx);
        trHead.appendChild(th);
      }});
      thead.appendChild(trHead);

      registros.forEach(r => {{
        const tr = document.createElement('tr');
        headers.forEach(h => {{
          const td = document.createElement('td');
          const val = r[h] || '';
          td.innerText = val;
          const num = parseFloat(val);
          if (!isNaN(num) && String(num) === val.trim()) {{
            td.classList.add('num');
            td.setAttribute('data-val', num);
          }}
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      document.getElementById('infoGenerica').innerText = `${{registros.length.toLocaleString()}} linhas | ${{headers.length}} colunas`;
      const btnGen = document.getElementById('tabBtnGenerica');
      btnGen.style.display = 'inline-block';
      btnGen.innerText = `CSV: ${{nomeArquivo}}`;
      trocarAba('tab-generica', btnGen);
      exibirBanner(`📂 CSV Livre "${{nomeArquivo}}" (${{registros.length.toLocaleString()}} linhas) carregado na aba dedicada!`);
    }}

    function filtrarTabelaGenerica(query) {{
      const q = query.toLowerCase();
      const rows = document.querySelectorAll('#tableGenerica tbody tr');
      rows.forEach(tr => {{
        tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
      }});
    }}

    // ==========================================
    // CONTROLE DE TEMA E DATASETS
    // ==========================================
    function toggleTheme() {{
      document.body.classList.toggle('dark');
      const isDark = document.body.classList.contains('dark');
      document.getElementById('btnTheme').innerText = isDark ? '☀️ Modo Claro' : '🌙 Modo Noturno';
      atualizarTudo();
    }}

    function selecionarDataset(chave) {{
      datasetAtivo = chave;
      if (!DATASETS[chave]) return;
      DADOS_BRUTOS = DATASETS[chave].brutos;
      DADOS_NORM = DATASETS[chave].norm;
      document.getElementById('kpiInstancias').innerText = DATASETS[chave].instancias;
      document.getElementById('kpiInstanciasDesc').innerText = DATASETS[chave].nome;
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
      document.getElementById('kpiSolvers').innerText = solversSet.size || '0';

      const ttp1Rows = DADOS_BRUTOS.filter(r => r.modelo === 'TTP1');
      if (ttp1Rows.length > 0) {{
        ttp1Rows.sort((a,b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));
        document.getElementById('kpiLeaderTTP1').innerText = ttp1Rows[0].solver;
      }} else {{
        document.getElementById('kpiLeaderTTP1').innerText = '-';
      }}

      const ttp2Rows = DADOS_BRUTOS.filter(r => r.modelo === 'TTP2');
      if (ttp2Rows.length > 0) {{
        ttp2Rows.sort((a,b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));
        document.getElementById('kpiLeaderTTP2').innerText = ttp2Rows[0].solver;
      }} else {{
        document.getElementById('kpiLeaderTTP2').innerText = '-';
      }}
    }}

    function trocarAba(tabId, btn) {{
      document.querySelectorAll('.tab-view').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-button').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      if (btn) btn.classList.add('active');
      if (tabId === 'tab-graficos') {{
        setTimeout(() => renderGraficos(modeloGraf), 50);
      }}
    }}

    // ==========================================
    // HEAT GRID COM PÓDIO E GRADIENTES
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
    // TABELA DE DADOS BRUTOS
    // ==========================================
    function renderTabelaRaw(modelo) {{
      const tbody = document.getElementById('tbodyRaw');
      tbody.innerHTML = '';
      const filtrados = DADOS_BRUTOS.filter(r => r.modelo === modelo);
      filtrados.sort((a, b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));

      filtrados.forEach((r) => {{
        const tr = document.createElement('tr');
        const gMed = parseFloat(r.media_G || 0);
        const gMedian = parseFloat(r.mediana_G || 0);
        const gStd = parseFloat(r.std_G || 0);
        const gQ1 = parseFloat(r.q1_G || 0);
        const gQ3 = parseFloat(r.q3_G || 0);
        const gMin = parseFloat(r.min_G || 0);
        const gMax = parseFloat(r.max_G || 0);
        const cpuTime = parseFloat(r.media_tempo_cpu_ms || 0);

        tr.innerHTML = `
          <td><b>${{r.solver}}</b></td>
          <td class="num" data-val="${{gMed}}">${{gMed.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gMedian}}"><b>${{gMedian.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</b></td>
          <td class="num" data-val="${{gStd}}">${{gStd.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gMin}}">${{gMin.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gQ1}}">${{gQ1.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gQ3}}">${{gQ3.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{gMax}}">${{gMax.toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{cpuTime}}">${{cpuTime.toLocaleString('pt-BR', {{maximumFractionDigits: 2}})}}</td>
          <td class="num" data-val="${{parseFloat(r.media_tempo_viagem || 0)}}">${{parseFloat(r.media_tempo_viagem || 0).toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{parseFloat(r.media_peso_acumulado || 0)}}">${{parseFloat(r.media_peso_acumulado || 0).toLocaleString('pt-BR', {{maximumFractionDigits: 1}})}}</td>
          <td class="num" data-val="${{parseFloat(r.media_pct_ocupacao_mochila || 0)}}">${{parseFloat(r.media_pct_ocupacao_mochila || 0).toFixed(1)}}%</td>
          <td class="num" data-val="${{parseFloat(r.media_qtd_itens || 0)}}">${{parseFloat(r.media_qtd_itens || 0).toFixed(1)}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // ==========================================
    // TABELA DE TRIAGEM TEÓRICA
    // ==========================================
    function renderTabelaTriagem() {{
      const tbody = document.getElementById('tbodyTriagem');
      tbody.innerHTML = '';

      DADOS_TRIAGEM.forEach((r) => {{
        const tr = document.createElement('tr');
        const razao = parseFloat(r.razao_rec_custo);
        const isPrejuizo = r.classificacao.toLowerCase().includes('preju');
        const badgeClass = isPrejuizo ? 'badge-prejuizo' : 'badge-ok';

        tr.innerHTML = `
          <td><b>${{r.instancia}}</b></td>
          <td class="num" data-val="${{r.n_cidades}}">${{r.n_cidades}}</td>
          <td class="num" data-val="${{r.n_itens}}">${{r.n_itens}}</td>
          <td class="num" data-val="${{parseFloat(r.taxa_aluguel)}}">${{parseFloat(r.taxa_aluguel).toFixed(4)}}</td>
          <td class="num" data-val="${{parseFloat(r.receita_mochila)}}">${{parseFloat(r.receita_mochila).toLocaleString('pt-BR', {{maximumFractionDigits: 0}})}}</td>
          <td class="num" data-val="${{parseFloat(r.custo_minimo_aluguel)}}">${{parseFloat(r.custo_minimo_aluguel).toLocaleString('pt-BR', {{maximumFractionDigits: 0}})}}</td>
          <td class="num" data-val="${{razao}}"><b>${{razao.toFixed(4)}}</b></td>
          <td style="text-align: center;"><span class="badge-status ${{badgeClass}}">${{r.classificacao}}</span></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    // ==========================================
    // GRÁFICOS (CHART.JS)
    // ==========================================
    function renderGraficos(modelo) {{
      const isDark = document.body.classList.contains('dark');
      const textColor = isDark ? '#f9fafb' : '#0f172a';
      const mutedColor = isDark ? '#94a3b8' : '#64748b';
      const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

      const filtrados = DADOS_BRUTOS.filter(r => r.modelo === modelo);
      filtrados.sort((a, b) => parseFloat(b.mediana_G) - parseFloat(a.mediana_G));
      const top10 = filtrados.slice(0, 10);

      // Gráfico de Barras
      const ctxBar = document.getElementById('chartBarLucro').getContext('2d');
      if (chartBarInstance) chartBarInstance.destroy();

      chartBarInstance = new Chart(ctxBar, {{
        type: 'bar',
        data: {{
          labels: top10.map(r => r.solver.length > 28 ? r.solver.substring(0, 26) + '...' : r.solver),
          datasets: [{{
            label: 'Mediana G',
            data: top10.map(r => parseFloat(r.mediana_G)),
            backgroundColor: isDark ? 'rgba(59, 130, 246, 0.85)' : 'rgba(30, 58, 138, 0.85)',
            borderColor: isDark ? '#60a5fa' : '#1e3a8a',
            borderWidth: 1,
            borderRadius: 3
          }}]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              callbacks: {{
                label: function(ctx) {{
                  return `Mediana: ${{ctx.raw.toLocaleString('pt-BR')}} pts`;
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              grid: {{ display: false }},
              ticks: {{ color: mutedColor, maxRotation: 40, font: {{ size: 10 }} }}
            }},
            y: {{
              grid: {{ color: gridColor }},
              ticks: {{ color: mutedColor, font: {{ size: 10 }} }}
            }}
          }}
        }}
      }});

      // Gráfico de Dispersão
      const ctxScatter = document.getElementById('chartScatter').getContext('2d');
      if (chartScatterInstance) chartScatterInstance.destroy();

      const scatterData = filtrados.map(r => ({{
        x: Math.max(0.01, parseFloat(r.media_tempo_cpu_ms)),
        y: parseFloat(r.mediana_G),
        solver: r.solver
      }}));

      chartScatterInstance = new Chart(ctxScatter, {{
        type: 'scatter',
        data: {{
          datasets: [{{
            label: 'Solucionadores (CPU vs Lucro G)',
            data: scatterData,
            backgroundColor: isDark ? '#38bdf8' : '#0284c7',
            borderColor: isDark ? '#0284c7' : '#0369a1',
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
    // ORDENAÇÃO E FILTROS ROBUSTOS
    // ==========================================
    function extrairValorParaOrdenacao(cell) {{
      if (cell.hasAttribute('data-val')) {{
        const n = parseFloat(cell.getAttribute('data-val'));
        if (!isNaN(n)) return n;
      }}
      let txt = cell.innerText.replace(/[🥇🥈🥉\\s]/g, '').trim();
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

    // Inicialização da Página
    window.addEventListener('DOMContentLoaded', () => {{
      renderTabelaTriagem();
      atualizarTudo();
    }});
  </script>
</body>
</html>
"""
    caminho_index = _pasta_visualizador / "index.html"
    with open(caminho_index, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ Dashboard web gerado com sucesso em:\n   {caminho_index}")
    print(f"   Dataset Ativo: {modo_selecionado.upper()}")
    return caminho_index


def main():
    parser = argparse.ArgumentParser(
        description="Gerador do Dashboard Web Interativo Universal do TTP."
    )
    parser.add_argument(
        "--modo",
        choices=["equilibradas", "todas"],
        default="equilibradas",
        help="Modo padrão de exibição do site: 'equilibradas' ou 'todas'",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Caminho de QUALQUER arquivo CSV (Grid Search, Bateria de Testes, ou Estatísticas) para carregar no dashboard",
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
        csv_qualquer_path=args.csv,
    )


if __name__ == "__main__":
    main()
