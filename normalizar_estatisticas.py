"""
Módulo de Normalização de Estatísticas Comparativas do Traveling Thief Problem (TTP).

Lê o arquivo de estatísticas consolidadas (results/comparativo_estatisticas_ttp1_ttp2.csv)
e normaliza as métricas no intervalo [0, 1] para cada modelo (TTP1 e TTP2) separadamente:
- O melhor desempenho em cada métrica recebe exatamente 1.0000.
- Os demais recebem valores no formato 0.XXXX em função do melhor.

Critérios de Orientação:
- Métricas de Ganho (maior é melhor): media_G, min_G, q1_G, mediana_G, q3_G, max_G.
  Fórmula: norm(x) = (x - min) / (max - min)
- Métricas de Custo/Dispersão (menor é melhor): media_tempo_cpu_ms, std_G.
  Fórmula: norm(x) = (max - x) / (max - min)

Arquivo de saída padrão:
  results/comparativo_estatisticas_normalizadas.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Configura stdout para UTF-8 de forma segura no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_raiz = Path(__file__).resolve().parent

# Colunas a serem normalizadas
COLUNAS_MAXIMIZAR = ["media_G", "min_G", "q1_G", "mediana_G", "q3_G", "max_G"]
COLUNAS_MINIMIZAR = ["media_tempo_cpu_ms", "std_G"]
TODAS_COLUNAS_NORM = [
    "media_G", "std_G", "min_G", "q1_G", "mediana_G", "q3_G", "max_G", "media_tempo_cpu_ms"
]


def normalizar_grupo(
    linhas_modelo: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Normaliza as métricas de um conjunto de solucionadores pertencentes a um mesmo modelo.
    Garante que o melhor valor seja 1.0000 e os demais estejam em [0.0, 1.0].
    """
    if not linhas_modelo:
        return []

    # Extrai limites mínimo e máximo de cada coluna numérica
    limites: Dict[str, Tuple[float, float]] = {}
    for col in TODAS_COLUNAS_NORM:
        valores = [float(r[col]) for r in linhas_modelo if col in r and r[col] != ""]
        if valores:
            limites[col] = (min(valores), max(valores))
        else:
            limites[col] = (0.0, 1.0)

    linhas_normalizadas = []
    for r in linhas_modelo:
        registro_norm = {
            "modelo": r.get("modelo", ""),
            "solver": r.get("solver", ""),
        }

        # Métricas onde maior é melhor
        for col in COLUNAS_MAXIMIZAR:
            v_min, v_max = limites[col]
            v = float(r.get(col, 0.0))
            if v_max > v_min:
                score = (v - v_min) / (v_max - v_min)
            else:
                score = 1.0
            registro_norm[col] = round(score, 4)

        # Métricas onde menor é melhor (tempo de CPU e desvio padrão)
        for col in COLUNAS_MINIMIZAR:
            v_min, v_max = limites[col]
            v = float(r.get(col, 0.0))
            if v_max > v_min:
                score = (v_max - v) / (v_max - v_min)
            else:
                score = 1.0
            registro_norm[col] = round(score, 4)

        # Mantém colunas auxiliares caso existam
        for col_aux in ["media_tempo_viagem", "media_peso_acumulado", "media_pct_ocupacao_mochila", "media_qtd_itens"]:
            if col_aux in r:
                registro_norm[col_aux] = r[col_aux]

        linhas_normalizadas.append(registro_norm)

    # Ordena por mediana_G normalizada decrescente (ou media_G)
    linhas_normalizadas.sort(key=lambda item: float(item.get("mediana_G", 0.0)), reverse=True)
    return linhas_normalizadas


def normalizar_arquivo_estatisticas(
    caminho_entrada: Path,
    caminho_saida: Path,
) -> List[Dict[str, str]]:
    """
    Lê o CSV de entrada, divide por modelo, normaliza e grava no CSV de saída.
    """
    if not caminho_entrada.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {caminho_entrada}")

    linhas_por_modelo: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    with open(caminho_entrada, "r", encoding="utf-8") as f:
        # Detecta delimitador (; ou ,)
        primeira_linha = f.readline()
        delimitador = ";" if ";" in primeira_linha else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimitador)
        for row in reader:
            mod = row.get("modelo", "TTP1").strip()
            linhas_por_modelo[mod].append(row)

    resultado_total: List[Dict[str, str]] = []

    for mod in sorted(linhas_por_modelo.keys()):
        linhas_norm = normalizar_grupo(linhas_por_modelo[mod])
        resultado_total.extend(linhas_norm)

    # Salva o arquivo CSV normalizado
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    colunas_saida = ["modelo", "solver"] + TODAS_COLUNAS_NORM

    with open(caminho_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas_saida, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        for r in resultado_total:
            writer.writerow({
                "modelo": r["modelo"],
                "solver": r["solver"],
                "media_G": f"{r['media_G']:.4f}",
                "std_G": f"{r['std_G']:.4f}",
                "min_G": f"{r['min_G']:.4f}",
                "q1_G": f"{r['q1_G']:.4f}",
                "mediana_G": f"{r['mediana_G']:.4f}",
                "q3_G": f"{r['q3_G']:.4f}",
                "max_G": f"{r['max_G']:.4f}",
                "media_tempo_cpu_ms": f"{r['media_tempo_cpu_ms']:.4f}",
            })

    return resultado_total


def imprimir_relatorio_normalizado(dados_normalizados: List[Dict[str, str]]):
    """Exibe no terminal a tabela normalizada formatada."""
    por_modelo = defaultdict(list)
    for r in dados_normalizados:
        por_modelo[r["modelo"]].append(r)

    for mod, linhas in por_modelo.items():
        print("\n" + "=" * 110)
        print(f"{f'ESTATÍSTICAS NORMALIZADAS (0 a 1) - MODELO {mod}':^110}")
        print(f"{'(Melhor = 1.0000 | Pior = 0.0000 | Demais em função do melhor)':^110}")
        print("=" * 110)
        print(
            f"{'Solucionador':<26} | {'Média (G)':>10} | {'Mediana':>10} | {'DesvPad':>10} | "
            f"{'Q1 (25%)':>10} | {'Q3 (75%)':>10} | {'Máx':>10} | {'CPU Time':>10}"
        )
        print("-" * 110)
        for r in linhas:
            print(
                f"{r['solver']:<26} | "
                f"{float(r['media_G']):>10.4f} | "
                f"{float(r['mediana_G']):>10.4f} | "
                f"{float(r['std_G']):>10.4f} | "
                f"{float(r['q1_G']):>10.4f} | "
                f"{float(r['q3_G']):>10.4f} | "
                f"{float(r['max_G']):>10.4f} | "
                f"{float(r['media_tempo_cpu_ms']):>10.4f}"
            )
        print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Normalizador de Estatísticas do TTP para escala [0, 1] onde o melhor é 1.0000."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(_raiz / "results" / "comparativo_estatisticas_ttp1_ttp2.csv"),
        help="Caminho do CSV de estatísticas consolidado original.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_raiz / "results" / "comparativo_estatisticas_normalizadas.csv"),
        help="Caminho do CSV de saída com dados normalizados.",
    )

    args = parser.parse_args()
    caminho_in = Path(args.input).resolve()
    caminho_out = Path(args.output).resolve()

    print("\n" + "=" * 80)
    print(f"{'NORMALIZADOR DE ESTATÍSTICAS COMPARATIVAS DO TTP':^80}")
    print(f" • Arquivo de Entrada: {caminho_in}")
    print(f" • Arquivo de Saída:   {caminho_out}")
    print("=" * 80)

    try:
        dados_norm = normalizar_arquivo_estatisticas(caminho_in, caminho_out)
        imprimir_relatorio_normalizado(dados_norm)
        print(f"\n✅ Estatísticas normalizadas salvas com sucesso em:\n   {caminho_out}\n")
    except Exception as e:
        print(f"\n❌ Erro durante a normalização: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
