"""
Módulo de Normalização de Estatísticas Comparativas do Traveling Thief Problem (TTP).

Lê o arquivo de estatísticas consolidadas (results/comparativo_estatisticas_ttp1_ttp2.csv)
e normaliza as métricas no intervalo [0, 1] para cada modelo separadamente:
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

_raiz = Path(__file__).resolve().parents[2]

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
        nova_linha = dict(r)

        # Normaliza métricas onde maior é melhor
        for col in COLUNAS_MAXIMIZAR:
            if col in r and r[col] != "":
                val = float(r[col])
                v_min, v_max = limites[col]
                amplitude = v_max - v_min
                if amplitude > 1e-9:
                    score = (val - v_min) / amplitude
                else:
                    score = 1.0  # Se todos tiverem o mesmo valor, todos empatam no topo
                nova_linha[col] = f"{max(0.0, min(1.0, score)):.4f}"

        # Normaliza métricas onde menor é melhor (tempo CPU, desvio padrão)
        for col in COLUNAS_MINIMIZAR:
            if col in r and r[col] != "":
                val = float(r[col])
                v_min, v_max = limites[col]
                amplitude = v_max - v_min
                if amplitude > 1e-9:
                    score = (v_max - val) / amplitude
                else:
                    score = 1.0
                nova_linha[col] = f"{max(0.0, min(1.0, score)):.4f}"

        linhas_normalizadas.append(nova_linha)

    return linhas_normalizadas


def normalizar_arquivo_estatisticas(
    caminho_csv_entrada: Path,
    caminho_csv_saida: Path,
) -> List[Dict[str, str]]:
    """
    Lê o CSV de estatísticas brutas, agrupa por modelo e grava o CSV normalizado.
    """
    if not caminho_csv_entrada.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv_entrada}")

    linhas = []
    with open(caminho_csv_entrada, "r", encoding="utf-8", errors="ignore") as f:
        primeira_linha = f.readline()
        delimitador = ";" if ";" in primeira_linha else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimitador)
        fieldnames = reader.fieldnames
        for row in reader:
            linhas.append(row)

    if not linhas or not fieldnames:
        print(f"⚠️ O arquivo {caminho_csv_entrada} está vazio.")
        return []

    # Agrupa por modelo TTP (TTP1 vs TTP2)
    por_modelo = defaultdict(list)
    for row in linhas:
        modelo = row.get("modelo") or row.get("modelo_ttp") or "TTP1"
        por_modelo[modelo].append(row)

    linhas_totais_normalizadas = []
    for modelo, grupo in por_modelo.items():
        grupo_norm = normalizar_grupo(grupo)
        # Reordena por lucro mediano normalizado decrescente
        grupo_norm.sort(key=lambda x: float(x.get("mediana_G", 0.0)), reverse=True)
        linhas_totais_normalizadas.extend(grupo_norm)

    # Grava o CSV de saída preservando o cabeçalho e delimitador
    caminho_csv_saida.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_csv_saida, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimitador)
        writer.writeheader()
        writer.writerows(linhas_totais_normalizadas)

    return linhas_totais_normalizadas


def imprimir_relatorio_normalizado(linhas_normalizadas: List[Dict[str, str]]):
    """Exibe no terminal a tabela normalizada formatada para inspeção imediata."""
    if not linhas_normalizadas:
        return

    por_modelo = defaultdict(list)
    for r in linhas_normalizadas:
        mod = r.get("modelo") or r.get("modelo_ttp") or "TTP1"
        por_modelo[mod].append(r)

    for modelo, grupo in por_modelo.items():
        print("\n" + "=" * 110)
        print(f"{f'ESTATÍSTICAS NORMALIZADAS [0, 1] - MODELO: {modelo} (MELHOR = 1.0000)':^110}")
        print("=" * 110)
        print(
            f"{'Solucionador':<26} | {'Média G':>9} | {'Std G':>9} | {'Min G':>9} | "
            f"{'Q1 G':>9} | {'Mediana':>9} | {'Q3 G':>9} | {'Max G':>9} | {'Tempo CPU':>11}"
        )
        print("-" * 110)
        for rank, r in enumerate(grupo, 1):
            s_name = r.get("solver") or r.get("solucionador", "")
            med = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else "   "))
            print(
                f"{med}{s_name:<23} | {r.get('media_G',''):>9} | {r.get('std_G',''):>9} | "
                f"{r.get('min_G',''):>9} | {r.get('q1_G',''):>9} | {r.get('mediana_G',''):>9} | "
                f"{r.get('q3_G',''):>9} | {r.get('max_G',''):>9} | {r.get('media_tempo_cpu_ms',''):>11}"
            )
        print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Normalizador de Estatísticas do TTP para escala [0, 1] onde o melhor é 1.0000."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(_raiz / "results" / "web" / "comparativo_estatisticas_ttp1_ttp2.csv" if (_raiz / "results" / "web" / "comparativo_estatisticas_ttp1_ttp2.csv").exists() else _raiz / "results" / "comparativo_estatisticas_ttp1_ttp2.csv"),
        help="Caminho do CSV de estatísticas consolidado original.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_raiz / "results" / "web" / "comparativo_estatisticas_normalizadas.csv" if (_raiz / "results" / "web" / "comparativo_estatisticas_normalizadas.csv").exists() else _raiz / "results" / "comparativo_estatisticas_normalizadas.csv"),
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
