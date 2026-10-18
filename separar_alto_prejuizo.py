"""
Script de Separação de Instâncias por Equilíbrio Econômico (Receita Máxima / Custo Mínimo >= 0.5).

Objetivos:
1. Avalia cada instância da pasta 'ttp_simulator/bateria_testes' com base na relação:
     Razão = (Receita Potencial Máxima da Mochila Fracionária) / (Custo Mínimo de Aluguel na MST)
2. Separa as instâncias com alto prejuízo (Razão < 0.5) e move os arquivos para uma pasta
   dedicada 'ttp_simulator/bateria_testes/alto_prejuizo'.
   - Essa pasta é separada da quarentena: as instâncias dentro de 'alto_prejuizo' continuam
     disponíveis e podem ser executadas a qualquer momento passando a pasta explicitamente.
3. Filtra automaticamente os dados de 'results/bateria_solucoes_detalhada.csv' sem necessidade
   de reexecução computacional:
   - Salva os registros detalhados das instâncias equilibradas em 'results/bateria_equilibradas_detalhada.csv'
   - Gera o consolidado estatístico em 'results/comparativo_estatisticas_equilibradas.csv'
   - Gera o consolidado normalizado [0, 1] em 'results/comparativo_estatisticas_equilibradas_normalizadas.csv'
   - Exporta o relatório completo de triagem em 'results/triagem_alto_prejuizo.csv'

Comandos:
    python separar_alto_prejuizo.py             # Classifica, move arquivos e gera CSVs filtrados
    python separar_alto_prejuizo.py --listar    # Dry-run: apenas audita sem mover arquivos
    python separar_alto_prejuizo.py --restaurar # Devolve instâncias de 'alto_prejuizo' para 'bateria_testes'
"""

import argparse
import csv
import os
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_raiz = Path(__file__).resolve().parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from filtrar_instancias_lucro import analisar_instancia_lucro
from normalizar_estatisticas import normalizar_grupo
from src.scenarios.parser import InstanceParser


def calcular_estatisticas(valores: List[float]) -> Dict[str, float]:
    """Calcula estatísticas descritivas completas: média, desvio padrão e quartis."""
    if not valores:
        return {
            "media": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "mediana": 0.0,
            "q3": 0.0,
            "max": 0.0,
        }

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
        q1 = vals[0]
        q3 = vals[-1]

    return {
        "media": media,
        "std": std,
        "min": v_min,
        "q1": q1,
        "mediana": med,
        "q3": q3,
        "max": v_max,
    }


def auditar_instancias(
    bateria_dir: Path,
    pasta_alto_prejuizo: Path,
) -> Tuple[List[dict], List[dict]]:
    """
    Avalia todas as instâncias em bateria_dir e pasta_alto_prejuizo.
    Retorna (equilibradas, alto_prejuizo).
    """
    parser = InstanceParser()
    todos_arquivos: Dict[str, Path] = {}

    for arq in bateria_dir.iterdir():
        if arq.is_file() and arq.suffix in [".txt", ".ttp"]:
            todos_arquivos[arq.name] = arq

    if pasta_alto_prejuizo.exists():
        for arq in pasta_alto_prejuizo.iterdir():
            if arq.is_file() and arq.suffix in [".txt", ".ttp"]:
                todos_arquivos[arq.name] = arq

    equilibradas = []
    alto_prejuizo = []

    print(f"Avaliando {len(todos_arquivos)} instâncias teóricas...")
    for nome, caminho in sorted(todos_arquivos.items()):
        try:
            inst = parser.parse(caminho)
            info = analisar_instancia_lucro(inst)
            rec = info["valor_mochila_divisivel"]
            custo = info["custo_min_aluguel"]
            ratio = rec / custo if custo > 0 else (float("inf") if rec > 0 else 0.0)

            registro = {
                "arquivo": nome,
                "caminho_atual": caminho,
                "cidades": info["cidades"],
                "itens": info["itens"],
                "capacidade_W": info["capacidade_W"],
                "taxa_aluguel_R": info["taxa_aluguel_R"],
                "valor_mochila_divisivel": rec,
                "custo_mst": info["custo_mst"],
                "custo_min_aluguel": custo,
                "razao_receita_custo": ratio,
                "ub_lucro_liquido": info["ub_lucro_liquido"],
                "equilibrada": ratio >= 0.5,
            }

            if ratio >= 0.5:
                equilibradas.append(registro)
            else:
                alto_prejuizo.append(registro)
        except Exception as e:
            print(f"⚠️ Aviso: Falha ao analisar {nome}: {e}")

    return equilibradas, alto_prejuizo


def restaurar_alto_prejuizo(pasta_origem: Path, pasta_destino: Path) -> int:
    """Restaura todas as instâncias da subpasta alto_prejuizo para a bateria_testes."""
    if not pasta_origem.exists():
        return 0
    total = 0
    for arq in list(pasta_origem.iterdir()):
        if arq.is_file() and arq.suffix in [".txt", ".ttp"]:
            destino = pasta_destino / arq.name
            shutil.move(str(arq), str(destino))
            total += 1
    return total


def mover_instancias_alto_prejuizo(
    alto_prejuizo: List[dict],
    pasta_destino: Path,
) -> int:
    """Move os arquivos classificados como alto prejuízo para a pasta dedicada."""
    pasta_destino.mkdir(parents=True, exist_ok=True)
    total = 0
    for reg in alto_prejuizo:
        origem = reg["caminho_atual"]
        if origem.parent != pasta_destino:
            destino = pasta_destino / reg["arquivo"]
            shutil.move(str(origem), str(destino))
            total += 1
    return total


def mover_instancias_equilibradas(
    equilibradas: List[dict],
    pasta_bateria: Path,
) -> int:
    """Garante que as instâncias equilibradas estejam na raiz da bateria_testes."""
    total = 0
    for reg in equilibradas:
        origem = reg["caminho_atual"]
        if origem.parent != pasta_bateria:
            destino = pasta_bateria / reg["arquivo"]
            shutil.move(str(origem), str(destino))
            total += 1
    return total


def filtrar_e_consolidar_resultados(
    csv_total: Path,
    nomes_equilibradas: Set[str],
    output_detalhado: Path,
    output_stats: Path,
    output_norm: Path,
):
    """
    Filtra o CSV total para as instâncias equilibradas e gera os consolidados estatísticos.
    """
    if not csv_total.exists() or csv_total.stat().st_size == 0:
        print(f"❌ Arquivo de resultados totais '{csv_total}' não encontrado.")
        return

    print(f"\nFiltrando execuções de '{csv_total.name}' para as {len(nomes_equilibradas)} instâncias equilibradas...")

    linhas_filtradas = []
    dados_por_modelo_solver = defaultdict(lambda: {
        "lucros": [],
        "tempos_comp": [],
        "tempos_viagem": [],
        "pesos": [],
        "pct_ocupacao": [],
        "qtd_itens": [],
    })

    with open(csv_total, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            inst = row.get("arquivo_instancia", "").strip()
            if inst in nomes_equilibradas:
                linhas_filtradas.append(row)
                mod = row.get("modelo_ttp", "").strip()
                solv = row.get("solucionador", "").strip()
                try:
                    g = float(row.get("lucro_liquido_G", 0.0))
                    t_cpu = float(row.get("tempo_computacional_seg", 0.0)) * 1000.0  # ms
                    t_viagem = float(row.get("tempo_viagem_f", 0.0))
                    peso = float(row.get("peso_acumulado_mochila", 0.0))
                    pct_oc = float(row.get("pct_ocupacao_mochila", 0.0))
                    qtd_it = float(row.get("qtd_itens_coletados", 0.0))

                    entry = dados_por_modelo_solver[(mod, solv)]
                    entry["lucros"].append(g)
                    entry["tempos_comp"].append(t_cpu)
                    entry["tempos_viagem"].append(t_viagem)
                    entry["pesos"].append(peso)
                    entry["pct_ocupacao"].append(pct_oc)
                    entry["qtd_itens"].append(qtd_it)
                except ValueError:
                    continue

    print(f"Total de execuções extraídas: {len(linhas_filtradas)}")

    # 1. Salva CSV detalhado equilibrado
    output_detalhado.parent.mkdir(parents=True, exist_ok=True)
    with open(output_detalhado, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(linhas_filtradas)
    print(f"✅ CSV Detalhado Equilibrado salvo em: {output_detalhado}")

    # 2. Consolida estatísticas
    modelos = ["TTP1", "TTP2"]
    estatisticas_consolidadas = []

    for mod in modelos:
        solvers_stats = []
        for (m, s_name), metricas in dados_por_modelo_solver.items():
            if m != mod:
                continue
            sl = calcular_estatisticas(metricas["lucros"])
            sc = calcular_estatisticas(metricas["tempos_comp"])
            sv = calcular_estatisticas(metricas["tempos_viagem"])
            sp = calcular_estatisticas(metricas["pesos"])
            so = calcular_estatisticas(metricas["pct_ocupacao"])
            sq = calcular_estatisticas(metricas["qtd_itens"])

            solvers_stats.append((s_name, sl, sc, sv, sp, so, sq))

        # Ordena por mediana decrescente
        solvers_stats.sort(key=lambda item: item[1]["mediana"], reverse=True)

        for s_name, sl, sc, sv, sp, so, sq in solvers_stats:
            estatisticas_consolidadas.append({
                "modelo": mod,
                "solver": s_name,
                "media_G": sl["media"],
                "std_G": sl["std"],
                "min_G": sl["min"],
                "q1_G": sl["q1"],
                "mediana_G": sl["mediana"],
                "q3_G": sl["q3"],
                "max_G": sl["max"],
                "media_tempo_cpu_ms": sc["media"],
                "media_tempo_viagem": sv["media"],
                "media_peso_acumulado": sp["media"],
                "media_pct_ocupacao_mochila": so["media"],
                "media_qtd_itens": sq["media"],
            })

    output_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stats, "w", newline="", encoding="utf-8") as f:
        colunas = [
            "modelo", "solver", "media_G", "std_G", "min_G", "q1_G", "mediana_G", "q3_G", "max_G",
            "media_tempo_cpu_ms", "media_tempo_viagem", "media_peso_acumulado",
            "media_pct_ocupacao_mochila", "media_qtd_itens"
        ]
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        writer.writeheader()
        for st in estatisticas_consolidadas:
            writer.writerow({
                "modelo": st["modelo"],
                "solver": st["solver"],
                "media_G": f"{st['media_G']:.4f}",
                "std_G": f"{st['std_G']:.4f}",
                "min_G": f"{st['min_G']:.4f}",
                "q1_G": f"{st['q1_G']:.4f}",
                "mediana_G": f"{st['mediana_G']:.4f}",
                "q3_G": f"{st['q3_G']:.4f}",
                "max_G": f"{st['max_G']:.4f}",
                "media_tempo_cpu_ms": f"{st['media_tempo_cpu_ms']:.2f}",
                "media_tempo_viagem": f"{st['media_tempo_viagem']:.4f}",
                "media_peso_acumulado": f"{st['media_peso_acumulado']:.4f}",
                "media_pct_ocupacao_mochila": f"{st['media_pct_ocupacao_mochila']:.2f}",
                "media_qtd_itens": f"{st['media_qtd_itens']:.2f}",
            })
    print(f"✅ CSV Estatísticas Consolidadas salvo em: {output_stats}")

    # 3. Normalização [0, 1]
    por_modelo = defaultdict(list)
    for st in estatisticas_consolidadas:
        por_modelo[st["modelo"]].append(st)

    linhas_norm_total = []
    for mod in ["TTP1", "TTP2"]:
        if mod in por_modelo:
            linhas_norm_total.extend(normalizar_grupo(por_modelo[mod]))

    output_norm.parent.mkdir(parents=True, exist_ok=True)
    with open(output_norm, "w", newline="", encoding="utf-8") as f:
        colunas_norm = [
            "modelo", "solver", "media_G", "std_G", "min_G", "q1_G", "mediana_G", "q3_G", "max_G", "media_tempo_cpu_ms"
        ]
        writer = csv.DictWriter(f, fieldnames=colunas_norm, delimiter=";")
        writer.writeheader()
        for r in linhas_norm_total:
            writer.writerow({
                "modelo": r["modelo"],
                "solver": r["solver"],
                "media_G": f"{float(r['media_G']):.4f}",
                "std_G": f"{float(r['std_G']):.4f}",
                "min_G": f"{float(r['min_G']):.4f}",
                "q1_G": f"{float(r['q1_G']):.4f}",
                "mediana_G": f"{float(r['mediana_G']):.4f}",
                "q3_G": f"{float(r['q3_G']):.4f}",
                "max_G": f"{float(r['max_G']):.4f}",
                "media_tempo_cpu_ms": f"{float(r['media_tempo_cpu_ms']):.4f}",
            })
    print(f"✅ CSV Estatísticas Normalizadas salvo em: {output_norm}")


def exportar_triagem(
    equilibradas: List[dict],
    alto_prejuizo: List[dict],
    caminho_csv: Path,
):
    """Grava o relatório de triagem econômica de todas as instâncias avaliadas."""
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)
    todos = equilibradas + alto_prejuizo
    todos.sort(key=lambda x: x["razao_receita_custo"], reverse=True)

    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        colunas = [
            "arquivo", "cidades", "itens", "capacidade_W", "taxa_aluguel_R",
            "valor_mochila_divisivel", "custo_mst", "custo_min_aluguel",
            "razao_receita_custo", "ub_lucro_liquido", "classificacao"
        ]
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        writer.writeheader()
        for r in todos:
            writer.writerow({
                "arquivo": r["arquivo"],
                "cidades": r["cidades"],
                "itens": r["itens"],
                "capacidade_W": f"{r['capacidade_W']:.2f}",
                "taxa_aluguel_R": f"{r['taxa_aluguel_R']:.6f}",
                "valor_mochila_divisivel": f"{r['valor_mochila_divisivel']:.2f}",
                "custo_mst": f"{r['custo_mst']:.2f}",
                "custo_min_aluguel": f"{r['custo_min_aluguel']:.2f}",
                "razao_receita_custo": f"{r['razao_receita_custo']:.4f}",
                "ub_lucro_liquido": f"{r['ub_lucro_liquido']:.2f}",
                "classificacao": "EQUILIBRADA" if r["equilibrada"] else "ALTO_PREJUIZO",
            })
    print(f"✅ Relatório de Triagem salvo em: {caminho_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Separa instâncias de alto prejuízo (Receita / Custo < 0.5) e filtra resultados estatísticos."
    )
    parser.add_argument(
        "--limiar",
        type=float,
        default=0.5,
        help="Razão mínima (Receita / Custo) para classificar como equilibrada (padrão: 0.5)",
    )
    parser.add_argument(
        "--bateria",
        type=str,
        default=str(_raiz / "ttp_simulator" / "bateria_testes"),
        help="Diretório principal da bateria de testes",
    )
    parser.add_argument(
        "--pasta-alto-prejuizo",
        type=str,
        default=str(_raiz / "ttp_simulator" / "bateria_testes" / "alto_prejuizo"),
        help="Subpasta de destino para instâncias de alto prejuízo",
    )
    parser.add_argument(
        "--csv-total",
        type=str,
        default=str(_raiz / "results" / "bateria_solucoes_detalhada.csv"),
        help="CSV com todos os dados detalhados da bateria executada",
    )
    parser.add_argument(
        "--output-detalhado",
        type=str,
        default=str(_raiz / "results" / "bateria_equilibradas_detalhada.csv"),
        help="Caminho do CSV de soluções detalhadas das instâncias equilibradas",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        default=str(_raiz / "results" / "comparativo_estatisticas_equilibradas.csv"),
        help="Caminho do CSV estatístico consolidado das instâncias equilibradas",
    )
    parser.add_argument(
        "--output-norm",
        type=str,
        default=str(_raiz / "results" / "comparativo_estatisticas_equilibradas_normalizadas.csv"),
        help="Caminho do CSV normalizado das instâncias equilibradas",
    )
    parser.add_argument(
        "--output-triagem",
        type=str,
        default=str(_raiz / "results" / "triagem_alto_prejuizo.csv"),
        help="Caminho do CSV de triagem econômica de todas as instâncias",
    )
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Apenas lista e audita sem mover arquivos e sem filtrar resultados",
    )
    parser.add_argument(
        "--restaurar",
        action="store_true",
        help="Restaura todos os arquivos de 'alto_prejuizo' de volta para a bateria_testes",
    )

    args = parser.parse_args()
    bateria_dir = Path(args.bateria).resolve()
    pasta_alto_prejuizo = Path(args.pasta_alto_prejuizo).resolve()

    print("\n" + "=" * 90)
    print(f"{'TRIAGEM E SEPARAÇÃO DE INSTÂNCIAS TTP (RECEITA / CUSTO >= 0.5)':^90}")
    print(f" • Diretório de Testes:     {bateria_dir}")
    print(f" • Pasta de Alto Prejuízo:  {pasta_alto_prejuizo}")
    print(f" • Limiar Econômico:        Razão >= {args.limiar}")
    print("=" * 90 + "\n")

    if args.restaurar:
        print("Restaurando instâncias de 'alto_prejuizo' para 'bateria_testes'...")
        total_rest = restaurar_alto_prejuizo(pasta_alto_prejuizo, bateria_dir)
        print(f"\n✅ Total de instâncias restauradas com sucesso: {total_rest}\n")
        return

    equilibradas, alto_prejuizo = auditar_instancias(bateria_dir, pasta_alto_prejuizo)
    total_inst = len(equilibradas) + len(alto_prejuizo)

    print("\n" + "-" * 90)
    print(f"Resumo da Classificação Teórica (Total: {total_inst} instâncias):")
    print(f" • Equilibradas (Receita / Custo >= {args.limiar}): {len(equilibradas)} instâncias")
    print(f" • Alto Prejuízo (Receita / Custo < {args.limiar}): {len(alto_prejuizo)} instâncias")
    print("-" * 90 + "\n")

    if args.listar:
        print("Modo --listar ativado. Nenhum arquivo foi movido e nenhum CSV foi alterado.")
        return

    # Move arquivos
    print("Organizando arquivos nos diretórios...")
    mov_ap = mover_instancias_alto_prejuizo(alto_prejuizo, pasta_alto_prejuizo)
    mov_eq = mover_instancias_equilibradas(equilibradas, bateria_dir)
    print(f" • Instâncias movidas para 'alto_prejuizo': {mov_ap}")
    if mov_eq > 0:
        print(f" • Instâncias reposicionadas na bateria:    {mov_eq}")

    # Exporta triagem completa
    exportar_triagem(equilibradas, alto_prejuizo, Path(args.output_triagem).resolve())

    # Filtra os dados existentes do CSV total
    nomes_eq = {r["arquivo"] for r in equilibradas}
    filtrar_e_consolidar_resultados(
        csv_total=Path(args.csv_total).resolve(),
        nomes_equilibradas=nomes_eq,
        output_detalhado=Path(args.output_detalhado).resolve(),
        output_stats=Path(args.output_stats).resolve(),
        output_norm=Path(args.output_norm).resolve(),
    )

    print("\n" + "=" * 90)
    print(f"{'OPERAÇÃO CONCLUÍDA COM SUCESSO!':^90}")
    print(" • Bateria Ativa em 'bateria_testes/': instâncias equilibradas (sem distorções extremas).")
    print(" • Instâncias de alto prejuízo isoladas em: 'bateria_testes/alto_prejuizo/'.")
    print(" • Dados filtrados e normalizados gerados em 'results/' sem necessidade de nova execução.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
