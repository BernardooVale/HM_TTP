"""
Bateria de testes comparativa entre heurísticas do TTP com análise estatística completa.
Padrão: TTP1 (TTP2 é opcional via --models TTP1,TTP2 ou --incluir-ttp2).

Solucionadores comparados:
- Baselines: Random, Greedy
- Heurística dos 37%:
    * heuristic_37 (Padrão)
    * heuristic_37_refinada (Com busca local PackIterative)
    * heuristic_37_simetrica (Com limiar simétrico em mochila vazia)
- Heurística da Queimada:
    * queimada (Padrão ótimo: 2-opt de rota, afinidade térmica, refinamento Drop & Add)
    * queimada_sem_2opt vs queimada_com_2opt (Impacto do 2-opt na rota)
    * queimada_sem_afinidade vs queimada_com_afinidade (Impacto da afinidade térmica Q/d^gamma)
    * queimada_regime_r (Adaptação dinâmica da afinidade à taxa R da instância)
    * queimada_reversa (Varredura reversa na seleção de itens)
    * queimada_legacy (Configuração inicial baseline)
    * queimada_backward_arestas / queimada_backward_buracos
- Estado da Arte da Literatura:
    * S2 Heuristic (Polyakovskiy et al., 2014)
    * S5 Heuristic (Polyakovskiy et al., 2014)
    * PackIterative (Faulkner et al., 2015)
    * Co-Solver (Faulkner et al., 2015)
"""

import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Adiciona caminhos do projeto ao sys.path
_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

import src.solvers  # Trigger registration
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.scenarios.parser import InstanceParser
from src.solvers.base import BaseSolver, SolverRegistry


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


def executar_bateria(
    pattern: Optional[str] = None,
    instances_list: Optional[List[str]] = None,
    solvers_list: Optional[List[str]] = None,
    models_list: Optional[List[str]] = None,
):
    test_dir = _sim_path / "bateria_testes"

    if instances_list:
        instance_files = [test_dir / name if not Path(name).is_absolute() else Path(name) for name in instances_list]
        instance_files = [f for f in instance_files if f.exists()]
    elif pattern:
        instance_files = sorted(list(test_dir.glob(pattern)))
    else:
        # Padrão: inclui arquivos .txt de teste
        instance_files = sorted(list(test_dir.glob("*.txt")) + list(test_dir.glob("*.ttp")))

    solvers_para_testar = solvers_list or [
        # Baselines
        "random",
        "greedy",
        # Heurística dos 37%
        "heuristic_37",
        "heuristic_37_refinada",
        "heuristic_37_simetrica",
        # Queimada - Comparações Paramétricas
        "queimada",
        "queimada_sem_2opt",
        "queimada_com_2opt",
        "queimada_sem_afinidade",
        "queimada_com_afinidade",
        "queimada_regime_r",
        "queimada_reversa",
        "queimada_legacy",
        "queimada_com_buracos",
        "queimada_sem_2opt_buracos",
        "queimada_backward_arestas",
        "queimada_backward_buracos",
        "queimada_backward_sem_2opt",
        "queimada_backward_sem_2opt_buracos",
        # Queimada - Variações Direcionadas
        "queimada_kruskal_padrao",
        "queimada_backward_direcionada",
        "queimada_backward_direcionada_buracos",
        "queimada_backward_direcionada_sem_2opt_buracos",
        "queimada_forward_direcionada",
        "queimada_forward_direcionada_buracos",
        "queimada_forward_direcionada_sem_2opt_buracos",
        # Estado da Arte da Literatura
        "s2_heuristic",
        "s5_heuristic",
        "pack_iterative",
        "cosolver",
    ]

    modelos = models_list or ["TTP1"]

    print(f"\n{'='*115}")
    print(f"{'BATERIA DE TESTES COMPARATIVA DO TRAVELING THIEF PROBLEM':^115}")
    print(f"Instâncias: {len(instance_files)} | Solucionadores: {len(solvers_para_testar)} | Modelos: {', '.join(modelos)}")
    print(f"{'='*115}\n")

    results_dir = _raiz / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dados_detalhados: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modelos}
    resultados_agrupados: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: defaultdict(lambda: {"lucros": [], "tempos_comp": [], "tempos_viagem": []})
        for m in modelos
    }

    # ==============================================================
    # LOOP PRINCIPAL DE EXECUÇÃO
    # ==============================================================
    for idx_inst, inst_file in enumerate(instance_files, 1):
        print(f"\n[{idx_inst}/{len(instance_files)}] Carregando '{inst_file.name}'...")
        instance = InstanceParser.carregar(str(inst_file))

        for modelo in modelos:
            print(f"  --- Avaliando sob modelo: {modelo} ---")
            avaliador = avaliar_ttp1 if modelo == "TTP1" else avaliar_ttp2

            for s_name in solvers_para_testar:
                solver = SolverRegistry.get(s_name)
                if hasattr(solver, "modelo"):
                    setattr(solver, "modelo", modelo)

                t0 = time.perf_counter()
                solucao = solver.solve(instance)
                tempo_exec = (time.perf_counter() - t0) * 1000.0

                res = avaliador(instance, solucao)
                lucro = res["lucro_liquido_G"]
                tempo_v = res["tempo_viagem_f"]
                peso = res.get("peso_acumulado") or res.get("peso_acumulado_mochila", 0.0)
                qtd = res["qtd_itens_coletados"]

                resultados_agrupados[modelo][s_name]["lucros"].append(lucro)
                resultados_agrupados[modelo][s_name]["tempos_comp"].append(tempo_exec)
                resultados_agrupados[modelo][s_name]["tempos_viagem"].append(tempo_v)

                dados_detalhados[modelo].append({
                    "instancia": inst_file.stem,
                    "modelo": modelo,
                    "solver": s_name,
                    "lucro_G": lucro,
                    "tempo_viagem": tempo_v,
                    "peso": peso,
                    "qtd_itens": qtd,
                    "tempo_ms": tempo_exec,
                })

                print(
                    f"  {s_name:<26} | Lucro G: {lucro:>10.2f} | T. Viagem: {tempo_v:>8.2f} | "
                    f"Itens: {qtd:>2} ({peso:>6.1f}kg) | {tempo_exec:>6.1f}ms"
                )

        # Salva CSV detalhado específico do modelo
        for modelo in modelos:
            csv_file_detalhe = results_dir / "queimada" / f"comparativo_detalhado_{modelo.lower()}.csv"
            with open(csv_file_detalhe, "w", encoding="utf-8") as f:
                f.write("instancia;modelo;solver;lucro_liquido_G;tempo_viagem_f;peso_acumulado;qtd_itens;tempo_exec_ms\n")
                for row in dados_detalhados[modelo]:
                    f.write(
                        f"{row['instancia']};{row['modelo']};{row['solver']};{row['lucro_G']:.4f};"
                        f"{row['tempo_viagem']:.4f};{row['peso']:.4f};{row['qtd_itens']};{row['tempo_ms']:.2f}\n"
                    )

    # ==============================================================
    # RELATÓRIOS CONSOLIDADOS COM MÉDIAS E QUARTIS
    # ==============================================================
    estatisticas_consolidadas = []

    for modelo in modelos:
        print(f"\n\n{'='*115}")
        print(f"{f'RESUMO CONSOLIDADO COM MÉDIAS E QUARTIS - {modelo}':^115}")
        print(f"{'='*115}")
        print(
            f"{'Solucionador':<26} | {'Média (G)':>10} | {'DesvPad':>9} | {'Q1 (25%)':>10} | "
            f"{'Mediana':>10} | {'Q3 (75%)':>10} | {'Min':>10} | {'Max':>10}"
        )
        print(f"{'-'*115}")

        solvers_stats = []
        for s_name in solvers_para_testar:
            lucros = resultados_agrupados[modelo][s_name]["lucros"]
            tempos_v = resultados_agrupados[modelo][s_name]["tempos_viagem"]
            tempos_cpu = resultados_agrupados[modelo][s_name]["tempos_comp"]

            stats_lucro = calcular_estatisticas(lucros)
            stats_viagem = calcular_estatisticas(tempos_v)
            stats_cpu = calcular_estatisticas(tempos_cpu)

            solvers_stats.append((s_name, stats_lucro, stats_viagem, stats_cpu))

        # Ordena do maior lucro mediano para o menor
        solvers_stats.sort(key=lambda item: item[1]["mediana"], reverse=True)

        for rank, (s_name, st_g, st_v, st_cpu) in enumerate(solvers_stats, 1):
            med = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else "   "))
            print(
                f"{med}{s_name:<23} | {st_g['media']:>10.2f} | {st_g['std']:>9.2f} | "
                f"{st_g['q1']:>10.2f} | {st_g['mediana']:>10.2f} | {st_g['q3']:>10.2f} | "
                f"{st_g['min']:>10.2f} | {st_g['max']:>10.2f}"
            )

            estatisticas_consolidadas.append({
                "modelo_ttp": modelo,
                "solucionador": s_name,
                "media_G": f"{st_g['media']:.4f}",
                "std_G": f"{st_g['std']:.4f}",
                "min_G": f"{st_g['min']:.4f}",
                "q1_G": f"{st_g['q1']:.4f}",
                "mediana_G": f"{st_g['mediana']:.4f}",
                "q3_G": f"{st_g['q3']:.4f}",
                "max_G": f"{st_g['max']:.4f}",
                "media_tempo_cpu_ms": f"{st_cpu['media']:.2f}",
                "media_tempo_viagem": f"{st_v['media']:.4f}",
            })

    # Grava CSV consolidado geral
    csv_consolidado = results_dir / "queimada" / "comparativo_heuristicas.csv"
    with open(csv_consolidado, "w", encoding="utf-8") as f:
        headers = [
            "modelo_ttp", "solucionador", "media_G", "std_G", "min_G", "q1_G",
            "mediana_G", "q3_G", "max_G", "media_tempo_cpu_ms", "media_tempo_viagem"
        ]
        f.write(";".join(headers) + "\n")
        for row in estatisticas_consolidadas:
            f.write(";".join([row[h] for h in headers]) + "\n")

    print(f"\n📁 Resultados estatísticos consolidados gravados em '{csv_consolidado}'.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bateria comparativa de solvers do Traveling Thief Problem.")
    parser.add_argument("--pattern", type=str, default=None, help="Padrão glob de instâncias (ex: '*.txt', '*.ttp', '*')")
    parser.add_argument("--instances", type=str, default=None, help="Lista de arquivos de instâncias separados por vírgula")
    parser.add_argument("--solvers", type=str, default=None, help="Lista de solvers separados por vírgula")
    parser.add_argument("--models", type=str, default=None, help="Modelos separados por vírgula (padrão: TTP1; ex: TTP1,TTP2)")
    parser.add_argument("--incluir-ttp2", action="store_true", help="Inclui TTP2 além do TTP1")
    parser.add_argument("--all", action="store_true", help="Executa em todos os arquivos de instâncias (.txt e .ttp)")

    args = parser.parse_args()

    inst_list = [s.strip() for s in args.instances.split(",")] if args.instances else None
    solv_list = [s.strip() for s in args.solvers.split(",")] if args.solvers else None

    if args.models:
        mod_list = [s.strip() for s in args.models.split(",")]
    elif args.incluir_ttp2:
        mod_list = ["TTP1", "TTP2"]
    else:
        mod_list = ["TTP1"]

    pat = "*" if args.all else args.pattern

    executar_bateria(
        pattern=pat,
        instances_list=inst_list,
        solvers_list=solv_list,
        models_list=mod_list,
    )
