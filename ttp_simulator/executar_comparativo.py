"""
Bateria de testes comparativa entre heuristicas do TTP:
- Baselines (Random, Greedy)
- Heurística dos 37% (Original vs Refinada)
- Heurística da Queimada (Original vs Reversa vs Refinada vs Reversa Refinada)
- Estado da Arte da Literatura:
    * S2 Heuristic (Polyakovskiy et al., 2014)
    * S5 Heuristic (Polyakovskiy et al., 2014)
    * PackIterative (Faulkner et al., 2015)
    * Co-Solver (Faulkner et al., 2015)
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
import src.solvers  # Trigger registration
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.core.evaluator_ttp1 import avaliar_ttp1


def executar_bateria():
    test_dir = Path("bateria_testes")
    instance_files = sorted(list(test_dir.glob("*.txt")))

    solvers_para_testar = [
        # Baselines
        "random",
        "greedy",
        # Heurística dos 37%
        "heuristic_37",
        "heuristic_37_refinada",
        # Queimada
        "queimada",
        "queimada_reversa",
        "queimada_refinada",
        "queimada_reversa_refinada",
        # Estado da Arte
        "s2_heuristic",
        "s5_heuristic",
        "pack_iterative",
        "cosolver",
    ]

    print(f"\n{'='*95}")
    print(f"BATERIA DE TESTES COMPARATIVA DO TRAVELING THIEF PROBLEM (TTP)")
    print(f"Instâncias encontradas: {len(instance_files)} | Solucionadores: {len(solvers_para_testar)}")
    print(f"{'='*95}\n")

    resultados_por_solver = defaultdict(lambda: {"lucros": [], "tempos_comp": [], "tempos_viagem": []})
    tabela_linhas = []

    for inst_file in instance_files:
        print(f"\n--- Instância: {inst_file.name} ---")
        instance = InstanceParser.carregar(str(inst_file))

        for s_name in solvers_para_testar:
            solver = SolverRegistry.get(s_name)

            t0 = time.perf_counter()
            solucao = solver.solve(instance)
            tempo_exec = (time.perf_counter() - t0) * 1000.0  # ms

            res = avaliar_ttp1(instance, solucao)
            lucro = res["lucro_liquido_G"]
            tempo_v = res["tempo_viagem_f"]
            peso = res["peso_acumulado"]
            qtd = res["qtd_itens_coletados"]

            resultados_por_solver[s_name]["lucros"].append(lucro)
            resultados_por_solver[s_name]["tempos_comp"].append(tempo_exec)
            resultados_por_solver[s_name]["tempos_viagem"].append(tempo_v)

            tabela_linhas.append({
                "instancia": inst_file.stem,
                "solver": s_name,
                "lucro_G": lucro,
                "tempo_viagem": tempo_v,
                "peso": peso,
                "qtd_itens": qtd,
                "tempo_ms": tempo_exec
            })

            print(f"  {s_name:<28} | Lucro G: {lucro:>10.2f} | T. Viagem: {tempo_v:>8.2f} | Itens: {qtd:>2} ({peso:>6.1f}kg) | {tempo_exec:>6.1f}ms")

    # Salva CSV com detalhes
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "comparativo_heuristicas.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("instancia;solver;lucro_liquido_G;tempo_viagem_f;peso_acumulado;qtd_itens;tempo_exec_ms\n")
        for row in tabela_linhas:
            f.write(f"{row['instancia']};{row['solver']};{row['lucro_G']:.4f};{row['tempo_viagem']:.4f};{row['peso']:.4f};{row['qtd_itens']};{row['tempo_ms']:.2f}\n")

    print(f"\n\n{'='*95}")
    print(f"{'RESUMO CONSOLIDADO (MÉDIAS GERAIS)':^95}")
    print(f"{'='*95}")
    print(f"{'Solucionador':<30} | {'Lucro Médio (G)':>16} | {'Tempo Médio Viagem':>18} | {'Tempo CPU Médio':>16}")
    print(f"{'-'*95}")

    solvers_ordenados = sorted(
        solvers_para_testar,
        key=lambda s: sum(resultados_por_solver[s]["lucros"]) / len(resultados_por_solver[s]["lucros"]),
        reverse=True
    )

    for s_name in solvers_ordenados:
        n_inst = len(resultados_por_solver[s_name]["lucros"])
        media_lucro = sum(resultados_por_solver[s_name]["lucros"]) / n_inst
        media_tempo_v = sum(resultados_por_solver[s_name]["tempos_viagem"]) / n_inst
        media_tempo_cpu = sum(resultados_por_solver[s_name]["tempos_comp"]) / n_inst
        print(f"{s_name:<30} | {media_lucro:>16.2f} | {media_tempo_v:>18.2f} | {media_tempo_cpu:>13.2f} ms")

    print(f"{'='*95}")
    print(f"Resultados completos salvos em: {csv_file}\n")


if __name__ == "__main__":
    executar_bateria()
