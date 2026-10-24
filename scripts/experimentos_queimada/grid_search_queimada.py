"""
Grid Search completo de todas as variações da Heurística da Queimada
e comparação com as heurísticas de estado da arte do TTP.
"""

import itertools
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

# Configura caminhos para importação
_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

import src.solvers  # Trigger registration
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.solvers.queimada import QueimadaSolver


def executar_grid_search_queimada():
    test_dir = _sim_path / "bateria_testes"
    instance_files = sorted(list(test_dir.glob("*.txt")))

    # Espaço de parâmetros da Queimada
    temperaturas = ["total_value", "knapsack_exact", "knapsack_greedy"]
    dissipacoes = ["adjacent_only", "dijkstra", "bfs"]
    direcoes = ["calor_final", "calor_inicio"]
    selecoes = ["greedy_relative", "knapsack_relative"]
    metodos_construcao = ["walk", "kruskal"]
    ordens_selecao = ["densidade", "reversa", "sequencial"]
    refinamentos = [False, True]

    # Gera todas as combinações únicas
    configuracoes = []
    for temp, diss, direcao, sel, metodo, ordem, ref in itertools.product(
        temperaturas, dissipacoes, direcoes, selecoes, metodos_construcao, ordens_selecao, refinamentos
    ):
        # Quando selecao == 'knapsack_relative', ordem_selecao não afeta a busca knapsack 0/1
        if sel == "knapsack_relative" and ordem != "densidade":
            continue

        cfg = {
            "temperatura": temp,
            "dissipacao": diss,
            "direcao": direcao,
            "selecao": sel,
            "metodo_construcao": metodo,
            "ordem_selecao": ordem,
            "refinamento": ref,
        }
        configuracoes.append(cfg)

    print(f"\n{'='*95}")
    print(f"GRID SEARCH COMPLETO DA HEURÍSTICA DA QUEIMADA ({len(configuracoes)} CONFIGURAÇÕES)")
    print(f"Instâncias de teste: {len(instance_files)} | Total de execuções da Queimada: {len(configuracoes) * len(instance_files)}")
    print(f"{'='*95}\n")

    # Estruturas para acumular estatísticas
    lucros_por_config = defaultdict(list)
    tempos_viagem_por_config = defaultdict(list)
    tempos_cpu_por_config = defaultdict(list)

    # Detalhes de cada execução para exportar em CSV
    linhas_csv = []

    # Carrega instâncias na memória
    instancias = [(f.stem, InstanceParser.carregar(str(f))) for f in instance_files]

    inicio_total = time.perf_counter()

    for idx_inst, (nome_inst, instance) in enumerate(instancias, 1):
        print(f"[{idx_inst}/{len(instancias)}] Avaliando em '{nome_inst}' ({len(instance.grafo.cidades)} cidades, {len(instance.itens)} itens)...")
        t_inst_0 = time.perf_counter()

        for idx_cfg, cfg in enumerate(configuracoes):
            solver = QueimadaSolver(**cfg)
            t0 = time.perf_counter()
            solucao = solver.solve(instance)
            t_ms = (time.perf_counter() - t0) * 1000.0

            res = avaliar_ttp1(instance, solucao)
            lucro = res["lucro_liquido_G"]
            tempo_v = res["tempo_viagem_f"]
            peso = res["peso_acumulado"]
            qtd = res["qtd_itens_coletados"]

            lucros_por_config[idx_cfg].append(lucro)
            tempos_viagem_por_config[idx_cfg].append(tempo_v)
            tempos_cpu_por_config[idx_cfg].append(t_ms)

            linhas_csv.append({
                "instancia": nome_inst,
                "config_id": idx_cfg,
                **cfg,
                "lucro_G": lucro,
                "tempo_viagem": tempo_v,
                "peso": peso,
                "qtd_itens": qtd,
                "tempo_ms": t_ms
            })

        t_inst_total = time.perf_counter() - t_inst_0
        print(f"      Concluído em {t_inst_total:.2f}s ({t_inst_total/len(configuracoes)*1000:.2f}ms/execução)")

    tempo_total_grid = time.perf_counter() - inicio_total
    print(f"\nGrid Search concluído em {tempo_total_grid:.2f} segundos!\n")

    # Salva CSV na pasta central results/
    results_dir = _raiz / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_file = results_dir / "grid_search_queimada.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        headers = [
            "instancia", "config_id", "temperatura", "dissipacao", "direcao",
            "selecao", "metodo_construcao", "ordem_selecao", "refinamento",
            "lucro_G", "tempo_viagem", "peso", "qtd_itens", "tempo_ms"
        ]
        f.write(";".join(headers) + "\n")
        for row in linhas_csv:
            line = [
                str(row["instancia"]),
                str(row["config_id"]),
                str(row["temperatura"]),
                str(row["dissipacao"]),
                str(row["direcao"]),
                str(row["selecao"]),
                str(row["metodo_construcao"]),
                str(row["ordem_selecao"]),
                str(row["refinamento"]),
                f"{row['lucro_G']:.4f}",
                f"{row['tempo_viagem']:.4f}",
                f"{row['peso']:.4f}",
                str(row["qtd_itens"]),
                f"{row['tempo_ms']:.2f}",
            ]
            f.write(";".join(line) + "\n")

    print(f"Resultados detalhados gravados em '{csv_file}'.\n")

    # Executa também os solvers de referência para a comparação final
    solvers_referencia = [
        "cosolver",
        "pack_iterative",
        "heuristic_37_refinada",
        "heuristic_37",
        "s2_heuristic",
        "s5_heuristic",
        "greedy",
        "random",
    ]
    referencia_lucros = defaultdict(list)
    referencia_tempos = defaultdict(list)
    referencia_cpus = defaultdict(list)

    for nome_inst, instance in instancias:
        for s_name in solvers_referencia:
            s = SolverRegistry.get(s_name)
            t0 = time.perf_counter()
            sol = s.solve(instance)
            t_ms = (time.perf_counter() - t0) * 1000.0
            r = avaliar_ttp1(instance, sol)
            referencia_lucros[s_name].append(r["lucro_liquido_G"])
            referencia_tempos[s_name].append(r["tempo_viagem_f"])
            referencia_cpus[s_name].append(t_ms)

    # Análise dos resultados
    medias_queimada = []
    for idx_cfg, cfg in enumerate(configuracoes):
        lucros = lucros_por_config[idx_cfg]
        media_g = sum(lucros) / len(lucros)
        media_viagem = sum(tempos_viagem_por_config[idx_cfg]) / len(lucros)
        media_cpu = sum(tempos_cpu_por_config[idx_cfg]) / len(lucros)
        medias_queimada.append({
            "idx": idx_cfg,
            "cfg": cfg,
            "media_g": media_g,
            "media_viagem": media_viagem,
            "media_cpu": media_cpu
        })

    medias_queimada.sort(key=lambda x: x["media_g"], reverse=True)

    print("=" * 110)
    print(f"{'TOP 10 MELHORES CONFIGURAÇÕES DA HEURÍSTICA DA QUEIMADA':^110}")
    print("=" * 110)
    print(f"{'Rank':<5} | {'Temp':<14} | {'Dissip':<14} | {'Direção':<12} | {'Construção':<10} | {'Seleção':<16} | {'Ordem':<10} | {'Refin':<6} | {'Lucro Médio G':>14}")
    print("-" * 110)
    for rank, item in enumerate(medias_queimada[:10], 1):
        c = item["cfg"]
        print(f"{rank:<5} | {c['temperatura']:<14} | {c['dissipacao']:<14} | {c['direcao']:<12} | {c['metodo_construcao']:<10} | {c['selecao']:<16} | {c['ordem_selecao']:<10} | {str(c['refinamento']):<6} | {item['media_g']:>14.2f}")
    print("=" * 110)

    # Análise marginal por parâmetro
    def calcular_efeito_parametro(param_nome):
        valores = defaultdict(list)
        for item in medias_queimada:
            val = item["cfg"][param_nome]
            valores[val].append(item["media_g"])
        return {val: sum(lst) / len(lst) for val, lst in valores.items()}

    print("\n" + "=" * 80)
    print(f"{'ANÁLISE DE IMPACTO MARGINAL POR PARÂMETRO (MÉDIA DE LUCRO G)':^80}")
    print("=" * 80)
    for param in ["metodo_construcao", "direcao", "dissipacao", "temperatura", "selecao", "ordem_selecao", "refinamento"]:
        efeitos = calcular_efeito_parametro(param)
        print(f"Parâmetro: {param.upper()}")
        for val, media in sorted(efeitos.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {str(val):<20}: {media:>12.2f}")
    print("=" * 80)

    # Comparativo: Melhor Queimada vs Estado da Arte
    melhor_queimada = medias_queimada[0]

    print("\n" + "=" * 90)
    print(f"{'COMPARATIVO: MELHOR CONFIGURAÇÃO DA QUEIMADA vs ESTADO DA ARTE':^90}")
    print("=" * 90)
    print(f"{'Solucionador':<35} | {'Lucro Médio (G)':>16} | {'Tempo Médio Viagem':>18} | {'Tempo CPU':>12}")
    print("-" * 90)

    tabela_final = [
        ("cosolver (Faulkner 2015)", sum(referencia_lucros["cosolver"]) / len(instancias), sum(referencia_tempos["cosolver"]) / len(instancias), sum(referencia_cpus["cosolver"]) / len(instancias)),
        ("heuristic_37_refinada", sum(referencia_lucros["heuristic_37_refinada"]) / len(instancias), sum(referencia_tempos["heuristic_37_refinada"]) / len(instancias), sum(referencia_cpus["heuristic_37_refinada"]) / len(instancias)),
        ("pack_iterative (Faulkner 2015)", sum(referencia_lucros["pack_iterative"]) / len(instancias), sum(referencia_tempos["pack_iterative"]) / len(instancias), sum(referencia_cpus["pack_iterative"]) / len(instancias)),
        (f"QUEIMADA (MELHOR #{melhor_queimada['idx']})", melhor_queimada["media_g"], melhor_queimada["media_viagem"], melhor_queimada["media_cpu"]),
        ("heuristic_37 (original)", sum(referencia_lucros["heuristic_37"]) / len(instancias), sum(referencia_tempos["heuristic_37"]) / len(instancias), sum(referencia_cpus["heuristic_37"]) / len(instancias)),
        ("s2_heuristic (Polyakovskiy 2014)", sum(referencia_lucros["s2_heuristic"]) / len(instancias), sum(referencia_tempos["s2_heuristic"]) / len(instancias), sum(referencia_cpus["s2_heuristic"]) / len(instancias)),
        ("s5_heuristic (Polyakovskiy 2014)", sum(referencia_lucros["s5_heuristic"]) / len(instancias), sum(referencia_tempos["s5_heuristic"]) / len(instancias), sum(referencia_cpus["s5_heuristic"]) / len(instancias)),
        ("QUEIMADA (BASELINE DEFAULT)", medias_queimada[-1]["media_g"], medias_queimada[-1]["media_viagem"], medias_queimada[-1]["media_cpu"]),
        ("greedy", sum(referencia_lucros["greedy"]) / len(instancias), sum(referencia_tempos["greedy"]) / len(instancias), sum(referencia_cpus["greedy"]) / len(instancias)),
        ("random", sum(referencia_lucros["random"]) / len(instancias), sum(referencia_tempos["random"]) / len(instancias), sum(referencia_cpus["random"]) / len(instancias)),
    ]

    tabela_final.sort(key=lambda x: x[1], reverse=True)

    for nome, g, tv, cpu in tabela_final:
        print(f"{nome:<35} | {g:>16.2f} | {tv:>18.2f} | {cpu:>9.2f} ms")
    print("=" * 90)


if __name__ == "__main__":
    executar_grid_search_queimada()
