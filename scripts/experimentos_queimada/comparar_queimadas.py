"""
Script de Comparação Direta entre a Nova Queimada (Backward Edges) e as demais Heurísticas.

Executa:
1. queimada (Kruskal padrão - base)
2. queimada_backward_arestas (Nova: encadeamento reverso de arestas mais quentes com 2-opt)
3. queimada_backward_buracos (Nova: com preenchimento térmico de buracos)
4. queimada_backward_sem_2opt
5. queimada_backward_sem_2opt_buracos
6. heuristic_37_refinada (Campeã dos 37%)
7. pack_iterative (Literatura)

Uso:
    python scripts/experimentos_queimada/comparar_queimadas.py
    python scripts/experimentos_queimada/comparar_queimadas.py --instancias 512.ttp instancia_01.txt cenario_medio_20c.txt
"""

import argparse
import sys
import time
from pathlib import Path

# Configura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona ttp_simulator e raiz ao path
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
from src.solvers.base import SolverRegistry


def comparar(instancias_nomes=None, modelo="TTP1"):
    pasta_testes = _sim_path / "bateria_testes"

    if not instancias_nomes:
        instancias_nomes = [
            "instancia_01.txt",
            "instancia_02.txt",
            "cenario_pequeno_10c.txt",
            "cenario_medio_20c.txt",
            "cenario_cluster_25c.txt",
            "512.ttp",
        ]

    solvers_alvo = [
        "queimada",
        "queimada_backward_arestas",
        "queimada_backward_buracos",
        "queimada_backward_sem_2opt",
        "queimada_backward_sem_2opt_buracos",
        "heuristic_37_refinada",
        "pack_iterative",
    ]

    avaliador = avaliar_ttp1 if modelo == "TTP1" else avaliar_ttp2

    print("\n" + "=" * 115)
    print(f"{f'COMPARATIVO DE HEURÍSTICAS ({modelo})':^115}")
    print("=" * 115)

    for nome in instancias_nomes:
        caminho = pasta_testes / nome
        if not caminho.exists():
            caminho = pasta_testes / "alto_prejuizo" / nome
        if not caminho.exists():
            print(f"⚠️ Instância {nome} não encontrada.")
            continue

        inst = InstanceParser.carregar(str(caminho))
        print(f"\n📂 Instância: {nome:<30} (Cidades: {len(inst.grafo.cidades)}, Itens: {len(inst.itens)}, R={inst.taxa_aluguel})")
        print("-" * 115)
        print(f"{'Solucionador':<30} | {'Lucro Líquido (G)':>18} | {'Tempo Viagem':>14} | {'Itens':>8} | {'Peso (kg)':>12} | {'Tempo CPU':>10}")
        print("-" * 115)

        resultados = []
        for s_nome in solvers_alvo:
            solver = SolverRegistry.get(s_nome)
            if hasattr(solver, "modelo"):
                setattr(solver, "modelo", modelo)

            t0 = time.perf_counter()
            sol = solver.solve(inst)
            cpu_s = time.perf_counter() - t0

            res = avaliador(inst, sol)
            resultados.append((s_nome, res, cpu_s))

        # Ordena pelo maior lucro líquido G
        resultados.sort(key=lambda x: x[1]["lucro_liquido_G"], reverse=True)

        for rank, (s_nome, res, cpu_s) in enumerate(resultados, 1):
            g = res["lucro_liquido_G"]
            t_viagem = res["tempo_viagem_f"]
            itens_cnt = res["qtd_itens_coletados"]
            peso = res.get("peso_acumulado") or res.get("peso_acumulado_mochila", 0.0)
            medalha = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else "   "))

            destaque = s_nome
            if "backward" in s_nome:
                destaque = f"★ {s_nome}"

            print(f"{medalha}{destaque:<28} | {g:>18.2f} | {t_viagem:>14.2f} | {itens_cnt:>8} | {peso:>12.1f} | {cpu_s*1000:>8.1f}ms")

    print("\n" + "=" * 115 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparativo de Heurísticas TTP")
    parser.add_argument("--instancias", nargs="*", default=None, help="Nomes dos arquivos de instâncias")
    parser.add_argument("--modelo", choices=["TTP1", "TTP2"], default="TTP1", help="Modelo TTP (padrão: TTP1; TTP2 é opcional)")
    args = parser.parse_args()

    comparar(instancias_nomes=args.instancias, modelo=args.modelo)
