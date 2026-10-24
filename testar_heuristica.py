"""
CLI Rápido e Sem Fofoca para Testar Solucionadores do Travelling Thief Problem (TTP).

Uso Rápido:
    # 1. Ver todos os solvers disponíveis
    python testar_heuristica.py --listar

    # 2. Testar seu solver em uma instância padrão rápida
    python testar_heuristica.py --solver queimada

    # 3. Testar seu solver e comparar lado a lado com os melhores da literatura
    python testar_heuristica.py --solver queimada --comparar

    # 4. Testar em uma instância específica
    python testar_heuristica.py --solver pack_iterative --instancia 512.ttp

    # 5. Avaliar seu solver em uma bateria rápida de instâncias
    python testar_heuristica.py --solver heuristic_37_refinada --bateria
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configura stdout para UTF-8 de forma segura no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona diretórios do projeto ao sys.path
_raiz = Path(__file__).resolve().parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

import src.solvers  # Trigger registration de todos os solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.core.models import SolucaoTTP, TTPInstance
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry


def resolver_caminho_instancia(nome_ou_caminho: str) -> Optional[Path]:
    """Localiza o arquivo de instância mesmo que o usuário passe apenas o nome base."""
    caminho = Path(nome_ou_caminho)
    if caminho.is_file():
        return caminho.resolve()

    pasta_bateria = _sim_path / "bateria_testes"
    candidatos = [
        pasta_bateria / nome_ou_caminho,
        pasta_bateria / f"{nome_ou_caminho}.txt",
        pasta_bateria / f"{nome_ou_caminho}.ttp",
        pasta_bateria / "alto_prejuizo" / nome_ou_caminho,
        pasta_bateria / "alto_prejuizo" / f"{nome_ou_caminho}.txt",
        pasta_bateria / "alto_prejuizo" / f"{nome_ou_caminho}.ttp",
    ]
    for c in candidatos:
        if c.is_file():
            return c.resolve()

    return None


def validar_solucao(instancia: TTPInstance, solucao: SolucaoTTP) -> Tuple[bool, List[str]]:
    """Verifica se a solução gerada é factível sob as regras matemáticas do TTP."""
    avisos = []
    n = instancia.num_cities

    # 1. Verifica tour
    if not solucao.tour:
        avisos.append("Tour está vazio!")
    else:
        if solucao.tour[0] != 1:
            avisos.append(f"O tour deve iniciar na cidade 1 (iniciou em {solucao.tour[0]}).")
        if len(solucao.tour) != n:
            avisos.append(f"O tour possui {len(solucao.tour)} cidades, mas a instância possui {n}.")
        cidades_unicas = set(solucao.tour)
        if len(cidades_unicas) != n:
            avisos.append(f"Cidades repetidas ou faltantes no tour ({len(cidades_unicas)}/{n} únicas).")

    # 2. Verifica capacidade da mochila
    mapa_itens = {it.id: it for it in instancia.itens}
    peso_total = 0.0
    ids_coletados = []

    if hasattr(solucao, "itens_coletados_ids") and solucao.itens_coletados_ids():
        ids_coletados = solucao.itens_coletados_ids()
    elif isinstance(solucao.plano_coleta, dict):
        for k, v in solucao.plano_coleta.items():
            if isinstance(v, (list, tuple, set)):
                ids_coletados.extend(v)
            elif isinstance(v, bool) and v:
                ids_coletados.append(k)

    for item_id in ids_coletados:
        it = mapa_itens.get(item_id)
        if it:
            peso_total += it.peso
        else:
            avisos.append(f"Item com ID {item_id} não existe na instância.")

    if peso_total > instancia.capacidade_mochila + 1e-6:
        excesso = peso_total - instancia.capacidade_mochila
        avisos.append(
            f"Capacidade da mochila estourada! Peso: {peso_total:.2f} kg > Limite: {instancia.capacidade_mochila:.2f} kg (+{excesso:.2f} kg)."
        )

    return (len(avisos) == 0, avisos)


def executar_solver_em_instancia(
    solver_nome: str,
    instancia_caminho: Path,
    modelo: str = "TTP1",
) -> Dict[str, Any]:
    """Executa um solver em uma instância e retorna as métricas avaliadas."""
    instancia = InstanceParser.carregar(str(instancia_caminho))
    solver = SolverRegistry.get(solver_nome)

    if hasattr(solver, "modelo"):
        setattr(solver, "modelo", modelo)

    t0 = time.perf_counter()
    solucao = solver.solve(instancia)
    tempo_cpu_ms = (time.perf_counter() - t0) * 1000.0

    avaliador = avaliar_ttp1 if modelo == "TTP1" else avaliar_ttp2
    resultado = avaliador(instancia, solucao)

    valida, problemas = validar_solucao(instancia, solucao)

    return {
        "solver": solver_nome,
        "instancia": instancia,
        "caminho": instancia_caminho,
        "solucao": solucao,
        "resultado": resultado,
        "tempo_cpu_ms": tempo_cpu_ms,
        "valida": valida,
        "problemas": problemas,
    }


def imprimir_tabela_comparativa(
    instancia_caminho: Path,
    solvers_para_rodar: List[str],
    modelo: str = "TTP1",
):
    """Executa múltiplos solvers na mesma instância e exibe tabela comparativa com medalhas."""
    instancia = InstanceParser.carregar(str(instancia_caminho))

    print("\n" + "=" * 110)
    print(f"{'COMPARATIVO DE SOLUCIONADORES':^110}")
    print(f" • Instância: {instancia_caminho.name} ({len(instancia.grafo.cidades)} cidades, {len(instancia.itens)} itens, R={instancia.taxa_aluguel})")
    print(f" • Modelo:    {modelo}")
    print("=" * 110)
    print(
        f"{'Pos':<4} | {'Solucionador':<26} | {'Lucro Líquido (G)':>17} | {'Tempo Viagem':>13} | "
        f"{'Itens':>8} | {'Peso (kg)':>11} | {'Tempo CPU':>10}"
    )
    print("-" * 110)

    resultados = []
    for s_nome in solvers_para_rodar:
        try:
            r = executar_solver_em_instancia(s_nome, instancia_caminho, modelo)
            resultados.append(r)
        except Exception as e:
            print(f"⚠️ Erro ao executar {s_nome}: {e}")

    # Ordena pelo maior lucro líquido G
    resultados.sort(key=lambda x: x["resultado"]["lucro_liquido_G"], reverse=True)

    for rank, r in enumerate(resultados, 1):
        s_nome = r["solver"]
        res = r["resultado"]
        g = res["lucro_liquido_G"]
        t_viagem = res["tempo_viagem_f"]
        qtd = res["qtd_itens_coletados"]
        peso = res.get("peso_acumulado") or res.get("peso_acumulado_mochila", 0.0)
        cpu_ms = r["tempo_cpu_ms"]

        medalha = "🥇 " if rank == 1 else ("🥈 " if rank == 2 else ("🥉 " if rank == 3 else f"#{rank:<2}"))
        print(f"{medalha:<4} | {s_nome:<26} | {g:>17.2f} | {t_viagem:>13.2f} | {qtd:>8} | {peso:>11.1f} | {cpu_ms:>8.1f}ms")

    print("=" * 110 + "\n")


def listar_solvers():
    """Exibe todos os solvers registrados organizados por categoria."""
    solvers = SolverRegistry.listar()
    print("\n" + "=" * 80)
    print(f"{'SOLUCIONADORES REGISTRADOS NO FRAMEWORK TTP':^80}")
    print("=" * 80)

    categorias = {
        "Baselines": ["random", "greedy"],
        "Heurística dos 37%": ["heuristic_37", "heuristic_37_refinada", "heuristic_37_simetrica"],
        "Heurística da Queimada": [s for s in solvers if s.startswith("queimada")],
        "Literatura Clássica": ["s2_heuristic", "s5_heuristic", "pack_iterative", "cosolver", "local_search"],
    }

    solvers_categorizados = set()
    for cat, lista in categorias.items():
        presentes = [s for s in lista if s in solvers]
        if presentes:
            print(f"\n📂 {cat}:")
            for s in presentes:
                solvers_categorizados.add(s)
                print(f"   • {s}")

    outros = [s for s in solvers if s not in solvers_categorizados]
    if outros:
        print("\n📂 Solucionadores Personalizados / Outros:")
        for s in outros:
            print(f"   • {s}")

    print(f"\nTotal cadastrados: {len(solvers)} solucionadores.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLI direto e sem fofoca para testar heurísticas no Traveling Thief Problem (TTP)."
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=None,
        help="Nome do solucionador a testar (ex: queimada, heuristic_37_refinada, pack_iterative)",
    )
    parser.add_argument(
        "--instancia",
        type=str,
        default="cenario_pequeno_10c.txt",
        help="Arquivo de instância para avaliação (padrão: cenario_pequeno_10c.txt)",
    )
    parser.add_argument(
        "--modelo",
        choices=["TTP1", "TTP2"],
        default="TTP1",
        help="Modelo matemático do TTP (padrão: TTP1; literatura usa quase exclusivamente TTP1)",
    )
    parser.add_argument(
        "--comparar",
        action="store_true",
        help="Executa o solver selecionado lado a lado com os principais baselines da literatura",
    )
    parser.add_argument(
        "--bateria",
        action="store_true",
        help="Executa em uma seleção rápida de instâncias representativas",
    )
    parser.add_argument(
        "--listar",
        action="store_true",
        help="Lista todos os solvers registrados disponíveis",
    )
    parser.add_argument(
        "--detalhes",
        action="store_true",
        help="Exibe a rota completa e os IDs de itens coletados",
    )

    args = parser.parse_args()

    if args.listar:
        listar_solvers()
        return

    if not args.solver:
        print("❌ Informe o solver a testar com --solver <nome> ou use --listar para ver opções.")
        print("Exemplo: python testar_heuristica.py --solver queimada --comparar")
        return

    solvers_disponiveis = SolverRegistry.listar()
    if args.solver not in solvers_disponiveis:
        print(f"❌ Solucionador '{args.solver}' não encontrado.")
        print(f"Solucionadores disponíveis: {', '.join(solvers_disponiveis[:8])}... (veja todos com --listar)")
        return

    # Modo 1: Bateria Rápida
    if args.bateria:
        instancias_bateria = [
            "instancia_01.txt",
            "cenario_pequeno_10c.txt",
            "cenario_cluster_15c.txt",
            "cenario_medio_20c.txt",
            "cenario_cluster_25c.txt",
            "512.ttp",
        ]
        print("\n" + "=" * 105)
        print(f"{f'BATERIA RÁPIDA DE TESTES - SOLVER: {args.solver} ({args.modelo})':^105}")
        print("=" * 105)
        print(f"{'Instância':<26} | {'Cidades':>7} | {'Itens':>7} | {'Lucro (G)':>14} | {'Tempo Viagem':>13} | {'Itens':>6} | {'CPU':>8}")
        print("-" * 105)

        for nome in instancias_bateria:
            caminho = resolver_caminho_instancia(nome)
            if not caminho:
                continue
            r = executar_solver_em_instancia(args.solver, caminho, args.modelo)
            res = r["resultado"]
            inst = r["instancia"]
            print(
                f"{nome:<26} | {inst.num_cities:>7} | {len(inst.itens):>7} | "
                f"{res['lucro_liquido_G']:>14.2f} | {res['tempo_viagem_f']:>13.2f} | "
                f"{res['qtd_itens_coletados']:>6} | {r['tempo_cpu_ms']:>6.1f}ms"
            )
        print("=" * 105 + "\n")
        return

    # Modo 2: Comparação lado a lado
    caminho_inst = resolver_caminho_instancia(args.instancia)
    if not caminho_inst:
        print(f"❌ Instância '{args.instancia}' não foi encontrada.")
        return

    if args.comparar:
        solvers_comp = [
            args.solver,
            "greedy",
            "heuristic_37_refinada",
            "pack_iterative",
            "queimada",
        ]
        # Remove duplicados preservando ordem
        vistos = set()
        solvers_unicos = []
        for s in solvers_comp:
            if s in solvers_disponiveis and s not in vistos:
                solvers_unicos.append(s)
                vistos.add(s)

        imprimir_tabela_comparativa(caminho_inst, solvers_unicos, args.modelo)
        return

    # Modo 3: Execução Individual
    print("\n" + "=" * 80)
    print(f"{'TESTE INDIVIDUAL DE SOLUCIONADOR DO TTP':^80}")
    print("=" * 80)
    print(f" • Solucionador: {args.solver}")
    print(f" • Instância:    {caminho_inst.name} ({caminho_inst})")
    print(f" • Modelo:       {args.modelo}")
    print("-" * 80)

    try:
        r = executar_solver_em_instancia(args.solver, caminho_inst, args.modelo)
    except Exception as e:
        print(f"\n❌ Erro na execução do solver: {e}\n")
        import traceback
        traceback.print_exc()
        return

    res = r["resultado"]
    inst = r["instancia"]

    cap_usada = res.get("peso_acumulado") or res.get("peso_acumulado_mochila", 0.0)
    cap_pct = (cap_usada / inst.capacidade_mochila * 100.0) if inst.capacidade_mochila > 0 else 0.0

    print("📊 RESULTADOS:")
    print(f"   • Lucro Líquido (Função Objetivo G): {res['lucro_liquido_G']:>14.4f}")
    print(f"   • Tempo Total de Viagem (f):          {res['tempo_viagem_f']:>14.4f}")
    print(f"   • Valor Bruto dos Itens (g):          {res.get('valor_bruto_itens_g', 0.0):>14.2f}")
    print(f"   • Peso Coletado na Mochila:           {cap_usada:>10.2f} / {inst.capacidade_mochila:.2f} kg ({cap_pct:.1f}%)")
    print(f"   • Quantidade de Itens Coletados:      {res['qtd_itens_coletados']:>10} / {len(inst.itens)}")
    print(f"   • Tempo de CPU:                       {r['tempo_cpu_ms']:>10.2f} ms")

    if r["valida"]:
        print("\n✅ Factibilidade: Solução VÁLIDA (todas as restrições foram respeitadas).")
    else:
        print("\n⚠️ Factibilidade: VIOLAÇÕES DETECTADAS:")
        for prob in r["problemas"]:
            print(f"   - {prob}")

    if args.detalhes:
        print("\n🧭 DETALHES DA SOLUÇÃO:")
        rota = r["solucao"].rota
        print(f"   • Tour ({len(rota)} cidades): {rota[:20]}{'...' if len(rota) > 20 else ''}")
        itens_coletados = r["solucao"].itens_coletados_ids()
        print(f"   • Itens Roubados ({len(itens_coletados)}): {itens_coletados[:30]}{'...' if len(itens_coletados) > 30 else ''}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
