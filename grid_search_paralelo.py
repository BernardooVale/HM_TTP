"""
Grid Search Paralelo de Alto Desempenho para a Heurística da Queimada no TTP.

Recursos:
- Multiprocessamento nativo com número configurável de workers (--workers).
- Suporte a múltiplos espaços de busca:
    * 'completo': 2.880 combinações únicas (incluindo backward_edges, 2-opt, afinidade térmica e otimização de buracos).
    * 'padrao': 288 combinações clássicas do grid_search_queimada.py.
    * 'rapido': combinações construtivas sem 2-opt global / sem Drop & Add pesado (ultra-rápido).
- Suporte a seleção flexível de instâncias de bateria_testes (todas, leves, sintéticas ou arquivos específicos).
- Poda automática de configurações redundantes (ex: afinidade térmica em walk, ordens em knapsack exato).
- Cache inteligente de instâncias por processo worker.
- Barra de progresso com taxa de throughput (execuções/segundo) e ETA.
- Ranking Top N das melhores configurações, análise de impacto marginal de parâmetros e exportação em CSV.

Uso:
    python grid_search_paralelo.py --help
    python grid_search_paralelo.py --instancias sinteticas --espaco completo
    python grid_search_paralelo.py --instancias leves --workers 8 --espaco completo
"""

import argparse
import concurrent.futures
import csv
import itertools
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configura stdout para UTF-8 no Windows de forma segura
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona caminhos do projeto ao sys.path
_raiz = Path(__file__).resolve().parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.solvers.queimada import QueimadaSolver


# Cache por processo worker para evitar recarga repetida de disco
_WORKER_CACHE: Dict[str, Any] = {}


def _worker_executar_config(args: Tuple[str, Dict[str, Any], int, str]) -> Dict[str, Any]:
    """
    Função de trabalho executada em paralelo pelos processos filhos.
    """
    caminho_arquivo, cfg, cfg_id, modelo = args

    if caminho_arquivo not in _WORKER_CACHE:
        _WORKER_CACHE[caminho_arquivo] = InstanceParser.carregar(caminho_arquivo)

    instance = _WORKER_CACHE[caminho_arquivo]
    nome_inst = Path(caminho_arquivo).name

    solver = QueimadaSolver(
        temperatura=cfg["temperatura"],
        dissipacao=cfg["dissipacao"],
        direcao=cfg["direcao"],
        metodo_construcao=cfg["metodo_construcao"],
        afinidade_termica=cfg.get("afinidade_termica", True),
        refinamento_rota=cfg.get("refinamento_rota", True),
        otimizar_buracos=cfg.get("otimizar_buracos", False),
        selecao=cfg["selecao"],
        ordem_selecao=cfg["ordem_selecao"],
        refinamento=cfg["refinamento"],
        modelo=modelo,
    )

    t0 = time.perf_counter()
    solucao = solver.solve(instance)
    tempo_cpu_ms = (time.perf_counter() - t0) * 1000.0

    if modelo == "TTP1":
        res = avaliar_ttp1(instance, solucao)
    else:
        res = avaliar_ttp2(instance, solucao)

    return {
        "cfg_id": cfg_id,
        "instancia": nome_inst,
        "lucro_G": res["lucro_liquido_G"],
        "tempo_viagem": res["tempo_viagem_f"],
        "peso": res["peso_acumulado"],
        "qtd_itens": res["qtd_itens_coletados"],
        "tempo_cpu_ms": tempo_cpu_ms,
    }


def gerar_espaco_configuracoes(tipo_espaco: str = "completo", deduplicar: bool = True) -> List[Dict[str, Any]]:
    """
    Gera o conjunto de configurações para o grid search.
    """
    temperaturas = ["total_value", "knapsack_exact", "knapsack_greedy"]
    dissipacoes = ["adjacent_only", "dijkstra", "bfs"]
    direcoes = ["calor_final", "calor_inicio"]
    selecoes = ["greedy_relative", "knapsack_relative"]
    ordens = ["densidade", "reversa", "sequencial"]

    if tipo_espaco == "padrao":
        # 288 configurações clássicas do grid_search_queimada.py
        metodos = ["walk", "kruskal"]
        refinamentos = [False, True]
        configuracoes = []
        for temp, diss, direcao, sel, metodo, ordem, ref in itertools.product(
            temperaturas, dissipacoes, direcoes, selecoes, metodos, ordens, refinamentos
        ):
            if deduplicar and sel == "knapsack_relative" and ordem != "densidade":
                continue
            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": True,
                "refinamento_rota": True,
                "otimizar_buracos": False,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": ref,
            })
        return configuracoes

    elif tipo_espaco == "rapido":
        # Versão ultra-rápida: compara métodos construtivos e otimização de buracos sem Drop&Add/2opt pesado
        metodos = ["walk", "kruskal", "backward_edges"]
        buracos_opts = [False, True]
        afinidades = [False, True]
        configuracoes = []
        for temp, diss, direcao, metodo, afinidade, buracos, sel, ordem in itertools.product(
            temperaturas, dissipacoes, direcoes, metodos, afinidades, buracos_opts, selecoes, ordens
        ):
            if deduplicar:
                if sel == "knapsack_relative" and ordem != "densidade":
                    continue
                if metodo == "walk" and afinidade:
                    continue
            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": afinidade,
                "refinamento_rota": False,
                "otimizar_buracos": buracos,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": False,
            })
        return configuracoes

    elif tipo_espaco == "completo":
        # Espaço completo com todas as novas opções estratégicas da Queimada
        metodos = ["walk", "kruskal", "backward_edges"]
        afinidades = [False, True]
        rotas_2opt = [False, True]
        buracos_opts = [False, True]
        refinamentos = [False, True]

        configuracoes = []
        for temp, diss, direcao, metodo, afin, r_2opt, buraco, sel, ordem, ref in itertools.product(
            temperaturas, dissipacoes, direcoes, metodos, afinidades, rotas_2opt, buracos_opts, selecoes, ordens, refinamentos
        ):
            if deduplicar:
                # Quando selecao é knapsack_relative, a ordem de varredura gulosa é irrelevante
                if sel == "knapsack_relative" and ordem != "densidade":
                    continue
                # Quando método é walk (guloso sequencial), afinidade térmica não é usada
                if metodo == "walk" and afin:
                    continue

            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": afin,
                "refinamento_rota": r_2opt,
                "otimizar_buracos": buraco,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": ref,
            })
        return configuracoes

    else:
        raise ValueError(f"Espaço de busca inválido: {tipo_espaco}. Escolha entre 'completo', 'padrao', 'rapido'.")


def resolver_instancias(
    opcao_instancias: str,
    ignorar_a280_pesada: bool = False,
    arquivos_especificos: Optional[List[str]] = None,
) -> List[Path]:
    """Retorna a lista de caminhos absolutos das instâncias a processar."""
    pasta_testes = _raiz / "ttp_simulator" / "bateria_testes"
    if not pasta_testes.exists():
        pasta_testes = _raiz / "bateria_testes"

    if arquivos_especificos:
        caminhos = []
        for nome in arquivos_especificos:
            p = pasta_testes / nome if not Path(nome).is_absolute() else Path(nome)
            if p.exists():
                caminhos.append(p)
            else:
                print(f"⚠️ Instância ignorada (não encontrada): {nome}")
        return caminhos

    todos_arquivos = [f for f in pasta_testes.iterdir() if f.is_file() and f.suffix in (".txt", ".ttp")]

    if opcao_instancias == "sinteticas":
        arquivos = [f for f in todos_arquivos if f.suffix == ".txt"]
    elif opcao_instancias == "leves":
        arquivos = [f for f in todos_arquivos if not f.name.startswith("a280_")]
    elif opcao_instancias == "todas":
        arquivos = todos_arquivos
    else:
        # Tenta interpretar como padrão glob ou nome único
        arquivos = list(pasta_testes.glob(opcao_instancias))
        if not arquivos:
            arquivos = todos_arquivos

    if ignorar_a280_pesada:
        arquivos = [f for f in arquivos if not (f.name.startswith("a280_n2790") or f.name.startswith("a280_n1395"))]

    return sorted(arquivos, key=lambda f: (f.stat().st_size, f.name))


def executar_grid_search(
    espaco: str = "completo",
    instancias_modo: str = "leves",
    instancias_lista: Optional[List[str]] = None,
    ignorar_a280_pesada: bool = True,
    workers: Optional[int] = None,
    modelo: str = "TTP1",
    top_n: int = 15,
    arquivo_csv: Optional[str] = None,
):
    """Executa o Grid Search paralelo completo."""
    workers = workers or os.cpu_count() or 4
    instancias = resolver_instancias(
        opcao_instancias=instancias_modo,
        ignorar_a280_pesada=ignorar_a280_pesada,
        arquivos_especificos=instancias_lista,
    )

    if not instancias:
        print("❌ Nenhuma instância encontrada para processamento. Abortando.")
        return

    configuracoes = gerar_espaco_configuracoes(tipo_espaco=espaco, deduplicar=True)
    total_instancias = len(instancias)
    total_configs = len(configuracoes)
    total_tarefas = total_instancias * total_configs

    print("\n" + "=" * 115)
    print(f"{'GRID SEARCH PARALELO DE ALTO RENDIMENTO - HEURÍSTICA DA QUEIMADA':^115}")
    print("=" * 115)
    print(f"• Espaço de busca:      {espaco.upper()} ({total_configs:,} configurações únicas)")
    print(f"• Instâncias alvo:      {total_instancias} arquivos de teste")
    print(f"• Modelo avaliado:      {modelo}")
    print(f"• Processamento:        {workers} workers paralelos (Multiprocessamento nativo)")
    print(f"• Total de execuções:   {total_tarefas:,} tarefas")
    print("=" * 115 + "\n")

    # Monta a lista completa de tarefas
    tarefas = []
    for cfg_id, cfg in enumerate(configuracoes):
        for inst_path in instancias:
            tarefas.append((str(inst_path.resolve()), cfg, cfg_id, modelo))

    # Estruturas de agregação
    lucros_por_config: Dict[int, List[float]] = defaultdict(list)
    tempos_v_por_config: Dict[int, List[float]] = defaultdict(list)
    tempos_cpu_por_config: Dict[int, List[float]] = defaultdict(list)
    registros_csv: List[Dict[str, Any]] = []

    concluidas = 0
    t_inicio = time.perf_counter()
    ultimo_print = t_inicio

    print("🚀 Iniciando processamento paralelo em lote...\n")

    try:
        # Usa chunksize para minimizar custo de IPC
        chunksize = max(1, min(64, total_tarefas // (workers * 4)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            resultados = executor.map(_worker_executar_config, tarefas, chunksize=chunksize)

            for res in resultados:
                concluidas += 1
                c_id = res["cfg_id"]
                lucros_por_config[c_id].append(res["lucro_G"])
                tempos_v_por_config[c_id].append(res["tempo_viagem"])
                tempos_cpu_por_config[c_id].append(res["tempo_cpu_ms"])

                cfg_ref = configuracoes[c_id]
                registros_csv.append({
                    "config_id": c_id,
                    "instancia": res["instancia"],
                    "lucro_G": res["lucro_G"],
                    "tempo_viagem": res["tempo_viagem"],
                    "peso": res["peso"],
                    "qtd_itens": res["qtd_itens"],
                    "tempo_cpu_ms": res["tempo_cpu_ms"],
                    **cfg_ref,
                })

                agora = time.perf_counter()
                if agora - ultimo_print >= 1.0 or concluidas == total_tarefas:
                    ultimo_print = agora
                    decorrido = agora - t_inicio
                    taxa = concluidas / max(decorrido, 0.001)
                    restante = (total_tarefas - concluidas) / max(taxa, 0.001)
                    pct = (concluidas / total_tarefas) * 100.0
                    print(
                        f"\rProgresso: [{concluidas:,}/{total_tarefas:,}] ({pct:5.1f}%) | "
                        f"Velocidade: {taxa:6.1f} exec/s | "
                        f"Decorrido: {decorrido:5.1f}s | "
                        f"ETA: {restante:5.1f}s",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupção manual detectada (Ctrl+C). Agregando resultados concluídos até o momento...")

    t_total = time.perf_counter() - t_inicio
    print(f"\n\n{'='*115}")
    print(f"✨ Concluído em {t_total:.2f}s ({concluidas:,} execuções finalizadas | Média: {concluidas / max(t_total, 0.001):.1f} exec/s)")
    print("=" * 115)

    if not lucros_por_config:
        print("Nenhum resultado para exibir.")
        return

    # Ranking das melhores configurações
    ranking = []
    for c_id, lista_lucros in lucros_por_config.items():
        media_lucro = sum(lista_lucros) / len(lista_lucros)
        media_viagem = sum(tempos_v_por_config[c_id]) / len(tempos_v_por_config[c_id])
        media_cpu = sum(tempos_cpu_por_config[c_id]) / len(tempos_cpu_por_config[c_id])
        ranking.append((media_lucro, media_viagem, media_cpu, c_id))

    ranking.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*115}")
    print(f"{f'TOP {min(top_n, len(ranking))} MELHORES CONFIGURAÇÕES DA HEURÍSTICA DA QUEIMADA ({modelo})':^115}")
    print("=" * 115)
    header = (
        f"{'Rank':<5} | {'Construção':<14} | {'Buracos':<7} | {'2-Opt':<5} | "
        f"{'Afin':<5} | {'Direção':<11} | {'Seleção':<15} | {'Ordem':<10} | {'Refin':<5} | {'Lucro Médio (G)':>14}"
    )
    print(header)
    print("-" * 115)

    for rank_idx, (med_g, med_tv, med_cpu, c_id) in enumerate(ranking[:top_n], 1):
        c = configuracoes[c_id]
        buraco_str = "SIM ★" if c.get("otimizar_buracos") else "NÃO"
        opt_str = "SIM" if c.get("refinamento_rota") else "NÃO"
        afin_str = "SIM" if c.get("afinidade_termica") else "NÃO"
        ref_str = "SIM" if c.get("refinamento") else "NÃO"

        print(
            f"{rank_idx:<5} | {c['metodo_construcao']:<14} | {buraco_str:<7} | {opt_str:<5} | "
            f"{afin_str:<5} | {c['direcao']:<11} | {c['selecao']:<15} | {c['ordem_selecao']:<10} | {ref_str:<5} | {med_g:>14.2f}"
        )

    # Análise de Impacto Marginal por Parâmetro
    print(f"\n{'='*115}")
    print(f"{'ANÁLISE DE IMPACTO MARGINAL POR PARÂMETRO (LUCRO MÉDIO G)':^115}")
    print("=" * 115)

    parametros_analise = [
        "metodo_construcao",
        "otimizar_buracos",
        "refinamento_rota",
        "afinidade_termica",
        "direcao",
        "selecao",
        "ordem_selecao",
        "refinamento",
        "temperatura",
        "dissipacao",
    ]

    for param in parametros_analise:
        acum_param = defaultdict(list)
        for c_id, lista_lucros in lucros_por_config.items():
            val = configuracoes[c_id].get(param, "default")
            acum_param[val].extend(lista_lucros)

        print(f"\n🔹 Parâmetro: {param.upper()}")
        itens_param = []
        for val, vals_lucro in acum_param.items():
            media_val = sum(vals_lucro) / len(vals_lucro) if vals_lucro else 0.0
            itens_param.append((val, media_val))
        itens_param.sort(key=lambda x: x[1], reverse=True)
        for val, m_val in itens_param:
            print(f"   • {str(val):<22} : {m_val:>14.2f}")

    # Salva CSV se solicitado ou por padrão em results/
    if arquivo_csv is None:
        caminho_results = _raiz / "results"
        caminho_results.mkdir(parents=True, exist_ok=True)
        arquivo_csv = str(caminho_results / f"grid_search_queimada_{espaco}.csv")

    if registros_csv:
        try:
            with open(arquivo_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(registros_csv[0].keys()))
                writer.writeheader()
                writer.writerows(registros_csv)
            print(f"\n📁 Resultados completos e detalhados exportados com sucesso para:\n   {arquivo_csv}")
        except Exception as e:
            print(f"\n⚠️ Falha ao salvar CSV: {e}")

    print("=" * 115 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid Search Paralelo da Heurística da Queimada no Traveling Thief Problem (TTP)"
    )
    parser.add_argument(
        "--espaco",
        choices=["completo", "padrao", "rapido"],
        default="completo",
        help="Espaço de configurações: 'completo' (2.880 únicas), 'padrao' (288), ou 'rapido' (sem 2-opt/Drop&Add pesado).",
    )
    parser.add_argument(
        "--instancias",
        default="leves",
        help="Instâncias: 'leves' (31 instâncias de até 100 cidades), 'sinteticas' (7 cenários .txt), 'todas' (34 arquivos), ou arquivos específicos.",
    )
    parser.add_argument(
        "--ignorar-a280-pesada",
        action="store_true",
        default=True,
        help="Ignora instâncias gigantes com > 1.000 itens (a280_n2790 e a280_n1395) para evitar gargalo computacional.",
    )
    parser.add_argument(
        "--incluir-todas-a280",
        action="store_false",
        dest="ignorar_a280_pesada",
        help="Força a inclusão das instâncias a280 completas mesmo com tempo estendido.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Número de processos de CPU paralelos (Padrão: {os.cpu_count() or 4}).",
    )
    parser.add_argument(
        "--modelo",
        choices=["TTP1", "TTP2"],
        default="TTP1",
        help="Modelo do TTP a avaliar: TTP1 (taxa de aluguel R) ou TTP2 (degradação temporal).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Quantidade de melhores configurações para listar no ranking final.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Caminho personalizado para salvar o relatório CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Se --instancias contiver múltiplos arquivos separados por espaço
    inst_lista = None
    if args.instancias not in ("leves", "sinteticas", "todas"):
        # Pode ser arquivo específico ou lista
        partes = args.instancias.split()
        if len(partes) > 1 or (len(partes) == 1 and ("." in partes[0] or "*" in partes[0])):
            inst_lista = partes

    executar_grid_search(
        espaco=args.espaco,
        instancias_modo=args.instancias,
        instancias_lista=inst_lista,
        ignorar_a280_pesada=args.ignorar_a280_pesada,
        workers=args.workers,
        modelo=args.modelo,
        top_n=args.top,
        arquivo_csv=args.csv,
    )
