"""
Orquestrador de Bateria para Instâncias Potencialmente Positivas (TTP1 e TTP2).

Este script:
1. Filtra as instâncias da pasta 'bateria_testes' utilizando a relaxação teórica:
   Mochila Divisível (Dantzig) - R * (Árvore Geradora Mínima / v_max) = Upper Bound (UB).
2. Seleciona EXCLUSIVAMENTE as instâncias cujo Limitante Superior (UB) é positivo (UB > 0),
   ou seja, instâncias onde é matematicamente possível obter lucro líquido positivo.
3. Executa a bateria de testes com todos os solucionadores (ou solvers selecionados),
   de forma idêntica ao 'executar_super_bateria.py', com suporte a paralelismo,
   retomada automática (Resume) e interrupção graciosa (Ctrl+C).
4. Grava os resultados em arquivos CSV DEDICADOS e SEPARADOS dos experimentos gerais:
   - Detalhes de execuções: results/bateria_lucrativas_detalhada.csv
   - Estatísticas consolidadas: results/comparativo_estatisticas_lucrativas.csv
   - Diagnóstico do filtro teórico: results/filtro_instancias_lucrativas.csv
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configura stdout para UTF-8 de forma segura no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona diretórios ao sys.path
_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

import src.solvers  # Trigger registration de todos os solvers
from src.core.models import Grafo, Item, TTPInstance
from src.runners.batch_runner import BatchRunner, _obter_metadados_tamanho
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry


class GerenciadorInterrupcaoFiltro:
    """Gerencia Ctrl+C (SIGINT) com salvamento seguro do progresso."""

    def __init__(self):
        self.interrompido = False
        self._manipulador_antigo = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._tratar_sinal)

    def _tratar_sinal(self, signum, frame):
        if not self.interrompido:
            self.interrompido = True
            print("\n" + "!" * 95)
            print("⚠️  INTERRUPÇÃO SOLICITADA (Ctrl+C)")
            print("Finalizando com segurança os processos em execução e salvando os dados no disco...")
            print("Pressione Ctrl+C novamente se desejar forçar a parada imediata.")
            print("!" * 95 + "\n")
        else:
            print("\n❌ Encerramento forçado solicitado. Saindo imediatamente...")
            if callable(self._manipulador_antigo):
                self._manipulador_antigo(signum, frame)
            else:
                sys.exit(1)


def calcular_mochila_divisivel(itens: List[Item], capacidade: float) -> Tuple[float, float]:
    """
    Calcula a mochila divisível (fracionária) ótima de forma gulosa.
    Retorna (valor_maximo_fracionario, peso_acumulado).
    """
    if not itens or capacidade <= 0.0:
        return 0.0, 0.0

    itens_ordenados = sorted(
        itens,
        key=lambda it: it.valor / it.peso if it.peso > 0 else float("inf"),
        reverse=True,
    )

    peso_acumulado = 0.0
    valor_acumulado = 0.0

    for it in itens_ordenados:
        if peso_acumulado + it.peso <= capacidade:
            peso_acumulado += it.peso
            valor_acumulado += it.valor
        else:
            if it.peso > 0:
                fracao = (capacidade - peso_acumulado) / it.peso
                valor_acumulado += fracao * it.valor
                peso_acumulado = capacidade
            break

    return valor_acumulado, peso_acumulado


def calcular_mst_prim(grafo: Grafo) -> float:
    """
    Calcula o custo da Árvore Geradora Mínima (MST) via algoritmo de Prim O(V^2).
    Representa o limitante inferior absoluto de qualquer ciclo hamiltoniano no grafo.
    """
    cidades = list(grafo.cidades.keys())
    n = len(cidades)
    if n <= 1:
        return 0.0

    start = cidades[0]
    min_dist = {c: grafo.distancia(start, c) for c in cidades}
    visitados = {start}
    custo_total_mst = 0.0

    while len(visitados) < n:
        u = min((c for c in cidades if c not in visitados), key=lambda c: min_dist[c])
        visitados.add(u)
        custo_total_mst += min_dist[u]

        for v in cidades:
            if v not in visitados:
                d = grafo.distancia(u, v)
                if d < min_dist[v]:
                    min_dist[v] = d

    return custo_total_mst


def analisar_instancia_lucro(instance: TTPInstance) -> dict:
    """
    Aplica a relaxação dual e calcula o limitante superior do lucro no TTP1.
    """
    n_cidades = len(instance.grafo.cidades)
    n_itens = len(instance.itens)
    cap = instance.capacidade_mochila
    r = instance.taxa_aluguel
    v_max = instance.v_max

    # 1. Mochila fracionária ótima
    kp_frac, kp_peso = calcular_mochila_divisivel(instance.itens, cap)

    # 2. Árvore Geradora Mínima
    custo_mst = calcular_mst_prim(instance.grafo)

    # 3. Tempo e custo mínimo de aluguel
    tempo_min_mst = custo_mst / v_max if v_max > 0 else 0.0
    custo_min_aluguel = r * tempo_min_mst

    # 4. Limitante Superior de Lucro Líquido
    ub_lucro = kp_frac - custo_min_aluguel
    potencialmente_lucrativa = ub_lucro > 0.0

    return {
        "arquivo": instance.nome,
        "cidades": n_cidades,
        "itens": n_itens,
        "capacidade_W": cap,
        "taxa_aluguel_R": r,
        "v_max": v_max,
        "valor_mochila_divisivel": kp_frac,
        "peso_mochila_divisivel": kp_peso,
        "custo_mst": custo_mst,
        "tempo_min_mst": tempo_min_mst,
        "custo_min_aluguel": custo_min_aluguel,
        "ub_lucro_liquido": ub_lucro,
        "potencialmente_lucrativa": potencialmente_lucrativa,
    }


def filtrar_instancias_positivas(
    bateria_dir: Path,
    output_filtro_csv: Path,
    max_cidades: int = 5000,
    incluir_quarentena: bool = False,
) -> Tuple[List[Path], List[dict]]:
    """
    Varre a bateria de instâncias, avalia a relaxação teórica e retorna apenas
    os caminhos das instâncias potencialmente lucrativas (UB > 0).
    """
    extensoes = ["*.ttp", "*.txt"]
    arquivos: List[Path] = []
    for ext in extensoes:
        for f in bateria_dir.glob(ext):
            if not f.is_file():
                continue
            if not incluir_quarentena and "quarentena" in f.parts:
                continue
            arquivos.append(f)

    # Remove duplicados preservando ordem estável
    vistos = set()
    arquivos_unicos = []
    for f in arquivos:
        if f.name not in vistos:
            vistos.add(f.name)
            arquivos_unicos.append(f)

    arquivos_unicos.sort(key=lambda a: (_obter_metadados_tamanho(a), a.name))

    print(f"\nVarrendo {len(arquivos_unicos)} instâncias para triagem teórica (UB > 0)...")

    arquivos_lucrativos: List[Path] = []
    todos_registros: List[dict] = []

    t0 = time.time()
    for idx, arq in enumerate(arquivos_unicos, 1):
        meta = _obter_metadados_tamanho(arq)
        if meta[0] > max_cidades:
            continue

        try:
            inst = InstanceParser.carregar(str(arq))
            res = analisar_instancia_lucro(inst)
            res["arquivo_origem"] = arq.name
            res["caminho_completo"] = str(arq)
            todos_registros.append(res)

            if res["potencialmente_lucrativa"]:
                arquivos_lucrativos.append(arq)
        except Exception as e:
            print(f"  ❌ Erro ao avaliar {arq.name}: {e}")

    t_filtro = time.time() - t0

    # Grava relatório do filtro em CSV
    output_filtro_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filtro_csv, "w", newline="", encoding="utf-8") as f:
        colunas = [
            "arquivo_origem", "cidades", "itens", "capacidade_W", "taxa_aluguel_R",
            "valor_mochila_divisivel", "custo_mst", "tempo_min_mst",
            "custo_min_aluguel", "ub_lucro_liquido", "potencialmente_lucrativa"
        ]
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        writer.writeheader()
        for r in todos_registros:
            writer.writerow({
                "arquivo_origem": r["arquivo_origem"],
                "cidades": r["cidades"],
                "itens": r["itens"],
                "capacidade_W": f"{r['capacidade_W']:.2f}",
                "taxa_aluguel_R": f"{r['taxa_aluguel_R']:.6f}",
                "valor_mochila_divisivel": f"{r['valor_mochila_divisivel']:.2f}",
                "custo_mst": f"{r['custo_mst']:.2f}",
                "tempo_min_mst": f"{r['tempo_min_mst']:.2f}",
                "custo_min_aluguel": f"{r['custo_min_aluguel']:.2f}",
                "ub_lucro_liquido": f"{r['ub_lucro_liquido']:.2f}",
                "potencialmente_lucrativa": "SIM" if r["potencialmente_lucrativa"] else "NAO",
            })

    total_avaliadas = len(todos_registros)
    total_lucrativas = len(arquivos_lucrativos)
    pct = (total_lucrativas / total_avaliadas * 100.0) if total_avaliadas > 0 else 0.0

    print(f"Triagem concluída em {t_filtro:.2f}s:")
    print(f" • Instâncias analisadas:            {total_avaliadas}")
    print(f" • Potencialmente Lucrativas (UB > 0): {total_lucrativas} ({pct:.1f}%)")
    print(f" • Garantidamente Negativas (UB <= 0): {total_avaliadas - total_lucrativas} ({100.0 - pct:.1f}%)")
    print(f" • Relatório da triagem salvo em:      {output_filtro_csv}\n")

    return arquivos_lucrativos, todos_registros


def executar_bateria_lucrativa(
    workers: Optional[int] = None,
    forcar: bool = False,
    solvers_lista: Optional[List[str]] = None,
    modelos_lista: Optional[List[str]] = None,
    output_solucoes: Optional[str] = None,
    output_stats: Optional[str] = None,
    output_filtro: Optional[str] = None,
    max_cidades: int = 5000,
    incluir_quarentena: bool = False,
    apenas_filtrar: bool = False,
):
    """
    Executa a triagem teórica e, em seguida, dispara a bateria de testes
    apenas para as instâncias potencialmente positivas.
    """
    gerenciador = GerenciadorInterrupcaoFiltro()
    num_workers = workers if workers is not None else min(os.cpu_count() or 4, 8)

    caminho_bateria = _raiz / "ttp_simulator" / "bateria_testes"
    pasta_resultados = _raiz / "results"
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    csv_detalhado = output_solucoes or str(pasta_resultados / "todos" / "lucrativas.csv")
    csv_estatisticas = output_stats or str(None)
    csv_filtro = output_filtro or str(pasta_resultados / "triagem" / "filtro_instancias_lucrativas.csv")

    modelos = modelos_lista or ["TTP1"]

    if not solvers_lista:
        solvers = [
            "random",
            "greedy",
            "heuristic_37",
            "heuristic_37_refinada",
            "heuristic_37_simetrica",
            "queimada",
            "queimada_sem_2opt",
            "queimada_com_2opt",
            "queimada_sem_afinidade",
            "queimada_com_afinidade",
            "queimada_regime_r",
            "queimada_reversa",
            "queimada_legacy",
            "s2_heuristic",
            "s5_heuristic",
            "pack_iterative",
            "cosolver",
            "local_search",
        ]
        for s_reg in SolverRegistry.listar():
            if s_reg not in solvers:
                solvers.append(s_reg)
    else:
        solvers = solvers_lista

    print("\n" + "=" * 95)
    print(f"{'BATERIA DE TESTES DE INSTÂNCIAS COM POTENCIAL DE LUCRO POSITIVO':^95}")
    print(f" • Diretório de Instâncias: {caminho_bateria}")
    print(f" • Workers Paralelos:       {num_workers}")
    print(f" • Resume Automático:       {'DESATIVADO (--forcar)' if forcar else 'ATIVADO'}")
    print(f" • Modelos Alvo:            {', '.join(modelos)}")
    print(f" • Solvers Registrados:     {len(solvers)} solucionadores")
    print(f" • CSV Detalhado:           {csv_detalhado}")
    print(f" • CSV Estatísticas:        {csv_estatisticas}")
    print("=" * 95 + "\n")

    # 1. Filtra as instâncias com UB > 0
    arquivos_alvo, registros = filtrar_instancias_positivas(
        bateria_dir=caminho_bateria,
        output_filtro_csv=Path(csv_filtro),
        max_cidades=max_cidades,
        incluir_quarentena=incluir_quarentena,
    )

    if apenas_filtrar:
        print("Opção --apenas-filtrar ativada. A bateria de simulações não será disparada.")
        return

    if not arquivos_alvo:
        print("⚠️ Nenhuma instância com potencial de lucro positivo encontrada.")
        return

    print(f"Iniciando bateria para as {len(arquivos_alvo)} instâncias potencialmente lucrativas...")
    print("-" * 95)
    for arq in arquivos_alvo:
        meta = _obter_metadados_tamanho(arq)
        print(f"  🎯 {arq.name:<45} (Cidades: {meta[0]:>5}, Itens: {meta[1]:>6})")
    print("-" * 95 + "\n")

    t_inicio_bateria = time.time()

    # 2. Configura e executa o BatchRunner exclusivamente para as instâncias lucrativas
    runner = BatchRunner(
        test_dir=str(caminho_bateria),
        solvers=solvers,
        models=modelos,
        runs=1,
        output_csv=csv_detalhado,
        base_seed=42,
        workers=num_workers,
        resume=(not forcar),
        output_stats_csv=csv_estatisticas,
    )

    # Substitui a listagem de arquivos do runner pelos arquivos filtrados ordenados por tamanho
    arquivos_alvo.sort(key=lambda a: (_obter_metadados_tamanho(a), a.name))
    runner._obter_arquivos_instancias = lambda: arquivos_alvo

    runner.run()

    t_total = time.time() - t_inicio_bateria
    print("\n" + "=" * 95)
    if gerenciador.interrompido:
        print(f"{'EXECUÇÃO INTERROMPIDA COM SEGURANÇA VIA CTRL+C':^95}")
        print("Todos os dados calculados foram preservados com sucesso no CSV.")
    else:
        print(f"{'BATERIA DE INSTÂNCIAS LUCRATIVAS CONCLUÍDA COM SUCESSO!':^95}")

    print(f" • Tempo total de execução: {t_total:.2f}s ({t_total/60:.2f}min)")
    print(f" • Resultados Detalhados:   {csv_detalhado}")
    print(f" • Resumo Estatístico:      {csv_estatisticas}")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Executa a Bateria de Testes TTP exclusivamente para instâncias potencialmente lucrativas (UB > 0)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos paralelos (padrão: CPU count, máx 8)",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Força reprocessamento de todas as instâncias filtradas, ignorando o CSV de resultados",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help="Lista de solvers separados por vírgula (padrão: todos os registrados)",
    )
    parser.add_argument(
        "--modelos",
        type=str,
        default=None,
        help="Modelos a avaliar separados por vírgula (padrão: TTP1; TTP2 é opcional)",
    )
    parser.add_argument(
        "--incluir-ttp2",
        action="store_true",
        help="Inclui o modelo TTP2 além do TTP1",
    )
    parser.add_argument(
        "--max-cidades",
        type=int,
        default=5000,
        help="Ignora instâncias com mais de N cidades (padrão: 5000)",
    )
    parser.add_argument(
        "--incluir-quarentena",
        action="store_true",
        help="Também avalia arquivos dentro da pasta quarentena",
    )
    parser.add_argument(
        "--apenas-filtrar",
        action="store_true",
        help="Apenas gera o relatório do filtro teórico de lucratividade sem executar a bateria",
    )
    parser.add_argument(
        "--output-solucoes",
        type=str,
        default=None,
        help="Caminho do CSV de soluções detalhadas (padrão: results/bateria_lucrativas_detalhada.csv)",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        default=None,
        help="Caminho do CSV estatístico consolidado (padrão: results/comparativo_estatisticas_lucrativas.csv)",
    )
    parser.add_argument(
        "--output-filtro",
        type=str,
        default=None,
        help="Caminho do CSV de triagem teórica (padrão: results/filtro_instancias_lucrativas.csv)",
    )

    args = parser.parse_args()

    solvers_lista = [s.strip() for s in args.solvers.split(",")] if args.solvers else None
    if args.modelos:
        modelos_lista = [m.strip() for m in args.modelos.split(",")]
    elif args.incluir_ttp2:
        modelos_lista = ["TTP1", "TTP2"]
    else:
        modelos_lista = ["TTP1"]

    executar_bateria_lucrativa(
        workers=args.workers,
        forcar=args.forcar,
        solvers_lista=solvers_lista,
        modelos_lista=modelos_lista,
        output_solucoes=args.output_solucoes,
        output_stats=args.output_stats,
        output_filtro=args.output_filtro,
        max_cidades=args.max_cidades,
        incluir_quarentena=args.incluir_quarentena,
        apenas_filtrar=args.apenas_filtrar,
    )


if __name__ == "__main__":
    main()
