"""
Orquestrador Unificado de Baterias de Testes do Traveling Thief Problem (TTP).

Executa sequencialmente:
1. Etapa 1 (ttp_simulator): Teste em lote de todas as instâncias para todas as soluções
   e variações de heurísticas (padrão: TTP1; TTP2 opcional via --incluir-ttp2), gerando
   métricas detalhadas e consolidação estatística (média, desvio padrão, mínimo, Q1, mediana, Q3, máximo, tempos).
   Suporta adição de novos solvers: ao incluir um novo solver, o runner detecta automaticamente
   que os anteriores já foram rodados e processa apenas o novo solver, incorporando-o ao CSV consolidado.
2. Etapa 2 (avaliacao_r): Avaliação do impacto de R em paralelo para determinar o
   ponto de cruzamento (Rc), gap de interdependência máximo (R_delta_max), faixas de 90%
   e a zona de forte acoplamento (intervalo ideal para R).

Recursos operacionais:
- Interrupção segura por teclado (Ctrl+C / SIGINT) com salvamento atômico do progresso.
- Retomada automática (Resume) de onde parou sem retrabalho.
- Multiprocessamento paralelo de alto rendimento configurável via --workers.
"""

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

# Configura stdout para UTF-8 de forma segura no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona caminhos do projeto ao sys.path
_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

import src.solvers  # Trigger registration de todos os solvers do ttp_simulator
from avaliacao_r.config import AvaliacaoRConfig
from avaliacao_r.runner import executar_bateria as executar_bateria_r
from src.runners.batch_runner import BatchRunner
from src.solvers.base import SolverRegistry


class GerenciadorInterrupcaoGlobal:
    """Gerencia interrupção por Ctrl+C (SIGINT) de forma limpa em todo o orquestrador."""

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


def executar_orquestrador(
    etapa: str = "todas",
    workers: Optional[int] = None,
    forcar: bool = False,
    instancias_padrao: Optional[str] = None,
    solvers_lista: Optional[List[str]] = None,
    modelos_lista: Optional[List[str]] = None,
    timeout_r: float = 120.0,
    output_solucoes: Optional[str] = None,
    output_stats: Optional[str] = None,
    diretorio_bateria: Optional[str] = None,
    incluir_alto_prejuizo: bool = False,
):
    gerenciador = GerenciadorInterrupcaoGlobal()
    num_workers = workers if workers is not None else min(os.cpu_count() or 4, 8)

    if diretorio_bateria:
        caminho_bateria = Path(diretorio_bateria).resolve()
    elif incluir_alto_prejuizo:
        caminho_bateria = _raiz / "ttp_simulator" / "bateria_testes" / "alto_prejuizo"
    else:
        caminho_bateria = _raiz / "ttp_simulator" / "bateria_testes"
    pasta_resultados = _raiz / "results"
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    csv_detalhado = output_solucoes or str(pasta_resultados / "todos" / "completo.csv")
    csv_estatisticas = output_stats or str(None)

    modelos = modelos_lista or ["TTP1"]

    # Se não fornecido solvers_lista, seleciona todos os solvers registrados
    if not solvers_lista:
        solvers = [
            # Baselines
            "random",
            "greedy",
            # Heurística dos 37% (Padrão Oficial Campeão + Variações)
            "heuristic_37",
            "heuristic_37_sem_refinamento",
            "heuristic_37_refinada",
            "heuristic_37_simetrica",
            # Queimada (Padrão Oficial Campeão + Variações de Ablação)
            "queimada",
            "queimada_kruskal_padrao",
            "queimada_backward_buracos",
            "queimada_backward_arestas",
            "queimada_backward_sem_2opt_buracos",
            "queimada_com_buracos",
            "queimada_sem_2opt",
            "queimada_com_2opt",
            "queimada_sem_afinidade",
            "queimada_com_afinidade",
            "queimada_regime_r",
            "queimada_reversa",
            "queimada_legacy",
            # Literatura
            "s2_heuristic",
            "s5_heuristic",
            "pack_iterative",
            "cosolver",
            "local_search",
        ]
        # Adiciona quaisquer outros solvers dinamicamente registrados no SolverRegistry
        for s_reg in SolverRegistry.listar():
            if s_reg not in solvers:
                solvers.append(s_reg)
    else:
        solvers = solvers_lista

    print("\n" + "=" * 95)
    print(f"{'ORQUESTRADOR UNIFICADO DE EXPERIMENTOS DO TRAVELING THIEF PROBLEM':^95}")
    print(f" • Diretório de Instâncias: {caminho_bateria}")
    print(f" • Modo / Etapas: {etapa.upper()}")
    print(f" • Processos Paralelos (Workers): {num_workers}")
    print(f" • Retomada Automática (Resume): {'DESATIVADA (--forcar)' if forcar else 'ATIVADA'}")
    print(f" • Solucionadores Alvo ({len(solvers)}): {', '.join(solvers[:6])}... (+{max(0, len(solvers)-6)} outros)")
    print(f" • Modelos: {', '.join(modelos)} (padrão: TTP1)")
    print("=" * 95 + "\n")

    t_inicio_total = time.time()

    # =========================================================================
    # ETAPA 1: BATERIA DE SOLUCIONADORES (ttp_simulator)
    # =========================================================================
    if etapa in ["todas", "solucoes"] and not gerenciador.interrompido:
        print("\n" + "#" * 95)
        print(f"{'INICIANDO ETAPA 1: BATERIA COMPARATIVA DE SOLUCIONADORES':^95}")
        print("#" * 95)

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

        if instancias_padrao:
            # Filtra instâncias pelo padrão fornecido
            arquivos_filtrados = [
                f for f in runner._obter_arquivos_instancias()
                if Path(f).match(instancias_padrao) or instancias_padrao in Path(f).name
            ]
            runner._obter_arquivos_instancias = lambda: arquivos_filtrados

        runner.run()

    # =========================================================================
    # ETAPA 2: BATERIA DE AVALIAÇÃO DE R (avaliacao_r)
    # =========================================================================
    if etapa in ["todas", "avaliacao_r"] and not gerenciador.interrompido:
        print("\n" + "#" * 95)
        print(f"{'INICIANDO ETAPA 2: AVALIAÇÃO DO IMPACTO DA TAXA DE ALUGUEL (R)':^95}")
        print("#" * 95)

        config_r = AvaliacaoRConfig(
            diretorio_raiz=_raiz,
            diretorio_bateria=caminho_bateria,
            diretorio_resultados=_raiz / "avaliacao_r" / "resultados",
            timeout_segundos_por_instancia=timeout_r,
            num_workers=num_workers,
        )

        executar_bateria_r(
            config=config_r,
            forcar=forcar,
            padrao=instancias_padrao,
            workers=num_workers,
        )

    t_total = time.time() - t_inicio_total

    print("\n" + "=" * 95)
    if gerenciador.interrompido:
        print(f"{'EXECUÇÃO INTERROMPIDA PELO USUÁRIO':^95}")
        print(f"Progresso preservado com segurança nos arquivos CSV.")
    else:
        print(f"{'EXECUÇÃO CONCLUÍDA COM SUCESSO':^95}")
    print(f" • Tempo total de execução: {t_total:.2f} segundos ({t_total/60:.1f} minutos)")
    print(f" • Resultados das Soluções:   {csv_detalhado}")
    print(f" • Estatísticas Consolidadas: {csv_estatisticas}")
    if etapa in ["todas", "avaliacao_r"]:
        print(f" • Resultados da Avaliação R: {_raiz / 'avaliacao_r' / 'resultados'}")
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Orquestrador unificado de experimentos do Traveling Thief Problem (TTP)."
    )
    parser.add_argument(
        "--etapa",
        choices=["todas", "solucoes", "avaliacao_r"],
        default="todas",
        help="Etapas a executar: 'todas', apenas 'solucoes' (Etapa 1), ou apenas 'avaliacao_r' (Etapa 2)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de processos paralelos (padrão: autodetectado com base nos núcleos da CPU)",
    )
    parser.add_argument(
        "--forcar",
        action="store_true",
        help="Força reprocessamento de todas as instâncias, ignorando execuções anteriores no CSV",
    )
    parser.add_argument(
        "--padrao",
        type=str,
        default=None,
        help="Filtro glob para instâncias (ex: '*.txt', '*.ttp', 'instancia_01*')",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help="Lista de solvers separados por vírgula (padrão: todos os solvers registrados)",
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
        help="Habilita a execução conjunta de TTP2 além de TTP1 (por padrão, apenas TTP1 é executado)",
    )
    parser.add_argument(
        "--timeout-r",
        type=float,
        default=120.0,
        help="Tempo limite por instância na avaliação de R em segundos (padrão: 120s)",
    )
    parser.add_argument(
        "--output-solucoes",
        type=str,
        default=None,
        help="Caminho do CSV detalhado de saída da bateria de soluções",
    )
    parser.add_argument(
        "--output-stats",
        type=str,
        default=None,
        help="Caminho do CSV consolidado de estatísticas da bateria de soluções",
    )
    parser.add_argument(
        "--diretorio-bateria",
        type=str,
        default=None,
        help="Diretório customizado de instâncias para execução (padrão: ttp_simulator/bateria_testes)",
    )
    parser.add_argument(
        "--incluir-alto-prejuizo",
        action="store_true",
        help="Executa explicitamente a bateria sobre a pasta de instâncias 'alto_prejuizo'",
    )

    args = parser.parse_args()

    solv_list = [s.strip() for s in args.solvers.split(",")] if args.solvers else None

    if args.modelos:
        mod_list = [m.strip() for m in args.modelos.split(",")]
    elif args.incluir_ttp2:
        mod_list = ["TTP1", "TTP2"]
    else:
        mod_list = ["TTP1"]

    executar_orquestrador(
        etapa=args.etapa,
        workers=args.workers,
        forcar=args.forcar,
        instancias_padrao=args.padrao,
        solvers_lista=solv_list,
        modelos_lista=mod_list,
        timeout_r=args.timeout_r,
        output_solucoes=args.output_solucoes,
        output_stats=args.output_stats,
        diretorio_bateria=args.diretorio_bateria,
        incluir_alto_prejuizo=args.incluir_alto_prejuizo,
    )


if __name__ == "__main__":
    main()
