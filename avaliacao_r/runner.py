"""
Módulo Principal de Execução da Avaliação de R.

Recursos:
- Interrupção Segura via Ctrl+C (SIGINT): conclui a instância em andamento e salva o estado sem corromper dados.
- Retomada Automática (Resume): identifica instâncias já processadas e continua a partir da próxima.
- Multiprocessamento Paralelo de Alto Desempenho (ProcessPoolExecutor).
- Interface de Linha de Comando (CLI) robusta com suporte a UTF-8 no Windows.
"""

import argparse
import concurrent.futures
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configura stdout para UTF-8 de forma segura no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Garante inclusão dos diretórios no sys.path
_raiz = Path(__file__).resolve().parent.parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from avaliacao_r.config import AvaliacaoRConfig
from avaliacao_r.evaluator import InstanciaAvaliacaoResultado, avaliar_instancia_ttp
from avaliacao_r.relatorios import (
    imprimir_resumo_instancia,
    obter_instancias_ja_calculadas,
    salvar_todos_resultados,
)
from src.core.models import TTPInstance
from src.scenarios.parser import InstanceParser


def _avaliar_instancia_worker(args: Tuple[str, dict]) -> Optional[InstanciaAvaliacaoResultado]:
    """
    Função de trabalho executada por processos paralelos isolados.
    Carrega a instância e executa a avaliação completa ao longo da grade de R.
    """
    file_path_str, config_dict = args
    try:
        import sys
        from pathlib import Path

        _r = Path(__file__).resolve().parent.parent
        _s = _r / "ttp_simulator"
        if str(_r) not in sys.path:
            sys.path.insert(0, str(_r))
        if str(_s) not in sys.path:
            sys.path.insert(0, str(_s))

        from avaliacao_r.config import AvaliacaoRConfig
        from avaliacao_r.evaluator import avaliar_instancia_ttp
        from src.scenarios.parser import InstanceParser

        config = AvaliacaoRConfig(**config_dict)
        instance = InstanceParser.carregar(file_path_str)
        return avaliar_instancia_ttp(instance, config)
    except Exception as e:
        print(f"[ERRO AVALIACAO R] {Path(file_path_str).name}: {e}")
        return None


class GerenciadorInterrupcao:
    """Gerencia sinais de interrupção (Ctrl+C / SIGINT) de forma segura e não destrutiva."""

    def __init__(self):
        self.parada_solicitada = False
        self._manipulador_original = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._capturar_interrupcao)

    def _capturar_interrupcao(self, signum, frame):
        if not self.parada_solicitada:
            self.parada_solicitada = True
            print("\n" + "!" * 78)
            print("⚠️  INTERRUPÇÃO SOLICITADA (Ctrl+C)")
            print("Concluindo as tarefas em andamento pelos workers para salvar com segurança...")
            print("Pressione Ctrl+C novamente se desejar forçar o encerramento imediato.")
            print("!" * 78 + "\n")
        else:
            print("\n❌ Encerramento forçado solicitado. Saindo imediatamente...")
            if callable(self._manipulador_original):
                self._manipulador_original(signum, frame)
            else:
                sys.exit(1)


def carregar_instancias_bateria(diretorio: Path, padrao: Optional[str] = None) -> List[Path]:
    """
    Lista todos os arquivos de instâncias (.txt e .ttp) da pasta de bateria,
    ignorando subdiretórios (como 'quarentena') e ordenando por tamanho crescente
    (menores primeiro) para feedback contínuo e rápido.
    """
    if not diretorio.exists():
        return []

    if padrao:
        arquivos_brutos = list(diretorio.glob(padrao))
    else:
        extensoes = ["*.txt", "*.ttp"]
        arquivos_brutos = []
        for ext in extensoes:
            arquivos_brutos.extend(diretorio.glob(ext))

    arquivos = []
    vistos = set()
    for f in arquivos_brutos:
        if not f.is_file():
            continue
        if "quarentena" in f.parts:
            continue
        if f.name not in vistos:
            vistos.add(f.name)
            arquivos.append(f)

    # Ordena por tamanho crescente (número de cidades, itens e tamanho)
    from src.runners.batch_runner import _obter_metadados_tamanho
    arquivos.sort(key=lambda a: (_obter_metadados_tamanho(a), a.name))
    return arquivos


def executar_bateria(
    config: Optional[AvaliacaoRConfig] = None,
    forcar: bool = False,
    instancia_especifica: Optional[str] = None,
    padrao: Optional[str] = None,
    workers: Optional[int] = None,
):
    """
    Executa a avaliação do impacto de R para as instâncias da bateria de testes.
    Suporta multiprocessamento paralelo, recuperação de progresso (resume) e interrupção segura.
    """
    if config is None:
        config = AvaliacaoRConfig()

    num_workers = workers if workers is not None else config.num_workers
    gerenciador = GerenciadorInterrupcao()

    print("\n" + "=" * 85)
    print(f"{'AVALIAÇÃO DO IMPACTO DE R NO TRAVELLING THIEF PROBLEM (TTP)':^85}")
    print(f" • Diretório de Entrada: {config.diretorio_bateria}")
    print(f" • Diretório de Resultados: {config.diretorio_resultados}")
    print(f" • Limiar de Qualidade: {config.limiar_qualidade * 100:.0f}%")
    print(f" • Workers Paralelos: {num_workers}")
    print("=" * 85 + "\n")

    arquivos = carregar_instancias_bateria(config.diretorio_bateria, padrao=padrao)
    if not arquivos:
        print(f"❌ Nenhum arquivo de instância encontrado em: {config.diretorio_bateria}")
        return

    if instancia_especifica:
        arquivos = [a for a in arquivos if instancia_especifica.lower() in a.name.lower()]
        if not arquivos:
            print(f"❌ Nenhuma instância encontrada contendo o termo: '{instancia_especifica}'")
            return

    # Filtro de segurança para evitar estouro de memória em instâncias de dezenas de milhares de cidades
    max_c = getattr(config, "max_cidades_avaliacao", 5000)
    from src.runners.batch_runner import _obter_metadados_tamanho
    arquivos_validos = []
    for a in arquivos:
        meta = _obter_metadados_tamanho(a)
        if meta[0] > max_c:
            print(f">> Pulando {a.name} ({meta[0]} cidades > limite de segurança {max_c})")
        else:
            arquivos_validos.append(a)
    arquivos = arquivos_validos

    calculadas = set() if forcar else obter_instancias_ja_calculadas(config)
    total_encontradas = len(arquivos)

    # Filtra instâncias pendentes com correspondência normalizada
    def ja_calculada(p: Path) -> bool:
        s = p.stem
        return (
            s in calculadas
            or s.lower() in calculadas
            or s.lower().replace("_", "").replace("-", "") in calculadas
        )

    pendentes = [a for a in arquivos if forcar or not ja_calculada(a)]
    total_pendentes = len(pendentes)

    if calculadas and not forcar:
        print(f"Modo Resume ativado: {len(calculadas)} instâncias já calculadas encontradas.")

    print(f"Total de instâncias para processar nesta sessão: {total_pendentes} de {total_encontradas}")

    if total_pendentes == 0:
        print("🎉 Todas as instâncias selecionadas já foram processadas anteriormente!")
        return

    t_inicio = time.time()
    processadas_sessao = 0

    # Prepara dicionário de configuração serializável
    config_dict = {
        "diretorio_raiz": str(config.diretorio_raiz),
        "diretorio_bateria": str(config.diretorio_bateria),
        "diretorio_resultados": str(config.diretorio_resultados),
        "fatores_r": config.fatores_r,
        "limiar_qualidade": config.limiar_qualidade,
        "max_cidades_exato_ttp": config.max_cidades_exato_ttp,
        "max_cidades_avaliacao": getattr(config, "max_cidades_avaliacao", 5000),
        "timeout_segundos_por_instancia": config.timeout_segundos_por_instancia,
        "num_workers": num_workers,
    }

    if num_workers <= 1:
        # Modo sequencial mono-processo
        for arq in pendentes:
            if gerenciador.parada_solicitada:
                break

            nome = arq.stem
            processadas_sessao += 1
            print(f"\n[{processadas_sessao}/{total_pendentes}] ⏳ Processando '{nome}'...")

            res = _avaliar_instancia_worker((str(arq), config_dict))
            if res:
                salvar_todos_resultados(res, config)
                calculadas.add(nome)
                imprimir_resumo_instancia(res)
                print(f"✅ Instância '{nome}' salva com sucesso!")
    else:
        # Modo multiprocesso paralelo de alto rendimento
        tarefas = [(str(arq), config_dict) for arq in pendentes]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futuros = {executor.submit(_avaliar_instancia_worker, t): t for t in tarefas}

            try:
                for futuro in concurrent.futures.as_completed(futuros):
                    t = futuros[futuro]
                    nome_arq = Path(t[0]).stem
                    processadas_sessao += 1

                    try:
                        res = futuro.result()
                        if res:
                            salvar_todos_resultados(res, config)
                            calculadas.add(nome_arq)

                            elapsed_total = time.time() - t_inicio
                            taxa = processadas_sessao / max(elapsed_total, 0.001)
                            eta = (total_pendentes - processadas_sessao) / max(taxa, 0.001)

                            print(
                                f"[{processadas_sessao:>3}/{total_pendentes}] "
                                f"{nome_arq:<25} | Rc: {res.r_crossover:>8.4f} | "
                                f"MaxGap: {res.max_gap_valor*100:>5.1f}% | "
                                f"{res.tempo_execucao_s:>5.1f}s | "
                                f"ETA: {int(eta // 60)}m{int(eta % 60):02d}s"
                            )
                    except Exception as e:
                        print(f"[FALHA NA INSTÂNCIA] {nome_arq}: {e}")

                    if gerenciador.parada_solicitada:
                        print("\nCancelando tarefas pendentes na fila dos workers...")
                        for f in futuros:
                            if not f.done():
                                f.cancel()
                        break
            except KeyboardInterrupt:
                print("\nCapturado KeyboardInterrupt. Aguardando workers finalizarem...")
                executor.shutdown(wait=False, cancel_futures=True)

    tempo_total = time.time() - t_inicio
    print("\n" + "=" * 85)
    if gerenciador.parada_solicitada:
        print("🛑 PARADA SEGURA EXECUTADA")
        print("Todas as instâncias concluídas até o momento foram salvas no histórico.")
    else:
        print("🎉 BATERIA DE AVALIAÇÃO DE R CONCLUÍDA COM SUCESSO!")

    print(f" • Instâncias processadas nesta sessão: {processadas_sessao}")
    print(f" • Total acumulado salvo: {len(calculadas)}")
    print(f" • Tempo total de execução: {tempo_total:.2f}s ({tempo_total/60:.2f}min)")
    print(f" • Relatórios salvos em: {config.diretorio_resultados}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Avaliação do Impacto de R no Travelling Thief Problem (TTP)")
    parser.add_argument("--bateria", type=str, help="Caminho para o diretório de instâncias de teste")
    parser.add_argument("--instancia", type=str, help="Executa apenas a instância que contém este nome")
    parser.add_argument("--padrao", type=str, help="Padrão glob de arquivos (ex: '*.txt', '*.ttp')")
    parser.add_argument("--workers", type=int, default=None, help="Número de processos paralelos")
    parser.add_argument("--timeout", type=float, default=120.0, help="Tempo limite por instância em segundos")
    parser.add_argument("--limiar", type=float, default=0.90, help="Limiar de qualidade relativa (padrão: 0.90)")
    parser.add_argument("--max-cidades", type=int, default=5000, help="Dimensão máxima de cidades a avaliar (padrão: 5000)")
    parser.add_argument("--max-cidades-exato", type=int, default=8, help="Dimensão máxima para B&B exato do TTP (padrão: 8)")
    parser.add_argument("--forcar", action="store_true", help="Recalcula instâncias mesmo se já existirem no histórico")

    args = parser.parse_args()

    config = AvaliacaoRConfig()
    if args.bateria:
        config.diretorio_bateria = Path(args.bateria)
    if args.timeout:
        config.timeout_segundos_por_instancia = args.timeout
    if args.limiar:
        config.limiar_qualidade = args.limiar
    if args.workers:
        config.num_workers = args.workers
    if args.max_cidades:
        config.max_cidades_avaliacao = args.max_cidades
    if args.max_cidades_exato:
        config.max_cidades_exato_ttp = args.max_cidades_exato

    executar_bateria(
        config=config,
        forcar=args.forcar,
        instancia_especifica=args.instancia,
        padrao=args.padrao,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
