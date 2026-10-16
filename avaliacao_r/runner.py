"""
Módulo Principal de Execução da Avaliação de R.

Recursos:
- Interrupção Segura via Ctrl+C (SIGINT): conclui a instância em andamento e salva o estado sem corromper dados.
- Retomada Automática (Resume): identifica instâncias já processadas e continua a partir da próxima.
- Otimização e Paralelização configuráveis.
- Interface de Linha de Comando (CLI) intuitiva.
"""

import sys
import os
import signal
import time
import argparse
from pathlib import Path
from typing import List, Optional

# Garante inclusão dos diretórios no sys.path
_raiz = Path(__file__).resolve().parent.parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.scenarios.parser import InstanceParser
from src.core.models import TTPInstance
from avaliacao_r.config import AvaliacaoRConfig
from avaliacao_r.evaluator import avaliar_instancia_ttp, InstanciaAvaliacaoResultado
from avaliacao_r.relatorios import (
    obter_instancias_ja_calculadas,
    salvar_todos_resultados,
    imprimir_resumo_instancia
)


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
            print("Concluindo a execução da instância atual para salvar com segurança...")
            print("Pressione Ctrl+C novamente se desejar forçar encerramento imediato.")
            print("!" * 78 + "\n")
        else:
            print("\n❌ Encerramento forçado solicitado. Saindo imediatamente...")
            if callable(self._manipulador_original):
                self._manipulador_original(signum, frame)
            else:
                sys.exit(1)


def carregar_instancias_bateria(diretorio: Path) -> List[Path]:
    """Lista todos os arquivos de cenário (.txt) ordenados por dimensão crescente."""
    if not diretorio.exists():
        return []
    
    arquivos = list(diretorio.glob("*.txt"))
    # Ordena colocando instâncias menores primeiro para feedback rápido
    def prioridade(p: Path):
        nome = p.stem.lower()
        if "01" in nome or "02" in nome:
            return (0, nome)
        if "pequeno" in nome:
            return (1, nome)
        if "cluster_15" in nome:
            return (2, nome)
        if "medio" in nome:
            return (3, nome)
        return (4, nome)
        
    arquivos.sort(key=prioridade)
    return arquivos


def executar_bateria(config: Optional[AvaliacaoRConfig] = None, forcar: bool = False, instancia_especifica: Optional[str] = None):
    """
    Executa a avaliação para todos os cenários na pasta da bateria de testes.
    """
    if config is None:
        config = AvaliacaoRConfig()

    gerenciador = GerenciadorInterrupcao()
    
    print("\n" + "=" * 78)
    print("🚀 INICIANDO AVALIAÇÃO DO IMPACTO DE R NO TTP")
    print(f" • Diretório de Entrada: {config.diretorio_bateria}")
    print(f" • Diretório de Resultados: {config.diretorio_resultados}")
    print(f" • Limiar de Qualidade: {config.limiar_qualidade * 100:.0f}%")
    print(f" • Grade de Fatores R: {config.fatores_r}")
    print("=" * 78 + "\n")

    arquivos = carregar_instancias_bateria(config.diretorio_bateria)
    if not arquivos:
        print(f"❌ Nenhum arquivo .txt encontrado em: {config.diretorio_bateria}")
        return

    if instancia_especifica:
        arquivos = [a for a in arquivos if instancia_especifica.lower() in a.name.lower()]
        if not arquivos:
            print(f"❌ Nenhuma instância encontrada contendo o termo: '{instancia_especifica}'")
            return

    calculadas = set() if forcar else obter_instancias_ja_calculadas(config)
    total = len(arquivos)
    processadas_nesta_sessao = 0

    for idx, arq in enumerate(arquivos, 1):
        nome_instancia = arq.stem

        if nome_instancia in calculadas and not forcar:
            print(f"[{idx}/{total}] ⏩ Pulando '{nome_instancia}' (já calculada anteriormente).")
            continue

        print(f"\n[{idx}/{total}] ⏳ Processando '{nome_instancia}'...")
        try:
            # Carrega a instância usando o InstanceParser do simulador
            instance = InstanceParser.carregar(str(arq))
            
            # Executa a avaliação
            resultado = avaliar_instancia_ttp(instance, config)
            
            # Salva atomicamente no disco
            salvar_todos_resultados(resultado, config)
            calculadas.add(nome_instancia)
            processadas_nesta_sessao += 1
            
            # Imprime resumo
            imprimir_resumo_instancia(resultado)
            print(f"✅ Instância '{nome_instancia}' salva com sucesso!")

        except Exception as e:
            print(f"❌ Erro ao processar '{nome_instancia}': {e}")
            import traceback
            traceback.print_exc()

        # Verifica se houve solicitação de parada limpa por Ctrl+C
        if gerenciador.parada_solicitada:
            print("\n" + "=" * 78)
            print("🛑 PARADA SEGURA EXECUTADA")
            print(f"A instância atual foi concluída e salva com segurança.")
            print(f"Instâncias processadas nesta sessão: {processadas_nesta_sessao}")
            print(f"Total já acumulado: {len(calculadas)} de {total}")
            print("Ao executar novamente, o processador continuará de onde parou.")
            print("=" * 78 + "\n")
            return

    print("\n" + "=" * 78)
    print(f"🎉 BATERIA DE AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f" • Instâncias processadas nesta sessão: {processadas_nesta_sessao}")
    print(f" • Total de instâncias salvas: {len(calculadas)}")
    print(f" • Resultados disponíveis em: {config.diretorio_resultados}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Avaliação do Impacto de R no Travelling Thief Problem (TTP)")
    parser.add_argument("--bateria", type=str, help="Caminho para o diretório de instâncias de teste")
    parser.add_argument("--instancia", type=str, help="Executa apenas a instância que contém este nome")
    parser.add_argument("--timeout", type=float, default=120.0, help="Tempo limite por instância em segundos (padrão: 120s)")
    parser.add_argument("--limiar", type=float, default=0.90, help="Limiar de qualidade relativa (padrão: 0.90)")
    parser.add_argument("--forcar", action="store_true", help="Recalcula instâncias mesmo se já existirem no histórico")
    
    args = parser.parse_args()
    
    config = AvaliacaoRConfig()
    if args.bateria:
        config.diretorio_bateria = Path(args.bateria)
    if args.timeout:
        config.timeout_segundos_por_instancia = args.timeout
    if args.limiar:
        config.limiar_qualidade = args.limiar
        
    executar_bateria(config=config, forcar=args.forcar, instancia_especifica=args.instancia)


if __name__ == "__main__":
    main()
