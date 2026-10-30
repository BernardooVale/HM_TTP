import argparse
import sys
from pathlib import Path

# Add project root and ttp_simulator to path
_raiz = Path(__file__).resolve().parents[1]
_sim_path = Path(__file__).resolve().parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.runners.batch_runner import BatchRunner
from src.solvers.base import SolverRegistry
from src.utils.results_manager import detectar_contexto_e_historico
import src.solvers  # noqa


def main():
    parser = argparse.ArgumentParser(description='Bateria de Testes do TTP com Gerenciamento Inteligente de Histórico')
    parser.add_argument('--test-dir', default=str(_sim_path / 'bateria_testes'), help='Diretorio com arquivos de instancia (.txt/.ttp)')
    parser.add_argument('--solvers', default='all', help='Nomes dos solvers separados por virgula OU "all"')
    parser.add_argument('--models', default='TTP1', help='Nomes dos modelos separados por virgula (padrao: TTP1; TTP2 e opcional)')
    parser.add_argument('--runs', type=int, default=1, help='Numero de repeticoes por combinacao')
    parser.add_argument('--cenario', choices=['completo', 'equilibradas', 'lucrativas'], default='completo', help='Cenario da estrutura 3x3 (padrao: completo)')
    parser.add_argument('--output', default=None, help='Arquivo CSV de saida (padrao: detectado automaticamente por familia/cenario)')
    parser.add_argument('--history', default=None, help='Arquivo CSV de historico (padrao: detectado automaticamente por familia/cenario)')
    parser.add_argument('--no-history', action='store_true', help='Desativa consulta ao historico (forca recomputacao total)')
    parser.add_argument('--check-history', action='store_true', help='Apenas exibe relatorio de status do historico sem executar')
    parser.add_argument('--seed', type=int, default=42, help='Semente base')
    parser.add_argument('--workers', type=int, default=None, help='Numero de processos paralelos (padrao: auto CPU)')

    args = parser.parse_args()

    # Parse solvers
    if args.solvers.lower() == 'all':
        solvers = SolverRegistry.listar()
    else:
        solvers = [s.strip() for s in args.solvers.split(',')]

    # Parse models
    models = [m.strip() for m in args.models.split(',')]

    # Detecção inteligente de contexto (queimada vs. heuristica_37 vs. todos)
    contexto, caminho_hist_padrao, caminho_out_padrao = detectar_contexto_e_historico(
        solvers=solvers if args.solvers.lower() != 'all' else None,
        output_caminho=args.output,
        history_caminho=args.history if not args.no_history else None,
        cenario=args.cenario,
    )

    caminho_hist = str(caminho_hist_padrao) if not args.no_history else None
    caminho_out = str(caminho_out_padrao)

    print(f"\n[SISTEMA DE RESULTADOS]")
    print(f" • Contexto detectado: {contexto.upper()}")
    print(f" • Histórico conectado: {caminho_hist or 'DESATIVADO (--no-history)'}")
    print(f" • Arquivo de saída: {caminho_out}")

    runner = BatchRunner(
        test_dir=args.test_dir,
        solvers=solvers,
        models=models,
        runs=args.runs,
        output_csv=caminho_out,
        base_seed=args.seed,
        workers=args.workers,
        resume=not args.no_history,
        history_csv=caminho_hist,
        cenario=args.cenario,
        propagar_resultados=True,
    )

    if args.check_history:
        print(f"\n{'='*85}")
        print(f"{'STATUS DO HISTÓRICO PARA A CONFIGURAÇÃO SOLICITADA':^85}")
        print(f"{'='*85}")
        status = runner.verificar_status_historico()
        print(f"Instâncias: {status['total_instancias']} | Modelos: {len(models)} | Solvers: {len(solvers)}")
        print(f"Total de tarefas: {status['total_tarefas']} | Concluídas: {status['tarefas_concluidas']} ({status['pct_concluido']:.1f}%) | "
              f"Pendentes: {status['tarefas_pendentes']}")
        print(f"{'-'*85}")
        print(f"{'Solucionador':<35} | {'Feitas':>8} | {'Pendentes':>10} | {'Progresso':>10}")
        print(f"{'-'*85}")
        for s_name, st in sorted(status['status_solvers'].items(), key=lambda x: x[1]['pct'], reverse=True):
            print(f"{s_name:<35} | {st['feitas']:>8} | {st['pendentes']:>10} | {st['pct']:>9.1f}%")
        print(f"{'='*85}\n")
        return

    runner.run()


if __name__ == '__main__':
    main()
