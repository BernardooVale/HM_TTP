import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.runners.batch_runner import BatchRunner
from src.solvers.base import SolverRegistry

# Import all solvers to trigger registration
import src.solvers  # noqa

def main():
    parser = argparse.ArgumentParser(description='Bateria de Testes do TTP')
    parser.add_argument('--test-dir', default='bateria_testes/', help='Diretorio com arquivos de instancia .txt')
    parser.add_argument('--solvers', default='all', help='Nomes dos solvers separados por virgula OU "all"')
    parser.add_argument('--models', default='TTP1', help='Nomes dos modelos separados por virgula (TTP1,TTP2)')
    parser.add_argument('--runs', type=int, default=1, help='Numero de repeticoes por combinacao')
    parser.add_argument('--output', default='results/bateria_completa.csv', help='Arquivo CSV de saida')
    parser.add_argument('--seed', type=int, default=42, help='Semente base')
    
    args = parser.parse_args()
    
    # Parse solvers
    if args.solvers.lower() == 'all':
        solvers = SolverRegistry.listar()
    else:
        solvers = [s.strip() for s in args.solvers.split(',')]
        
    # Parse models
    models = [m.strip() for m in args.models.split(',')]
    
    runner = BatchRunner(
        test_dir=args.test_dir,
        solvers=solvers,
        models=models,
        runs=args.runs,
        output_csv=args.output,
        base_seed=args.seed
    )
    
    runner.run()

if __name__ == '__main__':
    main()
