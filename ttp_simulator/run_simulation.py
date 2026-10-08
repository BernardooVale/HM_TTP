import argparse
import sys
import time
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.utils.csv_logger import CSVLogger

# Import all solvers to trigger registration
import src.solvers  # noqa

def main():
    parser = argparse.ArgumentParser(description='Simulacao individual do TTP')
    parser.add_argument('--instance', required=True, help='Caminho do arquivo de instancia .txt')
    parser.add_argument('--solver', required=True, help='Nome do solucionador')
    parser.add_argument('--model', default='TTP1', choices=['TTP1', 'TTP2'], help='Modelo TTP')
    parser.add_argument('--seed', type=int, default=42, help='Semente aleatoria')
    parser.add_argument('--output', default='results/resultado_unico.csv', help='Arquivo CSV de saida')
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load instance
    instance = InstanceParser.carregar(args.instance)
    
    # Get solver
    solver = SolverRegistry.get(args.solver)
    
    # Solve
    start_time = time.time()
    solucao = solver.solve(instance)
    elapsed = time.time() - start_time
    
    # Evaluate
    if args.model == 'TTP1':
        resultado = avaliar_ttp1(instance, solucao)
    else:
        resultado = avaliar_ttp2(instance, solucao)
    
    # Log
    logger = CSVLogger(args.output)
    registro = CSVLogger.criar_registro(
        arquivo_instancia=args.instance,
        modelo_ttp=args.model,
        solucionador=solver.nome,
        seed=args.seed,
        tempo_computacional=elapsed,
        resultado=resultado,
        instance=instance
    )
    logger.registrar(registro)
    
    print(f"Simulacao concluida: {solver.nome} em {args.instance}")
    print(f"  Modelo: {args.model}")
    print(f"  Lucro liquido (G): {resultado['lucro_liquido_G']:.4f}")
    print(f"  Tempo viagem (f): {resultado['tempo_viagem_f']:.4f}")
    print(f"  Valor bruto (g): {resultado['valor_bruto_itens_g']:.4f}")
    print(f"  Itens coletados: {resultado['qtd_itens_coletados']}")
    print(f"  Tempo computacional: {elapsed:.4f}s")
    print(f"  Resultado salvo em: {args.output}")

if __name__ == '__main__':
    main()
