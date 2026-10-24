import argparse
import random
import sys
import time
from pathlib import Path

# Add project root and ttp_simulator to path
_raiz = Path(__file__).resolve().parents[1]
_sim_path = Path(__file__).resolve().parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.utils.csv_logger import CSVLogger
import src.solvers  # noqa


def main():
    parser = argparse.ArgumentParser(description='Simulacao individual do TTP')
    parser.add_argument('--instance', required=True, help='Caminho do arquivo de instancia (.txt/.ttp)')
    parser.add_argument('--solver', required=True, help='Nome do solucionador')
    parser.add_argument('--model', default='TTP1', choices=['TTP1', 'TTP2'], help='Modelo TTP (padrao: TTP1)')
    parser.add_argument('--seed', type=int, default=42, help='Semente aleatoria')
    parser.add_argument('--output', default=str(_raiz / 'results' / 'resultado_unico.csv'), help='Arquivo CSV de saida')
    args = parser.parse_args()

    random.seed(args.seed)

    # Load instance
    caminho_inst = Path(args.instance)
    if not caminho_inst.is_file():
        caminho_alt = _sim_path / 'bateria_testes' / args.instance
        if caminho_alt.is_file():
            caminho_inst = caminho_alt

    instance = InstanceParser.carregar(str(caminho_inst))

    # Get solver
    solver = SolverRegistry.get(args.solver)
    if hasattr(solver, "modelo"):
        setattr(solver, "modelo", args.model)

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
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(str(output_path))
    registro = CSVLogger.criar_registro(
        arquivo_instancia=str(caminho_inst),
        modelo_ttp=args.model,
        solucionador=solver.nome,
        seed=args.seed,
        tempo_computacional=elapsed,
        resultado=resultado,
        instance=instance
    )
    logger.registrar(registro)

    print(f"Simulacao concluida: {solver.nome} em {caminho_inst.name}")
    print(f"  Modelo: {args.model}")
    print(f"  Lucro liquido (G): {resultado['lucro_liquido_G']:.4f}")
    print(f"  Tempo viagem (f): {resultado['tempo_viagem_f']:.4f}")
    print(f"  Valor bruto (g): {resultado['valor_bruto_itens_g']:.4f}")
    print(f"  Itens coletados: {resultado['qtd_itens_coletados']}")
    print(f"  Tempo computacional: {elapsed:.4f}s")
    print(f"  Resultado salvo em: {output_path}")


if __name__ == '__main__':
    main()
