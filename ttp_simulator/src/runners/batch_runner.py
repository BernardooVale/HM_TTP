import random
import time
from pathlib import Path
from typing import List

from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.utils.csv_logger import CSVLogger

class BatchRunner:
    def __init__(
        self,
        test_dir: str,
        solvers: List[str],
        models: List[str],
        runs: int,
        output_csv: str,
        base_seed: int
    ):
        self.test_dir = Path(test_dir)
        self.solvers = solvers
        self.models = models
        self.runs = runs
        self.output_csv = output_csv
        self.base_seed = base_seed
        self.logger = CSVLogger(self.output_csv)

    def run(self):
        # Scan test_dir for .txt files
        instance_files = list(self.test_dir.glob('*.txt'))
        
        if not instance_files:
            print(f"Nenhum arquivo de instancia .txt encontrado em {self.test_dir}")
            return
            
        print(f"Iniciando bateria de testes. Instancias: {len(instance_files)}, "
              f"Solvers: {len(self.solvers)}, Modelos: {len(self.models)}, Repeticoes: {self.runs}")
              
        total_runs = len(instance_files) * len(self.solvers) * len(self.models) * self.runs
        current_run = 0

        for instance_file in instance_files:
            try:
                instance = InstanceParser.carregar(str(instance_file))
            except Exception as e:
                print(f"Erro ao carregar instancia {instance_file}: {e}")
                continue

            for solver_name in self.solvers:
                try:
                    solver = SolverRegistry.get(solver_name)
                except Exception as e:
                    print(f"Erro ao carregar solver {solver_name}: {e}")
                    continue

                for model in self.models:
                    for run_idx in range(self.runs):
                        current_run += 1
                        seed = self.base_seed + run_idx
                        
                        print(f"[{current_run}/{total_runs}] Executando: {instance_file.name} | "
                              f"Solver: {solver_name} | Modelo: {model} | Run: {run_idx+1}/{self.runs} | Seed: {seed}")
                              
                        random.seed(seed)
                        
                        start_time = time.time()
                        try:
                            solucao = solver.solve(instance)
                            elapsed = time.time() - start_time
                            
                            if model == 'TTP1':
                                resultado = avaliar_ttp1(instance, solucao)
                            else:
                                resultado = avaliar_ttp2(instance, solucao)
                                
                            registro = CSVLogger.criar_registro(
                                arquivo_instancia=str(instance_file.name),
                                modelo_ttp=model,
                                solucionador=solver.nome,
                                seed=seed,
                                tempo_computacional=elapsed,
                                resultado=resultado,
                                instance=instance
                            )
                            self.logger.registrar(registro)
                        except Exception as e:
                            print(f"Erro durante a execucao: {e}")
                            
        print(f"\nBateria de testes concluida. Resultados salvos em: {self.output_csv}")
