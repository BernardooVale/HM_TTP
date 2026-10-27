"""
Solucionador Co-Solver para o Traveling Thief Problem (TTP).

Referência bibliográfica:
Faulkner, H., Polyakovskiy, S., Schultz, T., & Wagner, M. (2015).
"Solving the Traveling Thief Problem: PackIterative and Co-Solver Approaches".
In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2015), pp. 385-392.
IEEE Congress on Evolutionary Computation (CEC 2015).
"""

from typing import Dict, List, Optional
from src.core.models import TTPInstance, SolucaoTTP
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.tsp import obter_melhor_rota_tsp
from src.solvers.pack_iterative import resolver_pack_iterative, refinar_plano_coleta


@SolverRegistry.register
class CoSolver(BaseSolver):
    """
    Heurística Co-Solver para o TTP (Faulkner et al., 2015).
    
    Abordagem de co-otimização alternada:
    1. Otimiza a rota TSP usando 2-opt com avaliação exata do objetivo global TTP1
       mantendo o plano de empacotamento fixo.
    2. Otimiza o plano de empacotamento usando PackIterative (Drop & Add) mantendo
       a rota fixa.
    3. Alterna entre as duas etapas até a convergência ou atingir o limite de iterações.
    """

    def __init__(self, max_rounds: int = 5, nome: Optional[str] = None):
        self.max_rounds = max_rounds
        self._nome = nome or "cosolver"

    @property
    def nome(self) -> str:
        return self._nome

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        # 1. Rota inicial de alta qualidade e plano inicial PackIterative
        rota = list(obter_melhor_rota_tsp(instance.grafo))
        plano = resolver_pack_iterative(instance, rota, max_refinamento_iter=30)

        melhor_obj = avaliar_ttp1(instance, SolucaoTTP(rota=rota, plano_coleta=plano)).get(
            "lucro_liquido_G", float("-inf")
        )

        n = len(rota)

        for _ in range(self.max_rounds):
            melhorou_round = False

            # --- ETAPA 1: Otimização da rota via 2-opt com plano fixo ---
            rota_melhorou = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Inversão 2-opt mantendo a cidade 1 na posição 0
                    nova_rota = rota[:i] + rota[i : j + 1][::-1] + rota[j + 1 :]
                    obj_cand = avaliar_ttp1(
                        instance, SolucaoTTP(rota=nova_rota, plano_coleta=plano)
                    ).get("lucro_liquido_G", float("-inf"))

                    if obj_cand > melhor_obj + 1e-6:
                        rota = nova_rota
                        melhor_obj = obj_cand
                        rota_melhorou = True
                        melhorou_round = True
                        break
                if rota_melhorou:
                    break

            # --- ETAPA 2: Otimização do empacotamento via PackIterative com rota fixa ---
            novo_plano = refinar_plano_coleta(instance, rota, plano, max_iter=20)
            obj_novo_plano = avaliar_ttp1(
                instance, SolucaoTTP(rota=rota, plano_coleta=novo_plano)
            ).get("lucro_liquido_G", float("-inf"))

            if obj_novo_plano > melhor_obj + 1e-6:
                plano = novo_plano
                melhor_obj = obj_novo_plano
                melhorou_round = True

            if not melhorou_round:
                break

        return SolucaoTTP(rota=rota, plano_coleta=plano)
