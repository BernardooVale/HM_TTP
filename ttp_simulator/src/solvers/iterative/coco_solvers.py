"""
Solucionador CoCo (Cooperative Coordination Solver) para o TTP1
(Namazi, Sanderson, Newton & Sattar).
"""

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.operators.coco_operators import (
    boundary_bit_flip,
    calcular_perfis_e_fronteira_coco,
    pgch_estendido_move,
)
from src.solvers.operators.cosolver2b_operators import gerar_grafo_delaunay
from src.solvers.operators.faulkner_operators import pack_iterative
from src.solvers.operators.pgch_operators import preparar_itens_ordenados_por_cidade
from src.solvers.tsp.CLK import ChainedLK


def _avaliar_coco(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
) -> float:
    ev = FastTTPEvaluator(instance, rota, modelo="TTP1")
    return ev.calcular_lucro({it_id for ids in plano_coleta.values() for it_id in ids})


@SolverRegistry.register
class CoCoSolver(BaseSolver):
    """
    CoCo (Cooperative Coordination Solver):
    - Inicialização: ChainedLK + PackIterative
    - TSPSolver: steepest-ascent sobre vizinhos da Triangulação de Delaunay com
      PGCH Estendido (Π e Ω) e limiar de aceitação α = 0.01%
    - KPSolver: Boundary Bit-Flip
    - Loop multi-start até esgotar time_limit
    """

    def __init__(
        self,
        time_limit: float = 5.0,
        seed: Optional[int] = None,
        alpha_aceitacao: float = 0.0001,
        clk_tries: int = 5,
        clk_kicks: int = 5,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.alpha_aceitacao = alpha_aceitacao
        self.clk_tries = clk_tries
        self.clk_kicks = clk_kicks

    @property
    def nome(self) -> str:
        return "coco_solver"

    def _tsp_solver(
        self,
        instance: TTPInstance,
        rota: List[int],
        plano: Dict[int, List[int]],
        itens_por_cidade,
        razoes: Dict[int, float],
        sentinela: float,
        grafo_delaunay: Dict[int, Set[int]],
        start_global: float,
    ) -> Tuple[List[int], Dict[int, List[int]], float]:
        n = len(rota)
        rota_atual = list(rota)
        plano_atual = {c: list(ids) for c, ids in plano.items() if ids}
        z_atual = _avaliar_coco(instance, rota_atual, plano_atual)

        if n <= 3:
            return rota_atual, plano_atual, z_atual

        while time.perf_counter() - start_global < self.time_limit:
            pos_map = {cid: idx for idx, cid in enumerate(rota_atual)}
            pi_p, omega_q, _ = calcular_perfis_e_fronteira_coco(
                rota_atual, plano_atual, itens_por_cidade, razoes, sentinela
            )

            melhor_z = z_atual
            melhor_par: Optional[Tuple[List[int], Dict[int, List[int]]]] = None

            for k1 in range(1, n - 1):
                if time.perf_counter() - start_global >= self.time_limit:
                    break
                cid_k1 = rota_atual[k1]
                vizinhos = grafo_delaunay.get(cid_k1, set())
                candidatos_k2 = sorted(
                    pos_map[v] for v in vizinhos if v in pos_map and pos_map[v] > k1
                )

                for k2 in candidatos_k2:
                    r_cand, p_cand = pgch_estendido_move(
                        rota_atual,
                        plano_atual,
                        k1,
                        k2,
                        pi_p,
                        omega_q,
                        itens_por_cidade,
                        razoes,
                    )
                    z_cand = _avaliar_coco(instance, r_cand, p_cand)
                    if z_cand > melhor_z:
                        melhor_z = z_cand
                        melhor_par = (r_cand, p_cand)

            if melhor_par is not None:
                limiar = max(1e-9, self.alpha_aceitacao * abs(z_atual))
                if (melhor_z - z_atual) >= limiar:
                    rota_atual, plano_atual = melhor_par
                    z_atual = melhor_z
                    continue
            break

        return rota_atual, plano_atual, z_atual

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start_global = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        itens_por_cidade, razoes, sentinela = preparar_itens_ordenados_por_cidade(
            instance
        )
        grafo_delaunay = gerar_grafo_delaunay(instance.grafo)

        melhor_rota_global: List[int] = []
        melhor_plano_global: Dict[int, List[int]] = {}
        melhor_z_global = float("-inf")

        restart_idx = 0
        while True:
            iter_seed = None if self.seed is None else self.seed + restart_idx
            clk = ChainedLK(
                grafo=instance.grafo,
                num_tries=self.clk_tries,
                num_kicks=self.clk_kicks,
                seed=iter_seed,
            )
            rota = list(clk.solve(random_start=(restart_idx > 0)).order)
            plano, z_atual = pack_iterative(instance, rota)

            if z_atual > melhor_z_global or not melhor_rota_global:
                melhor_rota_global = list(rota)
                melhor_plano_global = {c: list(ids) for c, ids in plano.items() if ids}
                melhor_z_global = z_atual

            while time.perf_counter() - start_global < self.time_limit:
                rota, plano, z_tsp = self._tsp_solver(
                    instance,
                    rota,
                    plano,
                    itens_por_cidade,
                    razoes,
                    sentinela,
                    grafo_delaunay,
                    start_global,
                )
                plano, z_kp, _ = boundary_bit_flip(
                    instance,
                    rota,
                    plano,
                    itens_por_cidade,
                    razoes,
                    sentinela,
                    rng=rng,
                    time_limit=self.time_limit,
                    start_time=start_global,
                )

                if z_kp > melhor_z_global:
                    melhor_rota_global = list(rota)
                    melhor_plano_global = {
                        c: list(ids) for c, ids in plano.items() if ids
                    }
                    melhor_z_global = z_kp

                if z_kp <= z_tsp + 1e-9:
                    break

            restart_idx += 1
            if time.perf_counter() - start_global >= self.time_limit:
                break

        return SolucaoTTP(rota=melhor_rota_global, plano_coleta=melhor_plano_global)
