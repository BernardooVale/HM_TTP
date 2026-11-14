"""
Solucionadores da arquitetura CTTP (Namazi et al., SoCS 2019):
- CTTP2OptSolver (`cttp_2opt`): 2-OPT Convencional (Descoordenado) no loop CTTP
- CTTPPGCHSolver (`cttp_pgch`): Heurística de Coordenação Guiada por Lucro (PGCH)
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.operators.faulkner_operators import pack_iterative
from src.solvers.operators.pgch_operators import (
    calcular_prefixo_minimo,
    pgch_move,
    preparar_itens_ordenados_por_cidade,
    rand_flip_one_item,
    two_opt_convencional_move,
)
from src.solvers.tsp.CLK import ChainedLK


def _avaliar_solucao_rapida(
    instance: TTPInstance,
    rota: List[int],
    plano_coleta: Dict[int, List[int]],
) -> float:
    ev = FastTTPEvaluator(instance, rota, modelo="TTP1")
    itens_ids = {it_id for ids in plano_coleta.values() for it_id in ids}
    return ev.calcular_lucro(itens_ids)


class _CTTPBaseSolver(BaseSolver):
    """
    Arquitetura coordenada CTTP (Namazi et al., 2019) com suporte a multi-start
    e alternância entre TSPSolver (2-OPT Convencional ou PGCH) e KPSolver
    (RandFlipOneItem com mesma duração temporal do TSPSolver).
    """

    use_pgch: bool = True

    def __init__(
        self,
        time_limit: float = 5.0,
        seed: Optional[int] = None,
        k_vizinhos: int = 20,
        alpha_aceitacao: float = 0.0001,
        clk_tries: int = 5,
        clk_kicks: int = 5,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.k_vizinhos = k_vizinhos
        self.alpha_aceitacao = alpha_aceitacao
        self.clk_tries = clk_tries
        self.clk_kicks = clk_kicks

    def _tsp_solver_pass(
        self,
        instance: TTPInstance,
        rota: List[int],
        plano: Dict[int, List[int]],
        itens_por_cidade,
        razoes,
        sentinela: float,
        closest_map: Dict[int, List[int]],
        start_global: float,
    ) -> Tuple[List[int], Dict[int, List[int]], float, bool]:
        """Executa o TSPSolver (2-OPT convencional ou PGCH) até convergir ou expirar."""
        n = len(rota)
        rota_atual = list(rota)
        plano_atual = {c: list(ids) for c, ids in plano.items() if ids}
        z_atual = _avaliar_solucao_rapida(instance, rota_atual, plano_atual)
        melhorou_geral = False

        if n <= 3:
            return rota_atual, plano_atual, z_atual, False

        while time.perf_counter() - start_global < self.time_limit:
            pos_map = {cid: idx for idx, cid in enumerate(rota_atual)}
            pi_cp = (
                calcular_prefixo_minimo(rota_atual, plano_atual, razoes, sentinela)
                if self.use_pgch
                else []
            )

            melhor_z_varredura = z_atual
            melhor_par: Optional[Tuple[List[int], Dict[int, List[int]]]] = None

            for k1 in range(1, n - 1):
                if time.perf_counter() - start_global >= self.time_limit:
                    break
                cid_k1 = rota_atual[k1]
                vizinhos = closest_map.get(cid_k1, [])
                candidatos_k2 = sorted(
                    {
                        pos_map[v]
                        for v in vizinhos
                        if v in pos_map and pos_map[v] > k1
                    }
                )
                if not candidatos_k2:
                    candidatos_k2 = list(range(k1 + 1, min(n, k1 + 1 + self.k_vizinhos)))

                for k2 in candidatos_k2:
                    if self.use_pgch:
                        r_cand, p_cand = pgch_move(
                            rota_atual,
                            plano_atual,
                            k1,
                            k2,
                            pi_cp,
                            itens_por_cidade,
                            razoes,
                        )
                    else:
                        r_cand, p_cand = two_opt_convencional_move(
                            rota_atual, plano_atual, k1, k2
                        )

                    z_cand = _avaliar_solucao_rapida(instance, r_cand, p_cand)
                    if z_cand > melhor_z_varredura:
                        melhor_z_varredura = z_cand
                        melhor_par = (r_cand, p_cand)

            if melhor_par is not None:
                delta_g = melhor_z_varredura - z_atual
                limiar = max(1e-9, self.alpha_aceitacao * abs(z_atual))
                if delta_g >= limiar:
                    rota_atual, plano_atual = melhor_par
                    z_atual = melhor_z_varredura
                    melhorou_geral = True
                    continue
            break

        return rota_atual, plano_atual, z_atual, melhorou_geral

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        start_global = time.perf_counter()
        rng = np.random.default_rng(self.seed)

        itens_por_cidade, razoes, sentinela = preparar_itens_ordenados_por_cidade(
            instance
        )
        closest_map = instance.grafo.pre_calculate_closest_nodes(
            num_closest=self.k_vizinhos
        )

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

            # Loop CTTP para este restart
            while time.perf_counter() - start_global < self.time_limit:
                t_tsp_ini = time.perf_counter()
                rota, plano, z_atual, mel_tsp = self._tsp_solver_pass(
                    instance,
                    rota,
                    plano,
                    itens_por_cidade,
                    razoes,
                    sentinela,
                    closest_map,
                    start_global,
                )
                duracao_tsp = time.perf_counter() - t_tsp_ini

                restante = self.time_limit - (time.perf_counter() - start_global)
                if restante <= 0:
                    break

                duracao_kp = min(max(duracao_tsp, 1e-4), restante)
                plano, z_kp, mel_kp = rand_flip_one_item(
                    instance,
                    rota,
                    plano,
                    duration_seconds=duracao_kp,
                    rng=rng,
                    min_iters=max(1, len(instance.itens)),
                )
                z_atual = z_kp

                if z_atual > melhor_z_global:
                    melhor_rota_global = list(rota)
                    melhor_plano_global = {
                        c: list(ids) for c, ids in plano.items() if ids
                    }
                    melhor_z_global = z_atual

                if not mel_tsp and not mel_kp:
                    break

            restart_idx += 1
            if time.perf_counter() - start_global >= self.time_limit:
                break

        return SolucaoTTP(rota=melhor_rota_global, plano_coleta=melhor_plano_global)


@SolverRegistry.register
class CTTP2OptSolver(_CTTPBaseSolver):
    """Arquitetura CTTP usando 2-OPT Convencional (Descoordenado)."""

    use_pgch = False

    @property
    def nome(self) -> str:
        return "cttp_2opt"


@SolverRegistry.register
class CTTPPGCHSolver(_CTTPBaseSolver):
    """Arquitetura CTTP usando PGCH (Profit Guided Coordination Heuristic)."""

    use_pgch = True

    @property
    def nome(self) -> str:
        return "cttp_pgch"
