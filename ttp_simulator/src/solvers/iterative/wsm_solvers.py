"""
Solucionador Weighted-Sum Method (WSM) de Chagas & Wagner (2021) para o BITTP/TTP1.
"""

import time
from typing import List, Optional

import numpy as np

from src.core.models import SolucaoTTP, TTPInstance
from src.solvers.base import BaseSolver, SolverRegistry
from src.solvers.operators.wsm_operators import (
    ParetoEntry,
    atualizar_arquivo_pareto,
    calcular_distancia_media_cidades,
    explorar_2opt_com_filtro_beta,
    explorar_bit_flip_amostrado,
    extrair_melhor_ttp1_do_arquivo,
    randomized_packing_algorithm,
    selecionar_pivo,
)
from src.solvers.tsp.CLK import ChainedLK


@SolverRegistry.register
class WSMSolver(BaseSolver):
    """
    Weighted-Sum Method (Chagas & Wagner, 2021):
    Mantém um arquivo Pareto S de soluções não-dominadas alternando:
    - Exploração: ChainedLK + η planos via RandomizedPackingAlgorithm(π, ρ, α, γ)
    - Explotação: pivô de S + 2-opt com filtro geométrico β + Bit-Flip amostrado λ

    `solve_pareto(instance)` retorna a lista de `ParetoEntry` (fronteira aproximada).
    `solve(instance)` retorna a `SolucaoTTP` do arquivo S com maior lucro TTP1.
    """

    def __init__(
        self,
        time_limit: float = 5.0,
        seed: Optional[int] = None,
        eta: int = 117,
        rho: int = 12,
        gamma: float = 41.0,
        beta: float = 0.001,
        lambd: float = 0.22,
        clk_tries: int = 5,
        clk_kicks: int = 5,
    ):
        self.time_limit = time_limit
        self.seed = seed
        self.eta = eta
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        self.lambd = lambd
        self.clk_tries = clk_tries
        self.clk_kicks = clk_kicks
        self.last_pareto_archive: List[ParetoEntry] = []

    @property
    def nome(self) -> str:
        return "wsm_solver"

    def solve_pareto(self, instance: TTPInstance) -> List[ParetoEntry]:
        start = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        itens_by_id = {it.id: it for it in instance.itens}
        taxa = float(instance.taxa_aluguel)
        l_medio = calcular_distancia_media_cidades(instance)

        arquivo: List[ParetoEntry] = []
        iter_idx = 0

        while True:
            # --- Fase de Exploração ---
            iter_seed = None if self.seed is None else self.seed + iter_idx
            clk = ChainedLK(
                grafo=instance.grafo,
                num_tries=self.clk_tries,
                num_kicks=self.clk_kicks,
                seed=iter_seed,
            )
            rota = list(clk.solve(random_start=(iter_idx > 0)).order)

            for k in range(max(1, self.eta)):
                if k > 0 and time.perf_counter() - start >= self.time_limit:
                    break
                alpha = float(rng.random())
                itens_ids, g, h = randomized_packing_algorithm(
                    instance,
                    rota,
                    rho=self.rho,
                    alpha=alpha,
                    gamma=self.gamma,
                    rng=rng,
                    itens_by_id=itens_by_id,
                )
                atualizar_arquivo_pareto(arquivo, rota, itens_ids, g, h)

            if not arquivo:
                break

            # --- Fase de Explotação ---
            if time.perf_counter() - start < self.time_limit:
                alpha_exp = float(rng.random())
                pivo = selecionar_pivo(arquivo, alpha_exp, taxa)

                melhor_2opt = explorar_2opt_com_filtro_beta(
                    instance, pivo, alpha_exp, self.beta, l_medio, itens_by_id
                )
                if melhor_2opt is not None:
                    r_nova, g_nova, h_nova = melhor_2opt
                    atualizar_arquivo_pareto(arquivo, r_nova, pivo.itens_ids, g_nova, h_nova)

                if time.perf_counter() - start < self.time_limit:
                    explorar_bit_flip_amostrado(
                        instance, pivo, self.lambd, arquivo, rng, itens_by_id
                    )

            iter_idx += 1
            if time.perf_counter() - start >= self.time_limit:
                break

        self.last_pareto_archive = arquivo
        return arquivo

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        arquivo = self.solve_pareto(instance)
        itens_by_id = {it.id: it for it in instance.itens}
        return extrair_melhor_ttp1_do_arquivo(instance, arquivo, itens_by_id)
