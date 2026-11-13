"""
Procedimentos de Pós-Otimização do ReinforceNS:
- IFP (Item Flipping Phase / Bit-Flip de Itens)
- RRP (Route Refining Phase / Refinamento de Rota via Equação 8)
- Suporte aos 4 modos fatoriais da Hipótese 2: P0, P1, P2 e P3.

Referências:
- Wu et al. (2024), "ReinforceNS: Reinforcement Learning-based Multi-start
  Neighborhood Search for Solving the Traveling Thief Problem", IJCAI-24,
  Seção 4.4, Figura 1 e Equação (8).
- Campos et al., "Heurísticas para o Travelling Thief Problem", Seção 4.1
  (Tabela 1: Combinações de pós-processamento P0, P1, P2, P3).
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import TTPInstance


@dataclass
class PostOptimizationResult:
    """Resultado detalhado da fase de pós-otimização (P0, P1, P2 ou P3)."""

    route: List[int]
    plan: Dict[int, List[int]]
    objective_before: float
    objective_after_ifp: float
    objective_after_rrp: float
    objective_final: float
    ifp_flips_accepted: int
    rrp_relocations_accepted: int
    elapsed_time: float


def compute_rrp_delta_psi_eq8(
    dist_mtx: np.ndarray,
    x_i: int,
    x_i_next: int,
    x_j_prev: int,
    x_j: int,
    x_j_next: int,
    v_xi: float,
    v_xj_prev: float,
) -> float:
    """
    Calcula em O(1) a variação exata no tempo de viagem Δψ = ψ(Π', P) - ψ(Π, P)
    ao deslocar uma cidade sem itens coletados `x_j` (entre `x_j_prev` e `x_j_next`)
    para a aresta anterior (`x_i`, `x_i_next`), conforme a Equação (8) de Wu et al. (2024):

        Δψ = (d(x_i, x_j) + d(x_j, x_{i+1})) / v_{x_i} + d(x_{j-1}, x_{j+1}) / v_{x_{j-1}}
             - (d(x_i, x_{i+1}) / v_{x_i} + (d(x_{j-1}, x_j) + d(x_j, x_{j+1})) / v_{x_{j-1}})
    """
    new_time = (
        float(dist_mtx[x_i, x_j]) + float(dist_mtx[x_j, x_i_next])
    ) / v_xi + float(dist_mtx[x_j_prev, x_j_next]) / v_xj_prev

    old_time = (
        float(dist_mtx[x_i, x_i_next]) / v_xi
        + (float(dist_mtx[x_j_prev, x_j]) + float(dist_mtx[x_j, x_j_next]))
        / v_xj_prev
    )

    return new_time - old_time


def item_flipping_phase(
    instance: TTPInstance,
    route: List[int],
    plan: Dict[int, List[int]],
    modelo: str = "TTP1",
    deadline: Optional[float] = None,
) -> Tuple[Dict[int, List[int]], float, int]:
    """
    Fase de Item Flipping (IFP) conforme Seção 4.4 de Wu et al. (2024):
    Mantendo a rota Π' fixa, percorre iterativamente cada item i testando
    inverter seu estado na mochila (incluir se ausente e respeitar W, ou
    remover se já incluído). Repete até uma passagem inteira sem melhoria.
    """
    evaluator = FastTTPEvaluator(instance, route, modelo=modelo)
    initial_ids: Set[int] = {
        it_id for ids in plan.values() for it_id in ids
    }
    evaluator.inicializar_estado(initial_ids)
    best_obj = evaluator.calcular_lucro_atual(modelo=modelo)

    cap_max = float(instance.capacidade_mochila)
    flips_accepted = 0

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            break

        improved_in_pass = False
        for item in instance.itens:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            it_id = item.id
            if it_id in evaluator.itens_coletados:
                evaluator.remover_item(it_id)
                cand_obj = evaluator.calcular_lucro_atual(modelo=modelo)
                if cand_obj > best_obj + 1e-9:
                    best_obj = cand_obj
                    improved_in_pass = True
                    flips_accepted += 1
                else:
                    evaluator.adicionar_item(it_id)
            else:
                if evaluator.peso_total_atual + item.peso <= cap_max + 1e-9:
                    evaluator.adicionar_item(it_id)
                    cand_obj = evaluator.calcular_lucro_atual(modelo=modelo)
                    if cand_obj > best_obj + 1e-9:
                        best_obj = cand_obj
                        improved_in_pass = True
                        flips_accepted += 1
                    else:
                        evaluator.remover_item(it_id)

        if not improved_in_pass:
            break

    new_plan: Dict[int, List[int]] = {}
    for item in instance.itens:
        if item.id in evaluator.itens_coletados:
            new_plan.setdefault(item.cidade_origem, []).append(item.id)

    return new_plan, best_obj, flips_accepted


def route_refining_phase(
    instance: TTPInstance,
    route: List[int],
    plan: Dict[int, List[int]],
    modelo: str = "TTP1",
    deadline: Optional[float] = None,
) -> Tuple[List[int], float, int]:
    """
    Fase de Route Refining (RRP) conforme Seção 4.4, Figura 1 e Equação (8)
    de Wu et al. (2024):
    Para cada cidade x_j sem itens coletados na mochila, pseudo-insere x_j em
    cada aresta anterior (x_i, x_{i+1}) do caminho Π_{x_j} (do depósito até x_j)
    e calcula a variação de tempo Δψ em O(1) via Equação (8).
    Se houver redução de tempo (Δψ < 0), insere x_j na aresta de maior ganho.
    Repete até que nenhuma cidade vazia apresente melhoria.
    """
    curr_route = list(route)
    n = len(curr_route)
    dist_mtx = instance.grafo.distance_matrix
    dim = dist_mtx.shape[0]

    cap_max = max(float(instance.capacidade_mochila), 1e-6)
    v_min = float(instance.v_min)
    v_max = float(instance.v_max)
    speed_factor = (v_max - v_min) / cap_max

    item_by_id = {it.id: it for it in instance.itens}
    city_picked_weight = np.zeros(dim, dtype=float)
    for c, ids in plan.items():
        if ids:
            city_picked_weight[c] = sum(item_by_id[it_id].peso for it_id in ids)

    def _compute_speeds(r: List[int]) -> np.ndarray:
        speeds = np.empty(len(r), dtype=float)
        w_acc = 0.0
        for idx, c in enumerate(r):
            w_acc += city_picked_weight[c]
            vel = v_max - w_acc * speed_factor
            if vel < v_min:
                vel = v_min
            speeds[idx] = vel
        return speeds

    relocations_accepted = 0

    if n >= 3 and modelo == "TTP1":
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break

            improved_in_pass = False
            speeds = _compute_speeds(curr_route)
            empty_cities = [
                c for c in curr_route[1:] if city_picked_weight[c] == 0.0
            ]

            for x_j in empty_cities:
                if deadline is not None and time.perf_counter() >= deadline:
                    break

                pos_j = curr_route.index(x_j)
                if pos_j <= 1:
                    continue

                x_j_prev = curr_route[pos_j - 1]
                x_j_next = curr_route[(pos_j + 1) % n]
                v_xj_prev = speeds[pos_j - 1]

                best_delta_psi = 0.0
                best_insert_edge_idx = None

                # Testa inserir x_j em cada aresta (x_i, x_{i+1}) de Π_{x_j} anterior a x_j
                for i in range(0, pos_j - 1):
                    x_i = curr_route[i]
                    x_i_next = curr_route[i + 1]
                    v_xi = speeds[i]

                    delta_psi = compute_rrp_delta_psi_eq8(
                        dist_mtx=dist_mtx,
                        x_i=x_i,
                        x_i_next=x_i_next,
                        x_j_prev=x_j_prev,
                        x_j=x_j,
                        x_j_next=x_j_next,
                        v_xi=v_xi,
                        v_xj_prev=v_xj_prev,
                    )
                    if delta_psi < best_delta_psi - 1e-9:
                        best_delta_psi = delta_psi
                        best_insert_edge_idx = i

                if best_insert_edge_idx is not None:
                    curr_route.pop(pos_j)
                    curr_route.insert(best_insert_edge_idx + 1, x_j)
                    speeds = _compute_speeds(curr_route)
                    improved_in_pass = True
                    relocations_accepted += 1

            if not improved_in_pass:
                break

    evaluator = FastTTPEvaluator(instance, curr_route, modelo=modelo)
    picked_ids: Set[int] = {it_id for ids in plan.values() for it_id in ids}
    final_obj = evaluator.calcular_lucro(picked_ids, modelo=modelo)
    return curr_route, final_obj, relocations_accepted


def apply_post_optimization(
    instance: TTPInstance,
    route: List[int],
    plan: Dict[int, List[int]],
    mode: str = "P3",
    modelo: str = "TTP1",
    deadline: Optional[float] = None,
) -> PostOptimizationResult:
    """
    Aplica a combinação de pós-otimização especificada:
      - "P0": nenhum pós-processamento
      - "P1": apenas Item Flipping Phase (IFP)
      - "P2": apenas Route Refining Phase (RRP)
      - "P3": IFP seguido de RRP (padrão do ReinforceNS)
    """
    start_time = time.perf_counter()
    mode_norm = mode.upper().strip()
    if mode_norm not in ("P0", "P1", "P2", "P3"):
        raise ValueError(
            f"Modo de pós-processamento inválido '{mode}'. Use P0, P1, P2 ou P3."
        )

    curr_route = list(route)
    curr_plan = {c: list(ids) for c, ids in plan.items() if ids}

    eval_init = FastTTPEvaluator(instance, curr_route, modelo=modelo)
    picked_init = {it_id for ids in curr_plan.values() for it_id in ids}
    obj_before = eval_init.calcular_lucro(picked_init, modelo=modelo)

    obj_after_ifp = obj_before
    obj_after_rrp = obj_before
    ifp_flips = 0
    rrp_relocs = 0

    if mode_norm in ("P1", "P3"):
        curr_plan, obj_after_ifp, ifp_flips = item_flipping_phase(
            instance=instance,
            route=curr_route,
            plan=curr_plan,
            modelo=modelo,
            deadline=deadline,
        )
        obj_after_rrp = obj_after_ifp

    if mode_norm in ("P2", "P3"):
        curr_route, obj_after_rrp, rrp_relocs = route_refining_phase(
            instance=instance,
            route=curr_route,
            plan=curr_plan,
            modelo=modelo,
            deadline=deadline,
        )

    return PostOptimizationResult(
        route=curr_route,
        plan=curr_plan,
        objective_before=obj_before,
        objective_after_ifp=obj_after_ifp,
        objective_after_rrp=obj_after_rrp,
        objective_final=obj_after_rrp,
        ifp_flips_accepted=ifp_flips,
        rrp_relocations_accepted=rrp_relocs,
        elapsed_time=time.perf_counter() - start_time,
    )
