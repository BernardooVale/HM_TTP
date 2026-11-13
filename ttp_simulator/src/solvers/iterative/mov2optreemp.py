"""
Operador conjugado de troca 2-opt com reempacotamento parcial (MOV2OPTREEMP)
para o ReinforceNS e ReinforceNS-PM.

Referências:
- Wu et al. (2024), "ReinforceNS: Reinforcement Learning-based Multi-start
  Neighborhood Search for Solving the Traveling Thief Problem", IJCAI-24,
  Seção 4.3, Algoritmo 2 (linhas 6-8) e Equação (5).
- Campos et al., "Heurísticas para o Travelling Thief Problem", Seção 3,
  Algoritmos 1 e 2 (MOV2OPTREEMP).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
from src.core.models import Item, SolucaoTTP, TTPInstance


@dataclass
class RLNSSolutionState:
    """
    Representa o estado completo de uma solução (Π, P) durante a busca local,
    mantendo estruturas auxiliares para aplicação rápida de MOV2OPTREEMP.
    """

    route: List[int]
    pos: np.ndarray
    plan: Dict[int, List[int]]
    city_picked_weight: np.ndarray
    city_picked_profit: np.ndarray
    end_dist: np.ndarray
    total_weight: float
    total_profit: float
    travel_time: float
    objective: float

    def copy(self, deep_plan: bool = False) -> "RLNSSolutionState":
        plan_copy = (
            {c: items.copy() for c, items in self.plan.items()}
            if deep_plan
            else self.plan.copy()
        )
        return RLNSSolutionState(
            route=self.route.copy(),
            pos=self.pos.copy(),
            plan=plan_copy,
            city_picked_weight=self.city_picked_weight.copy(),
            city_picked_profit=self.city_picked_profit.copy(),
            end_dist=self.end_dist.copy(),
            total_weight=float(self.total_weight),
            total_profit=float(self.total_profit),
            travel_time=float(self.travel_time),
            objective=float(self.objective),
        )

    def to_solucao_ttp(self) -> SolucaoTTP:
        clean_plan = {
            int(c): list(ids) for c, ids in self.plan.items() if ids
        }
        return SolucaoTTP(rota=list(self.route), plano_coleta=clean_plan)


class Mov2OptReempOperator:
    """
    Executa o operador MOV2OPTREEMP(Π, P, x_i, x_j, δ) conforme as linhas 6-8
    do Algoritmo 2 de Wu et al. (2024):
      1. Remove da mochila P apenas os itens pertencentes às cidades na sub-rota
         entre x_i e x_j.
      2. Aplica a troca 2-opt invertendo a sub-rota entre x_i e x_j (gerando Π̄).
      3. Recalcula os scores (Equação 5) dos itens das cidades na sub-rota invertida
         com as novas distâncias até o depósito ξ_c e insere gulosamente na mochila
         sem exceder W * δ (gerando P̄).
    """

    def __init__(
        self,
        instance: TTPInstance,
        k: float,
        beta: float,
        gamma: float,
        delta: float,
        modelo: str = "TTP1",
    ):
        self.instance = instance
        self.grafo = instance.grafo
        self.dist_mtx = instance.grafo.distance_matrix
        self.k = float(k)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.modelo = modelo

        self.cap_max = float(instance.capacidade_mochila)
        self.restricted_cap = self.delta * self.cap_max
        self.v_min = float(instance.v_min)
        self.v_max = float(instance.v_max)
        self.taxa_aluguel = float(instance.taxa_aluguel)
        self.constante_degradacao = max(
            float(instance.constante_degradacao), 1e-6
        )
        self.speed_factor = (
            (self.v_max - self.v_min) / self.cap_max
            if self.cap_max > 0
            else 0.0
        )

        dim = self.dist_mtx.shape[0]
        self.dim = dim

        # Pré-indexa itens por cidade e o termo fixo p_ik^k / w_ik^beta da Equação (5)
        self.city_items: Dict[int, List[Tuple[int, float, float, float]]] = {
            c: [] for c in range(1, dim)
        }
        self.item_by_id: Dict[int, Item] = {}
        for item in instance.itens:
            self.item_by_id[item.id] = item
            base_ratio = (item.valor ** self.k) / (item.peso ** self.beta)
            self.city_items[item.cidade_origem].append(
                (item.id, float(item.peso), float(item.valor), float(base_ratio))
            )

    def _compute_end_distances(self, route: List[int]) -> np.ndarray:
        n = len(route)
        end_dist = np.zeros(self.dim, dtype=float)
        if n == 0:
            return end_dist
        last_city = route[n - 1]
        end_dist[last_city] = float(self.dist_mtx[last_city, route[0]])
        for idx in range(n - 2, -1, -1):
            u = route[idx]
            v = route[idx + 1]
            end_dist[u] = float(self.dist_mtx[u, v]) + end_dist[v]
        return end_dist

    def _evaluate_route_and_weights(
        self,
        route: List[int],
        plan: Dict[int, List[int]],
        city_picked_weight: np.ndarray,
        total_profit: float,
    ) -> Tuple[float, float]:
        """
        Calcula (tempo_viagem, lucro_liquido) de forma idêntica ao avaliador oficial.
        """
        n = len(route)
        dist_mtx = self.dist_mtx
        v_max = self.v_max
        v_min = self.v_min
        speed_factor = self.speed_factor

        if self.modelo == "TTP2":
            tempo_coleta_cidade = np.zeros(self.dim, dtype=float)
            tempo_viagem = 0.0
            peso_acc = 0.0
            for idx in range(n):
                u = route[idx]
                v = route[idx + 1] if idx + 1 < n else route[0]
                tempo_coleta_cidade[u] = tempo_viagem
                peso_acc += city_picked_weight[u]
                vel = v_max - peso_acc * speed_factor
                if vel < v_min:
                    vel = v_min
                tempo_viagem += float(dist_mtx[u, v]) / vel

            tempo_fim = tempo_viagem
            valor_degradado_total = 0.0
            c_deg = self.constante_degradacao
            for city_id, item_ids in plan.items():
                if not item_ids:
                    continue
                t_col = tempo_coleta_cidade[city_id]
                degradacao = math.ceil((tempo_fim - t_col) / c_deg)
                for it_id in item_ids:
                    valor_degradado_total += (
                        self.item_by_id[it_id].valor - degradacao
                    )
            return tempo_viagem, valor_degradado_total

        peso_acc = 0.0
        tempo_viagem = 0.0
        for idx in range(n):
            u = route[idx]
            v = route[idx + 1] if idx + 1 < n else route[0]
            peso_acc += city_picked_weight[u]
            vel = v_max - peso_acc * speed_factor
            if vel < v_min:
                vel = v_min
            tempo_viagem += float(dist_mtx[u, v]) / vel

        objective = total_profit - self.taxa_aluguel * tempo_viagem
        return tempo_viagem, objective

    def build_initial_state(
        self,
        route: List[int],
        plan: Dict[int, List[int]],
    ) -> RLNSSolutionState:
        """
        Constrói o estado RLNSSolutionState inicial a partir de uma rota Π e plano P.
        """
        pos = np.full(self.dim, -1, dtype=int)
        for idx, c in enumerate(route):
            pos[c] = idx

        city_picked_weight = np.zeros(self.dim, dtype=float)
        city_picked_profit = np.zeros(self.dim, dtype=float)
        clean_plan: Dict[int, List[int]] = {c: [] for c in route}

        total_weight = 0.0
        total_profit = 0.0

        for c in route:
            raw_ids = plan.get(c, [])
            if not raw_ids:
                continue
            c_w = 0.0
            c_p = 0.0
            kept_ids: List[int] = []
            for it_id in raw_ids:
                item = self.item_by_id[it_id]
                if total_weight + item.peso <= self.cap_max + 1e-9:
                    kept_ids.append(it_id)
                    total_weight += item.peso
                    total_profit += item.valor
                    c_w += item.peso
                    c_p += item.valor
            clean_plan[c] = kept_ids
            city_picked_weight[c] = c_w
            city_picked_profit[c] = c_p

        end_dist = self._compute_end_distances(route)
        travel_time, objective = self._evaluate_route_and_weights(
            route=route,
            plan=clean_plan,
            city_picked_weight=city_picked_weight,
            total_profit=total_profit,
        )

        return RLNSSolutionState(
            route=list(route),
            pos=pos,
            plan=clean_plan,
            city_picked_weight=city_picked_weight,
            city_picked_profit=city_picked_profit,
            end_dist=end_dist,
            total_weight=total_weight,
            total_profit=total_profit,
            travel_time=travel_time,
            objective=objective,
        )

    def apply_move(
        self,
        state: RLNSSolutionState,
        city_a: int,
        city_b: int,
        in_place: bool = False,
    ) -> RLNSSolutionState:
        """
        Executa MOV2OPTREEMP(Π, P, city_a, city_b, δ) retornando o novo estado (Π̄, P̄).
        Mantém o depósito (route[0] == 1) fixo na posição 0.
        """
        target = state if in_place else state.copy()

        pos_a = int(target.pos[city_a])
        pos_b = int(target.pos[city_b])
        if pos_a == pos_b or pos_a <= 0 or pos_b <= 0:
            return target

        if pos_a > pos_b:
            pos_a, pos_b = pos_b, pos_a

        route = target.route
        n = len(route)
        sub_cities = route[pos_a : pos_b + 1]

        # Linha 6 do Algoritmo 2: remove da mochila os itens das cidades na sub-rota [pos_a .. pos_b]
        removed_weight = 0.0
        removed_profit = 0.0
        for c in sub_cities:
            removed_weight += target.city_picked_weight[c]
            removed_profit += target.city_picked_profit[c]
            target.city_picked_weight[c] = 0.0
            target.city_picked_profit[c] = 0.0
            target.plan[c] = []

        curr_weight = target.total_weight - removed_weight
        curr_profit = target.total_profit - removed_profit

        # Linha 7 do Algoritmo 2: troca 2-opt invertendo a sub-rota entre pos_a e pos_b
        reversed_sub = sub_cities[::-1]
        route[pos_a : pos_b + 1] = reversed_sub
        for offset, c in enumerate(reversed_sub):
            target.pos[c] = pos_a + offset

        # Atualiza as distâncias até o depósito (end_dist / ξ) para a sub-rota invertida
        # e para o prefixo antes de pos_a
        dist_mtx = self.dist_mtx
        end_dist = target.end_dist
        next_after_pb = route[pos_b + 1] if (pos_b + 1) < n else route[0]
        xi_after = end_dist[route[pos_b + 1]] if (pos_b + 1) < n else 0.0

        last_in_sub = reversed_sub[-1]
        end_dist[last_in_sub] = float(dist_mtx[last_in_sub, next_after_pb]) + xi_after
        for idx in range(len(reversed_sub) - 2, -1, -1):
            u = reversed_sub[idx]
            v = reversed_sub[idx + 1]
            end_dist[u] = float(dist_mtx[u, v]) + end_dist[v]

        for idx in range(pos_a - 1, -1, -1):
            u = route[idx]
            v = route[idx + 1]
            end_dist[u] = float(dist_mtx[u, v]) + end_dist[v]

        # Linha 8 do Algoritmo 2: pontua itens da sub-rota invertida pela Equação (5)
        # e adiciona gulosamente sem exceder W * δ
        gamma = self.gamma
        candidates_in_sub: List[Tuple[float, int, int, float, float]] = []
        for c in reversed_sub:
            c_items = self.city_items.get(c)
            if not c_items:
                continue
            xi_c = max(end_dist[c], 1e-9)
            inv_xi_gamma = xi_c ** (-gamma)
            for it_id, peso, valor, base_ratio in c_items:
                score = base_ratio * inv_xi_gamma
                candidates_in_sub.append((score, it_id, c, peso, valor))

        if candidates_in_sub:
            candidates_in_sub.sort(key=lambda x: x[0], reverse=True)
            max_allowed_weight = min(self.restricted_cap, self.cap_max)
            for _, it_id, c, peso, valor in candidates_in_sub:
                if curr_weight + peso <= max_allowed_weight + 1e-9:
                    target.plan[c].append(it_id)
                    target.city_picked_weight[c] += peso
                    target.city_picked_profit[c] += valor
                    curr_weight += peso
                    curr_profit += valor

        target.total_weight = curr_weight
        target.total_profit = curr_profit
        target.travel_time, target.objective = self._evaluate_route_and_weights(
            route=route,
            plan=target.plan,
            city_picked_weight=target.city_picked_weight,
            total_profit=curr_profit,
        )
        return target
