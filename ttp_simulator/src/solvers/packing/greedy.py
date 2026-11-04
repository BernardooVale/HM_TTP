import math
from itertools import product
from math import log
from typing import List

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from src.core.fast_evaluator import FastTTPEvaluator
from src.core.models import Item, Tour, TTPInstance


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class IGP:
    scenario: TTPInstance
    tour: Tour
    modelo: str = "TTP1"

    def sort_items(self, item_list: List[Item], k, beta, gamma):
        value_dict = {}
        for item in item_list:
            value_dict[item] = item.valor**k / (
                (item.peso**beta)
                * (
                    self.tour.end_distance(
                        item.cidade_origem, self.scenario.grafo
                    )
                    ** gamma
                )
            )
        sorted_item_list = item_list.copy()
        sorted_item_list = sorted(
            sorted_item_list,
            key=value_dict.get,
            reverse=True,
        )
        return sorted_item_list

    def greedy_packing(
        self,
        sorted_item_list: List[Item],
        packing_factor: float,
    ):
        packed_items = []
        current_weight = 0
        for item in sorted_item_list:
            if (
                current_weight + item.peso
                <= packing_factor * self.scenario.capacidade_mochila
            ):
                packed_items.append(item)
                current_weight += item.peso
        return packed_items, current_weight

    def find_best_adjustment(
        self,
        evaluator: FastTTPEvaluator,
        best_packed_items: List[Item],
        best_k: float,
        best_beta: float,
        theta: float,
        gamma: float,
        delta: float,
        best_score: float,
    ):
        adj_k_up, adj_beta_up = best_k + theta, best_beta + theta
        adj_k_down, adj_beta_down = best_k - theta, best_beta - theta
        k_values = [adj_k_up, adj_k_down]
        beta_values = [adj_beta_up, adj_beta_down]

        # Itera em combinações de ajustes
        for k, beta in product(k_values, beta_values):
            packed_items, _ = self.greedy_packing(
                self.sort_items(self.scenario.itens, k, beta, gamma), delta
            )
            score = evaluator.calcular_lucro({item.id for item in packed_items})
            if score > best_score:
                best_score = score
                best_packed_items = packed_items
                (
                    best_k,
                    best_beta,
                ) = (
                    k,
                    beta,
                )
        return (
            best_score,
            best_packed_items,
            best_k,
            best_beta,
        )

    @staticmethod
    def parse_solution(
        best_packed_items: List[Item],
    ):
        pack_dict = {}
        for item in best_packed_items:
            if item.cidade_origem not in pack_dict:
                pack_dict[item.cidade_origem] = []
            pack_dict[item.cidade_origem].append(item.id)
        return pack_dict

    def solve(
        self,
        theta: float = 5,
        num_iterations: int = 20,
    ):
        # Guardrail para caso de borda sem itens
        if not self.scenario.itens:
            return {}

        evaluator = FastTTPEvaluator(
            self.scenario, self.tour.order, modelo=self.modelo
        )
        item_list = self.scenario.itens
        best_score = -math.inf
        best_packed_items: List[Item] = []
        best_k = best_beta = best_gamma = best_delta = None

        total_dim = max(self.scenario.num_cities * len(self.scenario.itens), 2)
        num_samples = int(1000 / log(total_dim))

        for _ in range(num_samples):
            k, beta, gamma = np.random.rand(3)
            delta = np.random.uniform(0.5, 1)
            packed_items, _ = self.greedy_packing(
                self.sort_items(item_list, k, beta, gamma), delta
            )
            score = evaluator.calcular_lucro({item.id for item in packed_items})
            if score > best_score:
                best_score = score
                best_packed_items = packed_items
                best_k, best_beta, best_gamma, best_delta = (
                    k,
                    beta,
                    gamma,
                    delta,
                )

        if (
            best_k is None
            or best_beta is None
            or best_gamma is None
            or best_delta is None
        ):
            raise ValueError('No valid solution found')

        for _ in range(num_iterations):
            (
                best_score,
                best_packed_items,
                best_k,
                best_beta,
            ) = self.find_best_adjustment(
                evaluator=evaluator,
                best_packed_items=best_packed_items,
                best_k=best_k,
                best_beta=best_beta,
                theta=theta,
                gamma=best_gamma,
                delta=best_delta,
                best_score=best_score,
            )
            theta -= 0.75

        best_pack_dict = self.parse_solution(best_packed_items)
        return best_pack_dict
