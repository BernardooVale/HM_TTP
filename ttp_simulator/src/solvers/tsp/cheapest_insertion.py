# Module to implement cheapest regret contructive heuristic for TSP
from typing import List, Optional, Tuple

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from src.core.models import Grafo, Tour


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CheapInsertionRegret:
    grafo: Grafo
    T: List[int] = Field(init=False)
    T_dist: float = Field(init=False)

    def __post_init__(self):
        self.T = []
        self.T_dist = 0

    @property
    def dist_mtx(self):
        return self.grafo.distance_matrix

    def insert_vertex(self, k: int):
        best_insert_cost = np.inf
        best_pos = None
        for pos, vertex in enumerate(self.T):
            if pos == 0:
                continue
            prev_vertex = self.T[pos - 1]
            next_vertex = vertex
            increase = (
                self.dist_mtx[prev_vertex, k]
                + self.dist_mtx[k, next_vertex]
                - self.dist_mtx[prev_vertex, next_vertex]
            )
            if increase < best_insert_cost:
                best_insert_cost = increase
                best_pos = pos
        return best_pos, best_insert_cost

    def find_cheapest_insertion(
        self,
    ) -> Tuple[int, int, float]:
        vertex_in_subtour = set(self.T)
        best_insert_cost = np.inf
        best_pos = None
        best_k = None
        for k in range(self.dist_mtx.shape[0]):
            if k not in vertex_in_subtour:
                pos, insert_cost = self.insert_vertex(k)
                if insert_cost < best_insert_cost:
                    best_insert_cost = insert_cost
                    best_k = k
                    best_pos = pos
        if best_k is None or best_pos is None:
            raise ValueError('No valid insertion found.')

        return int(best_k), int(best_pos), best_insert_cost

    def find_best_deletion(
        self, del_edge: Tuple[int, int]
    ) -> Tuple[Optional[int], float]:

        best_deletion_value = -np.inf
        best_pos = None
        for pos, k in enumerate(self.T[:-1]):
            if pos == 0:
                continue
            if k not in del_edge:
                del_value = (
                    self.dist_mtx[self.T[pos - 1], k]
                    + self.dist_mtx[k, self.T[pos + 1]]
                    - self.dist_mtx[self.T[pos - 1], self.T[pos + 1]]
                )
                if del_value > best_deletion_value:
                    best_deletion_value = del_value
                    best_pos = pos

        if best_pos is None or best_deletion_value is None:
            return None, -np.inf

        return int(best_pos), best_deletion_value

    def find_regret_insertion(self):
        k, l = np.unravel_index(self.dist_mtx.argmin(), self.dist_mtx.shape)
        self.T += [int(k), int(l), int(k)]
        self.T_dist = self.dist_mtx[k, l] + self.dist_mtx[l, k]
        while len(self.T) <= self.dist_mtx.shape[0]:
            # Find the vertex to be added to T that minimizes c(T, k)
            # where c(T,k) is the cost of the cycle inserting new vertex k into k
            # minus the initial cost of the cycle T.
            best_insert_k, best_insert_pos, best_insert_cost = (
                self.find_cheapest_insertion()
            )
            i, j = self.T[best_insert_pos - 1], self.T[best_insert_pos]
            # Now looks through vertexes in T except for i, j that maximizes
            # c-(T, k), where c-(T, k) is the sub cycle path lenght reduction
            # from the cycle c(T) by removing vertex k from cycle T
            # c(T) - c(T \ {k})
            best_del_pos, best_del_value = self.find_best_deletion((i, j))
            if best_insert_cost >= best_del_value:
                self.T.insert(best_insert_pos, best_insert_k)
                self.T_dist += best_insert_cost
            elif best_del_pos is not None:
                self.T.pop(best_del_pos)
                self.T_dist -= best_del_value
                best_insert_k, best_insert_pos, best_insert_cost = (
                    self.find_cheapest_insertion()
                )
                self.T.insert(best_insert_pos, best_insert_k)
                self.T_dist += best_insert_cost
            else:
                raise ValueError('No valid deletion found.')

        tour = Tour(order=self.T[:-1], cost=self.T_dist)
        return tour
