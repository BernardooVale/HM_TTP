import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from src.core.models import Grafo, Tour


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class LKLocalSearch:
    grafo: Grafo
    tour: Tour
    num_tries: int = 5

    def __post_init__(self):
        self.best_cost = self.tour.cost
        self.solutions = set()

    def update_best_cost(self, new_cost: float):
        self.best_cost = min(self.best_cost, new_cost)

    @property
    def dist_mtx(self):
        return self.grafo.distance_matrix

    @staticmethod
    def make_pair(i, j):
        if i > j:
            return (j, i)
        return (i, j)

    def calculate_gain(self, edge_delete, edge_add):
        return self.dist_mtx[edge_delete] - self.dist_mtx[edge_add]

    def get_closest_candidates(
        self, t2i, tour: Tour, gain, broken, joined, num_closest: int = 20
    ) -> Dict[int, Tuple[float, float]]:
        """
        Find the closest neighbours of a node ordered by potential gain.  As a
        side-effect, also compute the partial improvement of joining a node.

        Parameters:

            - t2i: node to relink from
            - tour: current tour to optimise
            - gain: current gain
            - broken: set of edges to remove (X)
            - joined: set of edges to join (Y)

        Return: sorted list of neighbours based on potential improvement with
        next removal (highest diff), and then sort by highest potential gain Gi
        """

        neighbours = {}
        closeness_candidates_dict = self.grafo.pre_calculate_closest_nodes(
            num_closest
        )
        closeness_candidates = closeness_candidates_dict.get(t2i, [])
        # Candidatos ara fechar o tour (t_4)

        for node in closeness_candidates:
            y_i = self.make_pair(t2i, node)
            G_i = gain - self.dist_mtx[self.make_pair(t2i, node)]
            if G_i <= 0 or y_i in broken or tour.contains(y_i):
                continue

            for succ in tour.around(node):
                x_i = self.make_pair(node, succ)
                if x_i not in broken and x_i not in joined:
                    diff = self.dist_mtx[x_i] - self.dist_mtx[y_i]
                    # potential improvement for removing next x edge

                    if node not in neighbours:
                        neighbours[node] = [diff, G_i]
                    elif node in neighbours and diff > neighbours[node][0]:
                        neighbours[node][0] = diff
                        # Updates only diff. G_i remains the same but
                        # next x_i note could improve diff.

        return sorted(neighbours.items(), key=lambda x: x[1][0], reverse=True)

    def choose_Y(
        self,
        tour,
        t1,
        t2i,
        gain,
        broken,
        joined,
    ):
        """
        Choose an edge to add to the new tour.

        Parameters:

            - tour: current tour to optimise
            - t1: starting node for the current k-opt
            - t2i: tail of the last edge removed (t_2i)
            - gain: current gain (Gi)
            - broken: potential edges to remove (X)
            - joined: potential edges to add (Y)

        Return: whether we found an improved tour
        """
        ordered = self.get_closest_candidates(
            t2i,
            tour,
            gain,
            broken,
            joined,
        )

        if len(broken) == 2:
            # Check the five nearest neighbours when i = 2
            top = 5
        else:
            # Otherwise the closest only
            top = 1

        for node, (_, Gi) in ordered:
            yi = self.make_pair(t2i, node)
            added = deepcopy(joined)
            added.add(yi)

            # Stop at the first improving tour
            if self.choose_X(tour, t1, node, Gi, broken, added):
                return True

            top -= 1
            # Tried enough options
            if top == 0:
                return False

        return False

    def choose_X(
        self,
        tour: Tour,
        start_node: int,
        last_t: int,
        G_i: float,
        broken: set,
        joined: set,
    ) -> bool:
        """
        Choose an edge to omit from the tour.

        Parameters:

            - tour: current tour to optimise
            - start_node: starting node for the current k-opt
            - last_t: tail of the last edge added (t_2i-1)
            - gain: current gain (Gi)
            - broken: potential edges to remove (X)
            - joined: potential edges to add (Y)

        Return: whether we found an improved tour
        """

        if len(broken) == 4:
            # Cuts the search space after 4 iterations
            # to avoid exploding possibilities
            pred, succ = tour.around(last_t)

            # Give priority to the longest edge for x_4
            if self.dist_mtx[(pred, last_t)] > self.dist_mtx[(succ, last_t)]:
                around = [pred]
            else:
                around = [succ]
        else:
            around = tour.around(last_t)

        for t2i in around:
            xi = self.make_pair(last_t, t2i)
            # Gain at current iteration
            gain = G_i + self.dist_mtx[(last_t, t2i)]

            # Verify that X and Y are disjoint, though I also need to check
            # that we are not including an x_i again for some reason.
            if xi not in joined and xi not in broken:
                added = deepcopy(joined)
                removed = deepcopy(broken)

                removed.add(xi)
                added.add(
                    self.make_pair(t2i, start_node)
                )  # Try to relink the tour

                relink = gain - self.dist_mtx[(t2i, start_node)]
                is_tour, new_tour = tour.generate(removed, added)

                # Se a solução atual não fecha o ciclo, e tivermos no começo
                # da busca (len(added) > 2) tentamos outra
                # se a outra não fechar ciclo também não der, retorna falso
                if not is_tour and len(added) > 2:
                    continue

                # Stop the search if we come back to the same solution
                if str(new_tour) in self.solutions:
                    return False

                # Save the current solution if the tour is better, we need
                # `is_tour` again in the case where we have a non-sequential
                # exchange with i = 2
                if is_tour and relink > 0:
                    self.tour = Tour(
                        order=new_tour, cost=self.tour.cost - relink
                    )

                    return True
                else:
                    # Pass on the newly "removed" edge but not the relink
                    choice = self.choose_Y(
                        tour, start_node, t2i, gain, removed, joined
                    )

                    if len(broken) == 2 and choice:
                        return True
                    else:
                        # Single iteration for i > 2
                        return choice

        return False

    def lin_kernighan(self):
        best_tour = Tour(order=self.tour.order.copy(), cost=self.tour.cost)

        nodes_to_explore = set(self.tour.order)

        while nodes_to_explore:
            start_node = nodes_to_explore.pop()
            around_start_node = best_tour.around(start_node)
            for t_2 in around_start_node:
                broken_x_i = {self.make_pair(start_node, t_2)}
                gain = self.dist_mtx[start_node, t_2]
                closest_candidates_dict = self.get_closest_candidates(
                    t_2,
                    self.tour,
                    gain,
                    broken_x_i,
                    set(),
                )

                tries = self.num_tries
                for t_3, (_, G_i) in closest_candidates_dict:
                    if t_3 in around_start_node:
                        continue
                    joined_yi = {self.make_pair(t_2, t_3)}

                    if self.choose_X(
                        self.tour, start_node, t_3, G_i, broken_x_i, joined_yi
                    ):
                        return True

                    tries -= 1
                    if tries == 0:
                        break

        return False

    def solve(self):
        better = True
        self.solutions = set()

        while better:
            better = self.lin_kernighan()
            self.solutions.add(str(self.tour.order))

        return self.tour, self.tour.cost


class TourKicker(ABC):
    @abstractmethod
    def kick(
        self,
        tour: Tour,
        grafo: Grafo,
    ) -> Tour:
        pass


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DoubleBridge(TourKicker):
    @staticmethod
    def sample_first_edge_candidates(
        tour: Tour,
        first_edge_sample_size: float = 0.1,
    ):
        candidates_size = tour.length - 1
        candidate_list = np.random.choice(
            candidates_size,
            size=int(first_edge_sample_size * candidates_size),
            replace=False,
        )
        if len(candidate_list) == 0:
            raise ValueError('No candidates available for the first edge.')
        return candidate_list

    def select_first_edge(
        self,
        tour: Tour,
        grafo: Grafo,
        first_edge_sample_size: float = 0.1,
    ):
        candidates = self.sample_first_edge_candidates(
            tour, first_edge_sample_size
        )
        best_candidate = None
        max_potential = -math.inf
        for candidate in candidates:
            next_node = tour.next(candidate)
            closest_node = grafo.pre_calculate_closest_nodes(
                num_closest=20
            ).get(candidate, [None])[0]
            dist_next = grafo.distance_matrix[candidate, next_node]
            dist_closest = grafo.distance_matrix[candidate, closest_node]
            potential = dist_next - dist_closest
            if potential > max_potential:
                max_potential = potential
                best_candidate = candidate
        if best_candidate is None:
            raise ValueError('No valid edge found for the best candidate.')
        return best_candidate

    @staticmethod
    def select_double_bridge(
        best_candidate,
        tour: Tour,
        grafo: Grafo,
        num_candidates: int = 10,
    ):
        edges_to_delete = []
        edges_to_delete.append((
            int(best_candidate),
            int(tour.next(best_candidate)),
        ))
        added_nodes = [int(best_candidate), int(tour.next(best_candidate))]
        closest_node_list = grafo.pre_calculate_closest_nodes(
            num_closest=20
        ).get(best_candidate, [None])[:num_candidates]
        for node in closest_node_list:
            if node not in added_nodes and tour.next(node) not in added_nodes:
                next_node = tour.next(node)
                edges_to_delete.append((int(node), int(next_node)))
                added_nodes += [int(node), int(next_node)]
            if len(edges_to_delete) == 4:
                break
        sorted_deleted_edges = sorted(
            edges_to_delete,
            key=lambda e: tour.pos[e[0]],
        )
        return sorted_deleted_edges

    @staticmethod
    def execute_double_bridge(delete_edges, tour: Tour, grafo: Grafo):
        (n1, n2), (n3, n4), (n5, n6), (n7, n8) = delete_edges

        added_edges = [(n1, n6), (n2, n5), (n3, n8), (n4, n7)]
        # tour._make_edges()

        ok, new_tour = tour.generate(
            broken=set(delete_edges), joined=set(added_edges)
        )

        if not ok:
            raise ValueError(
                'Failed to generate a valid tour after double bridge.'
            )

        new_cost = sum(
            grafo.distance_matrix[
                new_tour[i], new_tour[(i + 1) % len(new_tour)]
            ]
            for i in range(len(new_tour))
        )

        return Tour(order=new_tour, cost=new_cost)

    def kick(
        self,
        tour: Tour,
        grafo: Grafo,
    ):
        best_candidate = self.select_first_edge(tour, grafo)
        delete_edges = self.select_double_bridge(
            best_candidate, tour=tour, grafo=grafo
        )
        return self.execute_double_bridge(delete_edges, tour=tour, grafo=grafo)
