from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.core.models import Grafo

# DOCUMENTAR


Edge = tuple[int, int]


@dataclass
class OneTreeResult:
    one_tree: set[Edge]
    mst: set[Edge]
    special_node: int
    root_neighbors: tuple[int, int]
    penalties: np.ndarray
    transformed_costs: np.ndarray
    lower_bound: float
    tree_cost: float


@dataclass
class MinTree:
    grafo: Grafo

    @staticmethod
    def normalize_edge(u: int, v: int) -> Edge:
        """
        Normaliza arestas para padronizar busca - arestas sempre com menor
        inteiro primeiro

        Parameters
        ----------
        u : int
            vertice 1
        v : int
            vertice 2

        Returns
        -------
        Edge
            aresta normalizada (u, v) com u < v
        """
        return (u, v) if u < v else (v, u)

    def calc_mst_minus_n(self, dist_matrix):
        """
        Calcula MST do grafo a menos de um vértice específico (sampled_n)

        Parameters
        ----------
        dist_matrix : np.ndarray
            matriz de distâncias entre os vértices

        Returns
        -------
        tuple[set[Edge], set[int], int]
            mst: conjunto de arestas da MST
            visited_cities: conjunto de cidades visitadas na MST
            sampled_n: vértice excluído da MST
        """
        # Subtract the node 'n' from the graph before calculating the MST
        sampled_n = min(self.grafo.cidades.keys())
        # dist_matrix[sampled_n, :] = float('inf')
        # dist_matrix[:, sampled_n] = float('inf')
        visited_cities = set()
        city = min(set(self.grafo.cidades.keys()) - {sampled_n})
        visited_cities.add(city)
        best_dist = {}
        parent = {}
        for v in self.grafo.cidades.keys():
            if v == sampled_n:
                continue
            best_dist[v] = dist_matrix[v, city]
            parent[v] = city
        mst = set()

        while len(visited_cities) < len(self.grafo.cidades.keys()) - 1:
            new_city = min(
                (v for v in best_dist if v not in visited_cities),
                key=lambda x: best_dist[x],
            )
            visited_cities.add(new_city)
            mst.add(self.normalize_edge(parent[new_city], new_city))
            for v, dist in best_dist.items():
                if v not in visited_cities:
                    if dist_matrix[v, new_city] < dist:
                        best_dist[v] = dist_matrix[v, new_city]
                        parent[v] = new_city

        return mst, visited_cities, sampled_n

    def calc_one_tree(self, max_iterations, step_size=0.1):
        """
        Calcula uma one-tree para o problema TSP usando penalidades de 2-degree.

        Parameters
        ----------
        max_iterations : int
            número máximo de iterações para o algoritmo
        step_size : float, optional
            tamanho do passo para atualização das penalidades, by default 0.1

        Returns
        -------
        OneTreeResult
            resultado contendo a one-tree, MST, nó especial, vizinhos da raiz,
            penalidades, custos transformados, custo da árvore e limite
            inferior
        """
        best_cost = -np.inf
        penalties = np.zeros(
            self.grafo.distance_matrix.shape[0],
            dtype=float,
        )

        for _ in range(max_iterations):
            transformed_costs = (
                self.grafo.distance_matrix
                + penalties[:, None]
                + penalties[None, :]
            )

            mst, _, sampled_n = self.calc_mst_minus_n(transformed_costs)

            valid_nodes = np.array(
                [node for node in self.grafo.cidades if node != sampled_n],
                dtype=int,
            )

            order = np.argsort(transformed_costs[sampled_n, valid_nodes])

            first_neighbor = int(valid_nodes[order[0]])
            second_neighbor = int(valid_nodes[order[1]])

            closest_edge = self.normalize_edge(sampled_n, first_neighbor)
            second_closest_edge = self.normalize_edge(
                sampled_n, second_neighbor
            )

            one_tree = mst.copy()
            one_tree.add(closest_edge)
            one_tree.add(second_closest_edge)

            root_neighbors = (
                first_neighbor,
                second_neighbor,
            )

            transformed_tree_cost = sum(
                transformed_costs[u, v] for u, v in one_tree
            )

            lower_bound = transformed_tree_cost - 2.0 * sum(
                penalties[node] for node in self.grafo.cidades
            )

            if lower_bound > best_cost:
                best_cost = lower_bound
                best_result = OneTreeResult(
                    one_tree=one_tree.copy(),
                    mst=mst.copy(),
                    special_node=sampled_n,
                    root_neighbors=root_neighbors,
                    penalties=penalties.copy(),
                    transformed_costs=transformed_costs.copy(),
                    tree_cost=float(transformed_tree_cost),
                    lower_bound=float(lower_bound),
                )

            if self.check_2_degree(one_tree):
                break

            penalties = self.update_penalties(
                penalties=penalties,
                one_tree=one_tree,
                step_size=step_size,
            )

        return best_result

    def check_2_degree(self, one_tree: set[tuple[int, int]]):
        """
        Verifica se todos os vértices na one-tree têm grau 2.

        Parameters
        ----------
        one_tree : set[tuple[int, int]]
            conjunto de arestas da one-tree

        Returns
        -------
        bool
            True se todos os vértices têm grau 2, False caso contrário
        """
        degrees = dict.fromkeys(self.grafo.cidades, 0)

        for u, v in one_tree:
            degrees[u] += 1
            degrees[v] += 1

        if all(degree == 2 for degree in degrees.values()):
            return True
        return False

    def calc_one_tree_cost(self):
        """
        Calcula o custo da one-tree usando o resultado da função calc_one_tree.

        Returns
        -------
        np.ndarray
            matriz de custos transformados da one-tree
        """
        best_result = self.calc_one_tree(max_iterations=100)

        return best_result.transformed_costs

    def update_penalties(
        self,
        penalties: np.ndarray,
        one_tree: set[tuple[int, int]],
        step_size: float = 0.1,
    ) -> np.ndarray:
        """
        Atualiza as penalidades dos vértices com base nos graus na one-tree.

        Parameters
        ----------
        penalties : np.ndarray
            array de penalidades atuais
        one_tree : set[tuple[int, int]]
            conjunto de arestas da one-tree
        step_size : float, optional
            tamanho do passo para atualização das penalidades, by default 0.1

        Returns
        -------
        np.ndarray
            array de penalidades atualizado
        """
        degrees = dict.fromkeys(self.grafo.cidades, 0)

        for u, v in one_tree:
            degrees[u] += 1
            degrees[v] += 1

        for node in self.grafo.cidades:
            penalties[node] += step_size * (degrees[node] - 2)

        return penalties

    @staticmethod
    def adj_list(
        tree: set[Edge],
        cost_matrix: np.ndarray,
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Constrói uma lista de adjacência a partir de um conjunto de arestas e
        uma matriz de custos.

        Parameters
        ----------
        tree : set[Edge]
            conjunto de arestas da árvore
        cost_matrix : np.ndarray
            matriz de custos entre os vértices

        Returns
        -------
        dict[int, list[tuple[int, float]]]
            lista de adjacência representando a árvore
        """
        adj: dict[int, list[tuple[int, float]]] = defaultdict(list)

        for u, v in tree:
            weight = float(cost_matrix[u, v])

            adj[u].append((v, weight))
            adj[v].append((u, weight))

        return adj

    @staticmethod
    def max_edge_on_path(
        adj: dict[int, list[tuple[int, float]]],
        start: int,
        end: int,
    ) -> float:
        """
        Encontra a aresta de peso máximo no caminho entre dois vértices em uma
        árvore representada por uma lista de adjacência. Se start e end forem
        ligados na MST, retorna a aresta, resultando em alpha zero. Se não,
        vamos ter arestas maiores do que as da MST, e penalizamos indicando
        que ela "foge" da árvore mínima geradora (MST).

        Parameters
        ----------
        adj : dict[int, list[tuple[int, float]]]
            lista de adjacência representando a árvore
        start : int
            vértice de início do caminho
        end : int
            vértice de fim do caminho

        Returns
        -------
        float
            peso da aresta de peso máximo no caminho entre start e end

        Raises
        ------
        ValueError
            se não houver caminho entre start e end
        """
        stack = [
            (
                start,
                None,
                -np.inf,
            )
        ]

        visited = set()

        while stack:
            node, parent, max_weight = stack.pop()

            if node == end:
                return float(max_weight)

            visited.add(node)

            for neighbor, weight in adj[node]:
                if neighbor == parent:
                    continue

                if neighbor in visited:
                    continue

                stack.append((
                    neighbor,
                    node,
                    max(max_weight, weight),
                ))

        raise ValueError(f'No path between {start} and {end}')

    def alpha_nearness(
        self,
        edge: Edge,
        result: OneTreeResult,
        mst_adj: dict[int, list[tuple[int, float]]],
    ) -> float:
        """
        Calcula a alpha-nearness de uma aresta em relação a uma árvore mínima geradora (MST)
        transformada.

        Parameters
        ----------
        edge : Edge
            aresta a ser avaliada
        result : OneTreeResult
            resultado da construção da one-tree
        mst_adj : dict[int, list[tuple[int, float]]]
            lista de adjacência representando a MST

        Returns
        -------
        float
            valor da alpha-nearness da aresta

        Raises
        ------
        ValueError
            se a alpha-nearness for negativa, indicando inconsistência entre a MST e a matriz de custos transformados
        """
        u, v = self.normalize_edge(*edge)

        if u == v:
            raise ValueError('Alpha-nearness indefinido para self-loop.')

        special_node = result.special_node
        first_neighbor, second_neighbor = result.root_neighbors
        transformed_costs = result.transformed_costs

        first_root_edge = self.normalize_edge(
            special_node,
            first_neighbor,
        )

        second_root_edge = self.normalize_edge(
            special_node,
            second_neighbor,
        )

        if (
            (u, v) in result.mst
            or (u, v) == first_root_edge
            or (u, v) == second_root_edge
        ):
            return 0.0

        if special_node in (u, v):
            other_node = v if u == special_node else u

            alpha = (
                transformed_costs[
                    special_node,
                    other_node,
                ]
                - transformed_costs[
                    special_node,
                    second_neighbor,
                ]
            )
        else:
            max_path_cost = self.max_edge_on_path(
                adj=mst_adj,
                start=u,
                end=v,
            )

            alpha = transformed_costs[u, v] - max_path_cost

        if alpha < -1e-9:
            raise ValueError(
                f'Alpha negativo para {(u, v)}: {alpha}. '
                'A MST e a matriz de custos transformados '
                'podem estar inconsistentes.'
            )

        return max(0.0, float(alpha))

    def calculate_all_alpha_nearness(
        self,
        max_iterations: int = 100,
        step_size: float = 0.1,
    ) -> tuple[
        dict[Edge, float],
        OneTreeResult,
    ]:
        """
        Calcula a alpha-nearness de todas as arestas em relação a uma árvore mínima geradora (MST)
        transformada.

        Parameters
        ----------
        max_iterations : int, optional
            número máximo de iterações para a construção da one-tree (padrão: 100)
        step_size : float, optional
            tamanho do passo para a atualização dos custos transformados (padrão: 0.1)

        Returns
        -------
        tuple[dict[Edge, float], OneTreeResult]
            dicionário com os valores de alpha-nearness para cada aresta e o resultado da construção da one-tree
        """
        result = self.calc_one_tree(
            max_iterations=max_iterations,
            step_size=step_size,
        )

        mst_adj = self.adj_list(
            tree=result.mst,
            cost_matrix=result.transformed_costs,
        )

        cities = sorted(self.grafo.cidades.keys())

        alpha_values: dict[Edge, float] = {}

        for i, u in enumerate(cities):
            for v in cities[i + 1 :]:
                edge = (u, v)

                alpha_values[edge] = self.alpha_nearness(
                    edge=edge,
                    result=result,
                    mst_adj=mst_adj,
                )

        return alpha_values, result
