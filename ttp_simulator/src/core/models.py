import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Item:
    id: int
    peso: float  # weight w_k
    valor: float  # profit p_k
    cidade_origem: int  # assigned city


@dataclass
class Cidade:
    id: int
    x: Optional[float] = None  # for EUC_2D
    y: Optional[float] = None  # for EUC_2D
    itens: list[Item] = field(
        default_factory=list
    )  # items available at this city A_i


@dataclass
class Grafo:
    cidades: dict[int, Cidade]  # city_id -> Cidade
    edge_weight_type: str  # 'EUC_2D' or 'EXPLICIT'
    distancias: dict[tuple[int, int], float] = field(default_factory=dict)  # explicit distances
    _x: list[float] = field(default_factory=list, init=False, repr=False)
    _y: list[float] = field(default_factory=list, init=False, repr=False)
    _is_dense_index: bool = field(default=False, init=False, repr=False)
    _coords: dict[int, tuple[float, float]] = field(default_factory=dict, init=False, repr=False)
    _distance_matrix: Optional[np.ndarray] = field(
        default=None,
        init=False,
        repr=False,
    )
    _closest_nodes_cache: dict[int, dict[int, list[int]]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def distance_matrix(self):
        if self._distance_matrix is None:
            self._distance_matrix = self.build_distance_matrix()
        return self._distance_matrix

    def build_distance_matrix(self) -> np.ndarray:
        """Builds distance matrix for the graph."""
        if self._distance_matrix is not None:
            return self._distance_matrix
        
        max_id = max(self.cidades.keys()) if self.cidades else 0
        dim = max_id + 1
        D = np.zeros((dim, dim))
        for i in self.cidades:
            for j in self.cidades:
                if i != j:
                    D[i, j] = self.distancia(i, j)
        self._distance_matrix = D
        return D

    def __post_init__(self):
        if self.edge_weight_type in ('EUC_2D', 'CEIL_2D') and self.cidades:
            max_id = max(self.cidades.keys())
            if max_id == len(self.cidades) and 1 in self.cidades:
                self._x = [0.0] * (max_id + 1)
                self._y = [0.0] * (max_id + 1)
                for cid, c in self.cidades.items():
                    self._x[cid] = float(c.x if c.x is not None else 0.0)
                    self._y[cid] = float(c.y if c.y is not None else 0.0)
                self._is_dense_index = True
            else:
                self._coords = {
                    cid: (float(c.x if c.x is not None else 0.0), float(c.y if c.y is not None else 0.0))
                    for cid, c in self.cidades.items()
                }
                self._is_dense_index = False

    def distancia(self, id1: int, id2: int) -> float:
        """Calculate distance between two cities."""
        if id1 == id2:
            return 0.0
        if self.edge_weight_type in ('EUC_2D', 'CEIL_2D'):
            if self._is_dense_index:
                distancia_x = self._x[id1] - self._x[id2]
                distancia_y = self._y[id1] - self._y[id2]
            else:
                coordenada_1 = self._coords[id1]
                coordenada_2 = self._coords[id2]
                distancia_x = coordenada_1[0] - coordenada_2[0]
                distancia_y = coordenada_1[1] - coordenada_2[1]
            distancia = math.sqrt(distancia_x * distancia_x + distancia_y * distancia_y)
            return float(math.ceil(distancia)) if self.edge_weight_type == 'CEIL_2D' else distancia
        else:  # EXPLICIT
            key = (min(id1, id2), max(id1, id2))
            return self.distancias[key]

    def pre_calculate_closest_nodes(
        self,
        num_closest: int = 20,
    ) -> dict[int, list[int]]:

        if num_closest in self._closest_nodes_cache:
            return self._closest_nodes_cache[num_closest]

        closest_nodes = {}

        for i in range(len(self.distance_matrix)):
            sorted_nodes = sorted(
                [
                    (j, self.distance_matrix[i, j])
                    for j in range(len(self.distance_matrix))
                    if j != i
                ],
                key=lambda x: x[1],
            )

            closest_nodes[i] = [node for node, _ in sorted_nodes[:num_closest]]

        self._closest_nodes_cache[num_closest] = closest_nodes

        return closest_nodes


@dataclass
class Mochila:
    capacidade_maxima: float  # W
    peso_atual: float = 0.0  # W_c
    itens_coletados: list[Item] = field(default_factory=list)

    @property
    def espaco_disponivel(self) -> float:
        return self.capacidade_maxima - self.peso_atual


@dataclass
class TTPInstance:
    nome: str
    grafo: Grafo
    itens: list[Item]
    capacidade_mochila: float  # W
    v_min: float
    v_max: float
    taxa_aluguel: float  # R (renting ratio)
    constante_degradacao: float  # C
    dimension: int = 0
    num_items: int = 0

    @property
    def num_cities(self) -> int:
        return self.dimension if self.dimension > 0 else len(self.grafo.cidades)

@dataclass
class SolucaoTTP:
    rota: list[int]               # ordered list of city IDs to visit
    plano_coleta: dict[int, list[int]]  # city_id -> list of item IDs to collect there
      
    @property
    def tour(self) -> list[int]:
        return self.rota

    @property
    def picking_plan(self) -> dict[int, list[int]]:
        return self.plano_coleta

    def itens_coletados_ids(self) -> list[int]:
        all_ids = []
        for ids in self.plano_coleta.values():
            if isinstance(ids, (list, tuple, set)):
                all_ids.extend(ids)
        return all_ids


@dataclass
class Tour:
    """
    Classe que facilita a manipulação de tours em problemas de roteamento.
    Relevante para tratar de forma eficiente flips e manter registro de
    ordenação do tour, considerando circularidade.
    """

    order: List[int]
    """É a ordem dos nós no tour."""
    cost: float
    """Custo do tour."""
    length: int = field(init=False)
    """Comprimento do tour."""
    length_minus_1: int = field(init=False)
    """Comprimento do tour menos 1, usado para implementações rápidas de módulo."""

    def __post_init__(self):
        self.length = len(self.order)
        self.length_minus_1 = self.length - 1
        self.pos = np.empty(self.length, dtype=int)
        for i, v in enumerate(self.order):
            self.pos[v] = i
        self._make_edges()

    @staticmethod
    def make_pair(i, j):
        if i > j:
            return (j, i)
        return (i, j)

    def _make_edges(self):
        self.edges = set()
        for i in range(self.length):
            self.edges.add(
                self.make_pair(
                    self.order[i], self.order[(i + 1) % self.length]
                )
            )

    def contains(self, edge):
        """
        Check if an edge belongs to the tour
        """
        return edge in self.edges

    def next(self, node: int) -> int:
        """Retorna o nó seguinte ao nó de input.

        Parameters
        ----------
        node : int
            Nó que se deseja se obter o seguinte.

        Returns
        -------
        int
            Nó seguinte ao nó de input.
        """
        return self.order[(self.pos[node] + 1) % self.length]

    def prev(self, node: int) -> int:
        """
        Retorna o nó anterior ao nó de input.

        Parameters
        ----------
        node : int
            Nó que se deseja se obter o anterior.

        Returns
        -------
        int
            Nó anterior ao nó de input.
        """
        return self.order[(self.pos[node] - 1) % self.length]

    def around(self, node: int) -> Tuple[int, int]:
        """Retorna os nós anterior e seguinte ao nó de input."""
        return self.prev(node), self.next(node)

    def generate(self, broken, joined):
        """
        Gera um tour temporário, considerando as arestas removidas e adicionadas.
        Primeiro constroi a sequencia do tour a partir das arestas disponíveis,
        e verifica se o numero de arestas é compatível com o esperado. Depois
        percorre as arestas para construir o tour final e garantir que não se
        repete nenhum nó.

        Parameters

        ----------
        broken : set
            Conjunto de arestas a serem removidas do tour.
        joined : set
            Conjunto de arestas a serem adicionadas ao tour.

        Returns
        -------
        tuple
            Uma tupla contendo um booleano indicando se um tour válido foi gerado
            e a lista de nós do novo tour.
        """
        broken = {self.make_pair(int(i), int(j)) for i, j in broken}

        joined = {self.make_pair(int(i), int(j)) for i, j in joined}

        # New edges: old edges minus broken, plus joined
        edges = (self.edges - broken) | joined

        if len(edges) < self.length:
            return False, []

        successors = {}
        node = 0

        while len(edges) > 0:
            for i, j in edges:
                if i == node:
                    successors[node] = j
                    node = j
                    break
                elif j == node:
                    successors[node] = i
                    node = i
                    break

            edges.remove((i, j))

        if len(successors) < self.length:
            return False, []

        succ = successors[0]
        new_tour = [0]
        visited = set(new_tour)

        while succ not in visited:
            visited.add(succ)
            new_tour.append(succ)
            succ = successors[succ]

        # If we visited all nodes without a loop we have a tour
        return len(new_tour) == self.length, new_tour

    def flip(self, segment_start: int, segment_end: int):
        """
        Forma inteligente de fazer o flip sem ter que ficar invertendo
        sequencias e sem se preocupar com circularidade do tour.
        pegamos o segmento que vamos fazer o flip, e literalmente
        passamos caminhando pelo indice pela esquerda e pela direita e vamos
        invertendo as posições, e atualizando os vetores do tour
        e das posições.

        Parameters
        ----------
        segment_start : int
            Nó inicial do segmento a ser invertido. (não é a posição)
        segment_end : int
            Nó final do segmento a ser invertido. (não é a posição)
        """
        idx_start, idx_end = self.pos[segment_start], self.pos[segment_end]
        if idx_start <= idx_end:
            segment_len = idx_end - idx_start + 1
        else:
            segment_len = self.length - idx_start + idx_end + 1

        # Se o segmento for par precisa inverter todos,
        # se for impar todos menos o do centro
        for i in range(segment_len // 2):
            idx_left = (idx_start + i) % self.length
            idx_right = (idx_end - i) % self.length

            node_left = self.order[idx_left]
            node_right = self.order[idx_right]

            self.order[idx_left] = node_right
            self.order[idx_right] = node_left

            self.pos[node_left] = idx_right
            self.pos[node_right] = idx_left

    def flip_and_update_cost(self, node_start, node_end, dist_mtx):
        if node_start == node_end:
            delta_cost = 0.0
        else:
            pos_a = self.pos[node_start]
            pos_b = self.pos[node_end]

            # Identify nodes defining edges to be broken and formed for 2-opt
            prev_node_of_start = self.order[
                (pos_a - 1 + self.length) % self.length
            ]
            next_node_of_end = self.order[(pos_b + 1) % self.length]

            # Handle case where the "segment" is the entire tour (flip is identity)
            if (
                prev_node_of_start == node_end
                and next_node_of_end == node_start
            ):
                delta_cost = 0.0
            else:
                # Cost change for a 2-opt move
                term_removed = (
                    dist_mtx[prev_node_of_start, node_start]
                    + dist_mtx[node_end, next_node_of_end]
                )
                term_added = (
                    dist_mtx[prev_node_of_start, node_end]
                    + dist_mtx[node_start, next_node_of_end]
                )
                delta_cost = term_added - term_removed

            self.flip(node_start, node_end)

            self.cost += delta_cost
            return delta_cost
