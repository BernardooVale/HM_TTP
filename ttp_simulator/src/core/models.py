from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class Item:
    id: int
    peso: float      # weight w_k
    valor: float      # profit p_k
    cidade_origem: int  # assigned city

@dataclass
class Cidade:
    id: int
    x: Optional[float] = None  # for EUC_2D
    y: Optional[float] = None  # for EUC_2D
    itens: list[Item] = field(default_factory=list)  # items available at this city A_i

@dataclass
class Grafo:
    cidades: dict[int, Cidade]  # city_id -> Cidade
    edge_weight_type: str  # 'EUC_2D' or 'EXPLICIT'
    distancias: dict[tuple[int, int], float] = field(default_factory=dict)  # explicit distances
    _x: list[float] = field(default_factory=list, init=False, repr=False)
    _y: list[float] = field(default_factory=list, init=False, repr=False)
    _is_dense_index: bool = field(default=False, init=False, repr=False)
    _coords: dict[int, tuple[float, float]] = field(default_factory=dict, init=False, repr=False)

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



@dataclass
class Mochila:
    capacidade_maxima: float  # W
    peso_atual: float = 0.0   # W_c
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
