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
                dx = self._x[id1] - self._x[id2]
                dy = self._y[id1] - self._y[id2]
            else:
                c1 = self._coords[id1]
                c2 = self._coords[id2]
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
            d = math.sqrt(dx * dx + dy * dy)
            return float(math.ceil(d)) if self.edge_weight_type == 'CEIL_2D' else d
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

@dataclass
class SolucaoTTP:
    rota: list[int]               # ordered list of city IDs to visit
    plano_coleta: dict[int, list[int]]  # city_id -> list of item IDs to collect there
    
    def itens_coletados_ids(self) -> list[int]:
        all_ids = []
        for ids in self.plano_coleta.values():
            all_ids.extend(ids)
        return all_ids
