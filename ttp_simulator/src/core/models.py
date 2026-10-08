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
    
    def distancia(self, id1: int, id2: int) -> float:
        """Calculate distance between two cities."""
        if self.edge_weight_type == 'EUC_2D':
            c1, c2 = self.cidades[id1], self.cidades[id2]
            return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
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
