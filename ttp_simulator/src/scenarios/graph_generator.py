import random
import math
from typing import Dict, Tuple
from src.core.models import Cidade, Grafo

def _criar_cidades_dict(coords: list[tuple[float, float]]) -> dict[int, Cidade]:
    cidades = {}
    for i, (x, y) in enumerate(coords, 1):
        cidades[i] = Cidade(id=i, x=x, y=y)
    return cidades

def gerar_grafo_normal(n_cidades: int, seed: int, max_coord: float = 1000.0) -> Grafo:
    random.seed(seed)
    coords = [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(n_cidades)]
    return Grafo(cidades=_criar_cidades_dict(coords), edge_weight_type='EUC_2D')

def gerar_grafo_esparso(n_cidades: int, seed: int, max_coord: float = 1000.0, extra_edges_pct: float = 0.3) -> Grafo:
    random.seed(seed)
    coords = [(random.uniform(0, max_coord), random.uniform(0, max_coord)) for _ in range(n_cidades)]
    cidades = _criar_cidades_dict(coords)
    
    def calc_dist(c1, c2):
        return math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    
    nodes = list(cidades.keys())
    random.shuffle(nodes)
    distancias = {}
    
    for i in range(len(nodes) - 1):
        n1, n2 = nodes[i], nodes[i+1]
        distancias[(min(n1, n2), max(n1, n2))] = calc_dist(cidades[n1], cidades[n2])
    
    num_extra = int((n_cidades * (n_cidades - 1) / 2) * extra_edges_pct)
    all_pairs = [(min(i, j), max(i, j)) for i in range(1, n_cidades + 1) for j in range(i + 1, n_cidades + 1)]
    random.shuffle(all_pairs)
    
    added_extras = 0
    for pair in all_pairs:
        if pair not in distancias:
            n1, n2 = pair
            distancias[pair] = calc_dist(cidades[n1], cidades[n2])
            added_extras += 1
            if added_extras >= num_extra:
                break
                
    return Grafo(cidades=cidades, edge_weight_type='EXPLICIT', distancias=distancias)

def gerar_grafo_equidistante(n_cidades: int, seed: int, raio: float = 500.0) -> Grafo:
    random.seed(seed)
    coords = []
    center_x, center_y = raio, raio
    for i in range(n_cidades):
        angle = 2 * math.pi * i / n_cidades
        x = center_x + raio * math.cos(angle)
        y = center_y + raio * math.sin(angle)
        coords.append((x, y))
    return Grafo(cidades=_criar_cidades_dict(coords), edge_weight_type='EUC_2D')

def gerar_grafo_concentrado(n_cidades: int, seed: int, n_clusters: int = 3, max_coord: float = 1000.0, spread: float = 50.0) -> Grafo:
    random.seed(seed)
    clusters = [(random.uniform(spread, max_coord - spread), random.uniform(spread, max_coord - spread)) for _ in range(n_clusters)]
    
    coords = []
    for _ in range(n_cidades):
        cx, cy = random.choice(clusters)
        x = cx + random.gauss(0, spread)
        y = cy + random.gauss(0, spread)
        coords.append((max(0, min(max_coord, x)), max(0, min(max_coord, y))))
        
    return Grafo(cidades=_criar_cidades_dict(coords), edge_weight_type='EUC_2D')
