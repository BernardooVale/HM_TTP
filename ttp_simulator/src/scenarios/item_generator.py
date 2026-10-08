import random
from typing import Dict, Tuple, List
from src.core.models import Item, Cidade

def gerar_itens(
    n_itens: int,
    cidades: dict[int, Cidade],
    seed: int,
    distribuicao: str = 'uniforme',  # 'uniforme', 'concentrado', 'esparso'
    valor_range: tuple[float, float] = (10.0, 500.0),
    peso_range: tuple[float, float] = (1.0, 50.0),
    intensidade: str = 'media',  # 'baixa', 'media', 'alta'
) -> list[Item]:
    random.seed(seed)
    
    available_cities = [c for cid, c in cidades.items() if cid != 1]
    if not available_cities:
        return []
        
    if distribuicao == 'uniforme':
        city_assignments = [available_cities[i % len(available_cities)] for i in range(n_itens)]
    elif distribuicao == 'concentrado':
        num_concentrados = max(1, len(available_cities) // 4)
        chosen_cities = random.sample(available_cities, num_concentrados)
        city_assignments = [random.choice(chosen_cities) for _ in range(n_itens)]
    elif distribuicao == 'esparso':
        num_esparso = max(1, len(available_cities) // 2)
        chosen_cities = random.sample(available_cities, num_esparso)
        city_assignments = [random.choice(chosen_cities) for _ in range(n_itens)]
    else:
        city_assignments = [random.choice(available_cities) for _ in range(n_itens)]
        
    random.shuffle(city_assignments)
    
    itens = []
    for i in range(n_itens):
        city = city_assignments[i]
        peso = random.uniform(*peso_range)
        
        if intensidade == 'baixa':
            val = random.triangular(valor_range[0], valor_range[1], valor_range[0])
        elif intensidade == 'media':
            val = random.uniform(*valor_range)
        elif intensidade == 'alta':
            val = random.triangular(valor_range[0], valor_range[1], valor_range[1])
        else:
            val = random.uniform(*valor_range)
            
        item = Item(id=i+1, peso=peso, valor=val, cidade_origem=city.id)
        itens.append(item)
        city.itens.append(item)
        
    return itens

def calcular_capacidade_mochila(itens: list[Item], tightness_ratio: float = 0.5) -> float:
    """W = Tr * sum(w_k for all items)""" 
    peso_total = sum(item.peso for item in itens)
    return tightness_ratio * peso_total
