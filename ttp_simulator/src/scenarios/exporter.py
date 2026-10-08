import os
from pathlib import Path
from src.core.models import TTPInstance

def exportar_cenario(
    instance: TTPInstance,
    filepath: str,
    formato: str = 'unico'  # 'unico' (Mode A) or 'multiplo' (Mode B)
) -> None:
    path = Path(filepath)
    
    if formato == 'unico':
        _export_single_file(instance, path)
    elif formato == 'multiplo':
        _export_multiple_files(instance, path)

def _export_single_file(instance: TTPInstance, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"PROBLEM NAME: {instance.nome}\n")
        f.write("KNAPSACK DATA TYPE: UNCORRELATED\n")
        f.write(f"DIMENSION: {instance.dimension}\n")
        f.write(f"NUMBER OF ITEMS: {instance.num_items}\n")
        f.write(f"CAPACITY OF KNAPSACK: {instance.capacidade_mochila}\n")
        f.write(f"MIN SPEED: {instance.v_min}\n")
        f.write(f"MAX SPEED: {instance.v_max}\n")
        f.write(f"RENTING RATIO: {instance.taxa_aluguel}\n")
        f.write(f"EDGE_WEIGHT_TYPE: {instance.grafo.edge_weight_type}\n")
        f.write(f"DEGRADATION CONSTANT: {instance.constante_degradacao}\n")
        
        f.write("NODE_COORD_SECTION\n")
        for cid, cidade in instance.grafo.cidades.items():
            if instance.grafo.edge_weight_type == 'EUC_2D':
                f.write(f"{cid}\t{cidade.x:.4f}\t{cidade.y:.4f}\n")
            else:
                f.write(f"{cid}\t0.0\t0.0\n") # Fallback for EXPLICIT
            
        f.write("ITEMS_SECTION\n")
        for item in instance.itens:
            f.write(f"{item.id}\t{item.valor:.4f}\t{item.peso:.4f}\t{item.cidade_origem}\n")

def _export_multiple_files(instance: TTPInstance, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Params
    with open(base_dir / "params.txt", 'w', encoding='utf-8') as f:
        f.write(f"PROBLEM NAME: {instance.nome}\n")
        f.write(f"DIMENSION: {instance.dimension}\n")
        f.write(f"NUMBER OF ITEMS: {instance.num_items}\n")
        f.write(f"CAPACITY OF KNAPSACK: {instance.capacidade_mochila}\n")
        f.write(f"MIN SPEED: {instance.v_min}\n")
        f.write(f"MAX SPEED: {instance.v_max}\n")
        f.write(f"RENTING RATIO: {instance.taxa_aluguel}\n")
        f.write(f"EDGE_WEIGHT_TYPE: {instance.grafo.edge_weight_type}\n")
        f.write(f"DEGRADATION CONSTANT: {instance.constante_degradacao}\n")

    # Nodes
    with open(base_dir / "nodes.txt", 'w', encoding='utf-8') as f:
        for cid, cidade in instance.grafo.cidades.items():
            if instance.grafo.edge_weight_type == 'EUC_2D':
                f.write(f"{cid}\t{cidade.x:.4f}\t{cidade.y:.4f}\n")
            else:
                f.write(f"{cid}\t0.0\t0.0\n")
            
    # Items
    with open(base_dir / "items.txt", 'w', encoding='utf-8') as f:
        for item in instance.itens:
            f.write(f"{item.id}\t{item.valor:.4f}\t{item.peso:.4f}\t{item.cidade_origem}\n")

    # Distances for EXPLICIT
    if instance.grafo.edge_weight_type == 'EXPLICIT':
        with open(base_dir / "distances.txt", 'w', encoding='utf-8') as f:
            for (n1, n2), dist in instance.grafo.distancias.items():
                f.write(f"{n1}\t{n2}\t{dist:.4f}\n")
