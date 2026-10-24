"""
Script para geração de cenários aleatórios variados para a bateria de testes do TTP.
"""

import sys
from pathlib import Path

_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.core.models import TTPInstance
from src.scenarios.exporter import exportar_cenario
from src.scenarios.graph_generator import gerar_grafo_concentrado, gerar_grafo_normal
from src.scenarios.item_generator import calcular_capacidade_mochila, gerar_itens


def gerar_cenarios_bateria():
    output_dir = _sim_path / "bateria_testes"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "nome": "cenario_pequeno_10c",
            "n_cidades": 10,
            "n_itens": 20,
            "seed": 101,
            "grafo_tipo": "normal",
            "distribuicao": "uniforme",
            "tightness": 0.5,
            "taxa_aluguel": 1.2,
            "v_min": 0.1,
            "v_max": 1.0,
        },
        {
            "nome": "cenario_cluster_15c",
            "n_cidades": 15,
            "n_itens": 35,
            "seed": 202,
            "grafo_tipo": "concentrado",
            "distribuicao": "concentrado",
            "tightness": 0.3,  # Mochila bem apertada
            "taxa_aluguel": 2.5,  # Aluguel mais alto
            "v_min": 0.1,
            "v_max": 1.0,
        },
        {
            "nome": "cenario_medio_20c",
            "n_cidades": 20,
            "n_itens": 50,
            "seed": 303,
            "grafo_tipo": "normal",
            "distribuicao": "uniforme",
            "tightness": 0.7,  # Mochila folgada
            "taxa_aluguel": 1.0,
            "v_min": 0.1,
            "v_max": 1.0,
        },
        {
            "nome": "cenario_cluster_25c",
            "n_cidades": 25,
            "n_itens": 60,
            "seed": 404,
            "grafo_tipo": "concentrado",
            "distribuicao": "esparso",
            "tightness": 0.4,
            "taxa_aluguel": 1.8,
            "v_min": 0.1,
            "v_max": 1.0,
        },
        {
            "nome": "cenario_desafiador_30c",
            "n_cidades": 30,
            "n_itens": 75,
            "seed": 505,
            "grafo_tipo": "normal",
            "distribuicao": "concentrado",
            "tightness": 0.5,
            "taxa_aluguel": 3.0,  # Alta penalidade de tempo
            "v_min": 0.1,
            "v_max": 1.0,
        },
    ]

    for cfg in configs:
        if cfg["grafo_tipo"] == "concentrado":
            grafo = gerar_grafo_concentrado(cfg["n_cidades"], seed=cfg["seed"])
        else:
            grafo = gerar_grafo_normal(cfg["n_cidades"], seed=cfg["seed"])

        itens = gerar_itens(
            n_itens=cfg["n_itens"],
            cidades=grafo.cidades,
            seed=cfg["seed"],
            distribuicao=cfg["distribuicao"],
            valor_range=(20.0, 600.0),
            peso_range=(5.0, 60.0),
        )

        cap_mochila = calcular_capacidade_mochila(itens, tightness_ratio=cfg["tightness"])

        instance = TTPInstance(
            nome=cfg["nome"],
            grafo=grafo,
            itens=itens,
            capacidade_mochila=cap_mochila,
            v_min=cfg["v_min"],
            v_max=cfg["v_max"],
            taxa_aluguel=cfg["taxa_aluguel"],
            constante_degradacao=1.0,
            dimension=cfg["n_cidades"],
            num_items=cfg["n_itens"],
        )

        filepath = output_dir / f"{cfg['nome']}.txt"
        exportar_cenario(instance, str(filepath))
        print(f"Gerado: {filepath.name} ({cfg['n_cidades']} cidades, {cfg['n_itens']} itens, cap={cap_mochila:.1f})")


if __name__ == "__main__":
    gerar_cenarios_bateria()
