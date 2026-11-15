"""
Gerador de instâncias aleatórias equilibradas (com perfis positivos competitivos
e negativos moderados/limiar) em 'ttp_simulator/bateria_testes/aleatorias_equilibradas'.

Critérios:
- Perfil 'competitivo' (~25 instâncias): calibra R (taxa_aluguel) usando a velocidade
  média de referência v_ref = (v_max + v_min) / 2 para que ratio_v_ref fique em [1.1, 2.5].
  Boas soluções tendem a lucro líquido positivo (G > 0), mas com forte trade-off entre
  peso coletado e velocidade.
- Perfil 'negativo_moderado' (~10 instâncias): calibra R usando v_max para que ratio_vmax
  fique em [0.55, 0.95]. Mantém ratio_vmax >= 0.5 (critério de equilibrada do projeto),
  mas tende a G < 0 na prática, testando minimização de prejuízo.

Comandos:
    python scripts/instancias/gerar_aleatorias_equilibradas.py
    python scripts/instancias/gerar_aleatorias_equilibradas.py --limpar
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from scripts.instancias.filtrar_instancias_lucro import (
    calcular_mochila_divisivel,
    calcular_mst_prim,
)
from src.core.models import TTPInstance
from src.scenarios.exporter import exportar_cenario
from src.scenarios.graph_generator import gerar_grafo_concentrado, gerar_grafo_normal
from src.scenarios.item_generator import calcular_capacidade_mochila, gerar_itens


def _especificacoes_bateria() -> List[Dict]:
    """Define as 35 configurações determinísticas (25 competitivas + 10 negativas moderadas)."""
    specs: List[Dict] = []

    faixas = [
        # (porte, qtd_comp, qtd_neg, cidades_range, itens_por_cidade_range)
        ("pequena", 7, 3, (15, 30), (1, 3)),
        ("media", 11, 4, (50, 100), (2, 5)),
        ("grande", 7, 3, (150, 250), (3, 5)),
    ]

    grafos = ["normal", "concentrado"]
    distribuicoes = ["uniforme", "concentrado", "esparso"]
    intensidades = ["baixa", "media", "alta"]
    tightnesses = [0.3, 0.5, 0.7]

    idx_global = 1
    for porte, qtd_comp, qtd_neg, (c_min, c_max), (ipc_min, ipc_max) in faixas:
        perfis = [("competitivo", qtd_comp), ("negativo_moderado", qtd_neg)]
        for perfil, qtd in perfis:
            for k in range(1, qtd + 1):
                seed = 7000 + idx_global * 37
                rng = random.Random(seed)

                n_cidades = rng.randint(c_min, c_max)
                itens_por_cidade = rng.randint(ipc_min, ipc_max)
                n_itens = n_cidades * itens_por_cidade

                grafo_tipo = grafos[(idx_global + k) % len(grafos)]
                distribuicao = distribuicoes[(idx_global + 2 * k) % len(distribuicoes)]
                intensidade = intensidades[(idx_global + k) % len(intensidades)]
                tightness = tightnesses[(idx_global + k) % len(tightnesses)]

                if perfil == "competitivo":
                    ratio_alvo = round(rng.uniform(1.1, 2.5), 4)
                    prefixo = "comp"
                else:
                    ratio_alvo = round(rng.uniform(0.55, 0.95), 4)
                    prefixo = "neg"

                nome = f"rand_{prefixo}_{porte}_{idx_global:02d}_{n_cidades}c_{n_itens}i"
                specs.append({
                    "idx": idx_global,
                    "nome": nome,
                    "porte": porte,
                    "perfil": perfil,
                    "seed": seed,
                    "n_cidades": n_cidades,
                    "n_itens": n_itens,
                    "grafo_tipo": grafo_tipo,
                    "distribuicao": distribuicao,
                    "intensidade": intensidade,
                    "tightness": tightness,
                    "ratio_alvo": ratio_alvo,
                    "v_min": 0.1,
                    "v_max": 1.0,
                })
                idx_global += 1

    return specs


def gerar_instancia_calibrada(spec: Dict) -> tuple[TTPInstance, Dict]:
    """Gera uma instância TTP e calibra taxa_aluguel (R) para atingir o perfil econômico alvo."""
    seed = spec["seed"]
    if spec["grafo_tipo"] == "concentrado":
        n_clusters = max(3, min(6, spec["n_cidades"] // 20 + 2))
        grafo = gerar_grafo_concentrado(spec["n_cidades"], seed=seed, n_clusters=n_clusters)
    else:
        grafo = gerar_grafo_normal(spec["n_cidades"], seed=seed)

    itens = gerar_itens(
        n_itens=spec["n_itens"],
        cidades=grafo.cidades,
        seed=seed + 1,
        distribuicao=spec["distribuicao"],
        valor_range=(20.0, 600.0),
        peso_range=(5.0, 60.0),
        intensidade=spec["intensidade"],
    )

    cap_mochila = calcular_capacidade_mochila(itens, tightness_ratio=spec["tightness"])
    kp_frac, kp_peso = calcular_mochila_divisivel(itens, cap_mochila)
    custo_mst = calcular_mst_prim(grafo)

    v_min = spec["v_min"]
    v_max = spec["v_max"]
    v_ref = (v_max + v_min) / 2.0

    # Calibração de R:
    # - competitivo: ratio_alvo = kp_frac / (R * custo_mst / v_ref)
    # - negativo_moderado: ratio_alvo = kp_frac / (R * custo_mst / v_max)
    v_calib = v_ref if spec["perfil"] == "competitivo" else v_max
    if custo_mst > 0 and kp_frac > 0:
        taxa_aluguel = (kp_frac * v_calib) / (spec["ratio_alvo"] * custo_mst)
    else:
        taxa_aluguel = 1.0

    instance = TTPInstance(
        nome=spec["nome"],
        grafo=grafo,
        itens=itens,
        capacidade_mochila=cap_mochila,
        v_min=v_min,
        v_max=v_max,
        taxa_aluguel=taxa_aluguel,
        constante_degradacao=1.0,
        dimension=spec["n_cidades"],
        num_items=spec["n_itens"],
    )

    custo_vmax = taxa_aluguel * (custo_mst / v_max)
    custo_vref = taxa_aluguel * (custo_mst / v_ref)
    custo_vmin = taxa_aluguel * (custo_mst / v_min)

    metricas = {
        "arquivo": f"{spec['nome']}.txt",
        "porte": spec["porte"],
        "perfil": spec["perfil"],
        "seed": seed,
        "cidades": spec["n_cidades"],
        "itens": spec["n_itens"],
        "grafo_tipo": spec["grafo_tipo"],
        "distribuicao_itens": spec["distribuicao"],
        "intensidade_valor": spec["intensidade"],
        "tightness": spec["tightness"],
        "capacidade_W": cap_mochila,
        "taxa_aluguel_R": taxa_aluguel,
        "valor_mochila_divisivel": kp_frac,
        "peso_mochila_divisivel": kp_peso,
        "custo_mst": custo_mst,
        "razao_vmax": kp_frac / custo_vmax if custo_vmax > 0 else 0.0,
        "razao_vref": kp_frac / custo_vref if custo_vref > 0 else 0.0,
        "razao_vmin": kp_frac / custo_vmin if custo_vmin > 0 else 0.0,
        "ub_lucro_vmax": kp_frac - custo_vmax,
    }
    return instance, metricas


def gerar_bateria_aleatoria_equilibrada(
    output_dir: Path,
    resumo_csv: Path,
    limpar: bool = False,
) -> List[Dict]:
    """Gera todas as instâncias na pasta dedicada e exporta o CSV de resumo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if limpar:
        for arq in output_dir.glob("*.txt"):
            arq.unlink()
        for arq in output_dir.glob("*.ttp"):
            arq.unlink()

    specs = _especificacoes_bateria()
    registros: List[Dict] = []

    print("\n" + "=" * 95)
    print(f"{'GERAÇÃO DE INSTÂNCIAS ALEATÓRIAS EQUILIBRADAS (TTP)':^95}")
    print(f" • Diretório de Saída: {output_dir}")
    print(f" • Total Planejado:    {len(specs)} instâncias (25 competitivas + 10 negativas moderadas)")
    print("=" * 95 + "\n")

    for spec in specs:
        instance, metricas = gerar_instancia_calibrada(spec)
        filepath = output_dir / metricas["arquivo"]
        exportar_cenario(instance, str(filepath), formato="unico")
        registros.append(metricas)

        print(
            f"  [{metricas['perfil']:<17}] {metricas['arquivo']:<38} | "
            f"C={metricas['cidades']:>3} I={metricas['itens']:>4} | "
            f"R={metricas['taxa_aluguel_R']:>7.3f} | "
            f"ratio(v_max)={metricas['razao_vmax']:.2f} "
            f"ratio(v_ref)={metricas['razao_vref']:.2f}"
        )

    resumo_csv.parent.mkdir(parents=True, exist_ok=True)
    colunas = [
        "arquivo", "porte", "perfil", "seed", "cidades", "itens",
        "grafo_tipo", "distribuicao_itens", "intensidade_valor", "tightness",
        "capacidade_W", "taxa_aluguel_R", "valor_mochila_divisivel",
        "peso_mochila_divisivel", "custo_mst", "razao_vmax", "razao_vref",
        "razao_vmin", "ub_lucro_vmax",
    ]
    with open(resumo_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
        writer.writeheader()
        for r in registros:
            writer.writerow({
                **r,
                "capacidade_W": f"{r['capacidade_W']:.2f}",
                "taxa_aluguel_R": f"{r['taxa_aluguel_R']:.6f}",
                "valor_mochila_divisivel": f"{r['valor_mochila_divisivel']:.2f}",
                "peso_mochila_divisivel": f"{r['peso_mochila_divisivel']:.2f}",
                "custo_mst": f"{r['custo_mst']:.2f}",
                "razao_vmax": f"{r['razao_vmax']:.4f}",
                "razao_vref": f"{r['razao_vref']:.4f}",
                "razao_vmin": f"{r['razao_vmin']:.4f}",
                "ub_lucro_vmax": f"{r['ub_lucro_vmax']:.2f}",
            })

    print(f"\n✅ {len(registros)} instâncias geradas em: {output_dir}")
    print(f"✅ Resumo salvo em: {resumo_csv}\n")
    return registros


def main():
    parser = argparse.ArgumentParser(
        description="Gera bateria de instâncias TTP aleatórias equilibradas (competitivas e negativas moderadas)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_sim_path / "bateria_testes" / "aleatorias_equilibradas"),
        help="Pasta de destino das instâncias geradas",
    )
    parser.add_argument(
        "--resumo-csv",
        type=str,
        default=str(_raiz / "results" / "triagem" / "aleatorias_equilibradas_resumo.csv"),
        help="CSV de resumo com os parâmetros e métricas econômicas geradas",
    )
    parser.add_argument(
        "--limpar",
        action="store_true",
        help="Remove arquivos .txt/.ttp antigos da pasta de destino antes de gerar",
    )
    args = parser.parse_args()

    gerar_bateria_aleatoria_equilibrada(
        output_dir=Path(args.output_dir).resolve(),
        resumo_csv=Path(args.resumo_csv).resolve(),
        limpar=args.limpar,
    )


if __name__ == "__main__":
    main()
