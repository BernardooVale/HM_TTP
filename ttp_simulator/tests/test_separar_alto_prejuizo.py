import csv
import sys
from pathlib import Path

_raiz = Path(__file__).resolve().parents[2]
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

import pytest
from separar_alto_prejuizo import (
    calcular_estatisticas,
    auditar_instancias,
    filtrar_e_consolidar_resultados,
)


def test_calcular_estatisticas_basico():
    valores = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = calcular_estatisticas(valores)
    assert stats["media"] == 30.0
    assert stats["mediana"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["q1"] == 20.0
    assert stats["q3"] == 40.0


def test_filtrar_e_consolidar_resultados(tmp_path):
    csv_total = tmp_path / "total.csv"
    with open(csv_total, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arquivo_instancia",
                "modelo_ttp",
                "solucionador",
                "lucro_liquido_G",
                "tempo_computacional_seg",
                "tempo_viagem_f",
                "peso_acumulado_mochila",
                "pct_ocupacao_mochila",
                "qtd_itens_coletados",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "arquivo_instancia": "inst_boa.txt",
            "modelo_ttp": "TTP1",
            "solucionador": "solver_a",
            "lucro_liquido_G": "150.0",
            "tempo_computacional_seg": "0.05",
            "tempo_viagem_f": "20.0",
            "peso_acumulado_mochila": "10.0",
            "pct_ocupacao_mochila": "50.0",
            "qtd_itens_coletados": "5",
        })
        writer.writerow({
            "arquivo_instancia": "inst_ruim.txt",
            "modelo_ttp": "TTP1",
            "solucionador": "solver_a",
            "lucro_liquido_G": "-99999.0",
            "tempo_computacional_seg": "0.10",
            "tempo_viagem_f": "5000.0",
            "peso_acumulado_mochila": "1.0",
            "pct_ocupacao_mochila": "1.0",
            "qtd_itens_coletados": "1",
        })

    out_detalhado = tmp_path / "detalhado.csv"
    out_stats = tmp_path / "stats.csv"
    out_norm = tmp_path / "norm.csv"

    filtrar_e_consolidar_resultados(
        csv_total=csv_total,
        nomes_equilibradas={"inst_boa.txt"},
        output_detalhado=out_detalhado,
        output_stats=out_stats,
        output_norm=out_norm,
    )

    assert out_detalhado.exists()
    assert out_stats.exists()
    assert out_norm.exists()

    with open(out_detalhado, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["arquivo_instancia"] == "inst_boa.txt"

    with open(out_norm, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
        assert len(rows) == 1
        assert rows[0]["solver"] == "solver_a"
        assert float(rows[0]["media_G"]) == 1.0
