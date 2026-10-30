import csv
import os
import tempfile
from pathlib import Path

import pytest
from src.core.models import SolucaoTTP, TTPInstance
from src.runners.batch_runner import BatchRunner
from src.solvers.base import BaseSolver, SolverRegistry
from src.utils.results_manager import (
    classificar_instancia_cenarios,
    obter_destinos_registro,
    obter_familias_solver,
    propagar_registro_3x3,
)


class SimpleUserSolver(BaseSolver):
    """Exemplo de como um usuário adiciona um novo solver em 5 linhas."""
    @property
    def nome(self) -> str:
        return "simple_user_solver"

    def solve(self, instance: TTPInstance) -> SolucaoTTP:
        rota = list(range(1, len(instance.grafo.cidades) + 1))
        # Coleta primeiro item disponível
        plano = {2: [instance.itens[0].id]} if instance.itens else {}
        return SolucaoTTP(rota=rota, plano_coleta=plano)


@pytest.fixture(autouse=True)
def registrar_user_solver():
    SolverRegistry.register(SimpleUserSolver)
    yield


def test_batch_runner():
    test_dir = Path(__file__).resolve().parent.parent / "bateria_testes"
    if not test_dir.exists():
        test_dir = Path("bateria_testes")
    instancia_txt = test_dir / "instancia_01.txt"
    assert instancia_txt.exists()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        out_csv = tf.name

    try:
        runner = BatchRunner(
            test_dir=str(test_dir),
            solvers=["simple_user_solver", "random"],
            models=["TTP1"],
            runs=1,
            output_csv=out_csv,
            base_seed=100,
            workers=2,
            resume=False,
            propagar_resultados=False,
        )
        runner._obter_arquivos_instancias = lambda: [instancia_txt]
        runner.run()

        with open(out_csv, "r", encoding="utf-8") as f:
            linhas = list(csv.DictReader(f))
            assert len(linhas) == 2
            solvers_gravados = {row["solucionador"] for row in linhas}
            assert "simple_user_solver" in solvers_gravados
            assert "random" in solvers_gravados

        # Testa o modo resume
        runner_resume = BatchRunner(
            test_dir=str(test_dir),
            solvers=["simple_user_solver"],
            models=["TTP1"],
            runs=1,
            output_csv=out_csv,
            base_seed=100,
            workers=1,
            resume=True,
            propagar_resultados=False,
        )
        runner_resume._obter_arquivos_instancias = lambda: [instancia_txt]
        runner_resume.run()

        with open(out_csv, "r", encoding="utf-8") as f:
            linhas_pos = list(csv.DictReader(f))
            assert len(linhas_pos) == 2

    finally:
        if os.path.exists(out_csv):
            os.remove(out_csv)


def test_cenarios_e_propagacao():
    # 1. Verifica classificação hierárquica
    assert classificar_instancia_cenarios("instancia_01.txt") == ["lucrativas", "equilibradas", "completo"]
    assert classificar_instancia_cenarios("138.ttp") == ["equilibradas", "completo"]
    assert classificar_instancia_cenarios("50.ttp") == ["completo"]

    # 2. Verifica famílias de solvers
    assert obter_familias_solver("queimada") == ["queimada", "todos"]
    assert obter_familias_solver("queimada_com_2opt") == ["queimada", "todos"]
    assert obter_familias_solver("heuristic_37_refinada") == ["heuristica_37", "todos"]
    assert obter_familias_solver("pack_iterative") == ["todos"]

    # 3. Verifica destinos da matriz 3x3
    destinos_q = obter_destinos_registro("queimada", "instancia_01.txt")
    nomes_destinos_q = {f"{p.parent.name}/{p.name}" for p in destinos_q}
    assert "queimada/lucrativas.csv" in nomes_destinos_q
    assert "queimada/equilibradas.csv" in nomes_destinos_q
    assert "queimada/completo.csv" in nomes_destinos_q
    assert "todos/lucrativas.csv" in nomes_destinos_q
    assert "todos/equilibradas.csv" in nomes_destinos_q
    assert "todos/completo.csv" in nomes_destinos_q
    assert len(destinos_q) == 6

    # 4. Verifica filtro de instâncias por cenário no BatchRunner
    test_dir = Path(__file__).resolve().parent.parent / "bateria_testes"
    if test_dir.exists():
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            temp_dummy = tf.name
        try:
            r_luc = BatchRunner(test_dir=str(test_dir), solvers=["random"], models=["TTP1"], runs=1,
                                output_csv=temp_dummy, base_seed=1, cenario="lucrativas")
            r_eq = BatchRunner(test_dir=str(test_dir), solvers=["random"], models=["TTP1"], runs=1,
                               output_csv=temp_dummy, base_seed=1, cenario="equilibradas")
            r_comp = BatchRunner(test_dir=str(test_dir), solvers=["random"], models=["TTP1"], runs=1,
                                output_csv=temp_dummy, base_seed=1, cenario="completo")

            inst_luc = r_luc._obter_arquivos_instancias()
            inst_eq = r_eq._obter_arquivos_instancias()
            inst_comp = r_comp._obter_arquivos_instancias()

            assert len(inst_luc) <= len(inst_eq)
            assert len(inst_eq) <= len(inst_comp)
            assert len(inst_luc) == 23
            assert len(inst_eq) >= 34
        finally:
            if os.path.exists(temp_dummy):
                os.remove(temp_dummy)
