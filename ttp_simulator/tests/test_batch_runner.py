import csv
import os
import tempfile
from pathlib import Path

import pytest
from src.core.models import SolucaoTTP, TTPInstance
from src.runners.batch_runner import BatchRunner
from src.solvers.base import BaseSolver, SolverRegistry


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


    test_dir = Path(__file__).resolve().parent.parent / "bateria_testes"
    if not test_dir.exists():
        test_dir = Path("bateria_testes")
    # Usa arquivo de teste pequeno existente
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
            resume=False
        )
        # Filtra para executar apenas sobre instancia_01.txt para teste rápido
        runner._obter_arquivos_instancias = lambda: [instancia_txt]
        runner.run()

        # Verifica se o CSV foi gerado com os resultados do custom solver
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
            resume=True
        )
        runner_resume._obter_arquivos_instancias = lambda: [instancia_txt]
        # Não deve adicionar novas linhas pois já existe
        runner_resume.run()

        with open(out_csv, "r", encoding="utf-8") as f:
            linhas_pos = list(csv.DictReader(f))
            assert len(linhas_pos) == 2

    finally:
        if os.path.exists(out_csv):
            os.remove(out_csv)
