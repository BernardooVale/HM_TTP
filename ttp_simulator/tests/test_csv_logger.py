import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import os
import tempfile
import csv
from src.utils.csv_logger import CSVLogger
from src.core.models import TTPInstance, Grafo, Cidade, Item

@pytest.fixture
def temp_csv_path():
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_csv_logger_create(temp_csv_path):
    logger = CSVLogger(temp_csv_path)
    assert os.path.exists(temp_csv_path)
    
    with open(temp_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        read_headers = next(reader)
        assert read_headers == CSVLogger.COLUNAS

def test_csv_logger_append(temp_csv_path):
    logger = CSVLogger(temp_csv_path)
    
    mock_instance = TTPInstance(
        nome='Test',
        grafo=Grafo(cidades={1: Cidade(1), 2: Cidade(2)}, edge_weight_type='EUC_2D'),
        itens=[Item(1, 10.0, 50.0, 1)],
        capacidade_mochila=100.0,
        v_min=0.1,
        v_max=1.0,
        taxa_aluguel=1.0,
        constante_degradacao=10.0
    )
    
    resultado = {
        'lucro_liquido_G': 150.25,
        'tempo_viagem_f': 45.5,
        'valor_bruto_itens_g': 200.0,
        'peso_acumulado': 30.0,
        'qtd_itens_coletados': 2
    }
    
    registro = CSVLogger.criar_registro(
        arquivo_instancia="instancia_01.txt",
        modelo_ttp="TTP1",
        solucionador="random",
        seed=42,
        tempo_computacional=0.12345,
        resultado=resultado,
        instance=mock_instance
    )
    
    logger.registrar(registro)
    
    with open(temp_csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]['arquivo_instancia'] == 'instancia_01.txt'
        assert reader[0]['solucionador'] == 'random'
        assert float(reader[0]['lucro_liquido_G']) == 150.25
        assert float(reader[0]['peso_acumulado_mochila']) == 30.0
        assert int(reader[0]['qtd_itens_coletados']) == 2
