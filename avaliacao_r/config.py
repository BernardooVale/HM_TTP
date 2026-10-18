"""
Configurações para a Avaliação do Impacto de R no Travelling Thief Problem (TTP).
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import List


@dataclass
class AvaliacaoRConfig:
    # Diretórios do projeto
    diretorio_raiz: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    diretorio_bateria: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "ttp_simulator" / "bateria_testes")
    diretorio_resultados: Path = field(default_factory=lambda: Path(__file__).resolve().parent / "resultados")
    
    # Grade de fatores multiplicadores para variar a taxa de aluguel R em torno do R nominal da instância
    # Cobre desde R quase nulo (KP domina) até R muito alto (TSP domina)
    fatores_r: List[float] = field(default_factory=lambda: [
        0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0
    ])
    
    # Limiar padrão para o critério de qualidade relativa (0.9 = 90% do ótimo do TTP)
    limiar_qualidade: float = 0.90
    
    # Limites computacionais para busca exata de problemas NP-difíceis
    # Dimensão máxima de cidades para executar B&B exato global do TTP completo
    # (N <= 8: 7! = 5.040 permutações é viável em segundos; N=10 gera 362.880 permutações x 17 pontos de R)
    max_cidades_exato_ttp: int = 8
    
    # Dimensão máxima de cidades para avaliação de R (evita alocação de matriz de 9GB para 33.810 cidades)
    max_cidades_avaliacao: int = 5000
    
    # Timeout por instância em segundos (evita travamento em instâncias grandes)
    timeout_segundos_por_instancia: float = 120.0
    
    # Número de workers para paralelismo (padrão: CPU count ou 4)
    num_workers: int = field(default_factory=lambda: min(os.cpu_count() or 4, 8))

    def __post_init__(self):
        # Garante que os caminhos são objetos Path
        if not isinstance(self.diretorio_raiz, Path):
            self.diretorio_raiz = Path(self.diretorio_raiz)
        if not isinstance(self.diretorio_bateria, Path):
            self.diretorio_bateria = Path(self.diretorio_bateria)
        if not isinstance(self.diretorio_resultados, Path):
            self.diretorio_resultados = Path(self.diretorio_resultados)
        
        # Garante a existência do diretório de resultados
        self.diretorio_resultados.mkdir(parents=True, exist_ok=True)
        (self.diretorio_resultados / "detalhes").mkdir(parents=True, exist_ok=True)
