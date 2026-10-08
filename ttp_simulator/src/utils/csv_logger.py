import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.core.models import TTPInstance

class CSVLogger:
    COLUNAS = [
        'id_execucao', 'timestamp', 'arquivo_instancia', 'modelo_ttp',
        'solucionador', 'seed', 'tempo_computacional_seg', 'lucro_liquido_G',
        'tempo_viagem_f', 'valor_bruto_itens_g', 'peso_acumulado_mochila',
        'capacidade_mochila_W', 'pct_ocupacao_mochila', 'qtd_itens_coletados',
        'total_itens_instancia', 'qtd_cidades'
    ]
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._inicializar_csv()
    
    def _inicializar_csv(self):
        if not self.output_path.exists():
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUNAS)
                writer.writeheader()
    
    def registrar(self, dados: Dict[str, Any]) -> None:
        """Append a result row to CSV."""
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUNAS)
            writer.writerow(dados)
    
    @staticmethod
    def criar_registro(
        arquivo_instancia: str,
        modelo_ttp: str,
        solucionador: str,
        seed: int,
        tempo_computacional: float,
        resultado: Dict[str, Any],
        instance: 'TTPInstance'
    ) -> Dict[str, Any]:
        return {
            'id_execucao': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'arquivo_instancia': arquivo_instancia,
            'modelo_ttp': modelo_ttp,
            'solucionador': solucionador,
            'seed': seed,
            'tempo_computacional_seg': round(tempo_computacional, 6),
            'lucro_liquido_G': round(resultado['lucro_liquido_G'], 4),
            'tempo_viagem_f': round(resultado['tempo_viagem_f'], 4),
            'valor_bruto_itens_g': round(resultado['valor_bruto_itens_g'], 4),
            'peso_acumulado_mochila': round(resultado['peso_acumulado'], 4),
            'capacidade_mochila_W': instance.capacidade_mochila,
            'pct_ocupacao_mochila': round(resultado['peso_acumulado'] / instance.capacidade_mochila * 100, 2) if instance.capacidade_mochila > 0 else 0,
            'qtd_itens_coletados': resultado['qtd_itens_coletados'],
            'total_itens_instancia': len(instance.itens),
            'qtd_cidades': len(instance.grafo.cidades)
        }
