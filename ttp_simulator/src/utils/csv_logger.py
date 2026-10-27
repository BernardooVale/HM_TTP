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
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLUNAS)
                writer.writeheader()
    
    def registrar(self, dados: Dict[str, Any]) -> None:
        """Append a result row to CSV."""
        with open(self.output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUNAS)
            writer.writerow(dados)

    def obter_chaves_registradas(self) -> set[tuple[str, str, str, int]]:
        """Retorna conjunto de tuplas (arquivo_instancia, modelo_ttp, solucionador, seed) já gravadas."""
        registradas = set()
        if not self.output_path.exists():
            return registradas
        try:
            with open(self.output_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    inst = row.get('arquivo_instancia', '')
                    mod = row.get('modelo_ttp', '')
                    solv = row.get('solucionador', '')
                    s_str = row.get('seed', '0')
                    try:
                        seed = int(s_str)
                    except ValueError:
                        seed = 0
                    registradas.add((inst, mod, solv, seed))
        except Exception:
            pass
        return registradas
    
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
