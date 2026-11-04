"""
Gerenciador de Melhores Configurações para Heurísticas do TTP.

Permite que solucionadores como QueimadaSolver e Heuristic37Solver carreguem
automaticamente suas melhores configurações calibradas em tempo de execução
sem depender de hard-code.

Sempre que um Grid Search ou benchmark interno descobre uma combinação que supera
a campeã anterior, ele atualiza automaticamente o arquivo JSON persistido.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Arquivo JSON persistido onde as melhores configurações são salvas
_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = _CONFIG_DIR / "melhores_configuracoes.json"

# Configurações padrão de fallback caso o JSON ainda não exista ou esteja inacessível
FALLBACK_CONFIGURACOES: Dict[str, Dict[str, Any]] = {
    "queimada": {
        "parametros": {
            "temperatura": "total_value",
            "dissipacao": "dijkstra",
            "direcao": "calor_final",
            "metodo_construcao": "backward_edges",
            "afinidade_termica": True,
            "refinamento_rota": True,
            "otimizar_buracos": True,
            "selecao": "greedy_relative",
            "ordem_selecao": "densidade",
            "refinamento": True,
            "gamma_afinidade": 1.0,
            "adaptar_regime_r": False,
        },
        "score_medio": 768.29,
        "origem": "grid_search_paralelo_completo",
        "data_atualizacao": "2026-10-29",
    },
    "heuristic_37": {
        "parametros": {
            "refinamento": True,
            "limiar_simetrico": False,
        },
        "score_medio": 634.42,
        "origem": "benchmark_refinado",
        "data_atualizacao": "2026-10-29",
    },
}


def _carregar_todas_configuracoes() -> Dict[str, Any]:
    """Carrega o dicionário completo do arquivo JSON."""
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                dados = json.load(f)
                if isinstance(dados, dict):
                    return dados
        except Exception as e:
            # Em caso de falha de leitura (ex: concorrência ou arquivo corrompido), usa fallback
            pass

    return dict(FALLBACK_CONFIGURACOES)


def obter_melhor_configuracao(nome_heuristica: str) -> Dict[str, Any]:
    """
    Retorna o dicionário de parâmetros da melhor configuração conhecida para a heurística.
    Ex: 'queimada' ou 'heuristic_37'.
    """
    todas = _carregar_todas_configuracoes()
    registro = todas.get(nome_heuristica)
    fallback = FALLBACK_CONFIGURACOES.get(nome_heuristica, {}).get("parametros", {})

    if registro and isinstance(registro, dict) and "parametros" in registro:
        # Mescla garantindo que todos os campos esperados existam
        config_mesclada = dict(fallback)
        config_mesclada.update(registro["parametros"])
        return config_mesclada

    return dict(fallback)


def salvar_melhor_configuracao(
    nome_heuristica: str,
    parametros: Dict[str, Any],
    score: Optional[float] = None,
    origem: Optional[str] = None,
    detalhes: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Salva uma nova configuração como a melhor para a heurística especificada.
    """
    todas = _carregar_todas_configuracoes()

    novo_registro = {
        "parametros": dict(parametros),
        "score_medio": score,
        "origem": origem or "manual",
        "data_atualizacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if detalhes:
        novo_registro["detalhes"] = detalhes

    todas[nome_heuristica] = novo_registro

    try:
        # Escrita segura atômica
        temp_file = _CONFIG_FILE.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(todas, f, indent=2, ensure_ascii=False)
        temp_file.replace(_CONFIG_FILE)
        return True
    except Exception as e:
        print(f"⚠️ Erro ao salvar configuração campeã em {_CONFIG_FILE}: {e}")
        return False


def atualizar_campeao_se_melhor(
    nome_heuristica: str,
    novos_parametros: Dict[str, Any],
    novo_score: float,
    origem: Optional[str] = None,
    detalhes: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Compara o novo score com o score da configuração atualmente persistida.
    Se o novo score for estritamente superior, atualiza o campeão e retorna True.
    Caso contrário, mantém o anterior e retorna False.
    """
    todas = _carregar_todas_configuracoes()
    registro_atual = todas.get(nome_heuristica, {})
    score_atual = registro_atual.get("score_medio")

    # Atualiza se não havia score anterior ou se o novo score superar o anterior com margem significativa
    deve_atualizar = (score_atual is None) or (novo_score > score_atual + 1e-4)

    if deve_atualizar:
        return salvar_melhor_configuracao(
            nome_heuristica=nome_heuristica,
            parametros=novos_parametros,
            score=novo_score,
            origem=origem,
            detalhes=detalhes,
        )

    return False


# Inicialização automática do arquivo caso não exista
if not _CONFIG_FILE.exists():
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as _f:
            json.dump(FALLBACK_CONFIGURACOES, _f, indent=2, ensure_ascii=False)
    except Exception:
        pass
