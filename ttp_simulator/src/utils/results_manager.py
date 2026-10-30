"""
Gerenciador Central de Resultados e Histórico do Traveling Thief Problem.
Organiza a estrutura de diretórios 3x3 em results/:
- Famílias: 'todos', 'queimada', 'heuristica_37'
- Cenários: 'completo.csv', 'equilibradas.csv', 'lucrativas.csv'
- Subpastas auxiliares: 'queimada/grid_search/' e 'triagem/'
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Raiz do projeto e diretório results/
_raiz = Path(__file__).resolve().parents[3]
RESULTS_DIR = _raiz / "results"

# Subpastas padronizadas 3x3
DIR_TODOS = RESULTS_DIR / "todos"
DIR_QUEIMADA = RESULTS_DIR / "queimada"
DIR_HEURISTICA_37 = RESULTS_DIR / "heuristica_37"
DIR_GRID_SEARCH = DIR_QUEIMADA / "grid_search"
DIR_TRIAGEM = RESULTS_DIR / "triagem"

# Aliases de compatibilidade legada
DIR_MASTER = DIR_TODOS
DIR_CENARIOS_EQUILIBRADAS = DIR_TODOS
DIR_CENARIOS_LUCRATIVAS = DIR_TODOS
DIR_WEB = DIR_TODOS

# Mapeamento canônico de nomes de arquivos (novos e legados) para seus caminhos finais exatos
MAPA_ARQUIVOS_CANONICO: Dict[str, Path] = {
    # Estrutura 3x3 direta (default todos)
    "completo.csv": DIR_TODOS / "completo.csv",
    "equilibradas.csv": DIR_TODOS / "equilibradas.csv",
    "lucrativas.csv": DIR_TODOS / "lucrativas.csv",
    # Aliases legados -> mapeados para a nova estrutura 3x3
    "bateria_solucoes_detalhada.csv": DIR_TODOS / "completo.csv",
    "bateria_equilibradas_detalhada.csv": DIR_TODOS / "equilibradas.csv",
    "bateria_lucrativas_detalhada.csv": DIR_TODOS / "lucrativas.csv",
    "historico_queimada.csv": DIR_QUEIMADA / "completo.csv",
    "comparativo_heuristicas.csv": DIR_QUEIMADA / "completo.csv",
    "comparativo_estatisticas_ttp1_ttp2.csv": DIR_TODOS / "completo.csv",
    "comparativo_estatisticas_normalizadas.csv": DIR_TODOS / "completo.csv",
    "comparativo_estatisticas_equilibradas.csv": DIR_TODOS / "equilibradas.csv",
    "comparativo_estatisticas_equilibradas_normalizadas.csv": DIR_TODOS / "equilibradas.csv",
    "comparativo_estatisticas_lucrativas.csv": DIR_TODOS / "lucrativas.csv",
    # Triagem de instâncias
    "triagem_alto_prejuizo.csv": DIR_TRIAGEM / "triagem_alto_prejuizo.csv",
    "filtro_instancias_lucrativas.csv": DIR_TRIAGEM / "filtro_instancias_lucrativas.csv",
    # Grid Search da Queimada
    "grid_search_queimada.csv": DIR_GRID_SEARCH / "grid_search_queimada.csv",
    "grid_search_queimada_completo.csv": DIR_GRID_SEARCH / "grid_search_queimada_completo.csv",
    "grid_search_queimada_direcionado.csv": DIR_GRID_SEARCH / "grid_search_queimada_direcionado.csv",
    "grid_search_queimada_padrao.csv": DIR_GRID_SEARCH / "grid_search_queimada_padrao.csv",
    "grid_search_queimada_rapido.csv": DIR_GRID_SEARCH / "grid_search_queimada_rapido.csv",
}


def obter_caminho_cenario(familia: str = "todos", cenario: str = "completo", criar_pai: bool = True) -> Path:
    """
    Retorna o caminho canônico na matriz 3x3:
    results/<familia>/<cenario>.csv
    """
    fam = familia.strip().lower()
    if fam in ("37", "heuristic_37", "heuristica_37"):
        pasta = DIR_HEURISTICA_37
    elif fam in ("queimada",):
        pasta = DIR_QUEIMADA
    else:
        pasta = DIR_TODOS

    cen = cenario.strip().lower().replace(".csv", "")
    if cen not in ("completo", "equilibradas", "lucrativas"):
        cen = "completo"

    if criar_pai:
        pasta.mkdir(parents=True, exist_ok=True)
    return pasta / f"{cen}.csv"


def resolver_caminho_csv(nome_ou_caminho: str, criar_pai: bool = True) -> Path:
    """
    Resolve um caminho de arquivo CSV para seu destino canônico organizado na estrutura 3x3.
    """
    p = Path(nome_ou_caminho)

    if p.is_file():
        return p.resolve()

    # Se foi passado algo como "queimada/equilibradas.csv" ou "todos/completo.csv"
    if len(p.parts) >= 2:
        cand = RESULTS_DIR / p
        if cand.is_file():
            return cand.resolve()
        if p.parts[-2].lower() in ("todos", "queimada", "heuristica_37", "grid_search", "triagem"):
            destino = RESULTS_DIR / p.parts[-2].lower() / p.name
            if criar_pai:
                destino.parent.mkdir(parents=True, exist_ok=True)
            return destino.resolve()

    nome_arquivo = p.name

    if nome_arquivo in MAPA_ARQUIVOS_CANONICO:
        caminho_canonico = MAPA_ARQUIVOS_CANONICO[nome_arquivo]
        if criar_pai:
            caminho_canonico.parent.mkdir(parents=True, exist_ok=True)
        return caminho_canonico.resolve()

    if p.parent == RESULTS_DIR or not p.is_absolute():
        caminho = RESULTS_DIR / nome_arquivo
        if criar_pai:
            caminho.parent.mkdir(parents=True, exist_ok=True)
        return caminho.resolve()

    if criar_pai and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def organizar_estrutura_results() -> Dict[str, str]:
    """Garante que os diretórios da estrutura 3x3 existem."""
    for pasta in [DIR_TODOS, DIR_QUEIMADA, DIR_HEURISTICA_37, DIR_GRID_SEARCH, DIR_TRIAGEM]:
        pasta.mkdir(parents=True, exist_ok=True)
    return {}


def detectar_contexto_e_historico(
    solvers: Optional[List[str]] = None,
    output_caminho: Optional[str] = None,
    history_caminho: Optional[str] = None,
    cenario: str = "completo",
) -> Tuple[str, Path, Path]:
    """
    Detecta automaticamente a família de execução ('queimada', 'heuristica_37' ou 'todos')
    e o cenário ('completo', 'equilibradas', 'lucrativas'), definindo o histórico e a saída padrão.

    Retorna: (contexto_familia, caminho_historico, caminho_output)
    """
    organizar_estrutura_results()

    familia = "todos"
    if solvers:
        nomes_solvers = [s.strip().lower() for s in solvers if s.strip()]
        if nomes_solvers:
            if all("queimada" in s for s in nomes_solvers):
                familia = "queimada"
            elif all("37" in s for s in nomes_solvers):
                familia = "heuristica_37"

    padrao_csv = obter_caminho_cenario(familia=familia, cenario=cenario)

    if history_caminho:
        caminho_hist = resolver_caminho_csv(history_caminho)
    else:
        caminho_hist = padrao_csv

    if output_caminho:
        caminho_out = resolver_caminho_csv(output_caminho)
    else:
        caminho_out = padrao_csv

    return familia, caminho_hist, caminho_out
