"""
Gerenciador Central de Resultados e Histórico do Traveling Thief Problem.
Organiza a estrutura de diretórios 3x3 em results/:
- Famílias: 'todos', 'queimada', 'heuristica_37'
- Cenários: 'completo.csv', 'equilibradas.csv', 'lucrativas.csv'
- Subpastas auxiliares: 'queimada/grid_search/' e 'triagem/'

Recursos de Propagação Inteligente:
- Relação de inclusão de instâncias: Lucrativas ⊂ Equilibradas ⊂ Completas.
- Relação de famílias de solvers: Queimada ⊂ Todos, Heurística 37 ⊂ Todos.
- Propagação automática de qualquer resultado executado para todos os CSVs válidos na matriz 3x3.
- Prevenção de duplicatas por chave (instancia, modelo_ttp, solucionador, seed).
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Colunas canônicas da matriz 3x3 de execução
COLUNAS_PADRAO_3X3 = [
    "id_execucao",
    "timestamp",
    "arquivo_instancia",
    "modelo_ttp",
    "solucionador",
    "seed",
    "tempo_computacional_seg",
    "lucro_liquido_G",
    "tempo_viagem_f",
    "valor_bruto_itens_g",
    "peso_acumulado_mochila",
    "capacidade_mochila_W",
    "pct_ocupacao_mochila",
    "qtd_itens_coletados",
    "total_itens_instancia",
    "qtd_cidades",
]

# Raiz do projeto e diretório results/
_raiz = Path(__file__).resolve().parents[3]
RESULTS_DIR = _raiz / "results"
SIM_PATH = _raiz / "ttp_simulator"
DIR_BATERIA = SIM_PATH / "bateria_testes"

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

# Caches globais em memória para instâncias e chaves de arquivos
_CACHE_INSTANCIAS_LUCRATIVAS: Optional[Set[str]] = None
_CACHE_INSTANCIAS_EQUILIBRADAS: Optional[Set[str]] = None
_CACHE_CHAVES_ARQUIVO: Dict[Path, Set[Tuple[str, str, str, int]]] = {}

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


def obter_instancias_lucrativas() -> Set[str]:
    """
    Retorna o conjunto de nomes de arquivos das instâncias matematicamente lucrativas (UB > 0).
    Lê a partir de results/triagem/filtro_instancias_lucrativas.csv com cache em memória.
    """
    global _CACHE_INSTANCIAS_LUCRATIVAS
    if _CACHE_INSTANCIAS_LUCRATIVAS is not None:
        return _CACHE_INSTANCIAS_LUCRATIVAS

    instancias = set()
    arq_triagem = DIR_TRIAGEM / "filtro_instancias_lucrativas.csv"
    if arq_triagem.exists():
        try:
            with open(arq_triagem, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    pot = row.get("potencialmente_lucrativa", "").strip().upper()
                    if pot in ("SIM", "TRUE", "1"):
                        nome = row.get("arquivo_origem", "").strip()
                        if nome:
                            instancias.add(nome)
        except Exception:
            pass

    _CACHE_INSTANCIAS_LUCRATIVAS = instancias
    return _CACHE_INSTANCIAS_LUCRATIVAS


def obter_instancias_equilibradas() -> Set[str]:
    """
    Retorna o conjunto de nomes de arquivos das instâncias equilibradas (Razão >= 0.5).
    Lê a partir de results/triagem/triagem_alto_prejuizo.csv e complementa com arquivos
    presentes na raiz de bateria_testes/ (caso a separação de alto prejuízo tenha sido executada).
    """
    global _CACHE_INSTANCIAS_EQUILIBRADAS
    if _CACHE_INSTANCIAS_EQUILIBRADAS is not None:
        return _CACHE_INSTANCIAS_EQUILIBRADAS

    instancias = set()
    arq_triagem = DIR_TRIAGEM / "triagem_alto_prejuizo.csv"
    if arq_triagem.exists():
        try:
            with open(arq_triagem, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter=";")
                for row in reader:
                    cls_val = row.get("classificacao", "").strip().upper()
                    if cls_val == "EQUILIBRADA":
                        nome = row.get("arquivo", "").strip()
                        if nome:
                            instancias.add(nome)
        except Exception:
            pass

    # Fallback/complemento: instâncias presentes na raiz de bateria_testes (fora de alto_prejuizo/quarentena)
    if DIR_BATERIA.exists():
        for arq in DIR_BATERIA.iterdir():
            if arq.is_file() and arq.suffix in (".txt", ".ttp"):
                instancias.add(arq.name)

    _CACHE_INSTANCIAS_EQUILIBRADAS = instancias
    return _CACHE_INSTANCIAS_EQUILIBRADAS


def classificar_instancia_cenarios(nome_instancia: str) -> List[str]:
    """
    Retorna a lista de cenários aos quais a instância pertence pela hierarquia de inclusão:
    Lucrativas ⊂ Equilibradas ⊂ Completas

    Retorno:
    - Se lucrativa: ['lucrativas', 'equilibradas', 'completo']
    - Se equilibrada: ['equilibradas', 'completo']
    - Se alto prejuízo / outra: ['completo']
    """
    nome = Path(nome_instancia).name.strip()
    lucrativas = obter_instancias_lucrativas()
    equilibradas = obter_instancias_equilibradas()

    if nome in lucrativas:
        return ["lucrativas", "equilibradas", "completo"]
    elif nome in equilibradas:
        return ["equilibradas", "completo"]
    else:
        return ["completo"]


def obter_familias_solver(solucionador: str) -> List[str]:
    """
    Retorna as famílias de solucionadores às quais um solver pertence:
    - Queimada: ['queimada', 'todos']
    - Heurística dos 37%: ['heuristica_37', 'todos']
    - Baselines / Literatura / Outros: ['todos']
    """
    solv = solucionador.strip().lower()
    if "queimada" in solv:
        return ["queimada", "todos"]
    elif "37" in solv or "heuristic_37" in solv:
        return ["heuristica_37", "todos"]
    else:
        return ["todos"]


def obter_destinos_registro(solucionador: str, nome_instancia: str) -> List[Path]:
    """
    Calcula todos os caminhos CSV na matriz 3x3 onde o resultado de uma execução
    deve ser gravado, combinando a família do solver com os cenários da instância.
    """
    familias = obter_familias_solver(solucionador)
    cenarios = classificar_instancia_cenarios(nome_instancia)

    destinos = []
    vistos = set()
    for fam in familias:
        for cen in cenarios:
            p = obter_caminho_cenario(familia=fam, cenario=cen, criar_pai=True)
            if p not in vistos:
                destinos.append(p)
                vistos.add(p)

    return destinos


def _obter_chaves_arquivo(caminho_csv: Path) -> Set[Tuple[str, str, str, int]]:
    """Carrega com cache as chaves (arquivo_instancia, modelo_ttp, solucionador, seed) de um CSV."""
    p_res = caminho_csv.resolve()
    if p_res in _CACHE_CHAVES_ARQUIVO:
        return _CACHE_CHAVES_ARQUIVO[p_res]

    chaves = set()
    if p_res.exists() and p_res.stat().st_size > 0:
        try:
            with open(p_res, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    inst = row.get("arquivo_instancia", "").strip()
                    mod = row.get("modelo_ttp", "").strip()
                    solv = row.get("solucionador", "").strip()
                    s_str = row.get("seed", "0").strip()
                    try:
                        seed = int(s_str)
                    except ValueError:
                        seed = 0
                    if inst and solv:
                        chaves.add((inst, mod, solv, seed))
        except Exception:
            pass

    _CACHE_CHAVES_ARQUIVO[p_res] = chaves
    return chaves


def propagar_registro_3x3(
    dados: Dict[str, Any],
    ignorar_caminhos: Optional[Set[Path]] = None,
) -> List[Path]:
    """
    Propaga um registro de execução para todos os CSVs válidos na matriz 3x3:
    1. Determina as famílias válidas (ex: 'queimada' e 'todos')
    2. Determina os cenários válidos (ex: 'lucrativas', 'equilibradas', 'completo')
    3. Para cada destino, se o registro ainda não constar no arquivo, adiciona atomicamente.
    Retorna a lista de caminhos adicionais onde o dado foi gravado.
    """
    solv = str(dados.get("solucionador", "")).strip()
    inst = str(dados.get("arquivo_instancia", "")).strip()
    mod = str(dados.get("modelo_ttp", "TTP1")).strip()
    s_raw = dados.get("seed", 0)
    try:
        seed = int(s_raw)
    except (ValueError, TypeError):
        seed = 0

    if not solv or not inst:
        return []

    chave = (inst, mod, solv, seed)
    destinos = obter_destinos_registro(solv, inst)
    gravados = []

    ignorar_resolvidos = {p.resolve() for p in ignorar_caminhos} if ignorar_caminhos else set()

    for dest in destinos:
        dest_res = dest.resolve()
        if dest_res in ignorar_resolvidos:
            continue

        chaves_existentes = _obter_chaves_arquivo(dest)
        if chave in chaves_existentes:
            continue

        # Garante diretório pai e cabeçalho se arquivo novo
        dest.parent.mkdir(parents=True, exist_ok=True)
        precisa_cabecalho = (not dest.exists()) or (dest.stat().st_size == 0)

        try:
            with open(dest, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUNAS_PADRAO_3X3)
                if precisa_cabecalho:
                    writer.writeheader()
                writer.writerow(dados)
            chaves_existentes.add(chave)
            gravados.append(dest)
        except Exception:
            pass

    return gravados


def sincronizar_matriz_resultados(verbose: bool = True) -> Dict[str, int]:
    """
    Varre todos os CSVs da matriz 3x3 em results/ e propaga retrospectivamente
    quaisquer execuções registradas em cenários ou famílias mais amplos para os
    seus arquivos irmãos/filhos válidos, garantindo consistência total e sem duplicações.
    """
    organizar_estrutura_results()
    total_propagados = 0
    por_destino: Dict[str, int] = {}

    todos_csvs = [
        obter_caminho_cenario("todos", "completo"),
        obter_caminho_cenario("todos", "equilibradas"),
        obter_caminho_cenario("todos", "lucrativas"),
        obter_caminho_cenario("queimada", "completo"),
        obter_caminho_cenario("queimada", "equilibradas"),
        obter_caminho_cenario("queimada", "lucrativas"),
        obter_caminho_cenario("heuristica_37", "completo"),
        obter_caminho_cenario("heuristica_37", "equilibradas"),
        obter_caminho_cenario("heuristica_37", "lucrativas"),
    ]

    for arq in todos_csvs:
        if not arq.exists() or arq.stat().st_size == 0:
            continue
        try:
            with open(arq, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    adicionados = propagar_registro_3x3(row, ignorar_caminhos={arq.resolve()})
                    for dest in adicionados:
                        d_str = str(dest.relative_to(RESULTS_DIR))
                        por_destino[d_str] = por_destino.get(d_str, 0) + 1
                        total_propagados += 1
        except Exception as e:
            if verbose:
                print(f"⚠️ Erro ao ler {arq.name} durante sincronização: {e}")

    if verbose:
        print(f"\n🔄 Sincronização da matriz 3x3 concluída: {total_propagados} registros propagados.")
        for d, count in por_destino.items():
            print(f"   • {d}: +{count} novos registros")

    return por_destino


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
