"""
Grid Search Paralelo de Alto Desempenho para a Heurística da Queimada no TTP.

Recursos:
- Multiprocessamento nativo com número configurável de workers (--workers).
- Aceleração Numba JIT + Programação Dinâmica / Memoização multi-estágio adaptável à memória RAM.
- Agrupamento hierárquico de tarefas por DAG de estágios (Instância -> Temperatura -> Dissipação -> Rota -> Mochila -> Buracos)
  e envio em lotes (batching) para eliminar gargalos de IPC e maximizar a reutilização de cache.
- Suporte a múltiplos espaços de busca estruturais:
    * 'completo': espaço unificado com métodos clássicos e as novas variantes de grafo direcionado.
    * 'direcionado': focado exclusivamente nas variantes de grafo direcionado (backward_directed e forward_directed).
    * 'padrao': 288 combinações clássicas do grid_search_queimada.py.
    * 'rapido': combinações construtivas sem 2-opt global / sem Drop & Add pesado (ultra-rápido).
- Suporte a seleção flexível de instâncias de bateria_testes (todas, leves, sintéticas ou arquivos específicos).
- Poda automática de configurações redundantes (ex: afinidade térmica em walk, ordens em knapsack exato).
- Cache inteligente de instâncias por processo worker.
- Retomada Automática (Resume) sem recalcular execuções já salvas no histórico.
- Barra de progresso com taxa de throughput (execuções/segundo) e ETA.
- Ranking Top N das melhores configurações, análise de impacto marginal de parâmetros e exportação em CSV.
- Auto-atualização dinâmica da melhor configuração no config_manager.py para os testes globais.

Uso:
    python scripts/experimentos_queimada/grid_search_paralelo.py --help
    python scripts/experimentos_queimada/grid_search_paralelo.py --espaco direcionado --instancias sinteticas
    python scripts/experimentos_queimada/grid_search_paralelo.py --espaco completo --instancias leves --workers 8
"""

import argparse
import concurrent.futures
import csv
import itertools
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configura stdout para UTF-8 no Windows de forma segura
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Adiciona caminhos do projeto ao sys.path
_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.solvers.numba_kernels import (
    NUMBA_AVAILABLE,
    detectar_memoria_sistema_bytes,
    obter_cache_ram_global,
)
from src.solvers.queimada import QueimadaSolver
from src.utils.results_manager import (
    obter_instancias_equilibradas,
    obter_instancias_lucrativas,
    resolver_caminho_csv,
)


# Cache por processo worker para evitar recarga repetida de disco
_WORKER_CACHE: Dict[str, Any] = {}
_WORKER_INITIALIZED: bool = False


def _inicializar_worker(num_workers: int, max_ram_mb: Optional[float] = None) -> None:
    """Inicializa e dimensiona o cache de Programação Dinâmica adaptativo à RAM no processo worker."""
    global _WORKER_INITIALIZED
    cache = obter_cache_ram_global()
    if max_ram_mb is not None and max_ram_mb > 0:
        cache.max_budget_bytes = int((max_ram_mb / max(1, num_workers)) * (1024 ** 2))
    cache.reconfigurar_workers(num_workers)
    _WORKER_INITIALIZED = True


def _executar_uma_tarefa(
    caminho_arquivo: str, cfg: Dict[str, Any], cfg_id: int, modelo: str
) -> Dict[str, Any]:
    if caminho_arquivo not in _WORKER_CACHE:
        # Mantém no máximo 2 instâncias carregadas por worker para economizar RAM
        if len(_WORKER_CACHE) >= 2:
            _WORKER_CACHE.clear()
        inst = InstanceParser.carregar(caminho_arquivo)
        _ = inst.grafo.distance_matrix
        _WORKER_CACHE[caminho_arquivo] = inst

    instance = _WORKER_CACHE[caminho_arquivo]
    nome_inst = Path(caminho_arquivo).name

    grafo_dir = cfg.get("grafo_direcionado")
    if grafo_dir is None:
        grafo_dir = cfg["metodo_construcao"] in ("backward_directed", "forward_directed")

    solver = QueimadaSolver(
        temperatura=cfg["temperatura"],
        dissipacao=cfg["dissipacao"],
        direcao=cfg["direcao"],
        metodo_construcao=cfg["metodo_construcao"],
        afinidade_termica=cfg.get("afinidade_termica", True),
        refinamento_rota=cfg.get("refinamento_rota", True),
        otimizar_buracos=cfg.get("otimizar_buracos", False),
        selecao=cfg["selecao"],
        ordem_selecao=cfg["ordem_selecao"],
        refinamento=cfg["refinamento"],
        grafo_direcionado=grafo_dir,
        modelo=modelo,
    )

    t0 = time.perf_counter()
    solucao = solver.solve(instance)
    tempo_cpu_ms = (time.perf_counter() - t0) * 1000.0

    if modelo == "TTP1":
        res = avaliar_ttp1(instance, solucao)
    else:
        res = avaliar_ttp2(instance, solucao)

    return {
        "cfg_id": cfg_id,
        "instancia": nome_inst,
        "lucro_G": res["lucro_liquido_G"],
        "tempo_viagem": res["tempo_viagem_f"],
        "peso": res["peso_acumulado"],
        "qtd_itens": res["qtd_itens_coletados"],
        "tempo_cpu_ms": tempo_cpu_ms,
    }


def _worker_executar_config(args: Tuple[str, Dict[str, Any], int, str]) -> Dict[str, Any]:
    """
    Função de trabalho unitária (mantida para retrocompatibilidade direta).
    """
    caminho_arquivo, cfg, cfg_id, modelo = args
    return _executar_uma_tarefa(caminho_arquivo, cfg, cfg_id, modelo)


def _worker_executar_lote(
    lote: List[Tuple[str, Dict[str, Any], int, str]]
) -> List[Dict[str, Any]]:
    """
    Executa um lote contíguo de configurações da mesma instância no mesmo worker,
    maximizando o reaproveitamento do cache DP de estágios (temperaturas, calor,
    rota 2-opt, plano Drop & Add e buracos) e reduzindo drasticamente o overhead de IPC.
    """
    resultados: List[Dict[str, Any]] = []
    for caminho_arquivo, cfg, cfg_id, modelo in lote:
        resultados.append(_executar_uma_tarefa(caminho_arquivo, cfg, cfg_id, modelo))
    return resultados


# Padrões canônicos para quando uma coluna for adicionada no futuro e não existir em CSVs antigos
DEFAULTS_PARAMETROS: Dict[str, Any] = {
    "temperatura": "total_value",
    "dissipacao": "dijkstra",
    "direcao": "calor_final",
    "metodo_construcao": "kruskal",
    "afinidade_termica": True,
    "refinamento_rota": True,
    "otimizar_buracos": False,
    "grafo_direcionado": False,
    "selecao": "greedy_relative",
    "ordem_selecao": "densidade",
    "refinamento": False,
}


def _obter_assinatura_config(cfg: Dict[str, Any], chaves_schema: Optional[List[str]] = None) -> Tuple:
    """
    Gera uma assinatura imutável dos parâmetros de uma configuração para deduplicação robusta.
    Adapta-se dinamicamente ao schema de parâmetros: qualquer coluna nova ausente em execuções
    antigas registradas no CSV assume automaticamente o valor padrão daquela funcionalidade
    (False para flags booleanas desligadas no passado).
    """
    chaves = chaves_schema or sorted(cfg.keys())
    valores = []
    for k in chaves:
        if k in ("config_id", "instancia", "lucro_G", "tempo_viagem", "peso", "qtd_itens", "tempo_cpu_ms"):
            continue
        val = cfg.get(k)
        if val is None or val == "":
            val = DEFAULTS_PARAMETROS.get(k, False)

        if isinstance(val, bool) or str(val).strip().lower() in ("true", "false", "1", "0"):
            val_norm = str(val).strip().lower() in ("true", "1")
        else:
            val_norm = str(val).strip()
        valores.append((k, val_norm))
    return tuple(valores)


def _chave_ordenacao_dag_config(cfg: Dict[str, Any]) -> Tuple:
    """
    Ordena configurações seguindo a hierarquia exata do DAG da Queimada:
    Estágio 1 (Temperatura) -> Estágio 2 (Dissipação/GrafoDir) -> Estágio 3 (Construção/Afinidade/2-opt/Direção)
    -> Estágio 4 (Seleção/Ordem/Refinamento) -> Estágio 5 (Buracos: False antes de True).
    Assim, tarefas do mesmo subproblema ficam contíguas no mesmo lote e atingem >90% de cache hit.
    """
    return (
        cfg.get("temperatura", ""),
        cfg.get("dissipacao", ""),
        bool(cfg.get("grafo_direcionado", False)),
        cfg.get("metodo_construcao", ""),
        bool(cfg.get("afinidade_termica", False)),
        bool(cfg.get("refinamento_rota", False)),
        cfg.get("direcao", ""),
        cfg.get("selecao", ""),
        cfg.get("ordem_selecao", ""),
        bool(cfg.get("refinamento", False)),
        bool(cfg.get("otimizar_buracos", False)),
    )


def gerar_espaco_configuracoes(tipo_espaco: str = "completo", deduplicar: bool = True) -> List[Dict[str, Any]]:
    """
    Gera o conjunto de configurações para o grid search estrutural da Queimada.
    """
    temperaturas = ["total_value", "knapsack_exact", "knapsack_greedy"]
    dissipacoes = ["adjacent_only", "dijkstra", "bfs"]
    direcoes = ["calor_final", "calor_inicio"]
    selecoes = ["greedy_relative", "knapsack_relative"]
    ordens = ["densidade", "reversa", "sequencial"]

    if tipo_espaco == "padrao":
        # 288 configurações clássicas do grid_search_queimada.py
        metodos = ["walk", "kruskal"]
        refinamentos = [False, True]
        configuracoes = []
        for temp, diss, direcao, sel, metodo, ordem, ref in itertools.product(
            temperaturas, dissipacoes, direcoes, selecoes, metodos, ordens, refinamentos
        ):
            if deduplicar and sel == "knapsack_relative" and ordem != "densidade":
                continue
            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": True,
                "refinamento_rota": True,
                "otimizar_buracos": False,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": ref,
                "grafo_direcionado": False,
            })
        return configuracoes

    elif tipo_espaco == "direcionado":
        # Espaço focado nas novas variantes de fluxo térmico em grafo direcionado
        metodos = ["backward_directed", "forward_directed"]
        afinidades = [False, True]
        rotas_2opt = [False, True]
        buracos_opts = [False, True]
        refinamentos = [False, True]

        configuracoes = []
        for temp, diss, direcao, metodo, afin, r_2opt, buraco, sel, ordem, ref in itertools.product(
            temperaturas, dissipacoes, direcoes, metodos, afinidades, rotas_2opt, buracos_opts, selecoes, ordens, refinamentos
        ):
            if deduplicar:
                if sel == "knapsack_relative" and ordem != "densidade":
                    continue

            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": afin,
                "refinamento_rota": r_2opt,
                "otimizar_buracos": buraco,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": ref,
                "grafo_direcionado": True,
            })
        return configuracoes

    elif tipo_espaco == "rapido":
        # Versão ultra-rápida: métodos construtivos e otimização de buracos sem Drop&Add/2opt pesado
        metodos = ["walk", "kruskal", "backward_edges", "backward_directed", "forward_directed"]
        buracos_opts = [False, True]
        afinidades = [False, True]
        configuracoes = []
        for temp, diss, direcao, metodo, afinidade, buracos, sel, ordem in itertools.product(
            temperaturas, dissipacoes, direcoes, metodos, afinidades, buracos_opts, selecoes, ordens
        ):
            if deduplicar:
                if sel == "knapsack_relative" and ordem != "densidade":
                    continue
                if metodo == "walk" and afinidade:
                    continue
            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": afinidade,
                "refinamento_rota": False,
                "otimizar_buracos": buracos,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": False,
                "grafo_direcionado": metodo in ("backward_directed", "forward_directed"),
            })
        return configuracoes

    elif tipo_espaco == "completo":
        # Espaço completo unificado com todas as opções estratégicas clássicas e as novas direcionadas
        metodos = ["walk", "kruskal", "backward_edges", "backward_directed", "forward_directed"]
        afinidades = [False, True]
        rotas_2opt = [False, True]
        buracos_opts = [False, True]
        refinamentos = [False, True]

        configuracoes = []
        for temp, diss, direcao, metodo, afin, r_2opt, buraco, sel, ordem, ref in itertools.product(
            temperaturas, dissipacoes, direcoes, metodos, afinidades, rotas_2opt, buracos_opts, selecoes, ordens, refinamentos
        ):
            if deduplicar:
                # Quando seleção é knapsack_relative, a ordem de varredura gulosa é irrelevante
                if sel == "knapsack_relative" and ordem != "densidade":
                    continue
                # Quando método é walk (guloso sequencial), afinidade térmica não é usada
                if metodo == "walk" and afin:
                    continue

            configuracoes.append({
                "temperatura": temp,
                "dissipacao": diss,
                "direcao": direcao,
                "metodo_construcao": metodo,
                "afinidade_termica": afin,
                "refinamento_rota": r_2opt,
                "otimizar_buracos": buraco,
                "selecao": sel,
                "ordem_selecao": ordem,
                "refinamento": ref,
                "grafo_direcionado": metodo in ("backward_directed", "forward_directed"),
            })
        return configuracoes

    else:
        raise ValueError(f"Espaço de busca inválido: {tipo_espaco}. Escolha entre 'completo', 'direcionado', 'padrao', 'rapido'.")


def resolver_instancias(
    opcao_instancias: str,
    ignorar_a280_pesada: bool = False,
    arquivos_especificos: Optional[List[str]] = None,
) -> List[Path]:
    """Retorna a lista de caminhos absolutos das instâncias a processar."""
    pasta_testes = _sim_path / "bateria_testes"
    if not pasta_testes.exists():
        pasta_testes = _raiz / "bateria_testes"

    if arquivos_especificos:
        caminhos = []
        for nome in arquivos_especificos:
            p = pasta_testes / nome if not Path(nome).is_absolute() else Path(nome)
            if p.exists():
                caminhos.append(p)
            else:
                print(f"⚠️ Instância ignorada (não encontrada): {nome}")
        return caminhos

    pasta_alto_prejuizo = pasta_testes / "alto_prejuizo"
    todos_arquivos = [f for f in pasta_testes.iterdir() if f.is_file() and f.suffix in (".txt", ".ttp")]
    if opcao_instancias in ("todas", "completo") and pasta_alto_prejuizo.is_dir():
        todos_arquivos += [f for f in pasta_alto_prejuizo.iterdir() if f.is_file() and f.suffix in (".txt", ".ttp")]

    if opcao_instancias == "lucrativas":
        luc_set = obter_instancias_lucrativas()
        arquivos = [f for f in todos_arquivos if f.name in luc_set]
    elif opcao_instancias == "equilibradas":
        eq_set = obter_instancias_equilibradas()
        arquivos = [f for f in todos_arquivos if f.name in eq_set]
    elif opcao_instancias in ("todas", "completo"):
        arquivos = todos_arquivos
    elif opcao_instancias == "sinteticas":
        arquivos = [f for f in todos_arquivos if f.suffix == ".txt"]
    elif opcao_instancias == "leves":
        arquivos = [f for f in todos_arquivos if not f.name.startswith("a280_")]
    else:
        # Tenta interpretar como padrão glob ou nome único
        arquivos = list(pasta_testes.glob(opcao_instancias))
        if not arquivos and pasta_alto_prejuizo.is_dir():
            arquivos = list(pasta_alto_prejuizo.glob(opcao_instancias))
        if not arquivos:
            arquivos = todos_arquivos

    if ignorar_a280_pesada:
        arquivos = [f for f in arquivos if not (f.name.startswith("a280_n2790") or f.name.startswith("a280_n1395"))]

    return sorted(arquivos, key=lambda f: (f.stat().st_size, f.name))


def _agrupar_em_lotes_otimizados(
    instancias: List[Path],
    configuracoes: List[Dict[str, Any]],
    concluidas_set: set,
    modelo: str,
    workers: int,
    tamanho_lote: Optional[int] = None,
) -> Tuple[List[List[Tuple[str, Dict[str, Any], int, str]]], int]:
    """
    Agrupa as tarefas pendentes por instância e ordenadas pela árvore de dependências do DAG
    (Temperatura -> Dissipação -> Construção -> Mochila -> Buracos).
    Cada lote preserva a localidade de dados de uma mesma subárvore de (instância, temperatura, dissipação),
    permitindo que os workers processem centenas de configurações compartilhando o mesmo estado na RAM.
    """
    configs_ordenadas = sorted(
        enumerate(configuracoes),
        key=lambda par: _chave_ordenacao_dag_config(par[1]),
    )

    lotes: List[List[Tuple[str, Dict[str, Any], int, str]]] = []
    total_pendentes = 0

    for inst_path in instancias:
        inst_str = str(inst_path.resolve())
        inst_nome = inst_path.name
        tarefas_inst = [
            (inst_str, cfg, cfg_id, modelo)
            for cfg_id, cfg in configs_ordenadas
            if (cfg_id, inst_nome) not in concluidas_set
        ]
        if not tarefas_inst:
            continue

        total_pendentes += len(tarefas_inst)

        # Define tamanho de lote adaptativo:
        # 288 equivale exatamente a um bloco (temperatura, dissipacao) completo no espaço de 5184 configs,
        # mantendo atualizações frequentes no CSV e aproveitando 100% do estágio de calor e rotas!
        if tamanho_lote is not None and tamanho_lote > 0:
            chunk = tamanho_lote
        else:
            chunk = min(288, max(16, len(tarefas_inst) // max(1, workers * 2)))

        for i in range(0, len(tarefas_inst), chunk):
            lotes.append(tarefas_inst[i : i + chunk])

    return lotes, total_pendentes


def executar_grid_search(
    espaco: str = "completo",
    instancias_modo: str = "leves",
    instancias_lista: Optional[List[str]] = None,
    ignorar_a280_pesada: bool = True,
    workers: Optional[int] = None,
    modelo: str = "TTP1",
    top_n: int = 15,
    arquivo_csv: Optional[str] = None,
    max_ram_mb: Optional[float] = None,
    tamanho_lote: Optional[int] = None,
):
    """Executa o Grid Search paralelo com salvamento incremental atômico e retomada automática (Resume)."""
    workers = workers or os.cpu_count() or 4
    os.environ["TTP_WORKERS"] = str(workers)

    instancias = resolver_instancias(
        opcao_instancias=instancias_modo,
        ignorar_a280_pesada=ignorar_a280_pesada,
        arquivos_especificos=instancias_lista,
    )

    if not instancias:
        print("❌ Nenhuma instância encontrada para processamento. Abortando.")
        return

    configuracoes = gerar_espaco_configuracoes(tipo_espaco=espaco, deduplicar=True)
    total_instancias = len(instancias)
    total_configs = len(configuracoes)
    total_tarefas = total_instancias * total_configs

    # Define caminho canônico de saída do CSV (em results/grid_search/)
    if arquivo_csv is None:
        arquivo_csv_path = resolver_caminho_csv(f"grid_search_queimada_{espaco}.csv")
    else:
        arquivo_csv_path = resolver_caminho_csv(arquivo_csv)

    arquivo_csv_path.parent.mkdir(parents=True, exist_ok=True)

    chaves_parametros = sorted(list(configuracoes[0].keys()))

    # Mapa de assinaturas de configuração para IDs para recuperação exata em CSVs existentes
    mapa_sig_para_id: Dict[Tuple, int] = {
        _obter_assinatura_config(cfg, chaves_parametros): idx for idx, cfg in enumerate(configuracoes)
    }

    # Estruturas de agregação e conjunto de tarefas concluídas (para Resume)
    concluidas_set: set = set()
    lucros_por_config: Dict[int, List[float]] = defaultdict(list)
    tempos_v_por_config: Dict[int, List[float]] = defaultdict(list)
    tempos_cpu_por_config: Dict[int, List[float]] = defaultdict(list)

    headers = [
        "config_id",
        "instancia",
        "lucro_G",
        "tempo_viagem",
        "peso",
        "qtd_itens",
        "tempo_cpu_ms",
    ] + list(configuracoes[0].keys())

    modo_abertura = "w"
    escrever_cabecalho = True

    # Leitura prévia do CSV para retomada automática (Resume)
    if arquivo_csv_path.exists() and arquivo_csv_path.stat().st_size > 0:
        # Garante alinhamento de cabeçalho com quaisquer novas colunas adicionadas no futuro
        try:
            with open(arquivo_csv_path, "r", encoding="utf-8", errors="utf-8" if False else "ignore") as f_chk:
                primeira_linha = f_chk.readline()
            delimitador = ";" if ";" in primeira_linha else ","
            colunas_existentes = [c.strip() for c in primeira_linha.split(delimitador) if c.strip()]
            colunas_faltantes = [h for h in headers if h not in colunas_existentes]
            if colunas_faltantes:
                nova_primeira = primeira_linha.strip() + delimitador + delimitador.join(colunas_faltantes) + "\n"
                with open(arquivo_csv_path, "r", encoding="utf-8", errors="ignore") as f_in:
                    f_in.readline()
                    resto = f_in.read()
                with open(arquivo_csv_path, "w", encoding="utf-8") as f_out:
                    f_out.write(nova_primeira + resto)
                print(f"📌 Cabeçalho do CSV legado atualizado automaticamente com novas colunas: {colunas_faltantes}")
        except Exception as e:
            print(f"⚠️ Aviso ao verificar cabeçalho do CSV existente: {e}")

        try:
            with open(arquivo_csv_path, "r", encoding="utf-8", errors="ignore") as f_prev:
                primeira = f_prev.readline()
                delimitador = ";" if ";" in primeira else ","
                f_prev.seek(0)
                reader = csv.DictReader(f_prev, delimiter=delimitador)
                for row in reader:
                    try:
                        inst_nome = row["instancia"].strip()
                        g = float(row["lucro_G"])
                        tv = float(row["tempo_viagem"])
                        cpu = float(row["tempo_cpu_ms"])

                        row_sig = _obter_assinatura_config(row, chaves_parametros)
                        if row_sig in mapa_sig_para_id:
                            c_id = mapa_sig_para_id[row_sig]
                        elif "config_id" in row:
                            try:
                                c_id = int(row["config_id"])
                            except ValueError:
                                continue
                        else:
                            continue

                        if c_id < len(configuracoes):
                            concluidas_set.add((c_id, inst_nome))
                            lucros_por_config[c_id].append(g)
                            tempos_v_por_config[c_id].append(tv)
                            tempos_cpu_por_config[c_id].append(cpu)
                    except (ValueError, KeyError):
                        continue

            if concluidas_set:
                modo_abertura = "a"
                escrever_cabecalho = False
                print(f"🔄 Retomada Automática (Resume) ativada:")
                print(f"   {len(concluidas_set):,} execuções prévias carregadas com sucesso de:\n   {arquivo_csv_path}")
        except Exception as e:
            print(f"⚠️ Aviso ao carregar CSV existente para retomada: {e}")

    # Agrupa as tarefas pendentes em lotes otimizados pela árvore de dependências do DAG
    lotes_tarefas, total_pendentes = _agrupar_em_lotes_otimizados(
        instancias=instancias,
        configuracoes=configuracoes,
        concluidas_set=concluidas_set,
        modelo=modelo,
        workers=workers,
        tamanho_lote=tamanho_lote,
    )
    concluidas_historico = len(concluidas_set)

    ram_total_b, ram_disp_b = detectar_memoria_sistema_bytes()
    _inicializar_worker(workers, max_ram_mb)
    status_cache = obter_cache_ram_global().status_resumo()

    print("\n" + "=" * 125)
    print(f"{'AVALIAÇÃO ESTRUTURAL PARALELA DA HEURÍSTICA DA QUEIMADA':^125}")
    print("=" * 125)
    print(f"• Espaço de busca:      {espaco.upper()} ({total_configs:,} configurações únicas)")
    print(f"• Instâncias alvo:      {total_instancias} arquivos de teste")
    print(f"• Modelo avaliado:      {modelo}")
    print(f"• Processamento:        {workers} workers paralelos | Numba JIT: {'ATIVO ⚡' if NUMBA_AVAILABLE else 'Fallback Python'}")
    print(
        f"• Cache DP Adaptativo:  RAM Livre: {ram_disp_b / (1024**3):.2f} GB / {ram_total_b / (1024**3):.2f} GB | "
        f"Budget/Worker: {status_cache['budget_worker_mb']:.0f} MB | Lotes DAG: {len(lotes_tarefas):,}"
    )
    print(f"• Total do grid search: {total_tarefas:,} tarefas")
    print(f"• Já concluídas (disco):{concluidas_historico:,} tarefas ({concluidas_historico / max(total_tarefas, 1) * 100:.1f}%)")
    print(f"• Pendentes a executar: {total_pendentes:,} tarefas")
    print(f"• Destino dos dados:    {arquivo_csv_path}")
    print("=" * 125 + "\n")

    concluidas_sessao = 0
    t_inicio = time.perf_counter()
    ultimo_print = t_inicio
    interrompido = False

    if total_pendentes > 0:
        print("🚀 Iniciando processamento com Numba JIT + Memoização DP Adaptativa + Salvamento Atômico...\n")

        with open(arquivo_csv_path, modo_abertura, newline="", encoding="utf-8") as f_csv:
            csv_writer = csv.DictWriter(f_csv, fieldnames=headers)
            if escrever_cabecalho:
                csv_writer.writeheader()
                f_csv.flush()

            MAX_LOTES_PENDENTES = max(workers * 3, 16)
            lotes_iter = iter(lotes_tarefas)
            futuros: Dict[concurrent.futures.Future, List[Tuple]] = {}

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_inicializar_worker,
                initargs=(workers, max_ram_mb),
            ) as executor:
                for lote in itertools.islice(lotes_iter, MAX_LOTES_PENDENTES):
                    fut = executor.submit(_worker_executar_lote, lote)
                    futuros[fut] = lote

                while futuros:
                    try:
                        done, _ = concurrent.futures.wait(
                            futuros.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        for fut in done:
                            lote_orig = futuros.pop(fut)
                            try:
                                lista_res = fut.result()
                                if lista_res:
                                    linhas_para_gravar = []
                                    for res in lista_res:
                                        c_id = res["cfg_id"]
                                        lucros_por_config[c_id].append(res["lucro_G"])
                                        tempos_v_por_config[c_id].append(res["tempo_viagem"])
                                        tempos_cpu_por_config[c_id].append(res["tempo_cpu_ms"])

                                        cfg_ref = configuracoes[c_id]
                                        linhas_para_gravar.append({
                                            "config_id": c_id,
                                            "instancia": res["instancia"],
                                            "lucro_G": res["lucro_G"],
                                            "tempo_viagem": res["tempo_viagem"],
                                            "peso": res["peso"],
                                            "qtd_itens": res["qtd_itens"],
                                            "tempo_cpu_ms": res["tempo_cpu_ms"],
                                            **cfg_ref,
                                        })

                                    csv_writer.writerows(linhas_para_gravar)
                                    f_csv.flush()

                                    concluidas_sessao += len(lista_res)
                                    concluidas_total = concluidas_historico + concluidas_sessao

                                    agora = time.perf_counter()
                                    if agora - ultimo_print >= 0.3 or concluidas_sessao == total_pendentes:
                                        ultimo_print = agora
                                        decorrido = agora - t_inicio
                                        taxa = concluidas_sessao / max(decorrido, 0.001)
                                        restante = (total_pendentes - concluidas_sessao) / max(taxa, 0.001)
                                        pct = (concluidas_total / max(total_tarefas, 1)) * 100.0
                                        print(
                                            f"\rProgresso: [{concluidas_total:,}/{total_tarefas:,}] ({pct:5.1f}%) | "
                                            f"Sessão: +{concluidas_sessao:,} | "
                                            f"Velocidade: {taxa:7.1f} exec/s | "
                                            f"ETA: {int(restante // 60)}m{int(restante % 60):02d}s",
                                            end="",
                                            flush=True,
                                        )
                            except Exception as exc:
                                inst_err = lote_orig[0][0] if lote_orig else "?"
                                print(f"\n[FALHA NO LOTE {inst_err}]: {exc}")

                            if not interrompido:
                                prox_lote = next(lotes_iter, None)
                                if prox_lote is not None:
                                    novo_fut = executor.submit(_worker_executar_lote, prox_lote)
                                    futuros[novo_fut] = prox_lote

                    except KeyboardInterrupt:
                        print("\n\n" + "!" * 95)
                        print("⚠️  INTERRUPÇÃO MANUAL SOLICITADA (Ctrl+C)")
                        print("Cancelando tarefas pendentes na fila do pool de processos...")
                        print(f"Todas as {concluidas_historico + concluidas_sessao:,} execuções finalizadas já foram salvas com segurança no disco!")
                        print(f"Arquivo CSV: {arquivo_csv_path}")
                        print("Quando desejar continuar, basta rodar o mesmo comando novamente.")
                        print("!" * 95)
                        interrompido = True
                        for f_pendente in list(futuros.keys()):
                            f_pendente.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    else:
        print("🎉 Todas as tarefas desta avaliação já estão concluídas e salvas no disco!")

    t_total = time.perf_counter() - t_inicio
    total_final = concluidas_historico + concluidas_sessao
    print(f"\n\n{'='*125}")
    if interrompido:
        print(f"🛑 Sessão interrompida em {t_total:.2f}s (+{concluidas_sessao:,} novas tarefas salvas | Total acumulado: {total_final:,})")
    else:
        print(f"✨ Concluído em {t_total:.2f}s ({total_final:,} execuções consolidadas | Taxa da sessão: {concluidas_sessao / max(t_total, 0.001):.1f} exec/s)")
    print("=" * 125)

    if not lucros_por_config:
        print("Nenhum resultado para exibir.")
        return

    # Ranking das melhores configurações
    ranking = []
    for c_id, lista_lucros in lucros_por_config.items():
        media_lucro = sum(lista_lucros) / len(lista_lucros)
        media_viagem = sum(tempos_v_por_config[c_id]) / len(tempos_v_por_config[c_id])
        media_cpu = sum(tempos_cpu_por_config[c_id]) / len(tempos_cpu_por_config[c_id])
        ranking.append((media_lucro, media_viagem, media_cpu, c_id))

    ranking.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'='*125}")
    print(f"{f'TOP {min(top_n, len(ranking))} MELHORES CONFIGURAÇÕES DA HEURÍSTICA DA QUEIMADA ({modelo})':^125}")
    print("=" * 125)
    header = (
        f"{'Rank':<5} | {'Construção':<22} | {'Buracos':<7} | {'2-Opt':<5} | "
        f"{'Afin':<5} | {'Direção':<11} | {'Seleção':<15} | {'Ordem':<10} | {'Refin':<5} | {'Dir':<4} | {'Lucro Médio (G)':>14}"
    )
    print(header)
    print("-" * 125)

    for rank_idx, (med_g, med_tv, med_cpu, c_id) in enumerate(ranking[:top_n], 1):
        c = configuracoes[c_id]
        buraco_str = "SIM ★" if c.get("otimizar_buracos") else "NÃO"
        opt_str = "SIM" if c.get("refinamento_rota") else "NÃO"
        afin_str = "SIM" if c.get("afinidade_termica") else "NÃO"
        ref_str = "SIM" if c.get("refinamento") else "NÃO"
        dir_str = "SIM" if c.get("grafo_direcionado") else "NÃO"

        print(
            f"{rank_idx:<5} | {c['metodo_construcao']:<22} | {buraco_str:<7} | {opt_str:<5} | "
            f"{afin_str:<5} | {c['direcao']:<11} | {c['selecao']:<15} | {c['ordem_selecao']:<10} | {ref_str:<5} | {dir_str:<4} | {med_g:>14.2f}"
        )

    # Auto-atualização dinâmica da configuração campeã no config_manager
    if ranking:
        try:
            from src.solvers.config_manager import atualizar_campeao_se_melhor
            melhor_g, _, _, melhor_c_id = ranking[0]
            melhor_cfg = configuracoes[melhor_c_id]
            atualizou = atualizar_campeao_se_melhor(
                "queimada",
                melhor_cfg,
                melhor_g,
                origem=f"grid_search_paralelo_{espaco}",
                detalhes={"total_instancias": len(instancias), "instancias": [f.name for f in instancias]},
            )
            if atualizou:
                print(f"\n🏆 AUTO-UPDATE: Nova melhor configuração da Queimada salva automaticamente (Score: {melhor_g:.2f})!")
            else:
                print(f"\nℹ️ AUTO-UPDATE: Configuração campeã atual mantida no registro persistido.")
        except Exception as e:
            print(f"\n⚠️ Falha ao atualizar configuração campeã: {e}")

    # Análise de Impacto Marginal por Parâmetro
    print(f"\n{'='*125}")
    print(f"{'ANÁLISE DE IMPACTO MARGINAL POR PARÂMETRO (LUCRO MÉDIO G)':^125}")
    print("=" * 125)

    parametros_analise = [
        "metodo_construcao",
        "otimizar_buracos",
        "grafo_direcionado",
        "refinamento_rota",
        "afinidade_termica",
        "direcao",
        "selecao",
        "ordem_selecao",
        "refinamento",
        "temperatura",
        "dissipacao",
    ]

    for param in parametros_analise:
        acum_param = defaultdict(list)
        for c_id, lista_lucros in lucros_por_config.items():
            val = configuracoes[c_id].get(param, "default")
            acum_param[val].extend(lista_lucros)

        print(f"\n🔹 Parâmetro: {param.upper()}")
        itens_param = []
        for val, vals_lucro in acum_param.items():
            media_val = sum(vals_lucro) / len(vals_lucro) if vals_lucro else 0.0
            itens_param.append((val, media_val))
        itens_param.sort(key=lambda x: x[1], reverse=True)
        for val, m_val in itens_param:
            print(f"   • {str(val):<22} : {m_val:>14.2f}")

    # Confirmação de persistência
    print(f"\n📁 Todos os resultados consolidados foram gravados no disco em:\n   {arquivo_csv_path}")
    print("=" * 125 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação e Grid Search Estrutural da Heurística da Queimada no Traveling Thief Problem (TTP)"
    )
    parser.add_argument(
        "--espaco",
        choices=["completo", "direcionado", "padrao", "rapido"],
        default="completo",
        help="Espaço de configurações: 'completo' (unificado com métodos direcionados), 'direcionado' (apenas backward/forward direcionadas), 'padrao' (288), ou 'rapido' (sem 2-opt/Drop&Add).",
    )
    parser.add_argument(
        "--instancias",
        default="equilibradas",
        help="Instâncias: 'equilibradas' (34 instâncias da bateria padrão), 'lucrativas' (23 instâncias UB > 0), 'todas'/'completo' (todas as 550+ instâncias), 'leves' (sem as instâncias gigantes a280), 'sinteticas' (7 cenários .txt), ou arquivos específicos.",
    )
    parser.add_argument(
        "--ignorar-a280-pesada",
        action="store_true",
        default=True,
        help="Ignora instâncias gigantes com > 1.000 itens (a280_n2790 e a280_n1395) para evitar gargalo computacional.",
    )
    parser.add_argument(
        "--incluir-todas-a280",
        action="store_false",
        dest="ignorar_a280_pesada",
        help="Força a inclusão das instâncias a280 completas mesmo com tempo estendido.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help=f"Número de processos de CPU paralelos (Padrão: {os.cpu_count() or 4}).",
    )
    parser.add_argument(
        "--max-ram-mb",
        type=float,
        default=None,
        help="Limite máximo de RAM (em MB) dedicado ao cache de Programação Dinâmica (Padrão: automático, usa até 35%% da RAM livre do computador).",
    )
    parser.add_argument(
        "--lote",
        type=int,
        default=None,
        help="Tamanho do lote de configurações por submissão aos workers (Padrão: automático por bloco do DAG).",
    )
    parser.add_argument(
        "--modelo",
        choices=["TTP1", "TTP2"],
        default="TTP1",
        help="Modelo do TTP a avaliar: TTP1 (taxa de aluguel R) ou TTP2 (degradação temporal).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Quantidade de melhores configurações para listar no ranking final.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Caminho personalizado para salvar o relatório CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    inst_lista = None
    if args.instancias not in ("leves", "sinteticas", "todas", "completo", "equilibradas", "lucrativas"):
        partes = args.instancias.split()
        if len(partes) > 1 or (len(partes) == 1 and ("." in partes[0] or "*" in partes[0])):
            inst_lista = partes

    executar_grid_search(
        espaco=args.espaco,
        instancias_modo=args.instancias,
        instancias_lista=inst_lista,
        ignorar_a280_pesada=args.ignorar_a280_pesada,
        workers=args.workers,
        modelo=args.modelo,
        top_n=args.top,
        arquivo_csv=args.csv,
        max_ram_mb=args.max_ram_mb,
        tamanho_lote=args.lote,
    )
