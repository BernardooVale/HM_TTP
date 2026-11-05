"""
Grid Search Paralelo de Alto Desempenho para a Heurística dos 37% (Problema da Secretária) no TTP.

Recursos:
- Multiprocessamento nativo com número configurável de workers (--workers).
- Cache inteligente de instâncias e de rotas TSP por processo worker para evitar recálculos redundantes.
- Agrupamento em lotes (batching) contíguos da mesma instância para eliminar overhead de IPC e maximizar cache hit.
- Suporte a múltiplos espaços de busca estruturais:
    * 'padrao': 48 combinações equilibradas (taxas 30%, 37%, 45%, métodos TSP, 2-opt, limiar simétrico e refinamento).
    * 'completo': 288 combinações (varredura ampla de taxas de amostragem de 20% a 50%, todos os TSPs, 2-opt, limiares, pós-preenchimento e refinamento).
    * 'rapido': 12 combinações construtivas ultrarrápidas sem refinamento local.
    * 'focado': 36 combinações dedicadas à calibração fina da taxa de corte da secretária (15% a 50%).
- Retomada Automática (Resume) sem recalcular execuções já salvas no histórico CSV.
- Interrupção segura por teclado (Ctrl+C / SIGINT) com salvamento atômico do progresso.
- Ranking Top N das melhores configurações, análise de impacto marginal de parâmetros e exportação em CSV.
- Auto-atualização dinâmica da melhor configuração no config_manager.py para os testes globais.

Uso:
    python scripts/experimentos_37/grid_search_paralelo.py --help
    python scripts/experimentos_37/grid_search_paralelo.py --espaco rapido --instancias sinteticas
    python scripts/experimentos_37/grid_search_paralelo.py --espaco padrao --instancias equilibradas --workers 8
    python scripts/experimentos_37/grid_search_paralelo.py --espaco completo --instancias equilibradas --forcar-campeao
"""

import argparse
import concurrent.futures
import csv
import itertools
import math
import os
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from src.solvers.constructive.heuristic_37 import Heuristic37Solver
from src.solvers.tsp import (
    resolver_tsp_farthest_insertion,
    resolver_tsp_multistart_nn,
    obter_melhor_rota_tsp,
    otimizar_2opt,
)
from src.utils.results_manager import (
    obter_instancias_equilibradas,
    obter_instancias_lucrativas,
    resolver_caminho_csv,
)

# Caches de processo worker para evitar recarga de disco e recálculo de TSP
_WORKER_CACHE: Dict[str, Any] = {}
_TSP_CACHE: Dict[Tuple[str, str, bool], List[int]] = {}


def _executar_uma_tarefa(
    caminho_arquivo: str, cfg: Dict[str, Any], cfg_id: int, modelo: str
) -> Dict[str, Any]:
    global _WORKER_CACHE, _TSP_CACHE
    if caminho_arquivo not in _WORKER_CACHE:
        if len(_WORKER_CACHE) >= 4:
            _WORKER_CACHE.clear()
            _TSP_CACHE.clear()
        inst = InstanceParser.carregar(caminho_arquivo)
        _ = inst.grafo.distance_matrix
        _WORKER_CACHE[caminho_arquivo] = inst

    instance = _WORKER_CACHE[caminho_arquivo]
    nome_inst = Path(caminho_arquivo).name

    metodo_tsp = cfg.get("metodo_tsp", "melhor")
    ref_rota = cfg.get("refinamento_rota", False)
    chave_tsp = (caminho_arquivo, metodo_tsp, ref_rota)

    if chave_tsp not in _TSP_CACHE:
        if metodo_tsp == "farthest_insertion":
            r = resolver_tsp_farthest_insertion(instance.grafo)
        elif metodo_tsp == "multistart_nn":
            r = resolver_tsp_multistart_nn(instance.grafo)
        else:
            r = obter_melhor_rota_tsp(instance.grafo)
        if ref_rota:
            r = otimizar_2opt(r, instance.grafo)
        _TSP_CACHE[chave_tsp] = r
    rota = _TSP_CACHE[chave_tsp]

    solver = Heuristic37Solver(
        refinamento=cfg.get("refinamento", True),
        limiar_simetrico=cfg.get("limiar_simetrico", False),
        taxa_amostragem=cfg.get("taxa_amostragem", 0.37),
        metodo_tsp=metodo_tsp,
        refinamento_rota=ref_rota,
        preenchimento_pos=cfg.get("preenchimento_pos", True),
        modelo=modelo,
    )

    t0 = time.perf_counter()
    solucao = solver.solve(instance, rota=rota)
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


def _worker_executar_lote(
    lote: List[Tuple[str, Dict[str, Any], int, str]]
) -> List[Dict[str, Any]]:
    """Executa um lote de tarefas agrupadas no mesmo processo worker."""
    resultados: List[Dict[str, Any]] = []
    for caminho_arquivo, cfg, cfg_id, modelo in lote:
        resultados.append(_executar_uma_tarefa(caminho_arquivo, cfg, cfg_id, modelo))
    return resultados


DEFAULTS_PARAMETROS_37: Dict[str, Any] = {
    "taxa_amostragem": 0.37,
    "metodo_tsp": "melhor",
    "refinamento_rota": False,
    "limiar_simetrico": False,
    "preenchimento_pos": True,
    "refinamento": True,
}


def _obter_assinatura_config(cfg: Dict[str, Any], chaves_schema: Optional[List[str]] = None) -> Tuple:
    """Gera uma assinatura imutável dos parâmetros para detecção robusta de execuções anteriores."""
    chaves = chaves_schema or sorted(cfg.keys())
    valores = []
    for k in chaves:
        if k in ("config_id", "instancia", "lucro_G", "tempo_viagem", "peso", "qtd_itens", "tempo_cpu_ms"):
            continue
        val = cfg.get(k)
        if val is None or val == "":
            val = DEFAULTS_PARAMETROS_37.get(k, False)

        if isinstance(val, bool) or str(val).strip().lower() in ("true", "false", "1", "0"):
            val_norm = str(val).strip().lower() in ("true", "1")
        elif isinstance(val, (int, float)):
            val_norm = round(float(val), 4)
        else:
            s_val = str(val).strip()
            try:
                val_norm = round(float(s_val), 4)
            except ValueError:
                val_norm = s_val
        valores.append((k, val_norm))
    return tuple(valores)


def _chave_ordenacao_config(cfg: Dict[str, Any]) -> Tuple:
    """Ordena tarefas para manter configs com mesmo método TSP e refinamento de rota contíguas."""
    return (
        cfg.get("metodo_tsp", ""),
        bool(cfg.get("refinamento_rota", False)),
        float(cfg.get("taxa_amostragem", 0.37)),
        bool(cfg.get("limiar_simetrico", False)),
        bool(cfg.get("preenchimento_pos", True)),
        bool(cfg.get("refinamento", False)),
    )


def gerar_espaco_configuracoes(tipo_espaco: str = "padrao") -> List[Dict[str, Any]]:
    """Gera o espaço de busca combinatório para a Heurística dos 37%."""
    if tipo_espaco == "padrao":
        # 48 configurações equilibradas
        taxas = [0.30, 0.37, 0.45]
        metodos_tsp = ["melhor", "farthest_insertion"]
        rotas_2opt = [False, True]
        limiares = [False, True]
        refinamentos = [False, True]
        configs = []
        for taxa, tsp, r_2opt, limiar, ref in itertools.product(
            taxas, metodos_tsp, rotas_2opt, limiares, refinamentos
        ):
            configs.append({
                "taxa_amostragem": taxa,
                "metodo_tsp": tsp,
                "refinamento_rota": r_2opt,
                "limiar_simetrico": limiar,
                "preenchimento_pos": True,
                "refinamento": ref,
            })
        return configs

    elif tipo_espaco == "completo":
        # 288 configurações completas
        taxas = [0.20, 0.25, 0.30, 0.37, 0.45, 0.50]
        metodos_tsp = ["melhor", "farthest_insertion", "multistart_nn"]
        rotas_2opt = [False, True]
        limiares = [False, True]
        preenchimentos = [False, True]
        refinamentos = [False, True]
        configs = []
        for taxa, tsp, r_2opt, limiar, preench, ref in itertools.product(
            taxas, metodos_tsp, rotas_2opt, limiares, preenchimentos, refinamentos
        ):
            configs.append({
                "taxa_amostragem": taxa,
                "metodo_tsp": tsp,
                "refinamento_rota": r_2opt,
                "limiar_simetrico": limiar,
                "preenchimento_pos": preench,
                "refinamento": ref,
            })
        return configs

    elif tipo_espaco == "rapido":
        # 12 configurações construtivas ultra-rápidas sem refinamento local
        taxas = [0.25, 0.37, 0.50]
        limiares = [False, True]
        preenchimentos = [False, True]
        configs = []
        for taxa, limiar, preench in itertools.product(taxas, limiares, preenchimentos):
            configs.append({
                "taxa_amostragem": taxa,
                "metodo_tsp": "farthest_insertion",
                "refinamento_rota": False,
                "limiar_simetrico": limiar,
                "preenchimento_pos": preench,
                "refinamento": False,
            })
        return configs

    elif tipo_espaco == "focado":
        # 36 configurações dedicadas à variação fina da taxa da secretária
        taxas = [0.15, 0.20, 0.25, 0.30, 0.35, 0.37, 0.40, 0.45, 0.50]
        limiares = [False, True]
        refinamentos = [False, True]
        configs = []
        for taxa, limiar, ref in itertools.product(taxas, limiares, refinamentos):
            configs.append({
                "taxa_amostragem": taxa,
                "metodo_tsp": "melhor",
                "refinamento_rota": True,
                "limiar_simetrico": limiar,
                "preenchimento_pos": True,
                "refinamento": ref,
            })
        return configs

    else:
        raise ValueError(f"Espaço desconhecido: '{tipo_espaco}'. Escolha entre: 'padrao', 'completo', 'rapido', 'focado'.")


def resolver_instancias(
    opcao_instancias: str = "equilibradas",
    ignorar_a280_pesada: bool = True,
    arquivos_especificos: Optional[List[str]] = None,
) -> List[Path]:
    """Descobre e filtra instâncias do TTP conforme a seleção do usuário."""
    pasta_testes = _sim_path / "bateria_testes"

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
        arquivos = list(pasta_testes.glob(opcao_instancias))
        if not arquivos and pasta_alto_prejuizo.is_dir():
            arquivos = list(pasta_alto_prejuizo.glob(opcao_instancias))
        if not arquivos:
            arquivos = todos_arquivos

    if ignorar_a280_pesada:
        arquivos = [f for f in arquivos if not (f.name.startswith("a280_n2790") or f.name.startswith("a280_n1395"))]

    return sorted(arquivos, key=lambda f: (f.stat().st_size, f.name))


def _agrupar_em_lotes(
    instancias: List[Path],
    configuracoes: List[Dict[str, Any]],
    concluidas_set: Set[Tuple[int, str]],
    modelo: str,
    workers: int,
    tamanho_lote: Optional[int] = None,
) -> Tuple[List[List[Tuple[str, Dict[str, Any], int, str]]], int]:
    """Agrupa tarefas pendentes em lotes contíguos da mesma instância."""
    configs_ordenadas = sorted(
        enumerate(configuracoes),
        key=lambda par: _chave_ordenacao_config(par[1]),
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

        if tamanho_lote is not None and tamanho_lote > 0:
            chunk = tamanho_lote
        else:
            chunk = min(48, max(8, len(tarefas_inst) // max(1, workers * 2)))

        for i in range(0, len(tarefas_inst), chunk):
            lotes.append(tarefas_inst[i : i + chunk])

    return lotes, total_pendentes


def executar_grid_search(
    espaco: str = "padrao",
    instancias_modo: str = "equilibradas",
    instancias_lista: Optional[List[str]] = None,
    ignorar_a280_pesada: bool = True,
    workers: Optional[int] = None,
    modelo: str = "TTP1",
    top_n: int = 15,
    arquivo_csv: Optional[str] = None,
    tamanho_lote: Optional[int] = None,
    forcar_campeao: bool = False,
):
    """Executa o Grid Search paralelo da Heurística dos 37% com auto-atualização do campeão."""
    workers = workers or os.cpu_count() or 4

    instancias = resolver_instancias(
        opcao_instancias=instancias_modo,
        ignorar_a280_pesada=ignorar_a280_pesada,
        arquivos_especificos=instancias_lista,
    )

    if not instancias:
        print("❌ Nenhuma instância encontrada para processamento. Abortando.")
        return

    configuracoes = gerar_espaco_configuracoes(tipo_espaco=espaco)
    total_instancias = len(instancias)
    total_configs = len(configuracoes)
    total_tarefas = total_instancias * total_configs

    # Define caminho canônico de saída do CSV
    if arquivo_csv is None:
        arquivo_csv_path = resolver_caminho_csv(f"grid_search_37_{espaco}.csv")
    else:
        arquivo_csv_path = resolver_caminho_csv(arquivo_csv)

    arquivo_csv_path.parent.mkdir(parents=True, exist_ok=True)
    chaves_parametros = sorted(list(configuracoes[0].keys()))

    mapa_sig_para_id: Dict[Tuple, int] = {
        _obter_assinatura_config(cfg, chaves_parametros): idx for idx, cfg in enumerate(configuracoes)
    }

    concluidas_set: Set[Tuple[int, str]] = set()
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
    ] + chaves_parametros

    modo_abertura = "w"
    escrever_cabecalho = True

    # Leitura prévia do CSV para Retomada Automática (Resume)
    if arquivo_csv_path.exists() and arquivo_csv_path.stat().st_size > 0:
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

    lotes_tarefas, total_pendentes = _agrupar_em_lotes(
        instancias=instancias,
        configuracoes=configuracoes,
        concluidas_set=concluidas_set,
        modelo=modelo,
        workers=workers,
        tamanho_lote=tamanho_lote,
    )
    concluidas_historico = len(concluidas_set)

    print("\n" + "=" * 120)
    print(f"{'GRID SEARCH PARALELO - HEURÍSTICA DOS 37% (PROBLEMA DA SECRETÁRIA)':^120}")
    print("=" * 120)
    print(f" • Espaço Estrutural:   {espaco.upper()} ({total_configs} combinações)")
    print(f" • Instâncias Selecion: {instancias_modo} ({total_instancias} instâncias)")
    print(f" • Modelo Matemático:   {modelo}")
    print(f" • Processos (Workers): {workers}")
    print(f" • Tarefas Totais:      {total_tarefas:,} ({concluidas_historico:,} já prontas, {total_pendentes:,} pendentes)")
    print(f" • Lotes de Execução:   {len(lotes_tarefas)}")
    print(f" • Arquivo de Destino:  {arquivo_csv_path}")
    print("=" * 120 + "\n")

    tempo_inicio = time.perf_counter()
    executadas_agora = 0
    ultimo_pct = -1

    if total_pendentes > 0:
        with open(arquivo_csv_path, modo_abertura, newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=headers)
            if escrever_cabecalho:
                writer.writeheader()
                f_out.flush()

            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    futuros = [executor.submit(_worker_executar_lote, lote) for lote in lotes_tarefas]

                    for fut in concurrent.futures.as_completed(futuros):
                        res_lote = fut.result()
                        linhas_lote = []
                        for r in res_lote:
                            c_id = r["cfg_id"]
                            inst_nome = r["instancia"]
                            g = r["lucro_G"]
                            tv = r["tempo_viagem"]
                            cpu = r["tempo_cpu_ms"]

                            lucros_por_config[c_id].append(g)
                            tempos_v_por_config[c_id].append(tv)
                            tempos_cpu_por_config[c_id].append(cpu)
                            concluidas_set.add((c_id, inst_nome))

                            cfg = configuracoes[c_id]
                            linha = {
                                "config_id": c_id,
                                "instancia": inst_nome,
                                "lucro_G": f"{g:.2f}",
                                "tempo_viagem": f"{tv:.2f}",
                                "peso": f"{r['peso']:.2f}",
                                "qtd_itens": r["qtd_itens"],
                                "tempo_cpu_ms": f"{cpu:.2f}",
                                **cfg,
                            }
                            linhas_lote.append(linha)

                        writer.writerows(linhas_lote)
                        f_out.flush()
                        executadas_agora += len(res_lote)
                        total_concluidas = concluidas_historico + executadas_agora

                        tempo_decorrido = time.perf_counter() - tempo_inicio
                        taxa = executadas_agora / tempo_decorrido if tempo_decorrido > 0 else 0.0
                        restantes = total_pendentes - executadas_agora
                        eta_seg = restantes / taxa if taxa > 0 else 0.0
                        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seg)) if eta_seg < 86400 else "> 24h"

                        pct = (total_concluidas / total_tarefas) * 100.0 if total_tarefas > 0 else 100.0
                        
                        curr_step = int(pct // 5)
                        if curr_step > ultimo_pct or total_concluidas == total_tarefas:
                            ultimo_pct = curr_step
                            print(
                                f"\n⏳ Progresso: [{total_concluidas:,}/{total_tarefas:,}] ({pct:5.1f}%) | "
                                f"Taxa: {taxa:5.1f} op/s | ETA: {eta_str}",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r⏳ Progresso: {total_concluidas:,}/{total_tarefas:,} ({pct:5.1f}%) | "
                                f"Throughput: {taxa:5.1f} op/s | ETA: {eta_str}",
                                end="",
                                flush=True,
                            )

            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupção solicitada (Ctrl+C). Salvando dados pendentes e finalizando...")

    print("\n\n" + "=" * 120)
    print(f"{'CONSOLIDAÇÃO ESTATÍSTICA DO GRID SEARCH (HEURÍSTICA DOS 37%)':^120}")
    print("=" * 120)

    # Gera ranking final ordenado pela média de lucro líquido G
    ranking: List[Tuple[float, float, float, int]] = []
    for c_id, lista_g in lucros_por_config.items():
        if lista_g:
            media_g = sum(lista_g) / len(lista_g)
            media_tv = sum(tempos_v_por_config[c_id]) / len(tempos_v_por_config[c_id])
            media_cpu = sum(tempos_cpu_por_config[c_id]) / len(tempos_cpu_por_config[c_id])
            ranking.append((media_g, media_tv, media_cpu, c_id))

    ranking.sort(key=lambda x: x[0], reverse=True)

    print(f"\n🏆 TOP {min(top_n, len(ranking))} MELHORES CONFIGURAÇÕES DA HEURÍSTICA DOS 37%:\n")
    print(
        f"{'Rank':<5} | {'Taxa':<6} | {'TSP':<18} | {'2-opt':<5} | {'Simétrico':<9} | "
        f"{'Pós-Gulosa':<10} | {'Refino':<6} | {'Lucro Médio G':>14} | {'Tempo Viagem':>13} | {'CPU (ms)':>8}"
    )
    print("-" * 120)

    for rank_idx, (med_g, med_tv, med_cpu, c_id) in enumerate(ranking[:top_n], 1):
        c = configuracoes[c_id]
        opt_str = "SIM" if c.get("refinamento_rota") else "NÃO"
        sim_str = "SIM" if c.get("limiar_simetrico") else "NÃO"
        pos_str = "SIM" if c.get("preenchimento_pos") else "NÃO"
        ref_str = "SIM" if c.get("refinamento") else "NÃO"
        taxa_str = f"{c.get('taxa_amostragem', 0.37)*100:.0f}%"

        print(
            f"{rank_idx:<5} | {taxa_str:<6} | {c['metodo_tsp']:<18} | {opt_str:<5} | "
            f"{sim_str:<9} | {pos_str:<10} | {ref_str:<6} | {med_g:>14.2f} | {med_tv:>13.2f} | {med_cpu:>8.1f}"
        )

    # Auto-atualização dinâmica da configuração campeã no config_manager
    if ranking:
        try:
            from src.config.config_manager import atualizar_campeao_se_melhor, salvar_melhor_configuracao
            melhor_g, _, _, melhor_c_id = ranking[0]
            melhor_cfg = configuracoes[melhor_c_id]
            atualizou = False
            if forcar_campeao:
                atualizou = salvar_melhor_configuracao(
                    "heuristic_37",
                    melhor_cfg,
                    score=melhor_g,
                    origem=f"grid_search_paralelo_{espaco}",
                    detalhes={
                        "total_instancias": len(instancias),
                        "instancias": [f.name for f in instancias],
                        "espaco": espaco,
                        "motivo": "forcar_campeao_grid_search",
                    },
                )
            else:
                atualizou = atualizar_campeao_se_melhor(
                    "heuristic_37",
                    melhor_cfg,
                    melhor_g,
                    origem=f"grid_search_paralelo_{espaco}",
                    detalhes={
                        "total_instancias": len(instancias),
                        "instancias": [f.name for f in instancias],
                        "espaco": espaco,
                    },
                )

            if atualizou:
                print(f"\n🏆 AUTO-UPDATE: Nova melhor configuração da Heurística dos 37% salva automaticamente (Score: {melhor_g:.2f})!")
            else:
                print(f"\nℹ️ AUTO-UPDATE: Configuração campeã atual mantida no registro persistido.")
        except Exception as e:
            print(f"\n⚠️ Falha ao atualizar configuração campeã: {e}")

    # Análise de Impacto Marginal por Parâmetro
    print(f"\n{'='*120}")
    print(f"{'ANÁLISE DE IMPACTO MARGINAL POR PARÂMETRO (LUCRO MÉDIO G)':^120}")
    print("=" * 120)

    parametros_analise = [
        "taxa_amostragem",
        "metodo_tsp",
        "refinamento_rota",
        "limiar_simetrico",
        "preenchimento_pos",
        "refinamento",
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

    print(f"\n📁 Resultados consolidados gravados em:\n   {arquivo_csv_path}")
    print("=" * 120 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid Search Paralelo de Hiperparâmetros da Heurística dos 37% no Traveling Thief Problem (TTP)"
    )
    parser.add_argument(
        "--espaco",
        choices=["padrao", "completo", "rapido", "focado"],
        default="padrao",
        help="Espaço de configurações: 'padrao' (48), 'completo' (288), 'rapido' (12) ou 'focado' (36).",
    )
    parser.add_argument(
        "--instancias",
        default="equilibradas",
        help="Instâncias: 'equilibradas' (34 instâncias padrão), 'lucrativas', 'leves', 'sinteticas', 'todas'/'completo' ou arquivos específicos.",
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
    parser.add_argument(
        "--lote",
        type=int,
        default=None,
        help="Tamanho do lote de configurações por submissão aos workers.",
    )
    parser.add_argument(
        "--forcar-campeao",
        action="store_true",
        default=False,
        help="Força a gravação da configuração campeã no config_manager.py mesmo se a escala numérica anterior for diferente.",
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
        tamanho_lote=args.lote,
        forcar_campeao=args.forcar_campeao,
    )
