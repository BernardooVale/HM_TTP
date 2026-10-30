import concurrent.futures
import csv
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import src.solvers  # Trigger registration of all solvers
from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.evaluator_ttp2 import avaliar_ttp2
from src.scenarios.parser import InstanceParser
from src.solvers.base import SolverRegistry
from src.utils.csv_logger import CSVLogger


def _executar_tarefa_worker(args: Tuple[str, str, Optional[Tuple[str, str]], str, int, int]) -> Optional[Dict]:
    """
    Função de trabalho executada por processos paralelos isolados.
    Carrega a instância, executa o solver e avalia a solução de forma segura.
    """
    file_path_str, solver_name, solver_mod_info, model, run_idx, seed = args

    import random
    import time
    from pathlib import Path
    import src.solvers
    from src.core.evaluator_ttp1 import avaliar_ttp1
    from src.core.evaluator_ttp2 import avaliar_ttp2
    from src.scenarios.parser import InstanceParser
    from src.solvers.base import SolverRegistry
    from src.utils.csv_logger import CSVLogger

    try:
        if solver_name not in SolverRegistry._registry and solver_mod_info:
            mod_name, class_name = solver_mod_info
            import importlib
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, class_name)
            SolverRegistry.register(cls)

        instance = InstanceParser.carregar(file_path_str)
        solver = SolverRegistry.get(solver_name)
        if hasattr(solver, "modelo"):
            setattr(solver, "modelo", model)

        random.seed(seed)
        start_time = time.perf_counter()
        solucao = solver.solve(instance)
        elapsed = time.perf_counter() - start_time

        if model == "TTP1":
            resultado = avaliar_ttp1(instance, solucao)
        else:
            resultado = avaliar_ttp2(instance, solucao)

        return CSVLogger.criar_registro(
            arquivo_instancia=Path(file_path_str).name,
            modelo_ttp=model,
            solucionador=solver.nome,
            seed=seed,
            tempo_computacional=elapsed,
            resultado=resultado,
            instance=instance,
        )
    except Exception as e:
        print(f"[ERRO] {Path(file_path_str).name} | {solver_name} | {model} | Seed {seed}: {e}")
        return None


def calcular_estatisticas(valores: List[float]) -> Dict[str, float]:
    """Calcula estatísticas descritivas completas: média, desvio padrão e quartis."""
    if not valores:
        return {
            "media": 0.0,
            "std": 0.0,
            "min": 0.0,
            "q1": 0.0,
            "mediana": 0.0,
            "q3": 0.0,
            "max": 0.0,
        }

    vals = sorted(valores)
    n = len(vals)
    media = sum(vals) / n
    std = statistics.stdev(vals) if n > 1 else 0.0
    v_min = vals[0]
    v_max = vals[-1]

    if n >= 4:
        q = statistics.quantiles(vals, n=4, method="inclusive")
        q1, med, q3 = q[0], q[1], q[2]
    elif n == 1:
        q1 = med = q3 = vals[0]
    else:
        med = statistics.median(vals)
        q1 = vals[0]
        q3 = vals[-1]

    return {
        "media": media,
        "std": std,
        "min": v_min,
        "q1": q1,
        "mediana": med,
        "q3": q3,
        "max": v_max,
    }


def _obter_metadados_tamanho(arq: Path) -> Tuple[int, int, int]:
    """
    Extrai rapidamente dimensão (cidades), itens e tamanho do arquivo em bytes,
    lendo apenas as linhas de cabeçalho iniciais sem carregar o arquivo todo.
    """
    dim = 0
    itens = 0
    try:
        with open(arq, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(25):
                line = f.readline()
                if not line:
                    break
                line_u = line.upper()
                if "DIMENSION" in line_u and ":" in line:
                    parts = line.split(":", 1)
                    val = parts[1].strip().split()[0]
                    if val.isdigit():
                        dim = int(val)
                elif "NUMBER OF ITEMS" in line_u and ":" in line:
                    parts = line.split(":", 1)
                    val = parts[1].strip().split()[0]
                    if val.isdigit():
                        itens = int(val)
    except Exception:
        pass

    tam_bytes = arq.stat().st_size if arq.exists() else 0
    if dim == 0:
        dim = tam_bytes
    return (dim, itens, tam_bytes)


class BatchRunner:
    """
    Executor de baterias de testes em lote para o Traveling Thief Problem.
    Suporta multiprocessamento paralelo, carregamento assíncrono de .ttp e .txt,
    recuperação automática de progresso (resume), histórico configurável
    e salvamento incremental no CSV.
    """

    def __init__(
        self,
        test_dir: str,
        solvers: List[str],
        models: List[str],
        runs: int,
        output_csv: str,
        base_seed: int,
        workers: Optional[int] = None,
        resume: bool = True,
        output_stats_csv: Optional[str] = None,
        history_csv: Optional[str] = None,
        importar_historico: bool = True,
    ):
        self.test_dir = Path(test_dir)
        self.solvers = solvers
        self.models = models or ["TTP1"]
        self.runs = runs
        self.output_csv = str(Path(output_csv).resolve())
        self.output_stats_csv = output_stats_csv
        self.base_seed = base_seed
        self.workers = workers if workers is not None else min(os.cpu_count() or 4, 8)
        self.resume = resume
        self.history_csv = str(Path(history_csv).resolve()) if history_csv else None
        self.importar_historico = importar_historico
        self.logger = CSVLogger(self.output_csv)

    def _obter_arquivos_instancias(self) -> List[Path]:
        """
        Varre o diretório de testes por instâncias nos formatos .ttp e .txt.
        Ignora subpastas (como 'quarentena') e ordena as instâncias por tamanho
        crescente (número de cidades, itens e tamanho do arquivo), garantindo
        que as instâncias menores sejam executadas primeiro.
        """
        extensoes = ["*.txt", "*.ttp"]
        arquivos = []
        for ext in extensoes:
            for arq in self.test_dir.glob(ext):
                if not arq.is_file():
                    continue
                if "quarentena" in arq.parts:
                    continue
                if "alto_prejuizo" in arq.parts and self.test_dir.name != "alto_prejuizo":
                    continue
                arquivos.append(arq)

        # Remove duplicados preservando ordem estável
        vistos = set()
        arquivos_unicos = []
        for arq in arquivos:
            if arq.name not in vistos:
                vistos.add(arq.name)
                arquivos_unicos.append(arq)

        # Ordena por tamanho: dimensão (cidades), quantidade de itens, tamanho do arquivo e nome
        arquivos_unicos.sort(key=lambda a: (_obter_metadados_tamanho(a), a.name))
        return arquivos_unicos

    def verificar_status_historico(self) -> Dict[str, Any]:
        """
        Audita a base de histórico em relação aos solvers e instâncias solicitados
        sem executar simulações na CPU.
        """
        instance_files = self._obter_arquivos_instancias()
        concluidas = set()

        if Path(self.output_csv).is_file():
            concluidas.update(self.logger.obter_chaves_registradas())

        if self.history_csv and Path(self.history_csv).is_file():
            logger_hist = CSVLogger(self.history_csv)
            concluidas.update(logger_hist.obter_chaves_registradas())

        total_tarefas = len(instance_files) * len(self.solvers) * len(self.models) * self.runs
        tarefas_concluidas = 0
        status_por_solver = {}

        for s_name in self.solvers:
            tarefas_solver = len(instance_files) * len(self.models) * self.runs
            feitas_solver = 0
            for inst in instance_files:
                for mod in self.models:
                    for run_idx in range(self.runs):
                        seed = self.base_seed + run_idx
                        chave = (inst.name, mod, s_name, seed)
                        if chave in concluidas:
                            tarefas_concluidas += 1
                            feitas_solver += 1
            pct = (feitas_solver / max(tarefas_solver, 1)) * 100.0
            status_por_solver[s_name] = {
                "total": tarefas_solver,
                "feitas": feitas_solver,
                "pendentes": tarefas_solver - feitas_solver,
                "pct": pct,
            }

        return {
            "total_instancias": len(instance_files),
            "total_tarefas": total_tarefas,
            "tarefas_concluidas": tarefas_concluidas,
            "tarefas_pendentes": total_tarefas - tarefas_concluidas,
            "pct_concluido": (tarefas_concluidas / max(total_tarefas, 1)) * 100.0,
            "status_solvers": status_por_solver,
        }

    def run(self):
        instance_files = self._obter_arquivos_instancias()

        if not instance_files:
            print(f"Nenhum arquivo de instância (.txt ou .ttp) encontrado em {self.test_dir}")
            return

        print(f"\n{'='*95}")
        print(f"{'INICIANDO BATERIA DE TESTES DE SOLUCIONADORES DO TTP':^95}")
        print(f"Instâncias: {len(instance_files)} | Solvers: {len(self.solvers)} | "
              f"Modelos: {len(self.models)} | Runs: {self.runs} | Workers: {self.workers}")
        print(f"Destino de saída: {self.output_csv}")
        print(f"{'='*95}\n")

        # Verifica execuções já salvas no output_csv e no history_csv
        concluidas: Set[Tuple[str, str, str, int]] = set()
        historico_chaves: Set[Tuple[str, str, str, int]] = set()

        if self.resume:
            concluidas = self.logger.obter_chaves_registradas()
            if concluidas:
                print(f"[RESUME] {len(concluidas)} execuções já gravadas encontradas no CSV de saída.")

        # Conecta histórico externo (se existir e for diferente do output_csv)
        if self.history_csv and Path(self.history_csv).is_file():
            logger_hist = CSVLogger(self.history_csv)
            historico_chaves = logger_hist.obter_chaves_registradas()
            print(f"[HISTÓRICO] Base conectada: {Path(self.history_csv).name} ({len(historico_chaves)} registros)")

            # Se output_csv for diferente de history_csv, importa registros correspondentes
            if Path(self.history_csv).resolve() != Path(self.output_csv).resolve() and self.importar_historico:
                linhas_a_importar = []
                try:
                    with open(self.history_csv, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            inst = row.get("arquivo_instancia", "")
                            mod = row.get("modelo_ttp", "")
                            solv = row.get("solucionador", "")
                            s_str = row.get("seed", "0")
                            try:
                                seed = int(s_str)
                            except ValueError:
                                seed = 0
                            chave = (inst, mod, solv, seed)
                            if chave in historico_chaves and chave not in concluidas:
                                if solv in self.solvers and mod in self.models:
                                    linhas_a_importar.append(row)
                                    concluidas.add(chave)
                    if linhas_a_importar:
                        print(f"[HISTÓRICO] {len(linhas_a_importar)} registros reaproveitados do histórico para o CSV de saída.")
                        for r in linhas_a_importar:
                            self.logger.registrar(r)
                except Exception as e:
                    print(f"[AVISO] Falha ao ler histórico: {e}")
            else:
                concluidas.update(historico_chaves)

        # Constrói fila de tarefas pendentes
        tarefas: List[Tuple[str, str, Optional[Tuple[str, str]], str, int, int]] = []
        for instance_file in instance_files:
            for solver_name in self.solvers:
                solver_mod_info = SolverRegistry._registry_modules.get(solver_name)
                for model in self.models:
                    for run_idx in range(self.runs):
                        seed = self.base_seed + run_idx
                        chave = (instance_file.name, model, solver_name, seed)
                        if self.resume and chave in concluidas:
                            continue
                        tarefas.append((str(instance_file), solver_name, solver_mod_info, model, run_idx, seed))

        total_tarefas = len(tarefas)
        print(f"Total de tarefas pendentes para executar: {total_tarefas}")

        t_inicio = time.time()
        completas = 0
        interrompido = False

        if total_tarefas > 0:
            if self.workers <= 1:
                # Modo sequencial mono-processo
                try:
                    for t in tarefas:
                        completas += 1
                        registro = _executar_tarefa_worker(t)
                        if registro:
                            self.logger.registrar(registro)
                            lucro = registro["lucro_liquido_G"]
                            tempo_s = registro["tempo_computacional_seg"]
                            print(
                                f"[{completas}/{total_tarefas}] {Path(t[0]).name} | {t[1]} | "
                                f"{t[3]} | Lucro: {lucro:>10.2f} | {tempo_s:>6.2f}s"
                            )
                except KeyboardInterrupt:
                    print("\n⚠️ Interrupção via Ctrl+C no modo sequencial. Finalizando com segurança...")
                    interrompido = True
            else:
                # Modo multiprocesso paralelo de alto desempenho
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
                    futuros = {executor.submit(_executar_tarefa_worker, t): t for t in tarefas}

                    try:
                        for futuro in concurrent.futures.as_completed(futuros):
                            completas += 1
                            t = futuros[futuro]
                            try:
                                registro = futuro.result()
                                if registro:
                                    self.logger.registrar(registro)
                                    lucro = registro["lucro_liquido_G"]
                                    tempo_s = registro["tempo_computacional_seg"]
                                    pct = (completas / total_tarefas) * 100.0
                                    elapsed_total = time.time() - t_inicio
                                    taxa = completas / max(elapsed_total, 0.001)
                                    eta = (total_tarefas - completas) / max(taxa, 0.001)

                                    print(
                                        f"[{completas:>4}/{total_tarefas}] ({pct:>5.1f}%) "
                                        f"{Path(t[0]).name:<25} | {t[1]:<20} | {t[3]} | "
                                        f"G: {lucro:>10.1f} | {tempo_s:>5.2f}s | "
                                        f"ETA: {int(eta // 60)}m{int(eta % 60):02d}s"
                                    )
                            except Exception as e:
                                print(f"[FALHA NA TAREFA] {Path(t[0]).name} | {t[1]}: {e}")
                    except KeyboardInterrupt:
                        print("\n⚠️ Interrupção solicitada via Ctrl+C. Cancelando tarefas não iniciadas e salvando estado...")
                        interrompido = True
                        for f in futuros:
                            if not f.done():
                                f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)

        tempo_total = time.time() - t_inicio
        if interrompido:
            print(f"\n🛑 Execução interrompida com segurança. Tarefas concluídas salvas em: {self.output_csv}")
        else:
            print(f"\nBateria concluída em {tempo_total:.2f}s ({tempo_total/60:.2f}min). "
                  f"Resultados detalhados salvos em: {self.output_csv}")

        # Gera sempre o relatório estatístico consolidado a partir do CSV completo
        self.gerar_relatorio_estatistico_consolidado()

    def gerar_relatorio_estatistico_consolidado(self, output_stats_csv: Optional[str] = None):
        """
        Lê todas as execuções registradas em self.output_csv, agrupa por (modelo, solver)
        e gera a tabela de resumo estatístico completa (Média, Desvio, Mín, Q1, Mediana, Q3, Máx, etc.).
        Dessa forma, caso uma nova heurística seja adicionada, o relatório consolidado incluirá
        automaticamente tanto os solvers anteriores quanto o novo.
        """
        if not Path(self.output_csv).exists() or Path(self.output_csv).stat().st_size == 0:
            return

        dados_por_modelo_solver = defaultdict(lambda: {
            "lucros": [],
            "tempos_comp": [],
            "tempos_viagem": [],
            "pesos": [],
            "pct_ocupacao": [],
            "qtd_itens": [],
        })

        with open(self.output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mod = row.get("modelo_ttp", "TTP1")
                solv = row.get("solucionador", "")
                try:
                    g = float(row.get("lucro_liquido_G", 0.0))
                    t_comp = float(row.get("tempo_computacional_seg", 0.0)) * 1000.0  # ms
                    t_viagem = float(row.get("tempo_viagem_f", 0.0))
                    peso = float(row.get("peso_acumulado_mochila", 0.0))
                    pct_w = float(row.get("pct_ocupacao_mochila", 0.0))
                    qtd = float(row.get("qtd_itens_coletados", 0.0))

                    dados_por_modelo_solver[(mod, solv)]["lucros"].append(g)
                    dados_por_modelo_solver[(mod, solv)]["tempos_comp"].append(t_comp)
                    dados_por_modelo_solver[(mod, solv)]["tempos_viagem"].append(t_viagem)
                    dados_por_modelo_solver[(mod, solv)]["pesos"].append(peso)
                    dados_por_modelo_solver[(mod, solv)]["pct_ocupacao"].append(pct_w)
                    dados_por_modelo_solver[(mod, solv)]["qtd_itens"].append(qtd)
                except ValueError:
                    continue

        if not dados_por_modelo_solver:
            return

        modelos = sorted(list({k[0] for k in dados_por_modelo_solver.keys()}))
        estatisticas_consolidadas = []

        for mod in modelos:
            print(f"\n\n{'='*115}")
            print(f"{f'RESUMO ESTATÍSTICO CONSOLIDADO - MODELO {mod}':^115}")
            print(f"{'='*115}")
            print(
                f"{'Solucionador':<26} | {'Média (G)':>10} | {'DesvPad':>9} | {'Q1 (25%)':>10} | "
                f"{'Mediana':>10} | {'Q3 (75%)':>10} | {'Mín':>10} | {'Máx':>10} | {'T. CPU (ms)':>11}"
            )
            print(f"{'-'*115}")

            solvers_stats = []
            for (m, s_name), metricas in dados_por_modelo_solver.items():
                if m != mod:
                    continue
                sl = calcular_estatisticas(metricas["lucros"])
                sc = calcular_estatisticas(metricas["tempos_comp"])
                sv = calcular_estatisticas(metricas["tempos_viagem"])
                sp = calcular_estatisticas(metricas["pesos"])
                so = calcular_estatisticas(metricas["pct_ocupacao"])
                sq = calcular_estatisticas(metricas["qtd_itens"])

                solvers_stats.append((s_name, sl, sc, sv, sp, so, sq))

            # Ordena por mediana decrescente
            solvers_stats.sort(key=lambda item: item[1]["mediana"], reverse=True)

            for s_name, sl, sc, sv, sp, so, sq in solvers_stats:
                print(
                    f"{s_name:<26} | {sl['media']:>10.2f} | {sl['std']:>9.2f} | {sl['q1']:>10.2f} | "
                    f"{sl['mediana']:>10.2f} | {sl['q3']:>10.2f} | {sl['min']:>10.2f} | {sl['max']:>10.2f} | "
                    f"{sc['media']:>11.2f}"
                )
                estatisticas_consolidadas.append({
                    "modelo": mod,
                    "solver": s_name,
                    "total_execucoes": len(sl.get("lucros", [])),
                    "media_G": sl["media"],
                    "std_G": sl["std"],
                    "min_G": sl["min"],
                    "q1_G": sl["q1"],
                    "mediana_G": sl["mediana"],
                    "q3_G": sl["q3"],
                    "max_G": sl["max"],
                    "media_tempo_cpu_ms": sc["media"],
                    "media_tempo_viagem": sv["media"],
                    "media_peso_acumulado": sp["media"],
                    "media_pct_ocupacao_mochila": so["media"],
                    "media_qtd_itens": sq["media"],
                })

            print(f"{'='*115}")

        destino_stats = output_stats_csv or self.output_stats_csv
        if destino_stats:
            caminho_stats = Path(destino_stats)
            caminho_stats.parent.mkdir(parents=True, exist_ok=True)
            with open(caminho_stats, "w", newline="", encoding="utf-8") as f:
                colunas = [
                    "modelo", "solver", "media_G", "std_G", "min_G", "q1_G", "mediana_G", "q3_G", "max_G",
                    "media_tempo_cpu_ms", "media_tempo_viagem", "media_peso_acumulado",
                    "media_pct_ocupacao_mochila", "media_qtd_itens"
                ]
                writer = csv.DictWriter(f, fieldnames=colunas, delimiter=";")
                writer.writeheader()
                for st in estatisticas_consolidadas:
                    writer.writerow({
                        "modelo": st["modelo"],
                        "solver": st["solver"],
                        "media_G": f"{st['media_G']:.4f}",
                        "std_G": f"{st['std_G']:.4f}",
                        "min_G": f"{st['min_G']:.4f}",
                        "q1_G": f"{st['q1_G']:.4f}",
                        "mediana_G": f"{st['mediana_G']:.4f}",
                        "q3_G": f"{st['q3_G']:.4f}",
                        "max_G": f"{st['max_G']:.4f}",
                        "media_tempo_cpu_ms": f"{st['media_tempo_cpu_ms']:.2f}",
                        "media_tempo_viagem": f"{st['media_tempo_viagem']:.4f}",
                        "media_peso_acumulado": f"{st['media_peso_acumulado']:.4f}",
                        "media_pct_ocupacao_mochila": f"{st['media_pct_ocupacao_mochila']:.2f}",
                        "media_qtd_itens": f"{st['media_qtd_itens']:.2f}",
                    })
            print(f"\nRelatório estatístico consolidado salvo com sucesso em:\n  - {caminho_stats}\n")
