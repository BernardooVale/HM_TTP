"""
Script Experimental para Avaliação das Hipóteses 1 e 2 do artigo:
"Heurísticas para o Travelling Thief Problem" (ReinforceNS vs ReinforceNS-PM).

Suporta:
1. Experimento da Hipótese 1 (Não-inferioridade pareada sob orçamento de tempo T_max,
   incluindo execuções-piloto para cálculo da escala S_i e IC bootstrap 95%).
2. Experimento da Hipótese 2 (Ablação fatorial 2x2 de pós-processamento P0, P1, P2, P3,
   medindo tanto o efeito imediato sobre a mesma solução da busca local quanto o
   efeito ponta a ponta sob T_max).

Exemplo de uso rápido:
    .venv/bin/python scripts/executar_experimento_reinforcens.py \
        --instancias 11.ttp \
        --repeticoes 3 \
        --piloto 2 \
        --time-limit 3.0 \
        --output results/experimento_reinforcens.csv
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_raiz = Path(__file__).resolve().parent.parent
_sim_path = _raiz / "ttp_simulator"
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))
if str(_sim_path) not in sys.path:
    sys.path.insert(0, str(_sim_path))

from src.core.evaluator_ttp1 import avaliar_ttp1
from src.core.models import SolucaoTTP
from src.scenarios.parser import InstanceParser
from src.solvers.iterative.reinforcens_post_opt import apply_post_optimization
from src.solvers.iterative.reinforcens_solver import (
    ReinforceNSPMSolver,
    ReinforceNSSolver,
)


def _resolver_instancias(nomes: List[str]) -> List[Path]:
    pasta_bateria = _sim_path / "bateria_testes"
    resolvidos: List[Path] = []
    for nome in nomes:
        p = Path(nome)
        if p.is_dir():
            resolvidos.extend(sorted(p.resolve().glob("*.ttp")))
            continue
        if p.is_file():
            resolvidos.append(p.resolve())
            continue
        pasta_sub = pasta_bateria / nome
        if pasta_sub.is_dir():
            resolvidos.extend(sorted(pasta_sub.resolve().glob("*.ttp")))
            continue
        for cand in [
            pasta_bateria / nome,
            pasta_bateria / f"{nome}.ttp",
            pasta_bateria / "reinforcens" / nome,
            pasta_bateria / "reinforcens" / f"{nome}.ttp",
            pasta_bateria / "alto_prejuizo" / nome,
            pasta_bateria / "alto_prejuizo" / f"{nome}.ttp",
        ]:
            if cand.is_file():
                resolvidos.append(cand.resolve())
                break
        else:
            raise FileNotFoundError(f"Instância não encontrada: {nome}")
    return resolvidos


def _paired_bootstrap_ci(
    values: List[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr))
    if len(arr) <= 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[idx], axis=1)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, low, high


def main():
    parser = argparse.ArgumentParser(
        description="Executa os experimentos pareados de H1 (ReinforceNS vs ReinforceNS-PM) e H2 (P0..P3)."
    )
    parser.add_argument(
        "--instancias",
        nargs="+",
        default=["11.ttp"],
        help="Lista de arquivos de instância (ex: 11.ttp 121.ttp).",
    )
    parser.add_argument(
        "--repeticoes",
        type=int,
        default=5,
        help="Número de repetições pareadas R por instância (default: 5).",
    )
    parser.add_argument(
        "--piloto",
        type=int,
        default=3,
        help="Número de execuções-piloto com sementes distintas para cálculo de S_i (default: 3).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
        help="Orçamento de tempo T_max em segundos por execução (default: 5.0).",
    )
    parser.add_argument(
        "--max-starts",
        type=int,
        default=None,
        help="Limite opcional de starts por execução (default: ilimitado até T_max).",
    )
    parser.add_argument(
        "--clk-tries",
        type=int,
        default=2,
        help="Número de tentativas no LKLocalSearch (default: 2).",
    )
    parser.add_argument(
        "--clk-kicks",
        type=int,
        default=1,
        help="Número de kicks DoubleBridge no ChainedLK (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/experimento_reinforcens_h1_h2.csv",
        help="Caminho do arquivo CSV de saída.",
    )
    args = parser.parse_args()

    inst_paths = _resolver_instancias(args.instancias)
    out_path = (_raiz / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    vis_rows: List[Dict[str, object]] = []
    deltas_h1: List[float] = []

    def _avaliar_para_vis(instance, sol: SolucaoTTP) -> Dict[str, float]:
        ev = avaliar_ttp1(instance, sol)
        cap = max(1e-9, float(instance.capacidade_mochila))
        peso = float(ev["peso_acumulado"])
        return {
            "lucro_liquido_G": float(ev["lucro_liquido_G"]),
            "tempo_viagem": float(ev["tempo_viagem_f"]),
            "peso_acumulado": peso,
            "pct_ocupacao_mochila": 100.0 * peso / cap,
            "qtd_itens": int(ev["qtd_itens_coletados"]),
        }

    for inst_path in inst_paths:
        instance = InstanceParser.carregar(str(inst_path))
        inst_name = inst_path.stem
        print(
            f"\n=== Instância: {inst_name} (n={instance.num_cities}, m={len(instance.itens)}) ==="
        )

        # 1. Execuções-piloto com sementes distintas (90001..) para definir S_i
        pilot_scores: List[float] = []
        for p_idx in range(args.piloto):
            p_seed = 90001 + p_idx * 101
            solver_pilot = ReinforceNSSolver(
                time_limit=args.time_limit,
                max_starts=args.max_starts,
                clk_num_tries=args.clk_tries,
                clk_num_kicks=args.clk_kicks,
                seed=p_seed,
            )
            sol_p = solver_pilot.solve(instance)
            res_p = avaliar_ttp1(instance, sol_p)
            pilot_scores.append(float(res_p["lucro_liquido_G"]))

        median_pilot = statistics.median(pilot_scores) if pilot_scores else 1.0
        scale_s_i = max(1.0, abs(median_pilot))
        print(f"  [Piloto] Mediana f_Q = {median_pilot:.2f} -> Escala S_i = {scale_s_i:.2f}")

        # 2. Repetições pareadas r = 1 .. R para Hipótese 1 e Hipótese 2 (efeito imediato)
        for r in range(1, args.repeticoes + 1):
            rep_seed = 1000 + r

            solver_q = ReinforceNSSolver(
                time_limit=args.time_limit,
                max_starts=args.max_starts,
                clk_num_tries=args.clk_tries,
                clk_num_kicks=args.clk_kicks,
                seed=rep_seed,
            )
            solver_pm = ReinforceNSPMSolver(
                time_limit=args.time_limit,
                max_starts=args.max_starts,
                clk_num_tries=args.clk_tries,
                clk_num_kicks=args.clk_kicks,
                seed=rep_seed,
            )

            sol_q = solver_q.solve(instance)
            sol_pm = solver_pm.solve(instance)

            vis_q = _avaliar_para_vis(instance, sol_q)
            vis_pm = _avaliar_para_vis(instance, sol_pm)
            f_q = vis_q["lucro_liquido_G"]
            f_pm = vis_pm["lucro_liquido_G"]
            delta_ir = 100.0 * (f_pm - f_q) / scale_s_i
            deltas_h1.append(delta_ir)

            # Efeito imediato de P0, P1, P2, P3 sobre a mesma solução inicial pareada
            solver_p0 = ReinforceNSSolver(
                post_opt_mode="P0",
                time_limit=args.time_limit,
                max_starts=1,
                clk_num_tries=args.clk_tries,
                clk_num_kicks=args.clk_kicks,
                seed=rep_seed,
            )
            sol_base_ls = solver_p0.solve(instance)
            res_p0 = apply_post_optimization(
                instance, sol_base_ls.rota, sol_base_ls.plano_coleta, mode="P0"
            )
            res_p1 = apply_post_optimization(
                instance, sol_base_ls.rota, sol_base_ls.plano_coleta, mode="P1"
            )
            res_p2 = apply_post_optimization(
                instance, sol_base_ls.rota, sol_base_ls.plano_coleta, mode="P2"
            )
            res_p3 = apply_post_optimization(
                instance, sol_base_ls.rota, sol_base_ls.plano_coleta, mode="P3"
            )

            vis_p0 = _avaliar_para_vis(instance, SolucaoTTP(rota=res_p0.route, plano_coleta=res_p0.plan))
            vis_p1 = _avaliar_para_vis(instance, SolucaoTTP(rota=res_p1.route, plano_coleta=res_p1.plan))
            vis_p2 = _avaliar_para_vis(instance, SolucaoTTP(rota=res_p2.route, plano_coleta=res_p2.plan))
            vis_p3 = _avaliar_para_vis(instance, SolucaoTTP(rota=res_p3.route, plano_coleta=res_p3.plan))

            for s_name, metrics in [
                ("reinforcens (Q-Learning + P3)", vis_q),
                ("reinforcens_pm (First Improvement + P3)", vis_pm),
                ("reinforcens_p0 (Sem Pós-Opt)", vis_p0),
                ("reinforcens_p1 (Apenas IFP)", vis_p1),
                ("reinforcens_p2 (Apenas RRP)", vis_p2),
                ("reinforcens_p3 (IFP + RRP)", vis_p3),
            ]:
                vis_rows.append({
                    "instancia": inst_name,
                    "solver": s_name,
                    "modelo": "TTP1",
                    "repeticao": r,
                    "seed": rep_seed,
                    "lucro_liquido_G": metrics["lucro_liquido_G"],
                    "tempo_computacional_seg": args.time_limit,
                    "tempo_viagem": metrics["tempo_viagem"],
                    "peso_acumulado": metrics["peso_acumulado"],
                    "pct_ocupacao_mochila": metrics["pct_ocupacao_mochila"],
                    "qtd_itens": metrics["qtd_itens"],
                })

            tel_q = solver_q.last_telemetry
            tel_pm = solver_pm.last_telemetry

            row = {
                "instancia": inst_name,
                "repeticao": r,
                "semente": rep_seed,
                "escala_instancia_S_i": scale_s_i,
                "lucro_inicial_CLK_IGP": tel_q.first_start_init_objective if tel_q else 0.0,
                "lucro_final_QLearning": f_q,
                "lucro_final_PrimeiraMelhoria": f_pm,
                "diferenca_H1_PM_vs_Q_pct": delta_ir,
                "ciclos_multistart_QLearning": tel_q.starts_completed if tel_q else 0,
                "ciclos_multistart_PrimeiraMelhoria": tel_pm.starts_completed if tel_pm else 0,
                "lucro_pos_opt_P0_nenhuma": res_p0.objective_final,
                "lucro_pos_opt_P1_apenas_IFP": res_p1.objective_final,
                "lucro_pos_opt_P2_apenas_RRP": res_p2.objective_final,
                "lucro_pos_opt_P3_IFP_e_RRP": res_p3.objective_final,
                "ganho_P1_apenas_IFP_pct": 100.0
                * (res_p1.objective_final - res_p0.objective_final)
                / scale_s_i,
                "ganho_P2_apenas_RRP_pct": 100.0
                * (res_p2.objective_final - res_p0.objective_final)
                / scale_s_i,
                "ganho_P3_IFP_e_RRP_pct": 100.0
                * (res_p3.objective_final - res_p0.objective_final)
                / scale_s_i,
            }
            rows.append(row)
            print(
                f"  [Rep {r:02d} | seed={rep_seed}] "
                f"f_0={row['lucro_inicial_CLK_IGP']:.2f} | "
                f"f_Q={f_q:.2f} | f_PM={f_pm:.2f} | "
                f"Δ_H1={delta_ir:+.3f}% | "
                f"P0={res_p0.objective_final:.2f} -> P3={res_p3.objective_final:.2f}",
                flush=True,
            )

        # Salvamento incremental ao fim de cada instância para evitar perda de progresso
        if rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            vis_path_inc = out_path.with_name(out_path.stem + "_visualizador.csv")
            with open(vis_path_inc, "w", newline="", encoding="utf-8") as f_vis:
                writer_vis = csv.DictWriter(f_vis, fieldnames=list(vis_rows[0].keys()))
                writer_vis.writeheader()
                writer_vis.writerows(vis_rows)

    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        # Também gera o CSV no formato longo compatível com os gráficos e Heat Grid do visualizador/gerar_site.py
        vis_path = out_path.with_name(out_path.stem + "_visualizador.csv")
        with open(vis_path, "w", newline="", encoding="utf-8") as f_vis:
            writer_vis = csv.DictWriter(f_vis, fieldnames=list(vis_rows[0].keys()))
            writer_vis.writeheader()
            writer_vis.writerows(vis_rows)

        # Atualiza automaticamente o visualizador/index.html se gerar_site.py existir
        gerar_site_script = _raiz / "visualizador" / "gerar_site.py"
        if gerar_site_script.is_file():
            import subprocess
            subprocess.run(
                [sys.executable, str(gerar_site_script), "--csv", str(vis_path)],
                check=False,
            )

        mean_d, ci_low, ci_high = _paired_bootstrap_ci(deltas_h1)
        print("\n============================================================")
        print(f"Resultados pareados (H1/H2) salvos em: {out_path}")
        print(f"Resultados para o Visualizador salvos em: {vis_path}")
        print(
            f"Hipótese 1 (Δ = f_PM - f_Q em % de S_i): "
            f"Média = {mean_d:+.3f}% | IC 95% Bootstrap = [{ci_low:+.3f}%, {ci_high:+.3f}%]"
        )
        print(
            f"Critério de não-inferioridade (limite inferior > -1.000%): "
            f"{'ATENDIDO' if ci_low > -1.0 else 'NÃO ATENDIDO'}"
        )
        print("============================================================")


if __name__ == "__main__":
    main()
