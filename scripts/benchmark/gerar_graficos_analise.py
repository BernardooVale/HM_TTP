"""
Gera gráficos analíticos comparando o desempenho dos solvers por:
1. Fronteira de eficiência (Tempo log x RDI) dividida em KP (G* > 0) vs TSP (G* <= 0);
2. Boxplot de RDI (%) por faixa de tamanho das instâncias (Pequenas, Médias, Grandes);
3. Confronto direto (Vitória / Empate / Derrota) de queimada e reinforcens_pm_queimada vs literatura em KP vs TSP;
4. Impacto da escolha do conjunto de instâncias (reinforcens vs equilibradas vs aleatorias_equilibradas) no RDI.
"""

import collections
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_raiz = Path(__file__).resolve().parents[2]
_sim_path = _raiz / "ttp_simulator"
sys.path.insert(0, str(_raiz))
sys.path.insert(0, str(_sim_path))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import avaliar_todos_solvers as ats
from src.scenarios.parser import InstanceParser


SUITES = [
    ("reinforcens", _raiz / "results/campeoes_reinforcens/execucoes_detalhadas.csv"),
    ("equilibradas", _raiz / "results/campeoes_equilibradas/execucoes_detalhadas.csv"),
    (
        "aleatorias_equilibradas",
        _raiz / "results/campeoes_aleatorias_equilibradas/execucoes_detalhadas.csv",
    ),
]

SOLVER_LABELS = {
    "reinforcens_pm": "ReinforceNS-PM",
    "reinforcens_pm_queimada": "PM + Queimada",
    "reinforcens_queimada": "RL + Queimada",
    "reinforcens": "ReinforceNS",
    "coco_solver": "CoCo",
    "cttp_2opt": "CTTP-2opt",
    "queimada": "Queimada",
    "faulkner_c3": "Faulkner-C3",
    "wsm_solver": "WSM",
    "cosolver": "CoSolver",
    "s5_refinada": "S5-Refinada",
    "pack_iterative": "PackIterative",
    "queimada_construtiva": "Queimada-Constr.",
}

SOLVER_COLORS = {
    "reinforcens_pm": "#1f77b4",
    "reinforcens_pm_queimada": "#d62728",
    "reinforcens_queimada": "#ff7f0e",
    "reinforcens": "#17becf",
    "coco_solver": "#2ca02c",
    "cttp_2opt": "#9467bd",
    "queimada": "#e377c2",
    "faulkner_c3": "#8c564b",
    "wsm_solver": "#bcbd22",
    "cosolver": "#7f7f7f",
    "s5_refinada": "#aec7e8",
    "pack_iterative": "#98df8a",
    "queimada_construtiva": "#f7b6d2",
}


def carregar_dados():
    inst_paths, _ = ats.coletar_instancias()
    inst_n: Dict[str, int] = {
        name: InstanceParser.carregar(p).num_cities for name, p in inst_paths.items()
    }

    dedup: Dict[Tuple[str, str, int], Tuple[float, float]] = {}
    suite_insts: Dict[str, set] = {s: set() for s, _ in SUITES}
    for suite_name, csv_path in SUITES:
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["status"] != "ok":
                    continue
                g = float(r["lucro_G"])
                t = float(r["tempo_s"])
                dedup[(r["solver"], r["instancia"], int(r["seed"]))] = (g, t)
                suite_insts[suite_name].add(r["instancia"])

    by_si = collections.defaultdict(list)
    time_si = collections.defaultdict(list)
    for (s, i, _), (g, t) in dedup.items():
        by_si[(s, i)].append(g)
        time_si[(s, i)].append(t)

    insts = sorted({k[1] for k in dedup})
    solvers = sorted({k[0] for k in dedup if k[0] != "local_search"})

    best_i = {
        i: max(statistics.mean(by_si[(s, i)]) for s in solvers if (s, i) in by_si)
        for i in insts
    }
    worst_i = {
        i: min(
            statistics.mean(by_si[(s, i)])
            for s in sorted({k[0] for k in dedup})
            if (s, i) in by_si
        )
        for i in insts
    }

    return inst_n, suite_insts, by_si, time_si, insts, solvers, best_i, worst_i


def calc_rdi(solver: str, inst: str, by_si, best_i, worst_i) -> float:
    m = statistics.mean(by_si[(solver, inst)])
    bi, wi = best_i[inst], worst_i[inst]
    return ((bi - m) / (bi - wi) * 100.0) if bi > wi else 0.0


def plot_1_fronteira_eficiencia(
    out_dir: Path, by_si, time_si, insts, solvers, best_i, worst_i
):
    kp_insts = [i for i in insts if best_i[i] > 0]
    tsp_insts = [i for i in insts if best_i[i] <= 0]

    custom_offsets = {
        (0, "PackIterative"): (-55, -20),
        (0, "Faulkner-C3"): (-88, 6),
        (0, "WSM"): (7, -4),
        (0, "ReinforceNS"): (-84, -12),
        (0, "CoCo"): (7, -14),
        (1, "PackIterative"): (-62, -16),
        (1, "S5-Refinada"): (8, -8),
        (1, "WSM"): (-70, 4),
        (1, "ReinforceNS"): (-88, 6),
        (1, "CTTP-2opt"): (8, 4),
        (1, "ReinforceNS-PM"): (-96, -8),
        (1, "PM + Queimada"): (-25, -28),
        (1, "CoCo"): (8, -14),
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), sharey=False)
    for panel_idx, (ax, title, sub) in enumerate([
        (axes[0], f"Grupo KP-Favorável (G* > 0, n={len(kp_insts)} inst.)", kp_insts),
        (
            axes[1],
            f"Grupo TSP-Favorável / Prejudicial (G* <= 0, n={len(tsp_insts)} inst.)",
            tsp_insts,
        ),
    ]):
        xs, ys, labels, cols = [], [], [], []
        for s in solvers:
            if s == "queimada_construtiva":
                continue
            rdis = [calc_rdi(s, i, by_si, best_i, worst_i) for i in sub if (s, i) in by_si]
            ts = [statistics.mean(time_si[(s, i)]) for i in sub if (s, i) in by_si]
            if not rdis:
                continue
            xs.append(statistics.mean(ts))
            ys.append(statistics.mean(rdis))
            labels.append(SOLVER_LABELS.get(s, s))
            cols.append(SOLVER_COLORS.get(s, "#333333"))

        for x, y, lbl, c in zip(xs, ys, labels, cols):
            size = 150 if lbl in ("Queimada", "PM + Queimada", "ReinforceNS-PM", "CoCo") else 95
            marker = "*" if "Queimada" in lbl else "o"
            ax.scatter(x, y, s=size, color=c, marker=marker, edgecolors="black", linewidths=0.7, zorder=4)
            dx, dy = custom_offsets.get((panel_idx, lbl), (6, 4))
            ax.annotate(
                f"{lbl}\n({y:.2f}%, {x:.1f}s)",
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
                alpha=0.92,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Tempo Total Médio (s) [escala log]", fontsize=10)
        ax.set_ylabel("RDI Médio (%)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        if panel_idx == 0:
            ax.set_ylim(-0.5, 18.0)
            ax.set_xlim(1.0, 44.0)
        else:
            ax.set_ylim(-0.5, 9.4)
            ax.set_xlim(0.3, 18.0)

    fig.suptitle(
        "Fronteira de Eficiência: Qualidade (RDI Médio) vs Tempo Total — KP vs TSP",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "01_fronteira_eficiencia_kp_vs_tsp.png", dpi=300)
    plt.close(fig)


def plot_2_boxplot_tamanho(
    out_dir: Path, inst_n, by_si, time_si, insts, best_i, worst_i
):
    top_solvers = [
        "reinforcens_pm",
        "reinforcens_pm_queimada",
        "coco_solver",
        "cttp_2opt",
        "reinforcens",
        "reinforcens_queimada",
        "faulkner_c3",
        "queimada",
    ]

    def bucket(n: int) -> str:
        if n <= 52:
            return "Pequenas (n <= 52)"
        if n <= 160:
            return "Médias (53 <= n <= 160)"
        return "Grandes (n > 160)"

    buckets = [
        "Pequenas (n <= 52)",
        "Médias (53 <= n <= 160)",
        "Grandes (n > 160)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), sharey=True)
    first_ylabels = None
    for ax, b in zip(axes, buckets):
        sub = [i for i in insts if bucket(inst_n[i]) == b]
        data = []
        ylabels = []
        for s in reversed(top_solvers):
            rdis = [calc_rdi(s, i, by_si, best_i, worst_i) for i in sub if (s, i) in by_si]
            ts = [statistics.mean(time_si[(s, i)]) for i in sub if (s, i) in by_si]
            data.append(rdis)
            t_med = statistics.mean(ts) if ts else 0.0
            ylabels.append(f"{SOLVER_LABELS.get(s, s)} ({t_med:.1f}s)")
        if ax is axes[0]:
            first_ylabels = ylabels

        bp = ax.boxplot(
            data,
            orientation="horizontal",
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3.5, alpha=0.55),
            medianprops=dict(color="black", linewidth=1.6),
        )
        for patch, s in zip(bp["boxes"], reversed(top_solvers)):
            patch.set_facecolor(SOLVER_COLORS.get(s, "#cccccc"))
            patch.set_alpha(0.75)

        if ax is not axes[0]:
            # Anota tempo médio ao lado direito da faixa
            for idx, s in enumerate(reversed(top_solvers), 1):
                ts = [statistics.mean(time_si[(s, i)]) for i in sub if (s, i) in by_si]
                t_med = statistics.mean(ts) if ts else 0.0
                ax.text(
                    0.98,
                    idx,
                    f"{t_med:.1f}s",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    color="#333333",
                )

        kp_cnt = sum(1 for i in sub if best_i[i] > 0)
        tsp_cnt = len(sub) - kp_cnt
        ax.set_title(f"{b}\n({len(sub)} inst: {kp_cnt} KP / {tsp_cnt} TSP)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("RDI (%)", fontsize=10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        if b != "Pequenas (n <= 52)":
            ax.set_xlim(-0.2, 12.0 if b == "Médias (53 <= n <= 160)" else 4.0)

    axes[0].set_yticks(range(1, len(top_solvers) + 1))
    axes[0].set_yticklabels(first_ylabels, fontsize=9)

    fig.suptitle(
        "Distribuição de RDI (%) por Faixa de Tamanho das Instâncias (com tempo médio)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / "02_boxplot_rdi_por_tamanho.png", dpi=300)
    plt.close(fig)


def plot_3_confronto_direto(out_dir: Path, by_si, insts, best_i):
    kp_insts = [i for i in insts if best_i[i] > 0]
    tsp_insts = [i for i in insts if best_i[i] <= 0]
    opponents = [
        "reinforcens_pm",
        "coco_solver",
        "cttp_2opt",
        "faulkner_c3",
        "wsm_solver",
        "cosolver",
        "s5_refinada",
        "pack_iterative",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    focal_solvers = [
        ("queimada", "Queimada Isolada"),
        ("reinforcens_pm_queimada", "PM + Queimada (Hot-Start)"),
    ]
    groups = [
        (kp_insts, f"KP-Favorável (G* > 0, {len(kp_insts)} inst.)"),
        (tsp_insts, f"TSP-Favorável / Prejudicial (G* <= 0, {len(tsp_insts)} inst.)"),
    ]

    for row_idx, (focal, focal_lbl) in enumerate(focal_solvers):
        for col_idx, (sub, grp_lbl) in enumerate(groups):
            ax = axes[row_idx, col_idx]
            wins_p, ties_p, loss_p, ylbls = [], [], [], []
            for opp in reversed(opponents):
                if opp == focal:
                    continue
                w = t = l = 0
                for i in sub:
                    if (focal, i) not in by_si or (opp, i) not in by_si:
                        continue
                    gf = statistics.mean(by_si[(focal, i)])
                    go = statistics.mean(by_si[(opp, i)])
                    if gf > go + 1e-6:
                        w += 1
                    elif go > gf + 1e-6:
                        l += 1
                    else:
                        t += 1
                tot = max(w + t + l, 1)
                wins_p.append(100.0 * w / tot)
                ties_p.append(100.0 * t / tot)
                loss_p.append(100.0 * l / tot)
                ylbls.append(f"vs {SOLVER_LABELS.get(opp, opp)}")

            y = np.arange(len(ylbls))
            ax.barh(y, wins_p, color="#2ca02c", alpha=0.85, label="Vitória")
            ax.barh(y, ties_p, left=wins_p, color="#bdbdbd", alpha=0.85, label="Empate")
            left2 = [a + b for a, b in zip(wins_p, ties_p)]
            ax.barh(y, loss_p, left=left2, color="#d62728", alpha=0.80, label="Derrota")

            for idx, (wp, tp, lp) in enumerate(zip(wins_p, ties_p, loss_p)):
                if wp >= 12:
                    ax.text(wp / 2, idx, f"{wp:.0f}%", va="center", ha="center", color="white", fontsize=8, fontweight="bold")
                if lp >= 12:
                    ax.text(wp + tp + lp / 2, idx, f"{lp:.0f}%", va="center", ha="center", color="white", fontsize=8, fontweight="bold")

            ax.set_yticks(y)
            ax.set_yticklabels(ylbls, fontsize=9)
            ax.set_xlim(0, 100)
            ax.set_title(f"{focal_lbl} — {grp_lbl}", fontsize=10.5, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("% das instâncias", fontsize=10)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Confronto Direto (Vitória / Empate / Derrota) por Regime da Instância",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "03_confronto_direto_queimada_kp_vs_tsp.png", dpi=300)
    plt.close(fig)


def plot_4_impacto_bateria(out_dir: Path, suite_insts, by_si, best_i, worst_i):
    sel_solvers = [
        "reinforcens_pm_queimada",
        "reinforcens_pm",
        "coco_solver",
        "cttp_2opt",
        "reinforcens",
        "reinforcens_queimada",
        "queimada",
        "faulkner_c3",
    ]
    suites_order = [
        ("reinforcens", "ReinforceNS (18 inst., 100% KP)"),
        ("equilibradas", "Equilibradas (34 inst., 76% TSP)"),
        ("aleatorias_equilibradas", "Aleatórias Equilibradas (35 inst.)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=False)
    x = np.arange(len(sel_solvers))
    width = 0.25
    suite_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, metric_name, fn in [
        (axes[0], "RDI Médio (%)", statistics.mean),
        (axes[1], "Mediana do RDI (%)", statistics.median),
    ]:
        pad = 0.12 if ax is axes[0] else 0.03
        for idx, ((s_key, s_lbl), col) in enumerate(zip(suites_order, suite_colors)):
            sub = sorted(suite_insts[s_key])
            vals = []
            for s in sel_solvers:
                rdis = [calc_rdi(s, i, by_si, best_i, worst_i) for i in sub if (s, i) in by_si]
                vals.append(fn(rdis) if rdis else 0.0)
            offset = (idx - 1) * width
            bars = ax.bar(x + offset, vals, width, label=s_lbl, color=col, alpha=0.85, edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + pad,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90 if (ax is axes[0] and v > 2.5) else 0,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([SOLVER_LABELS[s] for s in sel_solvers], rotation=28, ha="right", fontsize=9)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 9.8 if ax is axes[0] else 2.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(
        "Impacto do Conjunto de Instâncias no Desempenho Relativo dos Solvers",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "04_impacto_conjunto_instancias_rdi.png", dpi=300)
    plt.close(fig)


def plot_5_panorama_conjuntos(out_dir: Path, suite_insts, by_si, time_si, best_i, worst_i):
    sel_solvers = [
        "reinforcens_pm",
        "coco_solver",
        "reinforcens_pm_queimada",
        "reinforcens_queimada",
        "reinforcens",
        "faulkner_c3",
        "queimada",
        "cttp_2opt",
        "wsm_solver",
        "cosolver",
    ]
    suites_order = [
        ("reinforcens", "ReinforceNS\n(18 inst.)"),
        ("equilibradas", "Equilibradas\n(34 inst.)"),
        ("aleatorias_equilibradas", "Aleatórias Equil.\n(35 inst.)"),
    ]
    suite_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Calcula RDI (usando best_i/worst_i globais de todos os solvers), Tempo médio, V/E/D vs ReinforceNS-PM e Rank dentro de cada conjunto
    stats_suite = {}
    for s_key, _ in suites_order:
        sub = sorted(suite_insts[s_key])
        rows = {}
        for s in sel_solvers:
            rdis = []
            ts = []
            w = t = l = 0
            for i in sub:
                if (s, i) not in by_si:
                    continue
                ms = statistics.mean(by_si[(s, i)])
                bi, wi = best_i[i], worst_i[i]
                rdi = ((bi - ms) / (bi - wi) * 100.0) if bi > wi else 0.0
                rdis.append(rdi)
                ts.append(statistics.mean(time_si[(s, i)]))
                if s != "reinforcens_pm" and ("reinforcens_pm", i) in by_si:
                    mref = statistics.mean(by_si[("reinforcens_pm", i)])
                    if ms > mref + 1e-6:
                        w += 1
                    elif mref > ms + 1e-6:
                        l += 1
                    else:
                        t += 1
            rows[s] = {
                "rdi": statistics.mean(rdis) if rdis else 0.0,
                "time": statistics.mean(ts) if ts else 0.0,
                "wtl": (w, t, l),
                "n": len(rdis),
            }
        ordered = sorted(sel_solvers, key=lambda s: rows[s]["rdi"])
        for rank_idx, s in enumerate(ordered, 1):
            rows[s]["rank"] = rank_idx
        stats_suite[s_key] = rows

    # Ordena solvers no painel esquerdo pelo rank em ReinforceNS (de cima para baixo)
    solvers_ordered = sorted(sel_solvers, key=lambda s: stats_suite["reinforcens"][s]["rank"])

    fig = plt.figure(figsize=(19.0, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_bump = fig.add_subplot(gs[0, 1])

    # --- Painel Esquerdo: RDI Médio (%) + Tempo + V/E/D vs ReinforceNS-PM ---
    y_base = np.arange(len(solvers_ordered))[::-1]
    bar_h = 0.25

    for idx, ((s_key, s_lbl), col) in enumerate(zip(suites_order, suite_colors)):
        offset = (1 - idx) * bar_h  # ReinforceNS em cima, Equilibradas no meio, Aleatórias embaixo
        vals = [stats_suite[s_key][s]["rdi"] for s in solvers_ordered]
        bars = ax_bar.barh(
            y_base + offset,
            vals,
            height=bar_h,
            color=col,
            alpha=0.86,
            edgecolor="black",
            linewidth=0.5,
            label=s_lbl.replace("\n", " "),
        )
        for bar, s, v in zip(bars, solvers_ordered, vals):
            st = stats_suite[s_key][s]
            if s == "reinforcens_pm":
                wtl_txt = f"Ref. (n={st['n']})"
            else:
                w, t, l = st["wtl"]
                wtl_txt = f"{w}V/{t}E/{l}D"
            ax_bar.text(
                v + 0.14,
                bar.get_y() + bar.get_height() / 2,
                f"#{st['rank']} | {v:.2f}% | {st['time']:.1f}s | {wtl_txt}",
                va="center",
                ha="left",
                fontsize=7.3,
                color="#222222",
                fontweight="bold" if "queimada" in s else "normal",
            )

    ax_bar.set_yticks(y_base)
    ax_bar.set_yticklabels(
        [
            f"{SOLVER_LABELS[s]}\n(Rank #{stats_suite['reinforcens'][s]['rank']} → #{stats_suite['equilibradas'][s]['rank']} → #{stats_suite['aleatorias_equilibradas'][s]['rank']})"
            for s in solvers_ordered
        ],
        fontsize=8.8,
    )
    ax_bar.set_xlabel("RDI Médio (%)  [anotação: #Rank | RDI% | Tempo Médio | V/E/D vs ReinforceNS-PM]", fontsize=9.5)
    ax_bar.set_title("Desempenho por Conjunto: RDI (%), Tempo e V/E/D vs ReinforceNS-PM", fontsize=11, fontweight="bold")
    max_rdi = max(stats_suite[sk][s]["rdi"] for sk, _ in suites_order for s in solvers_ordered)
    ax_bar.set_xlim(0, max_rdi + 5.8)
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_bar.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # --- Painel Direito: Bump Chart de Variação de Rank ---
    x_pos = np.arange(len(suites_order))
    for s in solvers_ordered:
        ranks = [stats_suite[sk][s]["rank"] for sk, _ in suites_order]
        rdis = [stats_suite[sk][s]["rdi"] for sk, _ in suites_order]
        col = SOLVER_COLORS.get(s, "#333333")
        lw = 3.0 if "queimada" in s or s in ("reinforcens_pm", "coco_solver") else 1.8
        alpha = 0.95 if "queimada" in s or s in ("reinforcens_pm", "coco_solver") else 0.72
        ax_bump.plot(x_pos, ranks, color=col, linewidth=lw, alpha=alpha, zorder=3)
        for xi, rk, rdi_v in zip(x_pos, ranks, rdis):
            ax_bump.scatter(
                xi,
                rk,
                s=220,
                color=col,
                edgecolors="black",
                linewidths=0.9,
                zorder=4,
            )
            ax_bump.text(
                xi,
                rk,
                f"{rk}",
                ha="center",
                va="center",
                color="white",
                fontsize=8.5,
                fontweight="bold",
                zorder=5,
            )
        delta_r = ranks[0] - ranks[-1]
        delta_str = f"+{delta_r}" if delta_r > 0 else f"{delta_r}"
        ax_bump.text(
            x_pos[-1] + 0.12,
            ranks[-1],
            f"{SOLVER_LABELS[s]} (ΔRank: {delta_str})",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold" if "queimada" in s else "normal",
            color=col if "queimada" in s else "#222222",
        )

    ax_bump.set_xticks(x_pos)
    ax_bump.set_xticklabels([lbl for _, lbl in suites_order], fontsize=9.2, fontweight="bold")
    ax_bump.set_yticks(range(1, len(solvers_ordered) + 1))
    ax_bump.set_yticklabels([f"#{r}" for r in range(1, len(solvers_ordered) + 1)], fontsize=9)
    ax_bump.invert_yaxis()
    ax_bump.set_xlim(-0.22, 3.30)
    ax_bump.set_ylim(len(solvers_ordered) + 0.55, 0.45)
    ax_bump.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_bump.set_title("Variação de Rank entre Conjuntos (1 = melhor)", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Panorama Consolidado: RDI (%), Tempo, V/E/D e Variação de Rank entre Conjuntos de Instâncias",
        fontsize=13.5,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.09, wspace=0.16)
    fig.savefig(out_dir / "05_panorama_conjuntos_rdi_rank.png", dpi=300)
    plt.close(fig)


def main():
    out_dir = _raiz / "results/analise_kp_vs_tsp/graficos"
    out_dir.mkdir(parents=True, exist_ok=True)
    inst_n, suite_insts, by_si, time_si, insts, solvers, best_i, worst_i = carregar_dados()

    plot_1_fronteira_eficiencia(out_dir, by_si, time_si, insts, solvers, best_i, worst_i)
    plot_2_boxplot_tamanho(out_dir, inst_n, by_si, time_si, insts, best_i, worst_i)
    plot_3_confronto_direto(out_dir, by_si, insts, best_i)
    plot_4_impacto_bateria(out_dir, suite_insts, by_si, best_i, worst_i)
    plot_5_panorama_conjuntos(out_dir, suite_insts, by_si, time_si, best_i, worst_i)
    print(f"Graficos salvos em: {out_dir}")


if __name__ == "__main__":
    main()
