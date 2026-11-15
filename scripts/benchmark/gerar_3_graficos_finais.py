"""
Gera os 3 gráficos finais solicitados:
1. 01_queimada_vs_literatura.png — Queimada isolada vs literatura (destaque apenas para Queimada);
2. 02_queimada_com_hotstart.png — Incluindo os hot-starts (destaque apenas para PM + Queimada e RL + Queimada);
3. 03_reinforcens_vs_equilibradas.png — Trajetória 2D (Tempo log x RDI) de ReinforceNS -> Equilibradas.
"""

import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_raiz = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gerar_graficos_analise import (
    SOLVER_COLORS,
    SOLVER_LABELS,
    calc_rdi,
    carregar_dados,
)


def _plot_comparativo_geral(
    out_path: Path,
    title: str,
    solvers_list,
    highlight_solvers: dict,
    by_si,
    time_si,
    insts,
    best_i,
    worst_i,
):
    rows = []
    for s in solvers_list:
        rdis = [calc_rdi(s, i, by_si, best_i, worst_i) for i in insts if (s, i) in by_si]
        ts = [statistics.mean(time_si[(s, i)]) for i in insts if (s, i) in by_si]
        w = t = l = 0
        for i in insts:
            if (s, i) not in by_si or ("reinforcens_pm", i) not in by_si:
                continue
            if s == "reinforcens_pm":
                continue
            ms = statistics.mean(by_si[(s, i)])
            mr = statistics.mean(by_si[("reinforcens_pm", i)])
            if ms > mr + 1e-6:
                w += 1
            elif mr > ms + 1e-6:
                l += 1
            else:
                t += 1
        rows.append(
            {
                "solver": s,
                "label": SOLVER_LABELS.get(s, s),
                "rdi": statistics.mean(rdis),
                "time": statistics.mean(ts),
                "wtl": (w, t, l),
                "n": len(rdis),
            }
        )

    rows.sort(key=lambda r: r["rdi"])
    for idx, r in enumerate(rows, 1):
        r["rank"] = idx

    rows_plot = list(reversed(rows))
    y = np.arange(len(rows_plot))
    vals = [r["rdi"] for r in rows_plot]
    max_v = max(vals)

    fig, ax = plt.subplots(figsize=(13.8, 6.6))

    for yi, r in zip(y, rows_plot):
        s = r["solver"]
        is_hi = s in highlight_solvers
        col = highlight_solvers[s] if is_hi else "#c4cbd2"
        alpha = 0.92 if is_hi else 0.72
        lw = 1.4 if is_hi else 0.6
        ax.barh(yi, r["rdi"], height=0.62, color=col, alpha=alpha, edgecolor="#222222" if is_hi else "#666666", linewidth=lw, zorder=3)

    # Colunas alinhadas à direita das barras
    col_rdi_x = max_v + 1.2
    col_time_x = max_v + 3.4
    col_wtl_x = max_v + 5.6

    hdr_y = len(rows_plot) - 0.15
    ax.text(col_rdi_x, hdr_y, "RDI Médio", fontsize=9, fontweight="bold", ha="right", va="bottom", color="#333333")
    ax.text(col_time_x, hdr_y, "Tempo", fontsize=9, fontweight="bold", ha="right", va="bottom", color="#333333")
    ax.text(col_wtl_x, hdr_y, "V / E / D (vs Ref.)", fontsize=9, fontweight="bold", ha="left", va="bottom", color="#333333")

    for yi, r in zip(y, rows_plot):
        s = r["solver"]
        is_hi = s in highlight_solvers
        fw = "bold" if is_hi else "normal"
        tc = "#111111" if is_hi else "#444444"
        if s == "reinforcens_pm":
            wtl_txt = f"Ref. (n={r['n']})"
        else:
            w, t, l = r["wtl"]
            wtl_txt = f"{w}V / {t}E / {l}D"

        ax.text(col_rdi_x, yi, f"{r['rdi']:.2f}%", fontsize=9.2, fontweight=fw, ha="right", va="center", color=tc)
        ax.text(col_time_x, yi, f"{r['time']:.1f}s", fontsize=9.2, fontweight=fw, ha="right", va="center", color=tc)
        ax.text(col_wtl_x, yi, wtl_txt, fontsize=9.2, fontweight=fw, ha="left", va="center", color=tc)

    ax.set_yticks(y)
    ax.set_yticklabels([f"#{r['rank']}  {r['label']}" for r in rows_plot], fontsize=9.8)
    for ticklabel, r in zip(ax.get_yticklabels(), rows_plot):
        if r["solver"] in highlight_solvers:
            ticklabel.set_fontweight("bold")
            ticklabel.set_color(highlight_solvers[r["solver"]])

    ax.set_xlabel("RDI Médio (%) — menor é melhor", fontsize=10.5)
    ax.set_xlim(0, max_v + 9.6)
    ax.set_ylim(-0.6, len(rows_plot) + 0.35)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_3_reinforcens_vs_equilibradas_2d(out_dir: Path, suite_insts, by_si, time_si, best_i, worst_i):
    sel_solvers = [
        "reinforcens_pm_queimada",
        "reinforcens_pm",
        "coco_solver",
        "cttp_2opt",
        "reinforcens",
        "reinforcens_queimada",
        "queimada",
        "faulkner_c3",
        "wsm_solver",
        "cosolver",
    ]

    r_insts = sorted(suite_insts["reinforcens"])
    e_insts = sorted(suite_insts["equilibradas"])

    def get_xy(s, sub):
        rdis = [calc_rdi(s, i, by_si, best_i, worst_i) for i in sub if (s, i) in by_si]
        ts = [statistics.mean(time_si[(s, i)]) for i in sub if (s, i) in by_si]
        return statistics.mean(ts), statistics.mean(rdis)

    fig, ax = plt.subplots(figsize=(12.5, 7.2))

    # Offsets manuais para evitar sobreposição de rótulos
    label_offsets = {
        "reinforcens_pm_queimada": (-95, 10),
        "reinforcens_pm": (-10, -18),
        "coco_solver": (8, 6),
        "cttp_2opt": (8, -12),
        "reinforcens": (-85, 8),
        "reinforcens_queimada": (8, 8),
        "queimada": (8, 4),
        "faulkner_c3": (8, 4),
        "wsm_solver": (8, 4),
        "cosolver": (8, 4),
    }

    for s in sel_solvers:
        x1, y1 = get_xy(s, r_insts)
        x2, y2 = get_xy(s, e_insts)
        col = SOLVER_COLORS.get(s, "#333333")
        lbl = SOLVER_LABELS.get(s, s)
        is_q = "queimada" in s

        # Seta / traçado de ReinforceNS -> Equilibradas
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=2.0 if is_q else 1.4,
                ls="--",
                alpha=0.85,
                shrinkA=5,
                shrinkB=6,
            ),
            zorder=3,
        )

        # Ponto em ReinforceNS (círculo vazado)
        ax.scatter(
            x1,
            y1,
            s=95,
            facecolors="white",
            edgecolors=col,
            linewidths=2.0,
            zorder=4,
        )
        # Ponto em Equilibradas (estrela para Queimada, círculo preenchido para demais)
        ax.scatter(
            x2,
            y2,
            s=170 if is_q else 110,
            color=col,
            marker="*" if is_q else "o",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )

        dx, dy = label_offsets.get(s, (8, 4))
        ax.annotate(
            f"{lbl}\n({y1:.2f}% → {y2:.2f}%)",
            (x2, y2),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.3,
            fontweight="bold" if is_q else "normal",
            color="#111111",
        )

    # Legenda explicando os dois símbolos
    ax.scatter([], [], s=95, facecolors="white", edgecolors="#333333", linewidths=2.0, label="ReinforceNS (18 inst.)")
    ax.scatter([], [], s=110, color="#333333", edgecolors="black", linewidths=0.8, label="Equilibradas (34 inst.)")
    ax.plot([], [], color="#333333", ls="--", lw=1.6, label="Deslocamento (ReinforceNS → Equilibradas)")

    ax.set_xscale("log")
    ax.set_xlabel("Tempo Total Médio (s) [escala log]", fontsize=10.5)
    ax.set_ylabel("RDI Médio (%) — menor é melhor", fontsize=10.5)
    ax.set_title("Deslocamento de Desempenho: ReinforceNS vs. Equilibradas", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.55, 75.0)
    ax.set_ylim(-0.3, 10.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9.2, framealpha=0.92)

    plt.tight_layout()
    fig.savefig(out_dir / "03_reinforcens_vs_equilibradas.png", dpi=300)
    plt.close(fig)


def main():
    out_dir = _raiz / "results/analise_kp_vs_tsp/graficos_finais"
    out_dir.mkdir(parents=True, exist_ok=True)
    _, suite_insts, by_si, time_si, insts, _, best_i, worst_i = carregar_dados()

    lit_sem_hotstart = [
        "reinforcens_pm",
        "coco_solver",
        "cttp_2opt",
        "reinforcens",
        "queimada",
        "faulkner_c3",
        "wsm_solver",
        "cosolver",
        "s5_refinada",
        "pack_iterative",
    ]
    _plot_comparativo_geral(
        out_dir / "01_queimada_vs_literatura.png",
        "Queimada vs. Literatura (84 instâncias)",
        lit_sem_hotstart,
        {"queimada": "#d62728"},
        by_si,
        time_si,
        insts,
        best_i,
        worst_i,
    )

    com_hotstart = [
        "reinforcens_pm_queimada",
        "reinforcens_pm",
        "coco_solver",
        "reinforcens",
        "reinforcens_queimada",
        "cttp_2opt",
        "queimada",
        "faulkner_c3",
        "wsm_solver",
        "cosolver",
        "s5_refinada",
        "pack_iterative",
    ]
    _plot_comparativo_geral(
        out_dir / "02_queimada_com_hotstart.png",
        "Queimada com Hot-Start vs. Literatura (84 instâncias)",
        com_hotstart,
        {"reinforcens_pm_queimada": "#d62728", "reinforcens_queimada": "#ff7f0e"},
        by_si,
        time_si,
        insts,
        best_i,
        worst_i,
    )

    plot_3_reinforcens_vs_equilibradas_2d(out_dir, suite_insts, by_si, time_si, best_i, worst_i)

    r_insts = sorted(suite_insts["reinforcens"])
    _plot_comparativo_geral(
        out_dir / "01_queimada_vs_literatura_reinforcens.png",
        "Queimada vs. Literatura (ReinforceNS — 18 instâncias)",
        lit_sem_hotstart,
        {"queimada": "#d62728"},
        by_si,
        time_si,
        r_insts,
        best_i,
        worst_i,
    )
    _plot_comparativo_geral(
        out_dir / "02_queimada_com_hotstart_reinforcens.png",
        "Queimada com Hot-Start vs. Literatura (ReinforceNS — 18 instâncias)",
        com_hotstart,
        {"reinforcens_pm_queimada": "#d62728", "reinforcens_queimada": "#ff7f0e"},
        by_si,
        time_si,
        r_insts,
        best_i,
        worst_i,
    )
    print(f"Graficos finais atualizados em: {out_dir}")


if __name__ == "__main__":
    main()
