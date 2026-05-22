from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SEED = 42
SIGMA = 1.0
N_VALUES = [16, 64, 256, 1024, 4096]
ALPHA_VALUES = [1e-2, 1e-3, 1e-4, 1e-6]
ALPHA_REF = 1e-3
N_REF = 256
A_REF = 0.3
A_GRID = np.logspace(-2, 0, 60)
N_DENSE = np.unique(np.round(np.logspace(0.5, 4, 40)).astype(int))
N_SIMULATIONS = 50_000
HIST_SIMULATIONS = 200_000

OUTPUT_DIR = Path(__file__).parent / "figures"
TABLES_DIR = Path(__file__).parent / "tables"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


@dataclass
class EmpiricalResult:
    n: int
    a: float
    sigma: float
    alpha: float
    pd_empirical: float
    pfa_empirical: float
    pd_theoretical: float
    n_simulations: int


def theoretical_power(amplitude: float, sigma: float, n: int, alpha: float) -> float:
    z = stats.norm.ppf(1 - alpha)
    d = amplitude * np.sqrt(n) / sigma
    return float(stats.norm.sf(z - d))


def generate_code(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice([-1.0, 1.0], size=n)


def matched_filter(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    return x @ s


def simulate_detection(
    rng: np.random.Generator,
    s: np.ndarray,
    amplitude: float,
    sigma: float,
    alpha: float,
    n_simulations: int,
) -> EmpiricalResult:
    n = len(s)
    threshold = stats.norm.ppf(1 - alpha) * sigma * np.sqrt(n)

    noise_h0 = rng.normal(0.0, sigma, size=(n_simulations, n))
    t_h0 = noise_h0 @ s
    pfa = float(np.mean(t_h0 > threshold))

    noise_h1 = rng.normal(0.0, sigma, size=(n_simulations, n))
    t_h1 = (amplitude * s + noise_h1) @ s
    pd = float(np.mean(t_h1 > threshold))

    return EmpiricalResult(
        n=n,
        a=amplitude,
        sigma=sigma,
        alpha=alpha,
        pd_empirical=pd,
        pfa_empirical=pfa,
        pd_theoretical=theoretical_power(amplitude, sigma, n, alpha),
        n_simulations=n_simulations,
    )


def plot_test_statistic_distribution(
    rng: np.random.Generator,
    filename: str,
) -> None:
    s = generate_code(rng, N_REF)
    threshold = stats.norm.ppf(1 - ALPHA_REF) * SIGMA * np.sqrt(N_REF)

    noise_h0 = rng.normal(0.0, SIGMA, size=(HIST_SIMULATIONS, N_REF))
    t_h0 = noise_h0 @ s

    noise_h1 = rng.normal(0.0, SIGMA, size=(HIST_SIMULATIONS, N_REF))
    t_h1 = (A_REF * s + noise_h1) @ s

    fig, ax = plt.subplots(figsize=(10, 6))

    std = SIGMA * np.sqrt(N_REF)
    grid = np.linspace(-4 * std, A_REF * N_REF + 4 * std, 600)
    pdf_h0 = stats.norm.pdf(grid, loc=0, scale=std)
    pdf_h1 = stats.norm.pdf(grid, loc=A_REF * N_REF, scale=std)

    ax.hist(t_h0, bins=120, density=True, alpha=0.45, color=PALETTE[0],
            label=r"$T \mid H_0$ (empiryczne)")
    ax.hist(t_h1, bins=120, density=True, alpha=0.45, color=PALETTE[1],
            label=r"$T \mid H_1$ (empiryczne)")
    ax.plot(grid, pdf_h0, color=PALETTE[0], linewidth=2,
            label=rf"$\mathcal{{N}}(0, \sigma^2 N)$")
    ax.plot(grid, pdf_h1, color=PALETTE[1], linewidth=2,
            label=rf"$\mathcal{{N}}(AN, \sigma^2 N)$")
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=2,
               label=rf"próg $\tau$ przy $\alpha = {ALPHA_REF:g}$")

    ax.set_xlabel(r"Statystyka testu $T = \sum_i s_i x_i$")
    ax.set_ylabel("Gęstość")
    ax.set_title(
        f"Rozkład statystyki testu pod $H_0$ i $H_1$\n"
        f"$N = {N_REF}$, $A = {A_REF}$, $\\sigma = {SIGMA}$, "
        f"$\\alpha = {ALPHA_REF:g}$"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_power_vs_amplitude(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for color, n in zip(PALETTE, N_VALUES):
        powers = [theoretical_power(a, SIGMA, n, ALPHA_REF) for a in A_GRID]
        ax.plot(A_GRID, powers, "-", color=color, linewidth=2, label=f"$N = {n}$")

    ax.axhline(ALPHA_REF, color="gray", linestyle=":", linewidth=1,
               label=rf"$\alpha = {ALPHA_REF:g}$")
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="moc = 0.9")
    ax.set_xscale("log")
    ax.set_xlabel(r"Amplituda sygnału $A$ (przy $\sigma = 1$)")
    ax.set_ylabel("Moc testu (prawdopodobieństwo detekcji)")
    ax.set_title(
        f"Moc detekcji w funkcji amplitudy sygnału\n"
        f"$\\sigma = {SIGMA}$, $\\alpha = {ALPHA_REF:g}$"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_power_vs_n(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    a_values = [0.05, 0.1, 0.2, 0.5, 1.0]
    for color, a in zip(PALETTE, a_values):
        powers = [theoretical_power(a, SIGMA, int(n), ALPHA_REF) for n in N_DENSE]
        label = f"$A/\\sigma = {a:.2f}$"
        ax.plot(N_DENSE, powers, "o-", color=color, linewidth=2, markersize=4, label=label)

    ax.axhline(ALPHA_REF, color="gray", linestyle=":", linewidth=1,
               label=rf"$\alpha = {ALPHA_REF:g}$")
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="moc = 0.9")
    ax.set_xscale("log")
    ax.set_xlabel(r"Długość sygnału $N$")
    ax.set_ylabel("Moc testu (prawdopodobieństwo detekcji)")
    ax.set_title(
        f"Moc detekcji w funkcji długości sygnału\n"
        f"$\\sigma = {SIGMA}$, $\\alpha = {ALPHA_REF:g}$"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_power_vs_alpha(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    a_values = [0.1, 0.2, 0.3, 0.5]
    alpha_grid = np.logspace(-8, -1, 80)
    for color, a in zip(PALETTE, a_values):
        powers = [theoretical_power(a, SIGMA, N_REF, al) for al in alpha_grid]
        d = a * np.sqrt(N_REF) / SIGMA
        ax.plot(alpha_grid, powers, "-", color=color, linewidth=2,
                label=f"$A = {a}$, $d = {d:.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"Prawdopodobieństwo fałszywego alarmu $\alpha$")
    ax.set_ylabel("Moc testu")
    ax.set_title(
        f"Wpływ poziomu istotności na moc detekcji\n"
        f"$N = {N_REF}$, $\\sigma = {SIGMA}$, $d = A\\sqrt{{N}}/\\sigma$"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.02, 1.02)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_power_vs_sigma(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    sigma_grid = np.logspace(-1, 1.3, 100)
    cases = [
        (1.0, 64),
        (1.0, 256),
        (0.3, 256),
        (0.3, 1024),
        (0.1, 4096),
    ]

    for color, (a, n) in zip(PALETTE, cases):
        powers = [theoretical_power(a, s, n, ALPHA_REF) for s in sigma_grid]
        ax.plot(sigma_grid, powers, "-", color=color, linewidth=2,
                label=f"$A = {a}$, $N = {n}$")

    ax.axhline(ALPHA_REF, color="gray", linestyle=":", linewidth=1,
               label=rf"$\alpha = {ALPHA_REF:g}$")
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="moc = 0.9")
    ax.set_xscale("log")
    ax.set_xlabel(r"Odchylenie standardowe szumu $\sigma$")
    ax.set_ylabel("Moc testu")
    ax.set_title(
        f"Moc detekcji w funkcji poziomu szumu\n"
        f"$\\alpha = {ALPHA_REF:g}$"
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_processing_gain(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    d_grid = np.linspace(0, 7, 200)

    for color, alpha in zip(PALETTE, ALPHA_VALUES):
        z = stats.norm.ppf(1 - alpha)
        powers = stats.norm.sf(z - d_grid)
        ax.plot(d_grid, powers, "-", color=color, linewidth=2,
                label=rf"$\alpha = {alpha:g}$")

    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Efektywny SNR $d = A\sqrt{N}/\sigma$")
    ax.set_ylabel("Moc testu")
    ax.set_title(
        "Uniwersalna krzywa mocy w funkcji efektywnego SNR\n"
        r"(wszystkie kombinacje $A$, $N$, $\sigma$ dające ten sam $d$ dają tę samą moc)"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_weak_signal(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    weak_amplitudes = [0.01, 0.03, 0.1, 0.3]
    n_grid = np.unique(np.round(np.logspace(0, 5, 60)).astype(int))

    for color, a in zip(PALETTE, weak_amplitudes):
        powers = [theoretical_power(a, SIGMA, int(n), ALPHA_REF) for n in n_grid]
        n_for_half = int(np.ceil(((stats.norm.ppf(1 - ALPHA_REF)) ** 2) * (SIGMA / a) ** 2))
        ax.plot(n_grid, powers, "-", color=color, linewidth=2,
                label=f"$A = {a}\\sigma$ (moc 0.5 przy $N \\approx {n_for_half}$)")

    ax.axhline(0.9, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="moc = 0.9")
    ax.set_xscale("log")
    ax.set_xlabel(r"Długość sygnału $N$")
    ax.set_ylabel("Moc testu")
    ax.set_title(
        f"Detekcja sygnału słabego $A \\ll \\sigma$\n"
        f"$\\sigma = {SIGMA}$, $\\alpha = {ALPHA_REF:g}$"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_roc(filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))

    d_values = [0.5, 1.0, 2.0, 3.0, 4.0]
    pfa_grid = np.logspace(-8, 0, 400)

    for color, d in zip(PALETTE, d_values):
        z = stats.norm.isf(pfa_grid)
        pd = stats.norm.sf(z - d)
        ax.plot(pfa_grid, pd, "-", color=color, linewidth=2, label=f"$d = {d}$")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="losowy detektor")
    ax.set_xscale("log")
    ax.set_xlabel(r"Prawdopodobieństwo fałszywego alarmu $P_{FA}$")
    ax.set_ylabel(r"Prawdopodobieństwo detekcji $P_D$")
    ax.set_title("Krzywe ROC dla różnych efektywnych SNR $d = A\\sqrt{N}/\\sigma$")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e-8, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def run_verification(rng: np.random.Generator) -> list[EmpiricalResult]:
    test_cases = [
        (256, 0.1, 1e-2),
        (256, 0.2, 1e-2),
        (256, 0.3, 1e-3),
        (1024, 0.05, 1e-3),
        (1024, 0.1, 1e-3),
        (4096, 0.03, 1e-3),
        (4096, 0.05, 1e-2),
    ]

    results: list[EmpiricalResult] = []
    for n, a, alpha in test_cases:
        s = generate_code(rng, n)
        result = simulate_detection(rng, s, a, SIGMA, alpha, N_SIMULATIONS)
        results.append(result)
    return results


def save_verification_table(results: list[EmpiricalResult], filename: str) -> None:
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$N$ & $A$ & $\alpha$ & $P_D$ teoria & $P_D$ empir. & $P_{FA}$ empir. \\",
        r"\midrule",
    ]
    for r in results:
        row = (
            f"{r.n} & {r.a:.2f} & {r.alpha:.0e} & "
            f"{r.pd_theoretical:.4f} & {r.pd_empirical:.4f} & {r.pfa_empirical:.4f} \\\\"
        )
        lines.append(row)
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_verification_table(results: list[EmpiricalResult]) -> None:
    print(f"\n{'='*78}")
    print("Weryfikacja Monte Carlo: empiryczna vs teoretyczna moc detekcji")
    print(f"σ = {SIGMA}, liczba symulacji = {N_SIMULATIONS}")
    print(f"{'='*78}")
    print(f"{'N':>6} | {'A':>6} | {'α':>8} | {'Pd teor':>9} | {'Pd emp':>9} | {'Pfa emp':>9}")
    print("-" * 78)
    for r in results:
        print(
            f"{r.n:>6} | {r.a:>6.2f} | {r.alpha:>8.0e} | "
            f"{r.pd_theoretical:>9.4f} | {r.pd_empirical:>9.4f} | {r.pfa_empirical:>9.4f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng_hist = np.random.default_rng(SEED)
    rng_verify = np.random.default_rng(SEED + 1)

    print("Rysowanie rozkładu statystyki testu...")
    plot_test_statistic_distribution(rng_hist, "test_statistic.png")

    print("Rysowanie mocy w funkcji amplitudy...")
    plot_power_vs_amplitude("power_vs_amplitude.png")

    print("Rysowanie mocy w funkcji N...")
    plot_power_vs_n("power_vs_n.png")

    print("Rysowanie wpływu poziomu istotności...")
    plot_power_vs_alpha("power_vs_alpha.png")

    print("Rysowanie mocy w funkcji odchylenia standardowego szumu...")
    plot_power_vs_sigma("power_vs_sigma.png")

    print("Rysowanie krzywej processing gain...")
    plot_processing_gain("processing_gain.png")

    print("Rysowanie scenariusza słabego sygnału...")
    plot_weak_signal("weak_signal.png")

    print("Rysowanie krzywych ROC...")
    plot_roc("roc.png")

    print("Weryfikacja Monte Carlo...")
    results = run_verification(rng_verify)
    print_verification_table(results)
    save_verification_table(results, "verification.tex")

    print(f"\nWygenerowano wykresy w {OUTPUT_DIR.resolve()}")
    print(f"Tabele LaTeX w {TABLES_DIR.resolve()}")


if __name__ == "__main__":
    main()
