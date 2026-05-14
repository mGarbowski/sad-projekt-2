from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SEED = 42
N_SIMULATIONS = 10_000
N_MOMENT_SAMPLES = 100_000
N_CONVERGENCE_SAMPLES = 100_000
K = 50
N_VALUES = [2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 200, 500]
N_CONVERGENCE_VALUES = [2, 10, 50, 200]
ALPHA = 0.05
OUTPUT_DIR = Path(__file__).parent / "figures"
TABLES_DIR = Path(__file__).parent / "tables"

DIST = stats.expon(scale=1)
DIST_LABEL = r"$\mathrm{Exp}(1)$"
MU = float(DIST.mean())
SIGMA = float(DIST.std())

SKEW_BASE = 2.0
EXCESS_KURT_BASE = 6.0

TEST_LABELS = {
    "ks": "Kołmogorow-Smirnow",
    "sw": "Shapiro-Wilk",
    "ag": "D'Agostino",
}

TEST_COLORS = {
    "ks": "steelblue",
    "sw": "darkorange",
    "ag": "seagreen",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})


@dataclass
class PowerResults:
    n_values: list[int]
    powers: dict[str, list[float]]


@dataclass
class MomentResults:
    n_values: list[int]
    empirical_skewness: list[float]
    empirical_kurtosis: list[float]
    theoretical_skewness: list[float]
    theoretical_kurtosis: list[float]


def run_power_simulation(
    rng: np.random.Generator,
    n_values: list[int],
    n_simulations: int,
    h0_true: bool = False,
) -> PowerResults:
    powers: dict[str, list[float]] = {"ks": [], "sw": [], "ag": []}

    for n in n_values:
        sigma_xbar = SIGMA / np.sqrt(n)
        ks_rejects = sw_rejects = ag_rejects = 0

        for _ in range(n_simulations):
            if h0_true:
                means = rng.standard_normal(K) * sigma_xbar + MU
            else:
                samples = DIST.rvs(size=(K, n), random_state=rng)
                means = samples.mean(axis=1)

            standardized = (means - MU) / sigma_xbar

            if stats.kstest(standardized, "norm").pvalue < ALPHA:
                ks_rejects += 1
            if stats.shapiro(means).pvalue < ALPHA:
                sw_rejects += 1
            if stats.normaltest(means).pvalue < ALPHA:
                ag_rejects += 1

        powers["ks"].append(ks_rejects / n_simulations)
        powers["sw"].append(sw_rejects / n_simulations)
        powers["ag"].append(ag_rejects / n_simulations)

    return PowerResults(n_values=n_values, powers=powers)


def estimate_moments(
    rng: np.random.Generator,
    n_values: list[int],
    n_samples: int,
) -> MomentResults:
    emp_skew: list[float] = []
    emp_kurt: list[float] = []
    theo_skew: list[float] = []
    theo_kurt: list[float] = []

    for n in n_values:
        samples = DIST.rvs(size=(n_samples, n), random_state=rng)
        means = samples.mean(axis=1)
        emp_skew.append(float(stats.skew(means)))
        emp_kurt.append(float(stats.kurtosis(means)))
        theo_skew.append(SKEW_BASE / np.sqrt(n))
        theo_kurt.append(EXCESS_KURT_BASE / n)

    return MomentResults(
        n_values=n_values,
        empirical_skewness=emp_skew,
        empirical_kurtosis=emp_kurt,
        theoretical_skewness=theo_skew,
        theoretical_kurtosis=theo_kurt,
    )


def plot_power_curves(results: PowerResults, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for test_key, label in TEST_LABELS.items():
        ax.plot(
            results.n_values,
            results.powers[test_key],
            "o-",
            color=TEST_COLORS[test_key],
            linewidth=2,
            markersize=7,
            label=label,
        )

    ax.axhline(ALPHA, color="crimson", linestyle="--", linewidth=1.5, alpha=0.7,
               label=rf"poziom istotności $\alpha = {ALPHA}$")
    ax.set_xscale("log")
    ax.set_xlabel("Liczba uśrednianych zmiennych n")
    ax.set_ylabel(r"Moc testu (frakcja odrzuceń $H_0$)")
    ax.set_title(
        f"Moc testów normalności dla $\\bar{{X}}_n$, rozkład bazowy {DIST_LABEL}\n"
        f"K = {K}, liczba symulacji = {N_SIMULATIONS:,}".replace(",", " ")
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_type_one_error(results: PowerResults, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for test_key, label in TEST_LABELS.items():
        ax.plot(
            results.n_values,
            results.powers[test_key],
            "o-",
            color=TEST_COLORS[test_key],
            linewidth=2,
            markersize=7,
            label=label,
        )

    ax.axhline(ALPHA, color="crimson", linestyle="--", linewidth=1.5,
               label=rf"nominalny $\alpha = {ALPHA}$")
    ax.set_xscale("log")
    ax.set_xlabel("Liczba uśrednianych zmiennych n")
    ax.set_ylabel(r"Empiryczny błąd I rodzaju")
    ax.set_title(
        f"Weryfikacja błędu I rodzaju (dane z $N(0,1)$)\n"
        f"K = {K}, liczba symulacji = {N_SIMULATIONS:,}".replace(",", " ")
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.1, ALPHA * 2.5))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_moments(results: MomentResults, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(results.n_values, results.empirical_skewness, "o-",
            color="steelblue", linewidth=2, markersize=6, label="Empiryczna")
    ax.plot(results.n_values, results.theoretical_skewness, "--",
            color="crimson", linewidth=2, label=r"Teoretyczna $2/\sqrt{n}$")
    ax.set_xscale("log")
    ax.set_xlabel("Liczba uśrednianych zmiennych n")
    ax.set_ylabel("Skośność")
    ax.set_title(r"Skośność $\bar{X}_n$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    ax = axes[1]
    ax.plot(results.n_values, results.empirical_kurtosis, "o-",
            color="darkorange", linewidth=2, markersize=6, label="Empiryczna")
    ax.plot(results.n_values, results.theoretical_kurtosis, "--",
            color="crimson", linewidth=2, label=r"Teoretyczna $6/n$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Liczba uśrednianych zmiennych n")
    ax.set_ylabel("Kurtoza nadmiarowa")
    ax.set_title(r"Kurtoza nadmiarowa $\bar{X}_n$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"Estymacja skośności i kurtozy nadmiarowej dla rozkładu bazowego {DIST_LABEL}\n"
        f"Liczba prób na n: {N_MOMENT_SAMPLES:,}".replace(",", " "),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_convergence(
    rng: np.random.Generator,
    n_values: list[int],
    n_samples: int,
    filename: str,
) -> None:
    n_cols = len(n_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    grid = np.linspace(-4, 4, 400)
    pdf_ref = stats.norm.pdf(grid)

    for col, n in enumerate(n_values):
        samples = DIST.rvs(size=(n_samples, n), random_state=rng)
        means = samples.mean(axis=1)
        standardized = (means - MU) / (SIGMA / np.sqrt(n))

        ax_hist = axes[0, col]
        ax_hist.hist(
            standardized,
            bins=80,
            density=True,
            color="steelblue",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.3,
        )
        ax_hist.plot(grid, pdf_ref, color="crimson", linewidth=2, label=r"$N(0,1)$")
        ax_hist.set_xlim(-4, 4)
        ax_hist.set_title(rf"$n = {n}$")
        ax_hist.set_xlabel(r"$z = (\bar{X}_n - \mu)/(\sigma/\sqrt{n})$")
        if col == 0:
            ax_hist.set_ylabel("Gęstość")
        ax_hist.legend(fontsize=9, loc="upper right")
        ax_hist.grid(True, alpha=0.3)

        ax_qq = axes[1, col]
        stats.probplot(standardized, dist="norm", plot=ax_qq)
        ax_qq.set_title("")
        ax_qq.set_xlabel("Kwantyle teoretyczne")
        if col == 0:
            ax_qq.set_ylabel("Kwantyle empiryczne")
        else:
            ax_qq.set_ylabel("")
        ax_qq.get_lines()[0].set_markersize(3)
        ax_qq.get_lines()[0].set_alpha(0.4)
        ax_qq.get_lines()[1].set_color("crimson")
        ax_qq.grid(True, alpha=0.3)

    fig.suptitle(
        f"Zbieżność rozkładu $\\bar{{X}}_n$ do $N(0,1)$ dla rozkładu bazowego {DIST_LABEL}\n"
        f"Liczba prób na n: {n_samples:,}".replace(",", " "),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_power_zoom(results: PowerResults, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for test_key, label in TEST_LABELS.items():
        ax.plot(
            results.n_values,
            results.powers[test_key],
            "o-",
            color=TEST_COLORS[test_key],
            linewidth=2,
            markersize=7,
            label=label,
        )

    ax.axhline(ALPHA, color="crimson", linestyle="--", linewidth=1.5, alpha=0.7,
               label=rf"$\alpha = {ALPHA}$")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label="moc = 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("Liczba uśrednianych zmiennych n")
    ax.set_ylabel(r"Moc testu")
    ax.set_title(
        f"Strefa przejścia: moc testów normalności (zbliżenie)\n"
        f"K = {K}, rozkład bazowy {DIST_LABEL}"
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def save_power_table_tex(
    power: PowerResults,
    type_one: PowerResults,
    filename: str,
) -> None:
    lines = [
        r"\begin{tabular}{r|rrr|rrr}",
        r"\toprule",
        r" & \multicolumn{3}{c|}{Moc (dane z Exp(1))} & \multicolumn{3}{c}{Empiryczny $\alpha$ (dane z $N(0,1)$)} \\",
        r"$n$ & KS & SW & D'A & KS & SW & D'A \\",
        r"\midrule",
    ]
    for i, n in enumerate(power.n_values):
        row = (
            f"{n} & "
            f"{power.powers['ks'][i]:.3f} & "
            f"{power.powers['sw'][i]:.3f} & "
            f"{power.powers['ag'][i]:.3f} & "
            f"{type_one.powers['ks'][i]:.3f} & "
            f"{type_one.powers['sw'][i]:.3f} & "
            f"{type_one.powers['ag'][i]:.3f} \\\\"
        )
        lines.append(row)
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_moments_table_tex(results: MomentResults, filename: str) -> None:
    lines = [
        r"\begin{tabular}{r|rr|rr}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{Skośność} & \multicolumn{2}{c}{Kurtoza nadmiarowa} \\",
        r"$n$ & empiryczna & teoretyczna $2/\sqrt{n}$ & empiryczna & teoretyczna $6/n$ \\",
        r"\midrule",
    ]
    for i, n in enumerate(results.n_values):
        row = (
            f"{n} & "
            f"{results.empirical_skewness[i]:.4f} & "
            f"{results.theoretical_skewness[i]:.4f} & "
            f"{results.empirical_kurtosis[i]:.4f} & "
            f"{results.theoretical_kurtosis[i]:.4f} \\\\"
        )
        lines.append(row)
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_power_table(results: PowerResults, h0_true_results: PowerResults) -> None:
    print(f"\n{'='*78}")
    print(f"Moc testów normalności (P(odrzucenie H0 | H0 fałszywe)), α = {ALPHA}")
    print(f"Rozkład bazowy: Exp(1), K = {K}, symulacji = {N_SIMULATIONS}")
    print(f"{'='*78}")
    print(f"{'n':>5} | {'KS':>8} | {'Shapiro-Wilk':>14} | {'D Agostino':>12}")
    print("-" * 78)
    for i, n in enumerate(results.n_values):
        ks = results.powers["ks"][i]
        sw = results.powers["sw"][i]
        ag = results.powers["ag"][i]
        print(f"{n:>5} | {ks:>8.4f} | {sw:>14.4f} | {ag:>12.4f}")

    print(f"\n{'='*78}")
    print("Empiryczny błąd I rodzaju (dane z N(0,1) jako weryfikacja)")
    print(f"{'='*78}")
    print(f"{'n':>5} | {'KS':>8} | {'Shapiro-Wilk':>14} | {'D Agostino':>12}")
    print("-" * 78)
    for i, n in enumerate(h0_true_results.n_values):
        ks = h0_true_results.powers["ks"][i]
        sw = h0_true_results.powers["sw"][i]
        ag = h0_true_results.powers["ag"][i]
        print(f"{n:>5} | {ks:>8.4f} | {sw:>14.4f} | {ag:>12.4f}")


def print_moments_table(results: MomentResults) -> None:
    print(f"\n{'='*78}")
    print(f"Skośność i kurtoza nadmiarowa $\\bar X_n$ (rozkład bazowy Exp(1))")
    print(f"Liczba prób na n: {N_MOMENT_SAMPLES}")
    print(f"{'='*78}")
    print(f"{'n':>5} | {'skew emp':>10} | {'skew teor':>10} | {'kurt emp':>10} | {'kurt teor':>10}")
    print("-" * 78)
    for i, n in enumerate(results.n_values):
        print(
            f"{n:>5} | "
            f"{results.empirical_skewness[i]:>10.4f} | "
            f"{results.theoretical_skewness[i]:>10.4f} | "
            f"{results.empirical_kurtosis[i]:>10.4f} | "
            f"{results.theoretical_kurtosis[i]:>10.4f}"
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng_power = np.random.default_rng(SEED)
    rng_h0 = np.random.default_rng(SEED + 1)
    rng_moments = np.random.default_rng(SEED + 2)
    rng_convergence = np.random.default_rng(SEED + 3)

    print("Symulacja mocy testów (H0 fałszywe - dane z Exp(1))...")
    power_results = run_power_simulation(rng_power, N_VALUES, N_SIMULATIONS, h0_true=False)

    print("Symulacja błędu I rodzaju (H0 prawdziwe - dane z N(0,1))...")
    h0_results = run_power_simulation(rng_h0, N_VALUES, N_SIMULATIONS, h0_true=True)

    print("Estymacja skośności i kurtozy...")
    moment_results = estimate_moments(rng_moments, N_VALUES, N_MOMENT_SAMPLES)

    plot_power_curves(power_results, "power_curves.png")
    plot_power_zoom(power_results, "power_zoom.png")
    plot_type_one_error(h0_results, "type_one_error.png")
    plot_moments(moment_results, "moments.png")
    plot_convergence(rng_convergence, N_CONVERGENCE_VALUES, N_CONVERGENCE_SAMPLES, "convergence.png")

    save_power_table_tex(power_results, h0_results, "power.tex")
    save_moments_table_tex(moment_results, "moments.tex")

    print_power_table(power_results, h0_results)
    print_moments_table(moment_results)

    print(f"\nWygenerowano wykresy w {OUTPUT_DIR.resolve()}")
    print(f"Tabele LaTeX w {TABLES_DIR.resolve()}")


if __name__ == "__main__":
    main()
