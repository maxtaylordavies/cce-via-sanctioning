from collections import Counter
from functools import partial
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
from scipy.stats.qmc import LatinHypercube
import pandas as pd
from tqdm import tqdm

from src.utils import save_fig, jax_key_to_np_rng
from utils import get_p_success_fn, compute_rho_from_w_and_mu

from constants import *
from phase_diagrams import (
    get_delta_functions,
    get_eta_nullcline_curve,
    compute_nullcline_intersection,
    get_optimal_lambda_fn,
)

etas = jnp.linspace(0.0, 1.0, 1000)
INITIAL_D_HI = 100.0
D_EXPANSION_FACTOR = 2.0
MAX_D_EXPANSIONS = 20
N_D_BISECTION_STEPS = 50
N_LAMBDA_BISECTION_STEPS = 15
LAMBDA_SLOPE_PROBE_FRACTION = 0.05
N_INTERSECTION_D_VALS = 1000
INTERSECTION_D_OVERSHOOT = 1.01
relative_Ds = jnp.linspace(
    0.0,
    INTERSECTION_D_OVERSHOOT,
    N_INTERSECTION_D_VALS,
)
outcome_colours = ["xkcd:greenish", "xkcd:coral", "xkcd:periwinkle"]
success_pct_margins = (1.0, 5.0, 10.0)
success_colours = tuple(reversed(sns.light_palette(outcome_colours[0], n_colors=5)[2:]))
failure_mode_cmap = ListedColormap(outcome_colours)
failure_mode_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], failure_mode_cmap.N)
n_samples = 100

param_ranges = {
    "p0": (0.05, 0.95),
    "k": (0.0, 0.1),
    "rho": (1.0, 1.0),
    "mu": (0.0, 0.05),
    "w": (0.0, 1.0),
    "log_beta": (-2.0, 1.0),
    "c_1": (0.0, 1.0),
    "sl_pool_prop": (0.0, 1.0),
    "alpha": (0.0, 0.01),
    "pi_0": (0.0, 0.5),
}

param_dists = {
    name: lambda key, shape, minval=min_val, maxval=max_val: jax.random.uniform(
        key, shape=shape, minval=minval, maxval=maxval
    )
    for name, (min_val, max_val) in param_ranges.items()
}


def _growth_values_over_eta(dD_fn, D):
    return jax.vmap(lambda eta: dD_fn(D, eta))(etas)


def find_ceiling_adaptive(dD_fn):
    """Find the largest feasible D, assuming max growth is non-increasing in D."""

    def max_growth(D):
        return jnp.max(_growth_values_over_eta(dD_fn, D))

    growth_at_zero = max_growth(jnp.asarray(0.0))

    def expansion_step(_, state):
        D_lo, D_hi, searching = state
        growth_at_hi = max_growth(D_hi)
        should_expand = searching & jnp.isfinite(growth_at_hi) & (growth_at_hi >= 0.0)
        return (
            jnp.where(should_expand, D_hi, D_lo),
            jnp.where(should_expand, D_hi * D_EXPANSION_FACTOR, D_hi),
            should_expand,
        )

    D_lo, D_hi, _ = jax.lax.fori_loop(
        0,
        MAX_D_EXPANSIONS,
        expansion_step,
        (
            jnp.asarray(0.0),
            jnp.asarray(INITIAL_D_HI),
            jnp.asarray(True),
        ),
    )

    growth_at_hi = max_growth(D_hi)
    bracket_resolved = (
        jnp.isfinite(growth_at_zero)
        & (growth_at_zero > 0.0)
        & jnp.isfinite(growth_at_hi)
        & (growth_at_hi < 0.0)
    )

    def bisection_step(_, state):
        D_lo, D_hi, values_finite = state
        D_mid = 0.5 * (D_lo + D_hi)
        growth_at_mid = max_growth(D_mid)
        mid_is_finite = jnp.isfinite(growth_at_mid)
        update_bracket = bracket_resolved & values_finite & mid_is_finite
        mid_is_feasible = growth_at_mid >= 0.0
        return (
            jnp.where(update_bracket & mid_is_feasible, D_mid, D_lo),
            jnp.where(update_bracket & ~mid_is_feasible, D_mid, D_hi),
            values_finite & mid_is_finite,
        )

    D_lo, _, bisection_values_finite = jax.lax.fori_loop(
        0,
        N_D_BISECTION_STEPS,
        bisection_step,
        (D_lo, D_hi, jnp.asarray(True)),
    )

    # Use the feasible side of the bracket, consistent with the supremum definition.
    candidate_D_max = D_lo
    safe_D_max = jnp.where(bracket_resolved, candidate_D_max, 1.0)
    growth_values = _growth_values_over_eta(dD_fn, safe_D_max)
    candidate_eta_c = etas[jnp.argmax(growth_values)]

    ceiling_resolved = (
        bracket_resolved
        & bisection_values_finite
        & jnp.all(jnp.isfinite(growth_values))
        & (candidate_D_max > jnp.finfo(candidate_D_max.dtype).eps)
    )
    nan = jnp.asarray(jnp.nan, dtype=candidate_D_max.dtype)
    return (
        jnp.where(ceiling_resolved, candidate_D_max, nan),
        jnp.where(ceiling_resolved, candidate_eta_c, nan),
        ceiling_resolved,
    )


def classify_equilibrium(dD_fn, deta_fn, D_max, eta_c, ceiling_resolved):
    D_max = jnp.asarray(D_max)
    eta_c = jnp.asarray(eta_c)
    ceiling_resolved = jnp.asarray(ceiling_resolved)
    ceiling_valid = (
        ceiling_resolved
        & jnp.isfinite(D_max)
        & jnp.isfinite(eta_c)
        & (D_max > jnp.finfo(D_max.dtype).eps)
    )
    safe_D_max = jnp.where(ceiling_valid, D_max, 1.0)
    safe_eta_c = jnp.where(ceiling_valid, eta_c, 0.5)
    system_Ds = relative_Ds * safe_D_max

    eta_nullcline_vals = get_eta_nullcline_curve(deta_fn, system_Ds)
    success, D, eta, _ = compute_nullcline_intersection(
        system_Ds, eta_nullcline_vals, dD_fn
    )
    D_margin = safe_D_max * (success_pct_margins[-1] / 100)

    # 0 = no intersection, 1 = near ceiling, 2/3 = premature equilibrium,
    # 4 = unresolved or degenerate cultural ceiling.
    label = jnp.where(
        ~success,
        0,
        jnp.where(
            jnp.abs(D - safe_D_max) <= D_margin,
            1,
            jnp.where(eta < safe_eta_c, 2, 3),
        ),
    )
    label = jnp.where(ceiling_valid, label, 4)

    # Retain overshoot so success at each margin can use the same absolute-distance
    # criterion as the label calculation above.
    fraction = jnp.where(success & ceiling_valid, D / safe_D_max, jnp.nan)

    return label, fraction


@partial(jax.jit, static_argnames=("use_adaptive_lambda",))
def _get_single_point_label(
    fixed_params,
    sampled_params,
    N=N,
    fixed_lambda=0.0,
    use_adaptive_lambda=False,
):
    params = {**sampled_params, **fixed_params}

    if "w" in params:
        w, mu = params["w"], params["mu"]
        rho = compute_rho_from_w_and_mu(w, mu)
        params = {**params, "rho": rho}

    p_success_fn = get_p_success_fn(
        params["p0"],
        params["k"],
    )

    c_innov_fn = lambda D: params["c_1"] * D
    c_imit_fn = lambda D: 0.0

    beta = 10 ** params["log_beta"]
    pi0_value = params.get("pi_0", pi_0)

    if use_adaptive_lambda:
        lambda_fn = get_optimal_lambda_fn(
            p_success_fn=p_success_fn,
            c_innov_fn=c_innov_fn,
            c_imit_fn=c_imit_fn,
            value_fn=baseline_value_fn,
            rho=params["rho"],
            mu=params["mu"],
            beta=beta,
            alpha=params["alpha"],
            N=N,
            sl_pool_prop=params["sl_pool_prop"],
            pi0=pi0_value,
        )
    else:
        lambda_fn = lambda D: fixed_lambda

    dD_fn, deta_fn, _ = get_delta_functions(
        p_success_fn,
        c_innov_fn=c_innov_fn,
        c_imit_fn=c_imit_fn,
        lambda_fn=lambda_fn,
        rho=params["rho"],
        mu=params["mu"],
        beta=beta,
        N=N,
        sl_pool_prop=params["sl_pool_prop"],
        alpha=params["alpha"],
        pi0=pi0_value,
    )

    D_max, eta_c, ceiling_resolved = find_ceiling_adaptive(dD_fn)
    return classify_equilibrium(dD_fn, deta_fn, D_max, eta_c, ceiling_resolved)


def sample_systems_latin_hypercube(
    key, n_samples, include_names=None, exclude_names=None
):
    rng = jax_key_to_np_rng(key)

    if include_names is None:
        include_names = list(param_ranges.keys())

    raw_samples = LatinHypercube(d=len(include_names), rng=rng).random(n=n_samples).T
    samples = {
        name: param_ranges[name][0]
        + (
            (param_ranges[name][1] - param_ranges[name][0])
            * jnp.asarray(raw_samples[i])
        )
        for i, name in enumerate(include_names)
    }

    if exclude_names:
        for name in exclude_names:
            del samples[name]

    return samples


def compute_system_outcomes(systems, N=N, fixed_lambda=0.0, use_adaptive_lambda=False):
    labels, scores = jax.vmap(
        _get_single_point_label, in_axes=(None, 0, None, None, None)
    )(dict(), systems, N, fixed_lambda, use_adaptive_lambda)
    counts = Counter(labels.tolist())
    counts = {label: counts.get(label, 0) / labels.size for label in range(5)}
    scores = jnp.where(jnp.isfinite(scores), scores, jnp.nan)
    for margin in success_pct_margins:
        within_margin = jnp.isfinite(scores) & (jnp.abs(scores - 1.0) <= margin / 100)
        counts[f"success_{margin:g}"] = float(jnp.mean(within_margin))
    return counts, jnp.nanmean(scores)
    # return counts, counts[1]


def find_optimal_fixed_lambda_for_systems(systems, N=N):
    """Find the fixed lambda that maximises the mean system outcome score.

    This uses slope bisection under the assumption that the score is unimodal in
    lambda. The endpoints are checked explicitly because the optimum may be 0 or 1.
    """
    evaluations = {}

    def evaluate(lambda_val):
        lambda_val = float(lambda_val)
        if lambda_val not in evaluations:
            label_proportions, score = compute_system_outcomes(
                systems,
                N=N,
                fixed_lambda=jnp.asarray(lambda_val),
            )
            scalar_score = float(jax.device_get(score))
            comparison_score = (
                scalar_score if math.isfinite(scalar_score) else -math.inf
            )
            evaluations[lambda_val] = (
                label_proportions,
                score,
                comparison_score,
            )
        return evaluations[lambda_val]

    lambda_lo, lambda_hi = 0.0, 1.0
    for _ in range(N_LAMBDA_BISECTION_STEPS):
        lambda_mid = 0.5 * (lambda_lo + lambda_hi)
        probe_radius = LAMBDA_SLOPE_PROBE_FRACTION * (lambda_hi - lambda_lo)
        lambda_left = max(lambda_lo, lambda_mid - probe_radius)
        lambda_right = min(lambda_hi, lambda_mid + probe_radius)
        left_score = evaluate(lambda_left)[2]
        right_score = evaluate(lambda_right)[2]

        if left_score < right_score:
            lambda_lo = lambda_mid
        elif left_score > right_score:
            lambda_hi = lambda_mid
        else:
            # Preserve the midpoint when finite precision makes the local slope flat.
            lambda_lo, lambda_hi = lambda_left, lambda_right

    lambda_mid = 0.5 * (lambda_lo + lambda_hi)
    candidates = (lambda_mid, lambda_lo, lambda_hi, 0.0, 1.0)
    optimal_lambda = max(candidates, key=lambda value: evaluate(value)[2])
    label_proportions, score, _ = evaluate(optimal_lambda)
    return optimal_lambda, label_proportions, score


def do_stacked_barplot(summary_df):
    success_keys = tuple(f"success_{margin:g}" for margin in success_pct_margins)
    outcome_styles = (
        (success_keys[0], "Within 1% of ceiling", success_colours[0]),
        (success_keys[1], "Within 5% of ceiling", success_colours[1]),
        (success_keys[2], "Within 10% of ceiling", success_colours[2]),
        (2, "Underinnovation", outcome_colours[1]),
        (3, "Overinnovation", outcome_colours[2]),
        (0, "No equilibrium found", "grey"),
        (4, "Cultural ceiling unresolved", "purple"),
    )

    required_columns = {"N", "run_type", "lambda"} | {
        f"prop_{label}" for label, _, _ in outcome_styles
    }
    missing_columns = required_columns.difference(summary_df.columns)
    if missing_columns:
        raise ValueError(
            "Summary data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if summary_df.duplicated(["N", "run_type"]).any():
        raise ValueError("Summary data must contain one row per (N, run_type) pair.")

    population_sizes = tuple(pd.unique(summary_df["N"]))
    run_types = tuple(pd.unique(summary_df["run_type"]))
    indexed_summary = summary_df.set_index(["N", "run_type"])
    combinations = [
        (pop_size, run_type) for pop_size in population_sizes for run_type in run_types
    ]
    missing_combinations = [
        combination
        for combination in combinations
        if combination not in indexed_summary.index
    ]
    if missing_combinations:
        raise ValueError(
            "Summary data does not contain every (N, run_type) combination: "
            f"{missing_combinations}"
        )

    bar_width = 1.0
    inner_gap = 0.15 * bar_width
    group_gap = 0.5 * bar_width
    inner_spacing = bar_width + inner_gap
    group_width = len(run_types) * bar_width + (len(run_types) - 1) * inner_gap
    group_stride = group_width + group_gap
    group_starts = [
        population_idx * group_stride for population_idx in range(len(population_sizes))
    ]
    group_centres = [group_start + group_width / 2 for group_start in group_starts]
    x_positions = [
        group_starts[population_idx] + bar_width / 2 + run_type_idx * inner_spacing
        for population_idx in range(len(population_sizes))
        for run_type_idx in range(len(run_types))
    ]
    bottoms = [0.0] * len(combinations)

    fig, ax = plt.subplots(figsize=(13, 5))
    for label, outcome_name, colour in outcome_styles:
        proportions = [
            indexed_summary.loc[combination, f"prop_{label}"]
            for combination in combinations
        ]
        if label in success_keys[1:]:
            previous_key = success_keys[success_keys.index(label) - 1]
            proportions = [
                max(
                    proportion
                    - indexed_summary.loc[combination, f"prop_{previous_key}"],
                    0.0,
                )
                for proportion, combination in zip(proportions, combinations)
            ]
        ax.bar(
            x_positions,
            proportions,
            bottom=bottoms,
            width=bar_width,
            color=colour,
            label=outcome_name,
        )
        bottoms = [
            bottom + proportion for bottom, proportion in zip(bottoms, proportions)
        ]

    ax.set(
        xticks=x_positions,
        xlim=(-0.25 * bar_width, group_starts[-1] + group_width + 0.25 * bar_width),
        yticks=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        ylim=(0.0, 1.08),
    )
    inner_tick_labels = []
    for combination in combinations:
        run_type = combination[1]
        lambda_val = indexed_summary.loc[combination, "lambda"]
        if run_type == "baseline":
            label = "None\n[$\\lambda=0$]"
        elif run_type == "best_fixed":
            label = f"Best fixed\n[$\\lambda={lambda_val:.3f}$]"
        elif run_type == "adaptive":
            label = "Adaptive\n[$\\lambda=\\lambda^*(D)$]"
        else:
            label = str(run_type).replace("_", " ").title()
        inner_tick_labels.append(label)
    ax.set_xticklabels(inner_tick_labels)
    ax.set_xlabel("Value-capture norm", labelpad=8, fontsize=13)
    ax.set_ylabel("Proportion of systems", labelpad=8, fontsize=13)
    ax.tick_params(axis="both", which="both", length=0)
    ax.tick_params(axis="x", pad=5)
    ax.set_axisbelow(True)
    # ax.grid(axis="y", color="#CCCCCC", linewidth=2.0)
    sns.despine(ax=ax, left=True, bottom=True)

    for i, (group_centre, pop_size) in enumerate(zip(group_centres, population_sizes)):
        text = f"$M={pop_size}$"
        if i == 0:
            text = f"Population size {text}"
        ax.text(
            group_centre,
            1.02,
            text,
            ha="center",
            va="bottom",
            fontsize=15,
        )

    return fig


def main():
    key = jax.random.key(0)
    n_systems = int(1e3)
    population_sizes = (10, 100, 1000)
    data = []

    systems = sample_systems_latin_hypercube(key, n_systems, exclude_names=("w",))
    for pop_size in tqdm(population_sizes):
        baseline_props, _ = compute_system_outcomes(systems, N=pop_size)
        best_fixed_lambda, best_fixed_props, best_fixed_score = (
            find_optimal_fixed_lambda_for_systems(systems, N=pop_size)
        )
        adaptive_props, _ = compute_system_outcomes(
            systems, N=pop_size, use_adaptive_lambda=True
        )
        for row in (
            ("baseline", 0.0, baseline_props),
            ("best_fixed", best_fixed_lambda, best_fixed_props),
            ("adaptive", -1.0, adaptive_props),
        ):
            label, lambda_val, label_props = row
            data.append(
                {
                    "N": pop_size,
                    "run_type": label,
                    "lambda": lambda_val,
                    **{f"prop_{k}": v for k, v in label_props.items()},
                }
            )

    df = pd.DataFrame(data)

    fig = do_stacked_barplot(df)
    fig.suptitle(
        f"Effect of value-capture norms on the distribution of outcomes over parameter space (from {n_systems} sampled systems)",
        fontsize=15,
    )
    save_fig(fig, "global_summary_by_N_and_lambda", subfolder="theoretical/scratch")


if __name__ == "__main__":
    main()
