from collections import Counter
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
from scipy.stats.qmc import LatinHypercube
import pandas as pd
from tqdm import tqdm

from src.utils import save_fig, jax_key_to_np_rng, colour_midpoint
from utils import get_p_success_fn, compute_rho_from_w_and_mu

from constants import *
from phase_diagrams import (
    get_delta_functions,
    get_eta_nullcline_curve,
    compute_nullcline_intersection,
    get_optimal_lambda_fn,
)

sns.set_theme(style="whitegrid")

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
outcome_colours.append(colour_midpoint(outcome_colours[-2], outcome_colours[-1]))
success_pct_margins = (1.0, 5.0, 10.0)
success_colours = tuple(reversed(sns.light_palette(outcome_colours[0], n_colors=5)[2:]))
failure_mode_cmap = ListedColormap(outcome_colours)
failure_mode_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], failure_mode_cmap.N)
n_samples = 50
n_sweep_vals = 50

param_ranges = {
    "log_M": (1.5, 3.0),
    # "log_M": (2.0, 2.0),
    "p0": (0.0, 1.0),
    "k": (0.0, 0.1),
    "rho": (1.0, 1.0),
    "mu": (0.01, 0.05),
    "w": (0.0, 1.0),
    "log_beta": (-2.0, 1.0),
    "c": (0.0, 1.0),
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


def get_sweep_vals(param_name):
    min_val, max_val = param_ranges[param_name]
    return jnp.linspace(min_val, max_val, n_sweep_vals)


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

    c_innov_fn = lambda D: params["c"] * D
    c_imit_fn = lambda D: 0.0

    M = 10 ** params["log_M"]
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
            N=M,
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
        N=M,
        sl_pool_prop=params["sl_pool_prop"],
        alpha=params["alpha"],
        pi0=pi0_value,
    )

    D_max, eta_c, ceiling_resolved = find_ceiling_adaptive(dD_fn)
    return classify_equilibrium(dD_fn, deta_fn, D_max, eta_c, ceiling_resolved)


def _make_get_sample_label_grid():
    get_point_labels = jax.vmap(_get_single_point_label, in_axes=(None, 0))
    get_row_labels = jax.vmap(get_point_labels, in_axes=(0, None))
    return jax.vmap(get_row_labels, in_axes=(0, None))


_get_sample_label_grid = _make_get_sample_label_grid()


def summarize_outcomes(fractions, labels, threshold=0.9):
    fractions = jnp.where(jnp.isfinite(fractions), fractions, jnp.nan)
    is_success = jnp.nanmean(fractions, axis=-1) >= threshold

    under_counts = jnp.sum(labels == 2, axis=-1)
    over_counts = jnp.sum(labels == 3, axis=-1)

    is_under = under_counts > over_counts
    is_over = over_counts > under_counts

    return jnp.where(is_success, 0, jnp.where(is_under, 1, jnp.where(is_over, 2, 3)))


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


def sweep_k_and_c(key, ks, cs, p0):
    k_grid, c_grid = jnp.meshgrid(ks, cs, indexing="xy")
    fixed_params = {"k": k_grid, "c": c_grid, "p0": jnp.full_like(k_grid, p0)}
    sampled_params = sample_systems_latin_hypercube(
        key, n_samples, exclude_names=("k", "c", "p0", "w")
    )
    labels, fractions = _get_sample_label_grid(fixed_params, sampled_params)
    return summarize_outcomes(fractions, labels)


def summarize_system_outcomes(labels, scores):
    counts = Counter(labels.tolist())
    counts = {label: counts.get(label, 0) / labels.size for label in range(5)}
    scores = jnp.where(jnp.isfinite(scores), scores, jnp.nan)
    for margin in success_pct_margins:
        within_margin = jnp.isfinite(scores) & (jnp.abs(scores - 1.0) <= margin / 100)
        counts[f"success_{margin:g}"] = float(jnp.mean(within_margin))
    return counts, jnp.nanmean(scores)


def compute_system_outcomes(systems, fixed_lambda=0.0, use_adaptive_lambda=False):
    labels, scores = jax.vmap(_get_single_point_label, in_axes=(None, 0, None, None))(
        dict(), systems, fixed_lambda, use_adaptive_lambda
    )
    return summarize_system_outcomes(labels, scores)


def evaluate_fixed_lambdas(systems, fixed_lambdas):
    def evaluate(lambda_val):
        labels, scores = jax.vmap(_get_single_point_label, in_axes=(None, 0, None))(
            dict(), systems, lambda_val
        )
        mean_score = jnp.nanmean(scores)
        prop_success = jnp.mean(labels == 1)
        prop_under = jnp.mean(labels == 2)
        prop_over = jnp.mean(labels == 3)
        return mean_score, prop_success, prop_under, prop_over

    return jax.vmap(evaluate)(fixed_lambdas)


def find_optimal_fixed_lambda_for_systems(systems):
    """Find the fixed lambda that maximises the mean system outcome score.

    This uses slope bisection under the assumption that the score is unimodal in
    lambda. The endpoints are checked explicitly because the optimum may be 0 or 1.
    """
    evaluations = {}

    def evaluate(lambda_val):
        lambda_val = float(lambda_val)
        if lambda_val not in evaluations:
            label_proportions, score = compute_system_outcomes(
                systems, fixed_lambda=jnp.asarray(lambda_val)
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

    lambda_lo, lambda_hi = 0.0, 0.2
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


def _evaluate_per_system_fixed_lambdas(systems, fixed_lambdas):
    return jax.vmap(_get_single_point_label, in_axes=(None, 0, 0, None))(
        dict(), systems, fixed_lambdas, False
    )


@jax.jit
def _find_optimal_fixed_lambdas_per_system(systems):
    """Run independent fixed-lambda searches for a batch of sampled systems."""

    def evaluate(fixed_lambdas):
        return _evaluate_per_system_fixed_lambdas(systems, fixed_lambdas)

    def comparable_scores(scores):
        return jnp.where(jnp.isfinite(scores), scores, -jnp.inf)

    def search_step(_, bounds):
        lambda_lo, lambda_hi = bounds
        lambda_mid = 0.5 * (lambda_lo + lambda_hi)
        probe_radius = LAMBDA_SLOPE_PROBE_FRACTION * (lambda_hi - lambda_lo)
        lambda_left = lambda_mid - probe_radius
        lambda_right = lambda_mid + probe_radius
        _, left_scores = evaluate(lambda_left)
        _, right_scores = evaluate(lambda_right)
        left_scores = comparable_scores(left_scores)
        right_scores = comparable_scores(right_scores)

        slopes_up = left_scores < right_scores
        slopes_down = left_scores > right_scores
        scores_tied = ~(slopes_up | slopes_down)
        return (
            jnp.where(
                slopes_up, lambda_mid, jnp.where(scores_tied, lambda_left, lambda_lo)
            ),
            jnp.where(
                slopes_down, lambda_mid, jnp.where(scores_tied, lambda_right, lambda_hi)
            ),
        )

    lambda_lo = jnp.zeros_like(systems["p0"])
    lambda_hi = 0.1 * jnp.ones_like(systems["p0"])
    lambda_lo, lambda_hi = jax.lax.fori_loop(
        0,
        N_LAMBDA_BISECTION_STEPS,
        search_step,
        (lambda_lo, lambda_hi),
    )

    lambda_mid = 0.5 * (lambda_lo + lambda_hi)
    candidates = jnp.stack(
        (
            lambda_mid,
            lambda_lo,
            lambda_hi,
            jnp.zeros_like(lambda_mid),
            jnp.ones_like(lambda_mid),
        )
    )
    candidate_labels, candidate_scores = jax.vmap(evaluate)(candidates)
    best_indices = jnp.argmax(comparable_scores(candidate_scores), axis=0)
    gather_indices = best_indices[jnp.newaxis, :]

    optimal_lambdas = jnp.take_along_axis(candidates, gather_indices, axis=0)[0]
    optimal_labels = jnp.take_along_axis(candidate_labels, gather_indices, axis=0)[0]
    optimal_scores = jnp.take_along_axis(candidate_scores, gather_indices, axis=0)[0]
    return optimal_lambdas, optimal_labels, optimal_scores


def find_optimal_fixed_lambdas_per_system(systems):
    """Return each system's best fixed lambda and the aggregate outcomes."""
    optimal_lambdas, labels, scores = _find_optimal_fixed_lambdas_per_system(systems)
    label_proportions, mean_score = summarize_system_outcomes(labels, scores)
    return optimal_lambdas, label_proportions, mean_score


def plot_heatmap(
    ax,
    x_vals,
    y_vals,
    values,
    x_log=False,
    y_log=False,
):
    def get_tick_vals_and_labels(vals, log, n_sf=1):
        lo, hi = vals[0], vals[-1]
        ticks = [lo + (hi - lo) * i / 5 for i in range(6)]
        if not log:
            return ticks, [f"{tick:.{n_sf}g}" for tick in ticks]
        return ticks, [f"{(10**tick):.{n_sf}g}" for tick in ticks]

    image_kwargs = {
        "origin": "lower",
        "extent": (x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]),
        "aspect": "auto",
        "cmap": failure_mode_cmap,
        "norm": failure_mode_norm,
    }
    im = ax.imshow(values, **image_kwargs)

    xtick_vals, xtick_labels = get_tick_vals_and_labels(x_vals, x_log)
    ytick_vals, ytick_labels = get_tick_vals_and_labels(y_vals, y_log)
    ax.set(
        xticks=xtick_vals,
        xticklabels=xtick_labels,
        yticks=ytick_vals,
        yticklabels=ytick_labels,
    )

    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0)
    sns.despine(ax=ax, left=True, bottom=True)
    return im


def do_grouped_barplot(summary_df):
    run_types = [
        "baseline",
        "best_fixed",
        "best_per_system_fixed",
        "adaptive",
    ]

    run_type_labels = {
        "baseline": "No value capture: $\\lambda=0$",
        "best_fixed": "Best fixed over all sampled systems: $\\lambda=\\lambda^\\star$",
        "best_per_system_fixed": "Per-system best fixed: $\\lambda=\\lambda_s^\\star$",
        "adaptive": "Per-system adaptive: $\\lambda=\\lambda_s^\\star(D)$",
    }

    # Darkest = closest to the ceiling.
    # This derives three green shades from your existing near-ceiling colour.
    near_ceiling_colours = sns.light_palette(
        outcome_colours[0],
        n_colors=4,
        reverse=True,
    )[:3]

    fig, axs = plt.subplots(
        1, len(run_types), figsize=(4 * len(run_types), 5), sharey=True
    )
    plt.subplots_adjust(wspace=0.25)

    x_positions = np.arange(3)
    bar_width = 1.0

    for ax, run_type in zip(axs, run_types):
        df_subset = summary_df.loc[summary_df["run_type"] == run_type]

        if len(df_subset) != 1:
            raise ValueError(
                f"Expected one row for run_type={run_type!r}, "
                f"but found {len(df_subset)}. "
                "Filter summary_df to one population size first."
            )

        row = df_subset.iloc[0]
        near_ceiling_segments = [
            float(row["prop_success_1"]),
            float(row["prop_success_5"]) - float(row["prop_success_1"]),
            float(row["prop_success_10"]) - float(row["prop_success_5"]),
        ]

        # Draw the stacked near-ceiling bar.
        bottom = 0.0
        for height, colour in zip(near_ceiling_segments, near_ceiling_colours):
            ax.bar(
                x_positions[0],
                height,
                bottom=bottom,
                width=bar_width,
                color=colour,
                edgecolor="none",
            )
            bottom += height

        # Draw ordinary underinnovation and overinnovation bars.
        ax.bar(
            x_positions[1],
            row["prop_2"],
            width=bar_width,
            color=outcome_colours[1],
            edgecolor="none",
        )
        ax.bar(
            x_positions[2],
            row["prop_3"],
            width=bar_width,
            color=outcome_colours[2],
            edgecolor="none",
        )

        ax.set(
            title=run_type_labels.get(run_type, run_type),
            xticks=[],
            xticklabels=[],
            ylim=(0, 1.05),
        )
        sns.despine(ax=ax, left=True, bottom=True)

    axs[0].set_ylabel("Proportion of sampled systems")
    return fig


def main():
    key = jax.random.key(0)

    # SWEEP OVER K AND C FOR DIFFERENT P0 VALUES
    p0_vals = (0.1, 0.5, 1.0)
    fig, axs = plt.subplots(
        1, len(p0_vals) + 1, figsize=(4 * (len(p0_vals) + 1), 4.25), sharey=True
    )

    ks, cs = get_sweep_vals("k"), get_sweep_vals("c")
    for i, p0 in tqdm(
        enumerate(p0_vals), total=len(p0_vals), desc="Sweeping over k and c"
    ):
        labels = jax.block_until_ready(sweep_k_and_c(key, ks, cs, p0))
        plot_heatmap(axs[i], ks, cs, labels)
        axs[i].set(
            title=f"$p_0={p0:.2f}$", xlabel="$k$", ylabel="$c$" if i == 0 else None
        )

    fig.suptitle(
        "B. Classifying equilibria under combinations of innovation difficulty $p_\\text{success}(D) = p_0e^{-kD}$ and cost disparity $c_\\text{innov}(D) - c_\\text{imit}(D) = cD$",
        x=0.04,
        horizontalalignment="left",
    )
    save_fig(fig, "sweep_k_and_c", subfolder="new")

    # # GLOBAL PARAMETER-SPACE ANALYSIS
    # n_systems = int(1e3)
    # systems = sample_systems_latin_hypercube(key, n_systems, exclude_names=("w",))

    # baseline_props, _ = compute_system_outcomes(systems)
    # best_fixed_lambda, best_fixed_props, _ = find_optimal_fixed_lambda_for_systems(
    #     systems
    # )
    # optimal_lambdas, per_system_fixed_props, _ = find_optimal_fixed_lambdas_per_system(
    #     systems
    # )
    # adaptive_props, _ = compute_system_outcomes(systems, use_adaptive_lambda=True)

    # data = []
    # for label, lambda_val, label_props in (
    #     ("baseline", 0.0, baseline_props),
    #     ("best_fixed", best_fixed_lambda, best_fixed_props),
    #     ("best_per_system_fixed", float("nan"), per_system_fixed_props),
    #     ("adaptive", float("nan"), adaptive_props),
    # ):
    #     data.append(
    #         {
    #             "run_type": label,
    #             "lambda": lambda_val,
    #             **{f"prop_{k}": v for k, v in label_props.items()},
    #         }
    #     )

    # fig = do_grouped_barplot(pd.DataFrame(data))
    # fig.suptitle(
    #     f"A. Effect of value-capture norms on outcomes over parameter space (from {n_systems} sampled systems)",
    #     x=0.125,
    #     horizontalalignment="left",
    # )
    # save_fig(
    #     fig,
    #     "global_summary_by_value_capture_norm",
    #     subfolder="theoretical/scratch",
    #     tight=False,
    # )

    # # PLOTS TO SHOW INVERSE-U RELATIONSHIP BETWEEN FIXED LAMBDA AND SUCCESS RATE
    # fig, axs = plt.subplots(1, 2, figsize=(13.25, 4.5), sharey=False)

    # all_lambdas, all_success, all_under, all_over = [], [], [], []
    # for i in tqdm(range(10)):
    #     fixed_lambdas = jnp.linspace(i / 10, (i + 1) / 10, 100)
    #     _, success, under, over = jax.block_until_ready(
    #         evaluate_fixed_lambdas(systems, fixed_lambdas)
    #     )
    #     all_lambdas.append(fixed_lambdas)
    #     all_success.append(success)
    #     all_under.append(under)
    #     all_over.append(over)

    # all_lambdas = jnp.concatenate(all_lambdas)
    # all_success = jnp.concatenate(all_success)
    # all_under = jnp.concatenate(all_under)
    # all_over = jnp.concatenate(all_over)

    # for y, colour in zip((all_success, all_under, all_over), outcome_colours):
    #     sns.lineplot(x=all_lambdas, y=y, ax=axs[0], color=colour, linewidth=2.5)
    # axs[0].set(
    #     xlabel=r"Fixed value capture $\lambda$",
    #     ylabel="Proportion of sampled systems",
    # )

    # sns.histplot(
    #     optimal_lambdas,
    #     bins=50,
    #     stat="proportion",
    #     kde=True,
    #     color="black",
    #     line_kws={"color": "black", "linewidth": 2.5},
    #     ax=axs[1],
    # )
    # axs[1].set(
    #     xlabel=r"Per-system optimal fixed value capture $\hat{\lambda}_s$",
    #     ylabel=None,
    #     xlim=(0, 1),
    # )

    # for ax in axs:
    #     sns.despine(ax=ax, left=True, bottom=True)
    # fig.suptitle(
    #     f"B. A left-peaked inverse-U relationship between value-capture strength and cultural attainment",
    #     x=0.05,
    #     horizontalalignment="left",
    # )

    # save_fig(
    #     fig,
    #     "lambda_inverse_u",
    #     subfolder="theoretical/scratch",
    # )


if __name__ == "__main__":
    main()
