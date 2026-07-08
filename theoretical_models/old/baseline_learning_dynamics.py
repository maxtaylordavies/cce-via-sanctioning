from itertools import product

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_fig
from utils import progression_rate, compute_rate_optimal_eta, get_eta_star_fn

sns.set_style("whitegrid")

N = 100
beta = 1.0


def compute_y_lims_and_ticks(etas, eta_stars, global_eta_stars, Ds, eta_tick_size=0.1):
    # compute y-axis limits for eta
    eta_min = min(min(etas), min(eta_stars), min(global_eta_stars))
    eta_max = max(max(etas), max(eta_stars), max(global_eta_stars))

    eta_min = np.floor(eta_min / eta_tick_size) * eta_tick_size
    eta_max = np.ceil(eta_max / eta_tick_size) * eta_tick_size
    num_ticks = np.ceil((eta_max - eta_min) / eta_tick_size).astype(int) + 1

    # find matching y-axis limits for D
    D_tick_size = np.ceil((max(Ds) - min(Ds)) / (num_ticks - 1))
    ratio = D_tick_size / eta_tick_size
    D_min, D_max = eta_min * ratio, eta_max * ratio

    eta_lims = (eta_min - eta_tick_size / 2, eta_max + eta_tick_size / 2)
    D_lims = (D_min - D_tick_size / 2, D_max + D_tick_size / 2)

    eta_ticks = np.linspace(eta_min, eta_max, num_ticks)
    D_ticks = np.linspace(D_min, D_max, num_ticks)

    return eta_lims, D_lims, eta_ticks, D_ticks


def apply_y_lims_and_ticks(axs, eta_lims, D_lims, eta_ticks, D_ticks):
    axs[0].set(ylim=eta_lims, yticks=eta_ticks)
    axs[1].set(ylim=D_lims, yticks=D_ticks)


def plot_segmented_line(
    ax,
    x,
    y,
    *,
    x_lims,
    y_lims,
    dash_fraction=0.035,
    gap_fraction=0.02,
    color="black",
    linewidth=2.0,
    label=None,
    zorder=3,
):
    x = np.asarray(x)
    y = np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    x_range = x_lims[1] - x_lims[0]
    y_range = y_lims[1] - y_lims[0]
    scaled_x = (x - x_lims[0]) / x_range
    scaled_y = (y - y_lims[0]) / y_range
    segment_lengths = np.hypot(np.diff(scaled_x), np.diff(scaled_y))
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_lengths[-1]

    dash_start = 0.0
    while dash_start < total_length:
        dash_end = min(dash_start + dash_fraction, total_length)
        in_dash = (cumulative_lengths > dash_start) & (cumulative_lengths < dash_end)
        segment_distances = np.concatenate(
            [[dash_start], cumulative_lengths[in_dash], [dash_end]]
        )
        segment_x = np.interp(segment_distances, cumulative_lengths, x)
        segment_y = np.interp(segment_distances, cumulative_lengths, y)
        ax.plot(
            segment_x,
            segment_y,
            color=color,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=zorder,
        )
        dash_start += dash_fraction + gap_fraction

    if label is not None:
        ax.plot(
            [],
            [],
            color=color,
            marker="_",
            linestyle="None",
            markersize=10,
            markeredgewidth=linewidth,
            label=label,
        )


def get_step_fn(eta_star_fn, learning_rate=0.1, turnover_rate=0.001, pi_0=0.5, phi=0.1):
    def step_fn(current_eta, D):
        eta_star, _, p_success = eta_star_fn(D)
        global_eta_star = compute_rate_optimal_eta(p_success, phi)
        next_eta = (
            current_eta
            + (learning_rate * (eta_star - current_eta))
            + (turnover_rate * (pi_0 - current_eta))
        )
        G = progression_rate(current_eta, p_success, phi)
        next_D = D + G
        return next_D, next_eta, eta_star, global_eta_star

    return step_fn


def make_single_learning_plot(
    ax,
    p_success_fn,
    c_innov_fn,
    c_imit=0.0,
    phi=0.1,
    T=100,
    learning_rate=0.1,
    turnover_rate=0.001,
    pi_0=0.5,
):
    b_imit_fn = lambda D: D
    v_innov_fn = lambda D: 2 * D
    c_imit_fn = lambda D: c_imit

    eta_star_fn = get_eta_star_fn(
        v_innov_fn, b_imit_fn, p_success_fn, c_innov_fn, c_imit_fn, beta=beta
    )

    step_fn = get_step_fn(eta_star_fn, learning_rate, turnover_rate, pi_0, phi)

    D_0, eta_0 = 0.0, pi_0
    eta_star_0, _, p_success_0 = eta_star_fn(D_0)
    global_eta_star_0 = compute_rate_optimal_eta(p_success_0, phi)

    Ds, etas, eta_stars, global_eta_stars = (
        [D_0],
        [eta_0],
        [eta_star_0],
        [global_eta_star_0],
    )
    for _ in range(T):
        next_D, next_eta, eta_star, global_eta_star = step_fn(etas[-1], Ds[-1])
        Ds.append(next_D)
        etas.append(next_eta)
        eta_stars.append(eta_star)
        global_eta_stars.append(global_eta_star)

    ts = np.arange(len(Ds))
    eta_lims, D_lims, eta_ticks, D_ticks = compute_y_lims_and_ticks(
        etas, eta_stars, global_eta_stars, Ds
    )

    ax2 = ax.twinx()
    axs = [ax, ax2]
    for sub_ax in axs:
        sub_ax.set_axisbelow(True)
    axs[1].grid(False)

    sns.lineplot(
        x=ts,
        y=etas,
        ax=axs[0],
        label="$\\eta_t$",
        color="black",
        linewidth=2.0,
        zorder=3,
    )
    eta_line = axs[0].lines[-1]
    eta_line.set_solid_capstyle("round")

    plot_segmented_line(
        axs[0],
        ts,
        eta_stars,
        x_lims=(ts.min(), ts.max()),
        y_lims=eta_lims,
        gap_fraction=0.026,
        label="$\\eta^*(D_t)$",
        color="black",
        linewidth=2.0,
        zorder=3,
    )

    dot_stride = 4
    axs[0].plot(
        ts[::dot_stride],
        global_eta_stars[::dot_stride],
        label="$\\arg\\max_{\\eta} G(\\eta)$",
        color="xkcd:lime green",
        marker="o",
        markersize=3.4,
        markeredgewidth=0,
        linestyle="None",
        linewidth=2.0,
        zorder=3,
    )

    axs[0].set(
        xlabel="$t$",
        ylabel="$\\eta$",
    )
    sns.despine(ax=axs[0], left=True, bottom=True)

    sns.lineplot(
        x=ts,
        y=Ds,
        ax=axs[1],
        label="$D_t$",
        color="red",
        linewidth=2.0,
        zorder=3,
    )
    D_line = axs[1].lines[-1]
    D_line.set_solid_capstyle("round")
    axs[1].set(xlabel="$t$", ylabel="$D$")
    sns.despine(ax=axs[1], left=True, bottom=True)

    legend_handles, legend_labels = [], []
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        legend_handles.extend(handles)
        legend_labels.extend(labels)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    axs[0].legend(legend_handles, legend_labels, loc="upper right")
    for ax in axs:
        ax.tick_params(axis="y", which="both", length=0)

    return axs, (eta_lims, D_lims, eta_ticks, D_ticks)


def make_learning_plots(c_imit, phi, learning_rate):
    p_success_fns = {
        "$p_{success}(D) = 0.5$": lambda D: 0.5,
        "$p_{success}(D) = \\exp(-D)$": lambda D: np.exp(-D),
    }

    c_innov_fns = {
        "$c_{innov}(D) = 1$": lambda D: 1.0,
        "$c_{innov}(D) = D$": lambda D: D,
    }

    # get all combinations of the above functions
    key_combinations = list(product(p_success_fns.keys(), c_innov_fns.keys()))
    n_combos = len(key_combinations)

    fig, axs = plt.subplots(1, n_combos, figsize=(n_combos * 4, 3.5))
    axs = axs if n_combos > 1 else [axs]

    sub_axs_list, global_y_ax_stuff, max_eta = [], None, 0.0
    for i, (p_success_key, c_innov_key) in enumerate(key_combinations):
        p_success_fn = p_success_fns[p_success_key]
        c_innov_fn = c_innov_fns[c_innov_key]

        sub_axs, y_ax_stuff = make_single_learning_plot(
            axs[i],
            p_success_fn,
            c_innov_fn,
            c_imit=c_imit,
            phi=phi,
            learning_rate=learning_rate,
        )
        axs[i].set_title(f"{p_success_key}, {c_innov_key}", fontsize=10)

        sub_axs_list.append(sub_axs)
        if y_ax_stuff[0][1] > max_eta:
            global_y_ax_stuff = y_ax_stuff
            max_eta = y_ax_stuff[0][1]

    for sub_axs in sub_axs_list:
        apply_y_lims_and_ticks(sub_axs, *global_y_ax_stuff)

    return fig


def main():
    c_imit = 0
    learning_rate = 0.1

    for phi in [0.1, 0.5, 0.9]:
        fig = make_learning_plots(
            c_imit=c_imit,
            phi=phi,
            learning_rate=learning_rate,
        )
        fig.suptitle(
            f"Example learning dynamics $(\\phi={phi}, \\rho={learning_rate}, c_\\text{{imit}}={c_imit}, B(D)=V(D)=D)$",
            fontsize=12,
        )
        fig.tight_layout(w_pad=2.0)
        save_fig(
            fig, f"learning_dynamics_phi_{phi}", subfolder="theoretical", fmts=["png"]
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
