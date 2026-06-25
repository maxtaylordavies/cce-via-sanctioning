from itertools import product

import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils import save_fig

sns.set_style("whitegrid")

N = 100
beta = 1.0


def time_to_discovery(eta, p_success):
    p_fail = 1 - p_success
    p_no_successes = p_fail ** (N * eta)
    return 1 / (1 - p_no_successes + 1e-10)


def time_to_diffusion(eta, phi):
    eta_term = 1 / ((beta * (1 - eta)) + 1e-10)
    phi_term = (phi * (N - 1)) / (1 - phi)
    return eta_term * np.log(phi_term)


def progression_rate(eta, p_success, phi):
    rate = 1 / (time_to_discovery(eta, p_success) + time_to_diffusion(eta, phi))
    return np.where(eta == 0, 0.0, rate)


def compute_rate_optimal_eta(p_success, phi):
    etas = np.linspace(0.0, 1.0, 1000)
    rates = progression_rate(etas, p_success, phi)
    return etas[np.argmax(rates)]


def visualise_progression_rate_curves():
    p_success_values = [0.01, 0.1, 0.5]
    phi_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    eta_values = np.linspace(0.0, 1.0, 1000)

    data = {"p_success": [], "phi": [], "eta": [], "progression_rate": []}
    for p_success, phi in product(p_success_values, phi_values):
        progression_rates = progression_rate(eta_values, p_success, phi)
        data["p_success"].extend([p_success] * len(eta_values))
        data["phi"].extend([phi] * len(eta_values))
        data["eta"].extend(eta_values)
        data["progression_rate"].extend(progression_rates)

    df = pd.DataFrame(data)

    palette = sns.color_palette("crest", n_colors=len(phi_values))
    fig, axs = plt.subplots(
        1,
        len(p_success_values),
        figsize=(16, 3.5),
        sharey=True,
        constrained_layout=True,
    )
    for i, p_success in enumerate(p_success_values):
        sub_df = df[df["p_success"] == p_success]
        sns.lineplot(
            x="eta",
            y="progression_rate",
            hue="phi",
            palette=palette,
            data=sub_df,
            ax=axs[i],
            linewidth=2.0,
            # legend=False,
        )
        axs[i].set_xlabel("Innovator frequency $\\eta$")
        axs[i].set_ylabel("Cultural progression rate $G$")
        axs[i].set_title(f"$p_\\text{{success}}={p_success}$")
        sns.despine(ax=axs[i], left=True, bottom=True)

        peak_rows = []
        for phi_idx, phi in enumerate(phi_values):
            subset = sub_df[sub_df["phi"] == phi]
            max_idx = subset["progression_rate"].idxmax()
            peak_rows.append(
                {
                    "phi": phi,
                    "phi_idx": phi_idx,
                    "eta": subset.loc[max_idx, "eta"],
                    "progression_rate": subset.loc[max_idx, "progression_rate"],
                }
            )

        peak_eta_values = np.array([row["eta"] for row in peak_rows])
        peak_eta_range = peak_eta_values.max() - peak_eta_values.min()
        peak_eta_margin = max(0.01, peak_eta_range * 0.6)
        inset_eta_min = max(0, peak_eta_values.min() - peak_eta_margin)
        inset_eta_max = min(1, peak_eta_values.max() + peak_eta_margin)
        inset_df = sub_df[
            (sub_df["eta"] >= inset_eta_min) & (sub_df["eta"] <= inset_eta_max)
        ]
        inset_y_min = inset_df["progression_rate"].min()
        inset_y_max = inset_df["progression_rate"].max()
        inset_y_pad = (inset_y_max - inset_y_min) * 0.08
        inset_y_lower = inset_y_min - inset_y_pad
        inset_y_upper = inset_y_max + inset_y_pad

        axins = inset_axes(
            axs[i],
            width="38%",
            height="38%",
            loc="upper right",
            borderpad=1.1,
        )
        axins.set_facecolor("white")

        for row in peak_rows:
            subset = sub_df[sub_df["phi"] == row["phi"]]
            zoomed = subset[
                (subset["eta"] >= inset_eta_min) & (subset["eta"] <= inset_eta_max)
            ]
            axins.plot(
                zoomed["eta"],
                zoomed["progression_rate"],
                color=palette[row["phi_idx"]],
                linewidth=1.0,
                alpha=0.6,
            )
            axins.plot(
                [row["eta"], row["eta"]],
                [inset_y_lower, row["progression_rate"]],
                color=palette[row["phi_idx"]],
                linestyle="-",
                linewidth=1.5,
                alpha=1.0,
            )
            axins.plot(
                row["eta"] * 0.998,
                row["progression_rate"],
                color="gold",
                marker="*",
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.25,
                linestyle="None",
            )

        axins.set_xlim(inset_eta_min, inset_eta_max)
        axins.set_ylim(inset_y_lower, inset_y_upper)
        axins.set_title("zoomed view of optimal $\\eta$ values", fontsize=8, pad=4)
        axins.tick_params(axis="both", which="major", labelsize=7, length=2)
        axins.tick_params(axis="y", labelleft=False)
        for spine in axins.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)

    fig.suptitle(
        "Cultural progression rate $G$ as a function of innovator frequency $\\eta$",
        fontsize=12,
    )

    return fig


def compute_y_lims_and_ticks(etas, eta_stars, global_eta_stars, Ds, eta_tick_size=0.1):
    # compute y-axis limits for eta
    eta_min = min(min(etas), min(eta_stars), min(global_eta_stars))
    eta_max = max(max(etas), max(eta_stars), max(global_eta_stars))

    print(f"Raw eta limits: {eta_min:.3f}, {eta_max:.3f}")

    eta_min = np.floor(eta_min / eta_tick_size) * eta_tick_size
    eta_max = np.ceil(eta_max / eta_tick_size) * eta_tick_size
    num_ticks = np.ceil((eta_max - eta_min) / eta_tick_size).astype(int) + 1

    print(f"Adjusted eta limits: {eta_min:.3f}, {eta_max:.3f} with {num_ticks} ticks")

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


def make_single_learning_plot(
    ax,
    p_success_fn,
    c_innov_fn,
    c_imit=0.0,
    phi=0.1,
    T=100,
    learning_rate=0.1,
):
    b_imit_fn = lambda D: D
    v_innov_fn = lambda D: D

    def eta_star_fn(D):
        delta_r = (
            b_imit_fn(D) - c_imit - (p_success_fn(D) * v_innov_fn(D)) + c_innov_fn(D)
        )
        return 1 / (1 + np.exp(delta_r / beta))

    def next_eta_fn(current_eta, eta_star):
        return current_eta + learning_rate * (eta_star - current_eta)

    D_0, eta_0 = 0.0, 0.0
    p_0 = p_success_fn(D_0)
    eta_star_0 = eta_star_fn(D_0)
    global_eta_star_0 = compute_rate_optimal_eta(p_0, phi)

    Ds, etas, eta_stars, global_eta_stars = (
        [D_0],
        [eta_0],
        [eta_star_0],
        [global_eta_star_0],
    )
    for _ in range(T):
        p = p_success_fn(Ds[-1])
        eta_star = eta_star_fn(Ds[-1])
        global_eta_star = compute_rate_optimal_eta(p, phi)
        next_eta = next_eta_fn(etas[-1], eta_star)
        G = progression_rate(etas[-1], p, phi)
        next_D = Ds[-1] + G
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
        color="gold",
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


def make_learning_plots(V, c_imit, phi, learning_rate):
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
    rate_curves_fig = visualise_progression_rate_curves()
    save_fig(
        rate_curves_fig,
        "progression_rate_curves",
        subfolder="theoretical",
        fmts=["png"],
    )

    V = 1.0
    c_imit = 0
    learning_rate = 0.1

    for phi in [0.1, 0.5, 0.9]:
        fig = make_learning_plots(
            V=V,
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
