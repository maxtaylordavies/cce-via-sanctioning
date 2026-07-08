import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    BoundaryNorm,
    TwoSlopeNorm,
)
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
import seaborn as sns

from src.utils import save_fig
from utils import progression_rate, get_eta_star_fn, simulate_trajectory
from constants import *

# variable parameters/functions
p_success_fns = {  # label -> (function, D_max) for plotting purposes
    "$p_{success}(D) = 0.5$": (lambda D: jnp.full_like(D, 0.5), 60),
    "$p_{success}(D) = 0.1$": (lambda D: jnp.full_like(D, 0.1), 60),
    "$p_{success}(D) = 1/(D+1)$": (lambda D: 1 / (D + 1), 60),
    "$p_{success}(D) = \\exp(-D / 5)$": (lambda D: jnp.exp(-D / 5), 60),
}
loss_rate_fn = (
    lambda D, eta: turnover_rate * D * ((1 - preservation_rate) ** (N * (1 - eta)))
)
fixed_lambdas = jnp.array([0.0, 0.1, 0.3, 0.5])
etas = jnp.linspace(0.0, 1.0, 1000)

# set up some colormaps for plotting
decline_colour = "#c45704"
growth_colour = "#058a43"
neutral_colour = "#ffffff"
sign_cmap = ListedColormap([decline_colour, growth_colour])
mag_cmap = LinearSegmentedColormap.from_list(
    "custom_orange_green", [decline_colour, neutral_colour, growth_colour], N=256
)
sign_boundary_norm = BoundaryNorm(boundaries=[-0.5, 0.0, 0.5], ncolors=sign_cmap.N)
variable_lambda_colors = ["red", "blue"]  # for lambda=0 and lambda=optimal


def get_palette(n):
    return sns.color_palette("RdPu", n_colors=n + 2)[2:]


def compute_net_growth(D, eta, p_success):
    gain = progression_rate(eta, p_success, phi=phi, N=N)
    loss = loss_rate_fn(D, eta)
    return gain - loss


def compute_growth_optimal_eta(D, p_success):
    etas = jnp.linspace(0.0, 1.0, 1000)
    rates = jax.vmap(compute_net_growth, in_axes=(None, 0, None))(D, etas, p_success)
    return etas[jnp.argmax(rates)]


def compute_K(eta):
    K = 1 + ((1 - eta) * (N - 1))
    # K = phi * N
    return K


def get_delta_functions(
    p_success_fn,
    c_innov_fn=c_innov_fn,
    c_imit_fn=c_imit_fn,
    lambda_fn=lambda D: 0.0,
    lr=learning_rate,
):
    dD_dt = lambda D, eta: compute_net_growth(D, eta, p_success_fn(D))

    def v_innov_fn(D, eta):
        K = compute_K(eta)
        return (1 + (lambda_fn(D) * (K - 1))) * baseline_value_fn(D)

    b_imit_fn = lambda D: (1 - lambda_fn(D)) * baseline_value_fn(D)

    eta_star_fn = get_eta_star_fn(
        v_innov_fn, b_imit_fn, p_success_fn, c_innov_fn, c_imit_fn, beta=beta
    )

    def deta_dt(D, eta):
        eta_star = eta_star_fn(D, eta)[0]
        learning_term = (1 - turnover_rate) * lr * (eta_star - eta)
        turnover_term = turnover_rate * (pi_0 - eta)
        return learning_term + turnover_term

    return dD_dt, deta_dt, eta_star_fn


def get_optimal_lambda_fn(p_success_fn):
    def optimal_lambda_fn(D):
        v, p = baseline_value_fn(D), p_success_fn(D)
        target = compute_growth_optimal_eta(D, p)
        K = compute_K(target)
        log_term = beta * jnp.log((target / (1 - target + 1e-10)) + 1e-10)
        numerator = log_term + c_innov_fn(D) - c_imit_fn(D) + (v * (1 - p))
        denominator = v * (1 + (p * (K - 1)))
        return jnp.clip(numerator / (denominator + 1e-10), 0.0, 1.0)

    return optimal_lambda_fn


def plot_phase_diagram(ax, dD_dt, Ds, etas, sign_only=False):
    net_growth_grid = jax.vmap(lambda D: jax.vmap(lambda eta: dD_dt(D, eta))(etas))(Ds)

    # convert from JAX to numpy for matplotlib
    Ds_np = np.asarray(Ds)
    etas_np = np.asarray(etas)
    net_np = np.asarray(net_growth_grid)
    Z_sign = np.sign(net_np).T
    Z_net = net_np.T

    if sign_only:
        ax.pcolormesh(
            Ds_np,
            etas_np,
            Z_sign,
            cmap=sign_cmap,
            norm=sign_boundary_norm,
            shading="auto",
            alpha=0.7,
        )
    else:
        v = np.percentile(np.abs(Z_net), 95)
        mag_norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        ax.pcolormesh(
            Ds_np,
            etas_np,
            Z_net,
            cmap=mag_cmap,
            norm=mag_norm,
            shading="auto",
        )

    # add maintenance boundary contour
    ax.contour(
        Ds_np,
        etas_np,
        Z_net,
        levels=[0.0],
        colors="black",
        linewidths=2,
        linestyles=":",
    )

    return net_growth_grid


def annotate_phases(ax, panel_idx):
    params = [
        [(0.52, 0.53, -32), (0.48, 0.43, -32)],  # panel 0
        [(0.52, 0.65, -29), (0.48, 0.54, -29)],  # panel 1
    ]
    decline_params, growth_params = params[panel_idx]
    ax.text(
        decline_params[0],
        decline_params[1],
        "Cultural decline",
        transform=ax.transAxes,
        rotation=decline_params[2],
        ha="center",
        va="center",
        fontsize=14,
        fontweight="medium",
        color="black",
        zorder=6,
    )
    ax.text(
        growth_params[0],
        growth_params[1],
        "Cultural growth",
        transform=ax.transAxes,
        rotation=growth_params[2],
        ha="center",
        va="center",
        fontsize=14,
        fontweight="medium",
        color="black",
        zorder=6,
    )


def compute_plot_lims(Ds, etas, margin_prop=0.02):
    D_margin = (Ds.max() - Ds.min()) * margin_prop
    eta_margin = (etas.max() - etas.min()) * margin_prop
    x_lims = (Ds.min() - D_margin, Ds.max() + D_margin)
    y_lims = (etas.min() - eta_margin, etas.max() + eta_margin)
    return x_lims, y_lims


def add_trajectory_arrows(
    ax,
    trajectory,
    color,
    positions=(0.5,),
    arrow_length=0.035,
    arrowhead_size=21,
):
    trajectory = np.asarray(trajectory)
    finite = np.isfinite(trajectory).all(axis=1)
    trajectory = trajectory[finite]

    x_span = np.ptp(ax.get_xlim())
    y_span = np.ptp(ax.get_ylim())
    scaled = trajectory / np.array([x_span, y_span])
    segment_lengths = np.linalg.norm(np.diff(scaled, axis=0), axis=1)
    cumulative_lengths = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    moving = np.concatenate([[True], np.diff(cumulative_lengths) > 1e-12])
    trajectory = trajectory[moving]
    cumulative_lengths = cumulative_lengths[moving]
    total_length = cumulative_lengths[-1]
    if total_length == 0:
        return

    for position in positions:
        arrow_end = position * total_length
        arrow_start = max(0.0, arrow_end - (arrow_length * total_length))
        start = [
            np.interp(arrow_start, cumulative_lengths, trajectory[:, dim])
            for dim in range(2)
        ]
        end = [
            np.interp(arrow_end, cumulative_lengths, trajectory[:, dim])
            for dim in range(2)
        ]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "facecolor": color,
                "edgecolor": color,
                "linewidth": 0,
                "mutation_scale": arrowhead_size,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            zorder=4,
        )


# Visualising population trajectories under different learning rates
def add_title_with_legend(
    fig,
    title,
    legend_symbol,
    legend_vals,
    legend_colors,
    legend_styles=None,
    font_size=14,
):
    title = TextArea(
        f"{title}  [",
        textprops={"fontsize": font_size},
    )

    if legend_styles is None:
        legend_styles = ["-"] * len(legend_vals)

    entries = []
    for i, (val, color) in enumerate(zip(legend_vals, legend_colors)):
        line_box = DrawingArea(20, 10, 0, 0)
        line_box.add_artist(
            Line2D(
                [0, 20], [5, 5], color=color, linewidth=3, linestyle=legend_styles[i]
            )
        )
        suffix = "]" if i == len(legend_vals) - 1 else ""
        label = TextArea(
            f"$\\{legend_symbol} = {val}$" + suffix, textprops={"fontsize": font_size}
        )
        entries.append(
            HPacker(children=[line_box, label], align="center", pad=0, sep=5)
        )

    entries_box = HPacker(children=entries, align="center", pad=0, sep=16)
    title_row = HPacker(children=[title, entries_box], align="center", pad=0, sep=4)
    fig.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=title_row,
            bbox_to_anchor=(0.5, 1.02),
            bbox_transform=fig.transFigure,
            frameon=False,
            borderpad=0,
            pad=0,
        )
    )


def do_plots_without_trajectories():
    n_panels = len(p_success_fns)
    sign_fig, sign_axs = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5), sharey=True
    )
    mag_fig, mag_axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for i, (p_fn_label, (p_success_fn, D_max)) in enumerate(p_success_fns.items()):
        Ds = jnp.linspace(0.0, D_max, 1000)

        dD_fn, _, _ = get_delta_functions(p_success_fn)

        plot_phase_diagram(sign_axs[i], dD_fn, Ds, etas, sign_only=True)
        net_change_grid = plot_phase_diagram(
            mag_axs[i], dD_fn, Ds, etas, sign_only=False
        )

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        for ax in [sign_axs[i], mag_axs[i]]:
            ax.set(
                xlabel="Cultural complexity $D_t$",
                ylabel=ylabel,
                xlim=x_lims,
                ylim=y_lims,
                title=p_fn_label,
            )
            ax.grid(False)
            ax.tick_params(axis="both", which="both", length=0)
            sns.despine(ax=ax, left=True, bottom=True)

            if i == 1:
                for eta_val in [0.02, 0.1, 0.5]:
                    # add horizontal dashed line from x=0 to x=the point where the line intersects the maintenance boundary
                    eta_idx = np.argmin(np.abs(etas - eta_val))
                    D_intersect_idx = np.argmin(np.abs(net_change_grid[:, eta_idx]))
                    D_intersect = float(Ds[D_intersect_idx])
                    ax.hlines(
                        y=eta_val,
                        xmin=0.0,
                        xmax=D_intersect,
                        colors="m",
                        linestyles="--",
                        linewidth=1.5,
                    )
                    ax.annotate(
                        f"$\\eta = {eta_val:g}$",
                        xy=(1, eta_val),
                        xytext=(0, 3),
                        textcoords="offset points",
                        color="m",
                        fontsize=8,
                        fontweight="bold",
                        ha="left",
                        va="bottom",
                    )
                    ax.scatter(
                        D_intersect,
                        eta_val,
                        color="m",
                        s=40,
                        zorder=5,
                    )

            if i == 0:
                annotate_phases(ax, panel_idx=i)

    sign_fig.suptitle(
        "$\\text{Sgn}(\\frac{dD_t}{dt})$ as a function of cultural complexity $D_t$ and innovator frequency $\\eta_t$",
        fontsize=14,
    )
    mag_fig.suptitle(
        "Net rate of cultural change $\\frac{dD_t}{dt}$ as a function of cultural complexity $D_t$ and innovator frequency $\\eta_t$",
        fontsize=14,
    )

    return sign_fig, mag_fig


def do_plots_for_learning_rates(learning_rates, colors):
    n_panels = len(p_success_fns)
    sign_fig, sign_axs = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5), sharey=True
    )
    mag_fig, mag_axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for i, (p_fn_label, (p_success_fn, D_max)) in enumerate(p_success_fns.items()):
        Ds = jnp.linspace(0.0, D_max, 1000)
        dD_fn, _, _ = get_delta_functions(p_success_fn)

        plot_phase_diagram(sign_axs[i], dD_fn, Ds, etas, sign_only=True)
        plot_phase_diagram(mag_axs[i], dD_fn, Ds, etas, sign_only=False)

        for j, lr in enumerate(learning_rates):
            dD_fn, deta_fn, _ = get_delta_functions(p_success_fn, lr=lr)
            delta_fns = (dD_fn, deta_fn)

            traj_np = np.asarray(
                simulate_trajectory(
                    delta_fns, D0=0.0, eta0=0.5, T=2000, dt=0.1, D_max=D_max
                )
            )

            for ax in [sign_axs[i], mag_axs[i]]:
                ax.plot(
                    traj_np[:, 0],
                    traj_np[:, 1],
                    color=colors[j],
                    linewidth=2,
                    alpha=1.0,
                )
                add_trajectory_arrows(ax, traj_np, colors[j])
                ax.scatter(  # start point
                    traj_np[0, 0],
                    traj_np[0, 1],
                    facecolor="white",
                    edgecolor="black",
                    s=45,
                    zorder=5,
                )
                ax.scatter(  # end point
                    traj_np[-1, 0],
                    traj_np[-1, 1],
                    color=colors[j],
                    s=45,
                    zorder=5,
                )

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        for ax in [sign_axs[i], mag_axs[i]]:
            ax.set(
                xlabel="Cultural complexity $D_t$",
                ylabel=ylabel,
                xlim=x_lims,
                ylim=y_lims,
                title=p_fn_label,
            )
            ax.grid(False)
            ax.tick_params(axis="both", which="both", length=0)
            sns.despine(ax=ax, left=True, bottom=True)

            if i == 0:
                annotate_phases(ax, panel_idx=i)

    for fig in [sign_fig, mag_fig]:
        add_title_with_legend(
            fig,
            "Visualising population trajectories under different learning rates",
            "rho",
            learning_rates,
            colors,
        )

    return sign_fig, mag_fig


def do_plots_for_lambda_fns(all_lambda_fns, colors, styles=None, widths=None):
    n_panels = len(p_success_fns)
    sign_fig, sign_axs = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5), sharey=True
    )
    mag_fig, mag_axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for i, (p_fn_label, (p_success_fn, D_max)) in enumerate(p_success_fns.items()):
        lambda_fns = all_lambda_fns[i]
        Ds = jnp.linspace(0.0, D_max, 1000)

        dD_fn, deta_fn, _ = get_delta_functions(p_success_fn)
        delta_fns = (dD_fn, deta_fn)

        plot_phase_diagram(sign_axs[i], dD_fn, Ds, etas, sign_only=True)
        plot_phase_diagram(mag_axs[i], dD_fn, Ds, etas, sign_only=False)

        # add some trajectories
        for j, lambda_fn in enumerate(lambda_fns):
            dD_fn_, deta_fn_, _ = get_delta_functions(p_success_fn, lambda_fn=lambda_fn)
            delta_fns = (dD_fn_, deta_fn_)
            traj_np = np.asarray(
                simulate_trajectory(
                    delta_fns, D0=0.0, eta0=0.5, T=2000, dt=0.1, D_max=D_max
                )
            )

            for ax in [sign_axs[i], mag_axs[i]]:
                ax.plot(
                    traj_np[:, 0],
                    traj_np[:, 1],
                    color=colors[j],
                    linestyle=styles[j] if styles is not None else "-",
                    linewidth=widths[j] if widths is not None else 2,
                    alpha=1.0,
                )
                add_trajectory_arrows(ax, traj_np, colors[j])
                ax.scatter(  # start point
                    traj_np[0, 0],
                    traj_np[0, 1],
                    facecolor="white",
                    edgecolor="black",
                    s=45,
                    zorder=5,
                )
                if (
                    traj_np[-1, 0] < D_max
                ):  # only plot end point if it is within the plot limits
                    ax.scatter(  # end point
                        traj_np[-1, 0],
                        traj_np[-1, 1],
                        color=colors[j],
                        s=45,
                        zorder=5,
                    )

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        for ax in [sign_axs[i], mag_axs[i]]:
            ax.set(
                xlabel="Cultural complexity $D_t$",
                ylabel=ylabel,
                xlim=x_lims,
                ylim=y_lims,
                title=p_fn_label,
            )
            ax.grid(False)
            ax.tick_params(axis="both", which="both", length=0)
            sns.despine(ax=ax, left=True, bottom=True)

            if i == 0:
                annotate_phases(ax, panel_idx=i)

    for fig in [sign_fig, mag_fig]:
        add_title_with_legend(
            fig,
            "Visualising population trajectories under different innovator appropriation shares",
            "lambda",
            [f"{l:.1f}" for l in fixed_lambdas] + ["\\lambda^\\star(D)"],
            colors,
            ["-"] * len(fixed_lambdas) + ["--"],
        )

    return sign_fig, mag_fig


def main():
    # do plots with no trajectories (phase diagrams only)
    sign_fig, mag_fig = do_plots_without_trajectories()
    save_fig(sign_fig, "no_trajs_signs", subfolder="theoretical/phase_diagrams")
    save_fig(mag_fig, "no_trajs_magnitudes", subfolder="theoretical/phase_diagrams")

    # do plots for just lambda=0 (baseline case) with different learning rates
    learning_rates = [0.05, 0.1, 1.0]
    lr_colors = get_palette(len(learning_rates))
    sign_fig, mag_fig = do_plots_for_learning_rates(learning_rates, lr_colors)
    save_fig(sign_fig, "learning_rates_signs", subfolder="theoretical/phase_diagrams")
    save_fig(
        mag_fig, "learning_rates_magnitudes", subfolder="theoretical/phase_diagrams"
    )

    # do plots for fixed lambda values
    fixed_lambda_fns = [
        [lambda D, l=l: l for l in fixed_lambdas] for _ in range(len(p_success_fns))
    ]
    fixed_lambda_colors = get_palette(len(fixed_lambdas))
    sign_fig, mag_fig = do_plots_for_lambda_fns(fixed_lambda_fns, fixed_lambda_colors)
    save_fig(sign_fig, "fixed_lambda_signs", subfolder="theoretical/phase_diagrams")
    save_fig(mag_fig, "fixed_lambda_magnitudes", subfolder="theoretical/phase_diagrams")

    # do plots for optimal variable lambda
    variable_lambda_fns = [
        [lambda D: 0.0, get_optimal_lambda_fn(p_success_fn)]
        for p_success_fn, _ in p_success_fns.values()
    ]
    sign_fig, mag_fig = do_plots_for_lambda_fns(
        variable_lambda_fns, variable_lambda_colors
    )
    save_fig(sign_fig, "variable_lambda_signs", subfolder="theoretical/phase_diagrams")
    save_fig(
        mag_fig, "variable_lambda_magnitudes", subfolder="theoretical/phase_diagrams"
    )

    # combined
    combined_lambda_fns = [
        [lambda D, l=l: l for l in fixed_lambdas]
        + [get_optimal_lambda_fn(p_success_fn)]
        for p_success_fn, _ in p_success_fns.values()
    ]
    colors = fixed_lambda_colors + ["blue"]
    styles = ["-"] * len(fixed_lambdas) + ["--"]
    widths = [2] * len(fixed_lambdas) + [2.5]
    sign_fig, mag_fig = do_plots_for_lambda_fns(
        combined_lambda_fns, colors, styles, widths
    )
    save_fig(sign_fig, "combined_lambda_signs", subfolder="theoretical/phase_diagrams")
    save_fig(
        mag_fig, "combined_lambda_magnitudes", subfolder="theoretical/phase_diagrams"
    )


if __name__ == "__main__":
    main()
