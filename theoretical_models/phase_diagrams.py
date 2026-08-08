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
import seaborn as sns

from src.utils import save_fig
from utils import (
    progression_rate,
    get_eta_star_fn,
    add_title_with_legend,
    get_p_copy,
    get_p_success_fn,
    compute_rho_from_w_and_mu,
)
from constants import *

Ds = jnp.linspace(0.0, 80, 1000)
etas = jnp.linspace(0.0, 1.0, 1000)

p_success_fn_params = [(1.0, 0), (0.1, 0), (1.0, 0.1), (0.1, 0.1)]
fixed_lambdas = jnp.array([0.0, 0.1, 0.3, 0.5])

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


def get_p_success_label(p0, k):
    return f"$p_0={p0:g}, k={k:g}$"


def get_palette(n):
    return sns.color_palette("RdPu", n_colors=n + 2)[2:]


def loss_rate_fn(D, eta, mu=turnover_rate, alpha=0.01, N=N):
    # preservation_rate_for_N = 1 / N
    return mu * D * ((1 - alpha) ** (N * (1 - eta)))


def compute_net_growth(D, eta, p_success, mu=turnover_rate, alpha=0.01, N=N):
    gain = progression_rate(eta, p_success, phi=phi, N=N)
    loss = loss_rate_fn(D, eta, mu, alpha, N)
    return gain - loss


def compute_growth_optimal_eta(D, p_success, mu=turnover_rate, alpha=0.01, N=N):
    etas = jnp.linspace(0.0, 1.0, 1000)
    rates = jax.vmap(compute_net_growth, in_axes=(None, 0, None, None, None, None))(
        D, etas, p_success, mu, alpha, N
    )
    return etas[jnp.argmax(rates)]


def compute_K(eta, N=N):
    # K = 1 + ((1 - eta) * (N - 1))
    K = phi * N
    return K


def get_delta_functions(
    p_success_fn,
    c_innov_fn=c_innov_fn,
    c_imit_fn=c_imit_fn,
    lambda_fn=lambda D: 0.0,
    rho=learning_rate,
    mu=turnover_rate,
    beta=beta,
    N=N,
    sl_pool_prop=1.0,
    alpha=0.01,
    pi0=pi_0,
):
    dD_dt = lambda D, eta: compute_net_growth(D, eta, p_success_fn(D), mu, alpha, N)

    def v_innov_fn(D, eta):
        K = compute_K(eta, N=N)
        return (1 + (lambda_fn(D) * (K - 1))) * baseline_value_fn(D)

    b_imit_fn = lambda D: (1 - lambda_fn(D)) * baseline_value_fn(D)

    eta_star_fn = get_eta_star_fn(
        v_innov_fn,
        b_imit_fn,
        p_success_fn,
        c_innov_fn,
        c_imit_fn,
        beta,
        sl_pool_prop * N,
    )

    def deta_dt(D, eta):
        eta_star = eta_star_fn(D, eta)[0]
        learning_term = (1 - mu) * rho * (eta_star - eta)
        turnover_term = mu * (pi0 - eta)
        return learning_term + turnover_term

    return dD_dt, deta_dt, eta_star_fn


def solve_eta_nullcline_for_D(deta_fn, D, n_itr=50):
    lo, hi = jnp.asarray(0.0), jnp.asarray(1.0)

    def body_fn(_, state):
        lo, hi = state
        mid = (lo + hi) / 2
        g_mid = deta_fn(D, mid)

        new_lo = jnp.where(g_mid > 0.0, mid, lo)
        new_hi = jnp.where(g_mid > 0.0, hi, mid)

        return new_lo, new_hi

    lo, hi = jax.lax.fori_loop(0, n_itr, body_fn, (lo, hi))
    return (lo + hi) / 2


def get_eta_nullcline_curve(deta_fn, Ds, n_itr=50):
    return jax.vmap(lambda D: solve_eta_nullcline_for_D(deta_fn, D, n_itr))(Ds)


def compute_nullcline_intersection(Ds, eta_nullcline_vals, dD_fn):
    Ds = jnp.asarray(Ds)
    eta_nullcline_vals = jnp.asarray(eta_nullcline_vals)
    dD_on_curve = jax.vmap(lambda D, eta: dD_fn(D, eta))(Ds, eta_nullcline_vals)

    finite = (
        jnp.isfinite(Ds) & jnp.isfinite(eta_nullcline_vals) & jnp.isfinite(dD_on_curve)
    )
    n_points = Ds.shape[0]
    sentinel_idx = n_points

    def first_true_idx(mask):
        idxs = jnp.arange(mask.shape[0])
        return jnp.min(jnp.where(mask, idxs, sentinel_idx))

    exact_mask = finite & jnp.isclose(dD_on_curve, 0.0)
    exact_idx = first_true_idx(exact_mask)
    exact_success = exact_idx < n_points

    valid_pair = finite[:-1] & finite[1:]
    crossing_mask = valid_pair & (dD_on_curve[:-1] * dD_on_curve[1:] < 0.0)
    crossing_idx = first_true_idx(crossing_mask)
    crossing_success = crossing_idx < (n_points - 1)

    use_exact = exact_success & (
        (exact_idx <= crossing_idx) | jnp.logical_not(crossing_success)
    )
    success = exact_success | crossing_success

    safe_exact_idx = jnp.minimum(exact_idx, n_points - 1)
    safe_crossing_idx = jnp.minimum(crossing_idx, n_points - 2)

    D_exact = Ds[safe_exact_idx]
    eta_exact = eta_nullcline_vals[safe_exact_idx]

    D0 = Ds[safe_crossing_idx]
    D1 = Ds[safe_crossing_idx + 1]
    eta0 = eta_nullcline_vals[safe_crossing_idx]
    eta1 = eta_nullcline_vals[safe_crossing_idx + 1]
    g0 = dD_on_curve[safe_crossing_idx]
    g1 = dD_on_curve[safe_crossing_idx + 1]
    crossing_weight = -g0 / (g1 - g0)
    D_interpolated = D0 + crossing_weight * (D1 - D0)
    eta_interpolated = eta0 + crossing_weight * (eta1 - eta0)

    D_crossing = jnp.where(use_exact, D_exact, D_interpolated)
    eta_crossing = jnp.where(use_exact, eta_exact, eta_interpolated)
    intersection_idx = jnp.where(use_exact, exact_idx, crossing_idx)

    return (
        success,
        jnp.where(success, D_crossing, -1.0),
        jnp.where(success, eta_crossing, -1.0),
        jnp.where(success, intersection_idx, -1),
    )


def truncate_curve_at_D_boundary(Ds, eta_vals, dD_fn):
    Ds_np = np.asarray(Ds)
    eta_np = np.asarray(eta_vals)

    success, D_crossing, eta_crossing, crossing_idx = compute_nullcline_intersection(
        Ds_np, eta_np, dD_fn
    )
    if not bool(np.asarray(success)):
        return Ds_np, eta_np, None

    D_crossing = float(np.asarray(D_crossing))
    eta_crossing = float(np.asarray(eta_crossing))
    crossing_idx = int(np.asarray(crossing_idx))

    return (
        np.concatenate([Ds_np[: crossing_idx + 1], [D_crossing]]),
        np.concatenate([eta_np[: crossing_idx + 1], [eta_crossing]]),
        (D_crossing, eta_crossing),
    )


def get_optimal_lambda_fn(
    p_success_fn,
    c_innov_fn=c_innov_fn,
    c_imit_fn=c_imit_fn,
    value_fn=baseline_value_fn,
    rho=learning_rate,
    mu=turnover_rate,
    beta=beta,
    alpha=0.01,
    N=N,
    sl_pool_prop=1.0,
    pi0=pi_0,
    eps=1e-6,
):
    """
    Construct the adaptive value-appropriation policy lambda*(D).

    For each cultural complexity D, lambda*(D) attempts to make the
    culturally growth-optimal innovator frequency eta_c(D) a fixed point
    of the behavioural dynamics:

        deta/dt =
            (1 - mu) * rho * (pi_star(D, eta) - eta)
            + mu * (pi0 - eta).

    The returned lambda is clipped to [0, 1]. When the target cannot be
    implemented because of turnover, bounded lambda, or a zero-value
    denominator, the function returns the boundary lambda that moves the
    behavioural equilibrium toward the target.

    Assumptions
    -----------
    - beta > 0
    - value_fn(D) >= 0
    - increasing lambda makes innovation more attractive
    - get_p_copy uses the same copying model as get_delta_functions
    """

    # Weight placed on reward-based learning in the eta dynamics.
    learning_weight = (1.0 - mu) * rho

    # Must match the value passed to get_eta_star_fn inside
    # get_delta_functions.
    copy_pool_size = sl_pool_prop * N

    def optimal_lambda_fn(D):
        D = jnp.asarray(D)

        value = value_fn(D)
        p_success = p_success_fn(D)

        # Population-level innovation frequency that maximizes net
        # cultural growth at the current D.
        target_eta = compute_growth_optimal_eta(
            D,
            p_success,
            mu=mu,
            alpha=alpha,
            N=N,
        )

        # The policy probability required for target_eta to be a fixed
        # point once turnover is included:
        #
        # 0 = a(pi_req - target_eta) + mu(pi0 - target_eta)
        #
        # where a = (1 - mu)rho.
        safe_learning_weight = jnp.where(
            learning_weight > eps,
            learning_weight,
            1.0,
        )

        pi_required = (
            (learning_weight + mu) * target_eta - mu * pi0
        ) / safe_learning_weight

        # Evaluate the reward terms at the target innovation frequency.
        K = compute_K(target_eta, N=N)
        p_copy = get_p_copy(
            p_success,
            target_eta,
            copy_pool_size,
        )

        # For an interior required policy,
        #
        # beta * logit(pi_required)
        #     = r_innov - r_imit.
        #
        # Use float32-safe clipping only for evaluating the logarithm.
        pi_safe = jnp.clip(
            pi_required,
            eps,
            1.0 - eps,
        )

        required_log_odds = beta * (jnp.log(pi_safe) - jnp.log1p(-pi_safe))

        cost_gap = c_innov_fn(D) - c_imit_fn(D)

        numerator = required_log_odds + cost_gap + value * (p_copy - p_success)

        denominator = value * (p_copy + p_success * (K - 1.0))

        # Avoid division by zero at, for example, D=0 when V(D)=D.
        safe_denominator = jnp.where(
            denominator > eps,
            denominator,
            1.0,
        )

        lambda_unconstrained = numerator / safe_denominator
        lambda_interior = jnp.clip(
            lambda_unconstrained,
            0.0,
            1.0,
        )

        # If turnover implies pi_required <= 0, the requested target is
        # below the smallest behaviourally attainable frequency. Since
        # innovation probability increases with lambda, lambda=0 is the
        # closest feasible choice.
        #
        # Conversely, pi_required >= 1 requires the largest attainable
        # innovation probability, so lambda=1 is the closest choice.
        lambda_best_feasible = jnp.where(
            pi_required <= 0.0,
            0.0,
            jnp.where(
                pi_required >= 1.0,
                1.0,
                lambda_interior,
            ),
        )

        has_learning = learning_weight > eps
        has_lambda_leverage = denominator > eps
        inputs_are_finite = (
            jnp.isfinite(target_eta)
            & jnp.isfinite(pi_required)
            & jnp.isfinite(lambda_best_feasible)
        )

        # When reward learning has zero weight, or lambda has no effect
        # because the value denominator is zero, lambda is arbitrary.
        # Return zero as the neutral convention.
        valid = has_learning & has_lambda_leverage & inputs_are_finite & (beta > 0.0)

        return jnp.where(
            valid,
            lambda_best_feasible,
            0.0,
        )

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
        linewidths=2.5,
        linestyles=":",
    )

    return net_growth_grid


def annotate_phases(ax, panel_idx):
    params = [
        [(0.52, 0.42, -36), (0.48, 0.32, -36)],  # panel 0
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


def do_plots_without_trajectories():
    n_panels = len(p_success_fn_params)
    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for i, (p0, k) in enumerate(p_success_fn_params):
        p_success_fn = get_p_success_fn(p0, k)
        dD_fn, _, _ = get_delta_functions(p_success_fn)
        net_change_grid = plot_phase_diagram(axs[i], dD_fn, Ds, etas, sign_only=False)

        inset_ax = axs[i].inset_axes([0.63, 0.66, 0.31, 0.27])
        inset_ax.plot(
            np.asarray(Ds),
            np.asarray(p_success_fn(Ds)),
            color="black",
            linewidth=1.5,
        )
        inset_ax.set(
            xlim=(float(Ds[0]), float(Ds[-1])),
            ylim=(0.0, 1.0),
            xticks=(float(Ds[0]), float(Ds[-1])),
            yticks=(0.0, 1.0),
            xlabel="$D$",
            ylabel="$p_\\mathrm{success}(D)$",
        )
        inset_ax.set_facecolor("white")
        inset_ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        inset_ax.xaxis.label.set_size(8)
        inset_ax.yaxis.label.set_size(8)
        inset_ax.xaxis.labelpad = 0
        inset_ax.yaxis.labelpad = 0
        inset_ax.grid(False)
        for spine in inset_ax.spines.values():
            spine.set_color("#555555")
            spine.set_linewidth(0.7)
        inset_ax.set_in_layout(False)

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        axs[i].set(
            xlabel="Cultural complexity $D_t$",
            ylabel=ylabel,
            xlim=x_lims,
            ylim=y_lims,
            title=get_p_success_label(p0, k),
        )
        axs[i].grid(False)
        axs[i].tick_params(axis="both", which="both", length=0)
        sns.despine(ax=axs[i], left=True, bottom=True)

        if i == 1:
            for eta_val in [0.02, 0.1, 0.5]:
                # add horizontal dashed line from x=0 to x=the point where the line intersects the maintenance boundary
                eta_idx = np.argmin(np.abs(etas - eta_val))
                D_intersect_idx = np.argmin(np.abs(net_change_grid[:, eta_idx]))
                D_intersect = float(Ds[D_intersect_idx])
                axs[i].hlines(
                    y=eta_val,
                    xmin=0.0,
                    xmax=D_intersect,
                    colors="m",
                    linestyles="--",
                    linewidth=1.5,
                )
                axs[i].annotate(
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
                axs[i].scatter(
                    D_intersect,
                    eta_val,
                    color="m",
                    s=40,
                    zorder=5,
                )

        if i == 0:
            annotate_phases(axs[i], panel_idx=i)

    fig.suptitle(
        "Net rate of cultural change $\\frac{dD_t}{dt}$ as a function of cultural complexity $D_t$ and innovator frequency $\\eta_t$, for different $p_\\text{success}(D) = p_0e^{-kD}$",
        x=0.04,
        horizontalalignment="left",
        fontsize=15,
    )

    return fig


def do_plots_for_w_vals(w_vals, colors):
    n_panels = len(p_success_fn_params)
    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for i, (p0, k) in enumerate(p_success_fn_params):
        p_success_fn = get_p_success_fn(p0, k)
        dD_fn, _, _ = get_delta_functions(p_success_fn)
        plot_phase_diagram(axs[i], dD_fn, Ds, etas, sign_only=False)

        # plot nullclines for deta/dt
        for j, w in enumerate(w_vals):
            rho = compute_rho_from_w_and_mu(w)
            _, deta_fn, _ = get_delta_functions(p_success_fn, rho=rho)
            nullcline_vals = get_eta_nullcline_curve(deta_fn, Ds)
            nullcline_Ds, nullcline_vals, intersection = truncate_curve_at_D_boundary(
                Ds, nullcline_vals, dD_fn
            )
            sns.lineplot(
                x=nullcline_Ds,
                y=nullcline_vals,
                ax=axs[i],
                color=colors[j],
                linewidth=2.5,
                alpha=1.0,
            )
            if intersection is not None:
                axs[i].scatter(
                    intersection[0],
                    intersection[1],
                    color=colors[j],
                    linewidth=0.0,
                    s=60,
                    zorder=6,
                )

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        axs[i].set(
            xlabel="Cultural complexity $D_t$",
            ylabel=ylabel,
            xlim=x_lims,
            ylim=y_lims,
            title=get_p_success_label(p0, k),
        )
        axs[i].grid(False)
        axs[i].tick_params(axis="both", which="both", length=0)
        sns.despine(ax=axs[i], left=True, bottom=True)

        if i == 0:
            annotate_phases(axs[i], panel_idx=i)

    add_title_with_legend(
        fig,
        "Visualising population equilibria under different learning-turnover balances $w \\in [0, 1]$ (learning dominates turnover as $w \\to 1$)",
        "w",
        [f"{w:.1g}" for w in w_vals],
        colors,
    )

    return fig


def do_plots_for_c_vals(c_vals, colors):
    n_panels = len(p_success_fn_params)
    fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)

    for i, (p0, k) in enumerate(p_success_fn_params):
        p_success_fn = get_p_success_fn(p0, k)
        dD_fn, _, _ = get_delta_functions(p_success_fn)
        plot_phase_diagram(axs[i], dD_fn, Ds, etas, sign_only=False)

        # plot nullclines for deta/dt
        for j, c in enumerate(c_vals):
            c_innov_fn = lambda D: c * D
            _, deta_fn, _ = get_delta_functions(p_success_fn, c_innov_fn=c_innov_fn)
            nullcline_vals = get_eta_nullcline_curve(deta_fn, Ds)
            nullcline_Ds, nullcline_vals, intersection = truncate_curve_at_D_boundary(
                Ds, nullcline_vals, dD_fn
            )
            sns.lineplot(
                x=nullcline_Ds,
                y=nullcline_vals,
                ax=axs[i],
                color=colors[j],
                linewidth=2.5,
                alpha=1.0,
            )
            if intersection is not None:
                axs[i].scatter(
                    intersection[0],
                    intersection[1],
                    color=colors[j],
                    linewidth=0.0,
                    s=60,
                    zorder=6,
                )

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        axs[i].set(
            xlabel="Cultural complexity $D_t$",
            ylabel=ylabel,
            xlim=x_lims,
            ylim=y_lims,
            title=get_p_success_label(p0, k),
        )
        axs[i].grid(False)
        axs[i].tick_params(axis="both", which="both", length=0)
        sns.despine(ax=axs[i], left=True, bottom=True)

        if i == 0:
            annotate_phases(axs[i], panel_idx=i)

    add_title_with_legend(
        fig,
        "A. Example RL-induced population trajectories under different cost disparities $c_\\text{innov}(D) - c_\\text{imit}(D) = cD$",
        "c",
        [f"{c:.1g}" for c in c_vals],
        colors,
    )

    return fig


def do_plots_for_lambda_fns(all_lambda_fns, colors, styles=None, widths=None):
    n_panels = len(p_success_fn_params)
    if len(all_lambda_fns) != n_panels:
        raise ValueError(
            "Expected one list of lambda functions per phase-diagram panel, "
            f"but received {len(all_lambda_fns)} lists for {n_panels} panels."
        )

    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)

    for i, (p0, k) in enumerate(p_success_fn_params):
        lambda_fns = all_lambda_fns[i]
        if not all(callable(lambda_fn) for lambda_fn in lambda_fns):
            raise TypeError(f"All lambda functions for panel {i} must be callable.")

        p_success_fn = get_p_success_fn(p0, k)
        dD_fn, _, _ = get_delta_functions(p_success_fn)
        plot_phase_diagram(axs[i], dD_fn, Ds, etas, sign_only=False)

        # plot nullclines for deta/dt
        for j, lambda_fn in enumerate(lambda_fns):
            _, deta_fn, _ = get_delta_functions(p_success_fn, lambda_fn=lambda_fn)
            nullcline_vals = get_eta_nullcline_curve(deta_fn, Ds)
            nullcline_Ds, nullcline_vals, intersection = truncate_curve_at_D_boundary(
                Ds, nullcline_vals, dD_fn
            )
            sns.lineplot(
                x=nullcline_Ds,
                y=nullcline_vals,
                ax=axs[i],
                color=colors[j],
                linewidth=widths[j] if widths else 2.5,
                linestyle=styles[j] if styles else "-",
                alpha=1.0,
            )
            if intersection is not None:
                axs[i].scatter(
                    intersection[0],
                    intersection[1],
                    color=colors[j],
                    linewidth=0.0,
                    s=60,
                    zorder=6,
                )

        ylabel = "Innovator frequency $\\eta_t$" if i == 0 else None
        x_lims, y_lims = compute_plot_lims(Ds, etas)
        axs[i].set(
            xlabel="Cultural complexity $D_t$",
            ylabel=ylabel,
            xlim=x_lims,
            ylim=y_lims,
            title=get_p_success_label(p0, k),
        )
        axs[i].grid(False)
        axs[i].tick_params(axis="both", which="both", length=0)
        sns.despine(ax=axs[i], left=True, bottom=True)

        if i == 0:
            annotate_phases(axs[i], panel_idx=i)

    add_title_with_legend(
        fig,
        "Visualising population equilibria under different value-capture regimes for four example systems",
        "\\lambda",
        [f"{l:.1f}" for l in fixed_lambdas] + ["\\lambda^\\star(D)"],
        colors,
        ["-"] * len(fixed_lambdas) + ["--"],
    )

    return fig


def main():
    # do plots with no trajectories (phase diagrams only)
    fig = do_plots_without_trajectories()
    save_fig(fig, "phase-diagrams", subfolder="new")

    # # do plots for just lambda=0 (baseline case) with different learning rates
    # w_vals = [0.0, 0.5, 0.9, 1.0]
    # w_colours = get_palette(len(w_vals))
    # fig = do_plots_for_w_vals(w_vals, w_colours)
    # save_fig(fig, "w_vals", subfolder="theoretical/phase_diagrams")

    # # do plots for just lambda=0 (baseline case) with different innovation costs
    # c_vals = [0.0, 0.01, 0.1, 1.0]
    # c_colours = get_palette(len(c_vals))
    # fig = do_plots_for_c_vals(c_vals, c_colours)
    # save_fig(fig, "c_vals", subfolder="new")

    # # do plots for fixed lambda values
    # fixed_lambda_fns = [
    #     [lambda D, l=l: l for l in fixed_lambdas] for _ in range(len(p_success_fns))
    # ]
    # fixed_lambda_colors = get_palette(len(fixed_lambdas))
    # fig = do_plots_for_lambda_fns(fixed_lambda_fns, fixed_lambda_colors)
    # save_fig(fig, "fixed_lambda", subfolder="theoretical/phase_diagrams")

    # # do plots for optimal variable lambda
    # variable_lambda_fns = [
    #     [lambda D: 0.0, get_optimal_lambda_fn(p_success_fn)]
    #     for p_success_fn, _ in p_success_fns.values()
    # ]
    # fig = do_plots_for_lambda_fns(variable_lambda_fns, variable_lambda_colors)
    # save_fig(fig, "optimal_lambda", subfolder="theoretical/phase_diagrams")

    # # combined
    # combined_lambda_fns = [
    #     [lambda D, l=l: l for l in fixed_lambdas]
    #     + [get_optimal_lambda_fn(get_p_success_fn(p0, k))]
    #     for p0, k in p_success_fn_params
    # ]
    # colors = fixed_lambda_colors + ["blue"]
    # styles = ["-"] * len(fixed_lambdas) + ["--"]
    # widths = [2] * len(fixed_lambdas) + [2.5]
    # fig = do_plots_for_lambda_fns(combined_lambda_fns, colors, styles, widths)
    # save_fig(fig, "combined_lambda", subfolder="theoretical/phase_diagrams")


if __name__ == "__main__":
    main()
