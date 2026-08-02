import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns

from src.utils import save_fig
from utils import add_title_with_legend
from phase_diagrams import get_palette

sns.set_theme(style="whitegrid")

N_values = [10, 100, 1000]
rho_values = [0.02, 0.05, 0.1, 1.0]
p_success_fns = {  # label -> function
    "$p_{success}(D) = 0.5$": lambda D: jnp.full_like(D, 0.5),
    "$p_{success}(D) = 0.1$": lambda D: jnp.full_like(D, 0.1),
    "$p_{success}(D) = 1/(D+1)$": lambda D: 1 / (D + 1),
    "$p_{success}(D) = \\exp(-D)$": lambda D: jnp.exp(-D / 5),
}
etas = jnp.linspace(0.0, 1.0, 1000)
Ds = jnp.linspace(0.0, 100.0, 1000)

v_fn = lambda D: D
c_innov_fn = lambda D: 0.0
c_imit_fn = lambda D: 0.0
beta = 1.0
mu = 0.01
pi_0 = 0.5
M = 100

rp_present_colour = "red"
rp_absent_colour = "blue"
neutral_colour = "#ffffff"
delta_r_cmap = LinearSegmentedColormap.from_list(
    "green_orange", [rp_absent_colour, neutral_colour, rp_present_colour], N=256
)
delta_eta_cmap = LinearSegmentedColormap.from_list(
    "orange_green",
    [rp_present_colour, neutral_colour, rp_absent_colour],
    N=256,
)


def get_delta_r_function(p_fn):
    def delta_r_fn(D, eta):
        p = p_fn(D)
        p_copy = 1 - ((1 - p) ** (M * eta))
        return ((p_copy - p) * v_fn(D)) + c_innov_fn(D) - c_imit_fn(D)

    return delta_r_fn


def get_delta_eta_function(delta_r_fn, rho):
    def eta_star_fn(D, eta):
        return 1 / (1 + jnp.exp(delta_r_fn(D, eta) / beta))

    def delta_eta_fn(D, eta):
        eta_star = eta_star_fn(D, eta)
        rho_ = (1 - mu) * rho
        return (rho_ * (eta_star - eta)) + (mu * (pi_0 - eta))

    return delta_eta_fn


def compute_heatmap_values(fn):
    values = jax.vmap(lambda D: jax.vmap(lambda eta: fn(D, eta))(etas))(Ds)
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, np.nan)


def get_shared_centered_norm(heatmaps):
    finite_abs_values = [np.abs(values[np.isfinite(values)]) for values in heatmaps]
    finite_abs_values = [values for values in finite_abs_values if values.size > 0]

    if not finite_abs_values:
        max_abs_value = 1.0
    else:
        max_abs_value = max(values.max() for values in finite_abs_values)
        if max_abs_value == 0:
            max_abs_value = 1.0

    return TwoSlopeNorm(vmin=-max_abs_value, vcenter=0.0, vmax=max_abs_value)


def add_zero_boundary_curve(ax, values):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return
    if finite_values.min() >= 0 or finite_values.max() <= 0:
        return

    ax.contour(
        np.asarray(Ds),
        np.asarray(etas),
        values.T,
        levels=[0.0],
        colors="black",
        linewidths=2,
        linestyles=":",
    )


def remove_grids_fully(ax):
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0)
    sns.despine(ax=ax, left=True, bottom=True)


def plot_heatmaps(heatmap_fn_factory, cmap, sign_only=False):
    n_panels = len(p_success_fns)
    fig, axs = plt.subplots(
        1,
        n_panels,
        figsize=(3.5 * n_panels, 3.5),
        sharey=True,
    )

    heatmaps = []
    for p_success_label, p_success_fn in p_success_fns.items():
        heatmap_fn = heatmap_fn_factory(p_success_fn)
        values = compute_heatmap_values(heatmap_fn)
        if sign_only:
            values = np.sign(values)
        heatmaps.append((p_success_label, values))

    extent = [
        float(Ds.min()),
        float(Ds.max()),
        float(etas.min()),
        float(etas.max()),
    ]

    norm = get_shared_centered_norm([values for _, values in heatmaps])
    image = None
    for i, (p_success_label, values) in enumerate(heatmaps):
        image = axs[i].imshow(
            values.T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        # if not sign_only:
        #     add_zero_boundary_curve(axs[i], values)

        axs[i].set(
            xlabel="$D$", ylabel="$\\eta$" if i == 0 else None, title=p_success_label
        )
        remove_grids_fully(axs[i])

    # fig.subplots_adjust(right=0.86, wspace=0.16, hspace=0.18)
    # cbar_ax = fig.add_axes([0.88, 0.18, 0.02, 0.64])
    # fig.colorbar(image, cax=cbar_ax, label="$\\partial \\Delta r / \\partial D$")

    return fig, axs


def plot_eta_nullclines():
    n_panels = len(p_success_fns)
    fig, axs = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5), sharey=True)
    colours = get_palette(len(rho_values))
    for i, (p_success_label, p_success_fn) in enumerate(p_success_fns.items()):
        for j, rho in enumerate(rho_values):
            delta_r_fn = get_delta_r_function(p_fn=p_success_fn)
            delta_eta_fn = get_delta_eta_function(delta_r_fn=delta_r_fn, rho=rho)
            delta_eta_vals = compute_heatmap_values(delta_eta_fn)
            axs[i].contour(
                np.asarray(Ds),
                np.asarray(etas),
                delta_eta_vals.T,
                levels=[0.0],
                colors=colours[j],
                linewidths=2.0,
                linestyles="-",
            )
        axs[i].set(
            xlabel="$D$", ylabel="$\\eta$" if i == 0 else None, title=p_success_label
        )

    add_title_with_legend(
        fig,
        "Nullclines of $\\frac{d\\eta}{dt}$ for $\\mu="
        + str(mu)
        + ", \\pi_0="
        + str(pi_0)
        + "$",
        "\\rho",
        rho_values,
        colours,
    )
    save_fig(
        fig,
        "eta_nullclines",
        subfolder="theoretical/scratch",
    )


def main():
    # plot heatmaps for \Delta
    heatmap_fn_factory = lambda p_success_fn: get_delta_r_function(p_fn=p_success_fn)
    fig, axs = plot_heatmaps(heatmap_fn_factory, delta_r_cmap, sign_only=False)
    fig.suptitle("$\\Delta(D, \\eta)$ (red: positive, blue: negative)", fontsize=14)
    save_fig(
        fig,
        "delta_r_heatmaps",
        subfolder="theoretical/scratch",
    )

    # plot heatmaps for \frac{\partial\Delta}{\partial D}
    heatmap_fn_factory = lambda p_success_fn: jax.grad(
        get_delta_r_function(p_fn=p_success_fn),
        argnums=0,
    )
    fig, axs = plot_heatmaps(heatmap_fn_factory, delta_r_cmap, sign_only=True)
    fig.suptitle(
        "Sign of $\\frac{\\partial \\Delta}{\\partial D}$ (red: positive, blue: negative)",
        fontsize=14,
    )
    save_fig(
        fig,
        "delta_r_derivative_heatmaps",
        subfolder="theoretical/scratch",
    )

    # plot heatmaps for delta_eta
    heatmap_fn_factory = lambda p_success_fn: get_delta_eta_function(
        delta_r_fn=get_delta_r_function(p_fn=p_success_fn), rho=1.0
    )
    fig, axs = plot_heatmaps(heatmap_fn_factory, delta_eta_cmap, sign_only=True)
    fig.suptitle(
        "Sign of $\\frac{d\\eta_t}{dt}$ (red: negative, blue: positive)", fontsize=14
    )
    save_fig(
        fig,
        "delta_eta_heatmaps",
        subfolder="theoretical/scratch",
    )

    plot_eta_nullclines()


if __name__ == "__main__":
    main()
