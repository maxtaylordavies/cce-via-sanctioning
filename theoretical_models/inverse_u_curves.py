import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_fig
from utils import simulate_trajectory
from constants import *
from phase_diagrams import (
    p_success_fns,
    get_delta_functions,
    get_optimal_lambda_fn,
    fixed_lambdas,
    get_palette,
)

fixed_lambda_colors = get_palette(len(fixed_lambdas))

sns.set_theme(style="whitegrid")

lambdas = np.linspace(0.0, 1.0, 200)
simulation_T = 1000
simulation_dt = 0.1


def simulate_lambda_fn(lambda_fn, p_success_fn, D_max):
    dD_fn_, deta_fn_, _ = get_delta_functions(p_success_fn, lambda_fn=lambda_fn)
    delta_fns = (dD_fn_, deta_fn_)
    traj = simulate_trajectory(
        delta_fns, D0=0.0, eta0=0.5, T=simulation_T, dt=simulation_dt, D_max=D_max
    )
    return traj[-1, 0], traj[:, 1].mean()


def simulate_fixed_lambda(lambda_value, p_success_fn, D_max):
    lambda_fn = lambda D: lambda_value
    return simulate_lambda_fn(lambda_fn, p_success_fn, D_max)


def simulate_optimal_lambda(p_success_fn, D_max):
    lambda_fn = get_optimal_lambda_fn(p_success_fn)
    return simulate_lambda_fn(lambda_fn, p_success_fn, D_max)


def main():
    n_panels = len(p_success_fns)
    D_fig, D_axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    eta_fig, eta_axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    for i, (p_success_label, (p_success_fn, D_max)) in enumerate(p_success_fns.items()):
        vmapped_sim = jax.vmap(simulate_fixed_lambda, in_axes=(0, None, None))
        Ds, eta_bars = vmapped_sim(lambdas, p_success_fn, D_max)
        D_star, _ = simulate_optimal_lambda(p_success_fn, D_max)
        D_axs[i].axhline(D_star, color="blue", linestyle="--", linewidth=2.5)
        sns.lineplot(
            x=lambdas,
            y=Ds,
            ax=D_axs[i],
            color="black",
            # marker="o",
            linewidth=2.0,
            # markersize=5,
        )

        reference_Ds, _ = vmapped_sim(fixed_lambdas, p_success_fn, D_max)
        for j, (lambda_value, D_ref) in enumerate(zip(fixed_lambdas, reference_Ds)):
            D_axs[i].scatter(
                lambda_value,
                D_ref,
                color=fixed_lambda_colors[j],
                s=50,
                zorder=5,
            )

        D_axs[i].set(
            xlabel="Innovator appropriation share $\\lambda$",
            ylabel="Final cultural complexity $D_T$",
            title=p_success_label,
        )
        sns.despine(ax=D_axs[i], left=True, bottom=True)

        sns.lineplot(
            x=lambdas,
            y=eta_bars,
            ax=eta_axs[i],
            color="black",
            # marker="o",
            linewidth=2.0,
            # markersize=5,
        )
        eta_axs[i].set(
            xlabel="Innovator appropriation share $\\lambda$",
            ylabel="Average innovator frequency $\\bar{\\eta}$",
            title=p_success_label,
        )
        sns.despine(ax=eta_axs[i], left=True, bottom=True)

    D_fig.suptitle(
        "Final cultural complexity follows an inverse-U curve with respect to innovator appropriation share $\\lambda$",
        fontsize=15,
    )

    save_fig(
        D_fig,
        "final_cultural_complexity_vs_lambda",
        subfolder="theoretical",
    )
    save_fig(
        eta_fig,
        "average_innovator_frequency_vs_lambda",
        subfolder="theoretical",
    )


if __name__ == "__main__":
    main()
