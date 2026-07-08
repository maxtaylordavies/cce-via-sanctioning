from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.utils import save_fig
from utils import progression_rate

sns.set_style("whitegrid")

N = 100
beta = 1.0


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


def main():
    fig = visualise_progression_rate_curves()
    save_fig(
        fig,
        "progression_rate_curves",
        subfolder="theoretical",
        fmts=["png"],
    )


if __name__ == "__main__":
    main()
