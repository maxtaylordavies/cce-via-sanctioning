from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from utils import save_fig

sns.set_style("whitegrid")

DATA_DIR = Path("data/new")
FORAGING_LEVEL_CMAP = LinearSegmentedColormap.from_list(
    "foraging_level", ["#d8f0c8", "#1d5c2f"]
)


def load_raw_outputs():
    scalar_keys = {
        "T",
        "T_main",
        "T_extra",
        "grid_length",
        "num_rules_in_initial_library",
        "empty_recipe_id",
        "role_innovate",
        "role_imitate",
    }
    outputs = {}
    concat_buffers = {}

    for path in sorted(DATA_DIR.glob("*.npz")):
        file_outputs = np.load(path, allow_pickle=True)
        for key in file_outputs.files:
            value = file_outputs[key]
            if key in scalar_keys:
                if key not in outputs:
                    outputs[key] = value
                elif outputs[key] != value:
                    raise ValueError(
                        f"Inconsistent scalar value for {key!r} in {path.name}"
                    )
                continue

            if key == "fees":
                if key not in outputs:
                    outputs[key] = value
                elif not np.array_equal(outputs[key], value):
                    raise ValueError(f"Inconsistent fee grid in {path.name}")
                continue

            concat_buffers.setdefault(key, []).append(value)

    for key, values in concat_buffers.items():
        outputs[key] = np.concatenate(values, axis=0)

    return outputs


def load_data():
    return {
        "population": pd.read_csv(DATA_DIR / "population_data.csv"),
        "agent": pd.read_csv(DATA_DIR / "agent_data.csv"),
        "recipe_lineage": pd.read_csv(DATA_DIR / "recipe_lineage_data.csv"),
        "recipe_descendant": pd.read_csv(DATA_DIR / "recipe_descendant_data.csv"),
        "recipe_recombination": pd.read_csv(DATA_DIR / "recipe_recombination_data.csv"),
        "similarity_matrices": 1 - np.load(DATA_DIR / "jaccard_matrices.npy"),
        "raw_outputs": load_raw_outputs(),
    }


def get_fee_plotting_config(recipe_lineage_data):
    fee_key_order = np.sort(recipe_lineage_data["log10_fee_value"].unique())
    fee_label_map = {fee: f"{fee:.1f}" for fee in fee_key_order}
    fee_label_order = [fee_label_map[fee] for fee in fee_key_order]
    fee_palette = dict(
        zip(
            fee_label_order, sns.color_palette("viridis", n_colors=len(fee_label_order))
        )
    )
    return fee_label_map, fee_label_order, fee_palette


def add_fee_plot_column(df, fee_label_map, fee_label_order):
    if "log10_fee_value" not in df.columns:
        df["log10_fee_value"] = np.round(np.log10(df["fee"]), 3)
    df["fee_plot"] = 10.0 ** df["log10_fee_value"]
    df["log10_fee"] = pd.Categorical(
        df["log10_fee_value"].map(fee_label_map),
        categories=fee_label_order,
        ordered=True,
    )


def format_time_series_fee_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = []
    for label in labels:
        try:
            formatted_labels.append(f"{np.log10(float(label)):.1f}")
        except ValueError:
            formatted_labels.append(label)
    ax.legend(handles, formatted_labels, title="log10(fee)")


def box_and_strip_plot(
    df, y, ax, title, fee_label_order, fee_palette, y_jitter=None, strip_kws=None
):
    plot_df = df.copy()
    y_ = y
    if y_jitter is not None:
        y_ = y + "_jittered"
        plot_df[y_] = plot_df[y] + np.random.normal(0, y_jitter, size=len(plot_df))

    default_strip_kws = {"alpha": 0.05, "size": 3, "dodge": False, "jitter": True}
    if strip_kws is None:
        strip_kws = {}
    strip_kws = {**default_strip_kws, **strip_kws}

    sns.stripplot(
        data=plot_df,
        x="log10_fee",
        y=y_,
        order=fee_label_order,
        hue="log10_fee",
        hue_order=fee_label_order,
        palette=fee_palette,
        ax=ax,
        **strip_kws,
    )
    sns.boxplot(
        data=plot_df,
        x="log10_fee",
        y=y,
        order=fee_label_order,
        palette=fee_palette,
        showfliers=False,
        width=0.7,
        boxprops={"edgecolor": "black", "zorder": 3},
        whiskerprops={"color": "black", "zorder": 3},
        capprops={"color": "black", "zorder": 3},
        medianprops={"color": "black", "zorder": 3},
        ax=ax,
    )

    for patch, fee_label in zip(ax.patches[: len(fee_label_order)], fee_label_order):
        r, g, b = fee_palette[fee_label]
        patch.set_facecolor((r, g, b, 0.3))
        patch.set_edgecolor("black")

    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_title(title)
    ax.set_xlabel("log10(fee)")
    sns.despine(ax=ax, left=True, bottom=True)


def format_p_value(p_value):
    if np.isnan(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def mutual_information_agent_to_prob(values):
    values = np.asarray(values)
    if values.size == 0:
        return np.nan
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    mi = float(-(probs * np.log2(probs)).sum())
    max_mi = np.log2(values.size)
    if max_mi <= 0:
        return np.nan
    return mi / max_mi


def annotate_regression(ax, df, x, y):
    fit_df = df[[x, y]].dropna()
    if len(fit_df) < 3 or fit_df[x].nunique() < 2:
        text = "b = n/a\np = n/a"
    else:
        X = sm.add_constant(fit_df[x])
        model = sm.OLS(fit_df[y], X).fit()
        text = f"b = {model.params[x]:.3f}\np = {format_p_value(model.pvalues[x])}"

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def plot_yield_relationship(ax, df, x, fee_label_order, legend=False):
    sns.scatterplot(
        data=df,
        x=x,
        y="yield",
        hue="log10_fee",
        hue_order=fee_label_order,
        palette="viridis",
        s=80,
        ax=ax,
        legend=legend,
    )
    sns.regplot(
        data=df,
        x=x,
        y="yield",
        scatter=False,
        ci=None,
        line_kws={"color": "red", "alpha": 0.8},
        ax=ax,
    )
    annotate_regression(ax, df, x, "yield")
    sns.despine(ax=ax, left=True, bottom=True)


def build_recipe_plot_data(
    population_data,
    recipe_lineage_data,
    recipe_descendant_data,
    recipe_recombination_data,
    fee_label_map,
    fee_label_order,
):
    recipe_key_cols = ["seed", "log10_fee_value", "recipe_id"]
    extant_recipe_keys = recipe_lineage_data.loc[
        recipe_lineage_data["recipe_length"] >= 1, recipe_key_cols
    ].drop_duplicates()
    extant_recipe_lineage_data = recipe_lineage_data.merge(
        extant_recipe_keys, on=recipe_key_cols, how="inner"
    )

    for df in (
        extant_recipe_lineage_data,
        recipe_descendant_data,
        recipe_recombination_data,
        population_data,
    ):
        add_fee_plot_column(df, fee_label_map, fee_label_order)

    recipe_summary = (
        extant_recipe_lineage_data.groupby(["seed", "log10_fee_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "mean_n_innovation_events": x["n_innovation_events"].mean(),
                    "mean_n_unique_innovators": x["n_unique_innovators"].mean(),
                    "mean_recipe_length": x["recipe_length"].mean(),
                    "mean_recipe_age": x["recipe_age"].mean(),
                    "mean_recipe_ancestor_age": x["recipe_ancestor_age"].mean(),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(recipe_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_data.groupby(["seed", "log10_fee_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() * 0.9, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    fee_outcome_summary = recipe_summary.merge(
        final_population_data,
        on=["seed", "log10_fee_value", "fee_plot", "log10_fee"],
    )

    recipe_descendant_summary_non_extant = (
        recipe_descendant_data[recipe_descendant_data["is_extant"] == 0]
        .groupby(["seed", "log10_fee_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "max_n_unique_extant_descendants": x[
                        "n_unique_extant_descendants"
                    ].max(),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(
        recipe_descendant_summary_non_extant, fee_label_map, fee_label_order
    )
    descendant_outcome_summary_non_extant = recipe_descendant_summary_non_extant.merge(
        final_population_data,
        on=["seed", "log10_fee_value", "fee_plot", "log10_fee"],
    )

    recombination_summary = (
        recipe_recombination_data.groupby(["seed", "log10_fee_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "n_innovation_events": len(x),
                    "n_recombination_v2": x["is_recombination_v2"].sum(),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(recombination_summary, fee_label_map, fee_label_order)
    recombination_outcome_summary = recombination_summary.merge(
        final_population_data,
        on=["seed", "log10_fee_value", "fee_plot", "log10_fee"],
    )

    return (
        extant_recipe_lineage_data,
        fee_outcome_summary,
        recipe_descendant_summary_non_extant,
        descendant_outcome_summary_non_extant,
        recombination_summary,
        recombination_outcome_summary,
    )


def build_specialisation_plot_data(
    agent_data, population_data, fee_label_map, fee_label_order
):
    agent_df = agent_data.copy()
    population_df = population_data.copy()
    add_fee_plot_column(agent_df, fee_label_map, fee_label_order)
    add_fee_plot_column(population_df, fee_label_map, fee_label_order)

    agent_df["innov_prob"] = 1 - agent_df["role"]
    specialisation_summary = (
        agent_df.groupby(["seed", "log10_fee_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "specialisation": mutual_information_agent_to_prob(
                        x["innov_prob"].to_numpy()
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(specialisation_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_df.groupby(["seed", "log10_fee_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() * 0.9, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    specialisation_outcome_summary = specialisation_summary.merge(
        final_population_data,
        on=["seed", "log10_fee_value", "fee_plot", "log10_fee"],
    )
    return specialisation_summary, specialisation_outcome_summary


def build_whole_history_specialisation_plot_data(
    raw_outputs, population_data, fee_label_map, fee_label_order
):
    seeds = raw_outputs["seeds"]
    fees = raw_outputs["fees"]
    role_innovate = int(raw_outputs["role_innovate"])
    innov_probs = (raw_outputs["agent_roles"] == role_innovate).mean(axis=2)

    rows = []
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            per_slot_probs = innov_probs[seed_idx, fee_idx]
            rows.append(
                {
                    "seed": int(seed),
                    "fee": float(fee),
                    "specialisation_whole_history": mutual_information_agent_to_prob(
                        per_slot_probs
                    ),
                }
            )

    whole_history_summary = pd.DataFrame(rows)
    add_fee_plot_column(whole_history_summary, fee_label_map, fee_label_order)

    population_df = population_data.copy()
    add_fee_plot_column(population_df, fee_label_map, fee_label_order)
    final_population_data = (
        population_df.groupby(["seed", "log10_fee_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() * 0.9, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    whole_history_outcome_summary = whole_history_summary.merge(
        final_population_data,
        on=["seed", "log10_fee_value", "fee_plot", "log10_fee"],
    )
    return whole_history_summary, whole_history_outcome_summary


def plot_performance_metrics(pop_df):
    plot_df = pop_df.copy()
    for metric in ["r_innov", "r_imit", "yield", "yield_gini"]:
        plot_df[metric] = plot_df.groupby(["fee", "seed"])[metric].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

    plot_df = plot_df[
        (plot_df["t"] < plot_df["t"].max() - 500)
        & (plot_df["t"] % 20 == 0)
        & (plot_df["fee"] < plot_df["fee"].max())
    ]

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    axs = axs.flatten()
    palette = sns.color_palette("husl", n_colors=plot_df["fee"].nunique())
    titles = {
        "level": "Average level of foraged plants",
        "yield": "Average yield",
        "yield_gini": "Yield inequality (Gini)",
        "r_innov": "Average reward for innovating",
        "r_imit": "Average reward for imitating",
        "prop_innov": "Average proportion of innovators",
    }

    for i, metric in enumerate(titles):
        sns.lineplot(
            plot_df,
            x="t",
            y=metric,
            hue="fee",
            palette=palette,
            legend=i == 0,
            ax=axs[i],
        )
        axs[i].set_title(titles[metric])
        sns.despine(ax=axs[i], left=True, bottom=True)
        if i == 0:
            format_time_series_fee_legend(axs[i])

    save_fig(fig, "performance_metrics_over_time")
    return plot_df


def plot_final_performance_metrics(pop_df):
    final_pop_df = (
        pop_df.groupby(["fee", "seed"])
        .apply(lambda x: x[x["t"] >= x["t"].max() - 500].mean())
        .reset_index()
        .drop(columns=["t"])
    )

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    axs = axs.flatten()
    titles = {
        "level": "Average level of foraged plants",
        "yield": "Average yield",
        "yield_gini": "Yield inequality (Gini)",
        "prop_innov": "Average proportion of innovators",
    }
    metric_groups = [
        ("level",),
        ("yield",),
        ("yield_gini",),
        ("r_innov", "r_imit"),
        ("prop_innov",),
    ]

    for i, metric_group in enumerate(metric_groups):
        if len(metric_group) == 2:
            for metric, color in zip(metric_group, ["red", "blue"]):
                sns.lineplot(
                    final_pop_df,
                    x="fee",
                    y=metric,
                    marker="o",
                    color=color,
                    err_style="bars",
                    label=metric,
                    ax=axs[i],
                )
            axs[i].set_title("Final average reward by role")
            axs[i].legend()
        else:
            metric = metric_group[0]
            sns.lineplot(
                final_pop_df,
                x="fee",
                y=metric,
                marker="o",
                color="black",
                err_style="bars",
                ax=axs[i],
            )
            axs[i].set_title(f"Final {titles[metric].lower()}")
        axs[i].set_xscale("log", base=10)
        sns.despine(ax=axs[i], left=True, bottom=True)

    for ax in axs[len(metric_groups) :]:
        ax.set_visible(False)

    save_fig(fig, "final_performance_metrics")


def plot_agent_heatmaps(agent_df):
    plot_df = agent_df.copy()
    plot_df["row"] = plot_df["agent_idx"] // 10
    plot_df["col"] = plot_df["agent_idx"] % 10
    plot_df["innov_freq"] = 1 - plot_df["role"]

    per_agent_metrics = {
        "innov_freq": "Frequency of innovation attempts",
        "level": "Average foraging level",
        "yield": "Average yield",
    }
    fees = np.sort(plot_df["fee"].unique())

    for seed in np.sort(plot_df["seed"].unique()):
        seed_df = plot_df[plot_df["seed"] == seed]
        fig, axs = plt.subplots(
            len(per_agent_metrics),
            len(fees) + 1,
            figsize=(len(fees) * 3 + 1.0, 2.5 * len(per_agent_metrics)),
            gridspec_kw={"width_ratios": [1] * len(fees) + [0.08]},
        )
        heatmap_axs = axs[:, :-1]
        cbar_axs = axs[:, -1]

        metric_ranges = {
            metric: (seed_df[metric].min(), seed_df[metric].max())
            for metric in per_agent_metrics
        }

        for col, fee in enumerate(fees):
            fee_df = seed_df[seed_df["fee"] == fee]
            for row, metric in enumerate(per_agent_metrics):
                pivot = fee_df.pivot_table(
                    index="row", columns="col", values=metric, aggfunc="mean"
                )
                vmin, vmax = metric_ranges[metric]
                cmap = (
                    "coolwarm"
                    if metric == "innov_freq"
                    else FORAGING_LEVEL_CMAP if metric == "level" else "viridis"
                )
                sns.heatmap(
                    pivot,
                    ax=heatmap_axs[row, col],
                    cbar=col == len(fees) - 1,
                    cbar_ax=cbar_axs[row] if col == len(fees) - 1 else None,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                if row == 0:
                    heatmap_axs[row, col].set_title(
                        "Fee = $10^{" + f"{np.log10(fee):.1f}" + "}$", fontsize=16
                    )
                heatmap_axs[row, col].set_ylabel(
                    per_agent_metrics[metric] if col == 0 else None, fontsize=12
                )
                heatmap_axs[row, col].set_xlabel(None)
                heatmap_axs[row, col].set_xticks([])
                heatmap_axs[row, col].set_yticks([])

        save_fig(fig, f"seed_{seed}", subfolder="agent_heatmaps")


def plot_similarity(similarity_matrices, fees):
    mean_similarity = np.mean(similarity_matrices, axis=0)
    n_fees = mean_similarity.shape[0]
    grid_length = int(np.sqrt(mean_similarity.shape[1]))

    fig, axs = plt.subplots(2, n_fees, figsize=(n_fees * 4, 6), sharey="row")
    for fee_idx in range(n_fees):
        sim_mat = mean_similarity[fee_idx]
        distance_to_sims = defaultdict(list)
        for i in range(sim_mat.shape[0]):
            for j in range(sim_mat.shape[1]):
                if i == j:
                    continue
                pos_i = np.unravel_index(i, (grid_length, grid_length))
                pos_j = np.unravel_index(j, (grid_length, grid_length))
                dist = np.sum(np.abs(np.array(pos_i) - np.array(pos_j)))
                distance_to_sims[dist].append(sim_mat[i, j])

        dists = np.array(sorted(distance_to_sims))
        mean_sims = np.array([np.mean(distance_to_sims[d]) for d in dists])
        sns.lineplot(
            x=dists, y=mean_sims, marker="o", color="black", ax=axs[0, fee_idx]
        )
        sns.despine(ax=axs[0, fee_idx], left=True, bottom=True)
        axs[0, fee_idx].set(
            xlabel="Distance between agents",
            ylabel="Average library similarity",
            title="Fee = $10^{" + f"{np.log10(fees[fee_idx]):.1f}" + "}$",
        )

        axs[1, fee_idx].imshow(sim_mat, vmin=0, vmax=1)
        axs[1, fee_idx].axis("off")

    save_fig(fig, "jaccard_similarity")


def plot_recipe_level_metrics(
    extant_recipe_lineage_data,
    fee_outcome_summary,
    fee_label_order,
    fee_palette,
):
    metrics = [
        ("recipe_length", "recipe length", "length"),
        ("n_innovation_events", "# innovation events in recipe history", "count"),
        ("recipe_age", "timesteps since recipe created", "timesteps"),
        ("n_unique_innovators", "# unique innovators in recipe history", "count"),
        ("recipe_ancestor_age", "timesteps since oldest ancestor created", "timesteps"),
    ]
    fig, axs = plt.subplots(2, len(metrics), figsize=(4 * len(metrics), 8))
    for ax in axs[1, 1:]:
        ax.sharey(axs[1, 0])

    for i, (metric, title, label) in enumerate(metrics):
        box_and_strip_plot(
            extant_recipe_lineage_data,
            metric,
            axs[0, i],
            title,
            fee_label_order,
            fee_palette,
            y_jitter=[None, None, 0.25, None, None][i],
        )
        axs[0, i].set(ylabel=label)
        plot_yield_relationship(
            axs[1, i], fee_outcome_summary, f"mean_{metric}", fee_label_order, i == 0
        )
        axs[1, i].set(xlabel=f"{label} (mean)", ylabel=None)
        if i != 0:
            axs[1, i].set(yticklabels=[])

    axs[1, 0].set_ylabel("Final population yield")
    fig.suptitle("Analysis of recipes present at end of simulation", fontsize=16)
    save_fig(fig, "recipe_level_metrics")


def plot_population_level_metrics(
    recombination_summary,
    recombination_outcome_summary,
    recipe_descendant_summary_non_extant,
    descendant_outcome_summary_non_extant,
    fee_label_order,
    fee_palette,
):
    metrics = [
        (
            "n_innovation_events",
            "# innovation events\nover population history",
            "count",
        ),
        (
            "n_recombination_v2",
            "# recombination events\nover population history",
            "count",
        ),
        (
            "max_n_unique_extant_descendants",
            "max # unique surviving descendants\nover non-surviving recipes",
            "count",
        ),
    ]
    fig, axs = plt.subplots(2, len(metrics), figsize=(4 * len(metrics), 8))
    for ax in axs[1, 1:]:
        ax.sharey(axs[1, 0])

    rng = np.random.default_rng(0)
    for i, (metric, title, label) in enumerate(metrics):
        summary_df = (
            recipe_descendant_summary_non_extant
            if "descendant" in metric
            else recombination_summary
        )
        outcome_df = (
            descendant_outcome_summary_non_extant
            if "descendant" in metric
            else recombination_outcome_summary
        )
        plot_df = (
            summary_df[["seed", "fee_plot", "log10_fee", metric]]
            .drop_duplicates()
            .sort_values("fee_plot")
            .copy()
        )
        plot_df["fee_plot_jittered"] = 10.0 ** (
            np.log10(plot_df["fee_plot"]) + rng.normal(0.0, 0.025, size=len(plot_df))
        )

        sns.scatterplot(
            data=plot_df,
            x="fee_plot_jittered",
            y=metric,
            hue="log10_fee",
            hue_order=fee_label_order,
            palette=fee_palette,
            s=45,
            legend=False,
            ax=axs[0, i],
        )
        sns.lineplot(
            data=plot_df,
            x="fee_plot",
            y=metric,
            color="red",
            marker=None,
            errorbar=None,
            ax=axs[0, i],
        )
        axs[0, i].set_xscale("log", base=10)
        axs[0, i].set(xlabel="Fee", ylabel=label, title=title)
        sns.despine(ax=axs[0, i], left=True, bottom=True)

        plot_yield_relationship(axs[1, i], outcome_df, metric, fee_label_order, i == 0)
        axs[1, i].set(xlabel=f"{label} (mean)", ylabel=None)
        if i != 0:
            axs[1, i].set(yticklabels=[])

    axs[0, 0].set_ylabel("Count")
    axs[1, 0].set_ylabel("Final population yield")
    fig.suptitle("Population-level metrics", fontsize=16)
    save_fig(fig, "population_level_metrics")


def plot_specialisation_metrics(
    specialisation_summary,
    specialisation_outcome_summary,
    whole_history_summary,
    whole_history_outcome_summary,
    fee_label_order,
    fee_palette,
):
    metrics = [
        (
            "specialisation",
            "Population specialisation (mutual information)",
            "Normalised mutual information",
        ),
        (
            "specialisation_whole_history",
            "Whole-history specialisation by spatial slot (mutual information)",
            "Normalised mutual information",
        ),
    ]
    fig, axs = plt.subplots(2, len(metrics), figsize=(5 * len(metrics), 8))
    for ax in axs[1, 1:]:
        ax.sharey(axs[1, 0])
    rng = np.random.default_rng(0)

    for i, (metric, title, label) in enumerate(metrics):
        summary_df = (
            whole_history_summary
            if "whole_history" in metric
            else specialisation_summary
        )
        outcome_df = (
            whole_history_outcome_summary
            if "whole_history" in metric
            else specialisation_outcome_summary
        )
        plot_df = (
            summary_df[["seed", "fee_plot", "log10_fee", metric]]
            .drop_duplicates()
            .sort_values("fee_plot")
            .copy()
        )
        plot_df["fee_plot_jittered"] = 10.0 ** (
            np.log10(plot_df["fee_plot"]) + rng.normal(0.0, 0.025, size=len(plot_df))
        )
        sns.scatterplot(
            data=plot_df,
            x="fee_plot_jittered",
            y=metric,
            hue="log10_fee",
            hue_order=fee_label_order,
            palette=fee_palette,
            s=45,
            legend=False,
            ax=axs[0, i],
        )
        sns.lineplot(
            data=plot_df,
            x="fee_plot",
            y=metric,
            color="red",
            marker=None,
            errorbar=None,
            ax=axs[0, i],
        )
        axs[0, i].set_xscale("log", base=10)
        axs[0, i].set(xlabel="Fee", ylabel=label, title=title)
        sns.despine(ax=axs[0, i], left=True, bottom=True)

        plot_yield_relationship(
            axs[1, i],
            outcome_df,
            metric,
            fee_label_order,
            legend=i == 0,
        )
        axs[1, i].set(xlabel=label, ylabel=None)
        if i != 0:
            axs[1, i].set(yticklabels=[])

    axs[1, 0].set_ylabel("Final population yield")
    fig.suptitle("Population specialisation", fontsize=16)
    save_fig(fig, "population_specialisation")


def main():
    data = load_data()
    fee_label_map, fee_label_order, fee_palette = get_fee_plotting_config(
        data["recipe_lineage"]
    )

    filtered_pop_df = plot_performance_metrics(data["population"])
    plot_final_performance_metrics(filtered_pop_df)
    plot_agent_heatmaps(data["agent"])
    plot_similarity(data["similarity_matrices"], np.sort(data["agent"]["fee"].unique()))

    (
        extant_recipe_lineage_data,
        fee_outcome_summary,
        recipe_descendant_summary_non_extant,
        descendant_outcome_summary_non_extant,
        recombination_summary,
        recombination_outcome_summary,
    ) = build_recipe_plot_data(
        data["population"].copy(),
        data["recipe_lineage"].copy(),
        data["recipe_descendant"].copy(),
        data["recipe_recombination"].copy(),
        fee_label_map,
        fee_label_order,
    )

    plot_recipe_level_metrics(
        extant_recipe_lineage_data,
        fee_outcome_summary,
        fee_label_order,
        fee_palette,
    )
    plot_population_level_metrics(
        recombination_summary,
        recombination_outcome_summary,
        recipe_descendant_summary_non_extant,
        descendant_outcome_summary_non_extant,
        fee_label_order,
        fee_palette,
    )

    (
        specialisation_summary,
        specialisation_outcome_summary,
    ) = build_specialisation_plot_data(
        data["agent"].copy(),
        data["population"].copy(),
        fee_label_map,
        fee_label_order,
    )
    (
        whole_history_summary,
        whole_history_outcome_summary,
    ) = build_whole_history_specialisation_plot_data(
        data["raw_outputs"],
        data["population"].copy(),
        fee_label_map,
        fee_label_order,
    )
    plot_specialisation_metrics(
        specialisation_summary,
        specialisation_outcome_summary,
        whole_history_summary,
        whole_history_outcome_summary,
        fee_label_order,
        fee_palette,
    )


if __name__ == "__main__":
    main()
