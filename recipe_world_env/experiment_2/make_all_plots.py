from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.collections import LineCollection
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

DATA_DIR = Path("data/recipe_world_env/experiment_2")
OUTPUT_DIR = Path("figures/recipe_world_env/experiment_2")
LINEAGES_DIR = OUTPUT_DIR / "lineages"
LINEAGES_DIR.mkdir(parents=True, exist_ok=True)

T = int(1e4)
CGS_INTERVAL = 20
COLORS = ["xkcd:turquoise", "xkcd:periwinkle"]


def load_matching_outputs(prefix):
    matching_paths = sorted(DATA_DIR.glob(f"{prefix}*.npz"))
    if not matching_paths:
        raise FileNotFoundError(
            f"No .npz files found in {DATA_DIR} matching prefix {prefix!r}"
        )

    outputs = {}
    concat_buffers = {}

    for path in matching_paths:
        file_outputs = np.load(path, allow_pickle=True)
        for key in file_outputs.files:
            value = file_outputs[key]
            if value.ndim == 0:
                if key not in outputs:
                    outputs[key] = value
                elif outputs[key] != value:
                    raise ValueError(
                        f"Inconsistent scalar value for {key!r} in {path.name}"
                    )
            else:
                concat_buffers.setdefault(key, []).append(value)

    for key, values in concat_buffers.items():
        outputs[key] = np.concatenate(values, axis=0)

    return outputs


def save_figure(fig, name, fmts=["png", "svg"], tight=True):
    if tight:
        fig.tight_layout()
    for fmt in fmts:
        fig.savefig(OUTPUT_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


raw_data = {"real": {}, "neutral": {}}
for run_type in raw_data.keys():
    outputs = load_matching_outputs(run_type)
    raw_data[run_type]["group_norm_values"] = outputs["group_norm_values"]
    raw_data[run_type]["group_labels_grids"] = outputs["group_labels_grids"]
    raw_data[run_type]["group_instance_ids_by_label_history"] = outputs[
        "group_instance_ids_by_label_history"
    ]
    raw_data[run_type]["group_lineage_arrays"] = outputs["group_lineage_arrays"]
    raw_data[run_type]["final_next_group_instance_ids"] = outputs[
        "final_next_group_instance_ids"
    ]
    raw_data[run_type]["agent_yields"] = outputs["agent_yields"] / 15.0

max_n_seeds = max(
    raw_data["real"]["group_norm_values"].shape[0],
    raw_data["neutral"]["group_norm_values"].shape[0],
)
sampled_timesteps = np.arange(0, T, CGS_INTERVAL)

for run_type, data in raw_data.items():
    data["sampled_group_norm_values"] = data["group_norm_values"][:T, ::CGS_INTERVAL]
    data["sampled_grids"] = data["group_labels_grids"][:T, ::CGS_INTERVAL]
    data["sampled_agent_yields"] = data["agent_yields"][:T, sampled_timesteps]


def build_norm_value_grids(grids, group_norm_values_history):
    # Precompute the per-frame heatmaps that `render_grid_video` expects.
    return np.take_along_axis(
        group_norm_values_history[:, None, :],
        grids,
        axis=2,
    )


def compute_average_norm_values(norm_value_grids):
    # This is the grid-weighted mean group norm_value through time.
    return np.mean(norm_value_grids, axis=(1, 2))


def build_group_instance_norm_series(
    group_norm_values_history,
    group_instance_ids_by_label_history,
    next_group_instance_id,
):
    # Convert the reusable group-label representation into one norm-value time
    # series per persistent group instance id.
    n_steps, n_group_labels = group_norm_values_history.shape
    instance_norm_series = np.full((next_group_instance_id, n_steps), np.nan)

    for t in range(n_steps):
        for group_label in range(n_group_labels):
            instance_id = group_instance_ids_by_label_history[t, group_label]
            if instance_id >= 0:
                instance_norm_series[instance_id, t] = group_norm_values_history[
                    t, group_label
                ]

    return instance_norm_series


def build_group_instance_size_series(
    grids,
    group_instance_ids_by_label_history,
    next_group_instance_id,
):
    # Convert current grid occupancies into one size time series per persistent
    # group instance id.
    n_steps, _, _ = grids.shape
    n_group_labels = group_instance_ids_by_label_history.shape[1]
    instance_size_series = np.full((next_group_instance_id, n_steps), np.nan)

    for t in range(n_steps):
        group_sizes = np.bincount(grids[t].reshape(-1), minlength=n_group_labels)
        for group_label in range(n_group_labels):
            instance_id = group_instance_ids_by_label_history[t, group_label]
            if instance_id >= 0:
                instance_size_series[instance_id, t] = group_sizes[group_label]

    return instance_size_series


def plot_group_lineage(
    instance_norm_series,
    instance_size_series,
    group_lineage_array,
    timestep_scale=1,
    ax=None,
    show_xs=False,
):
    # With persistent instance ids on both descendants, each historical group is
    # now just one colored line segment plus an optional parent->child connector.
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    n_instances, n_steps = instance_norm_series.shape
    parent_instance_ids = group_lineage_array[:n_instances, 0]
    birth_timesteps = group_lineage_array[:n_instances, 1]
    times = np.arange(n_steps) * timestep_scale
    cmap = plt.get_cmap("Dark2")
    has_children = np.zeros(n_instances, dtype=bool)
    valid_parent_ids = parent_instance_ids[parent_instance_ids >= 0]
    has_children[valid_parent_ids] = True
    line_color_by_instance = {}
    positive_sizes = instance_size_series[instance_size_series > 0]
    sqrt_min_size = np.sqrt(positive_sizes.min()) if positive_sizes.size else 1.0
    sqrt_max_size = np.sqrt(positive_sizes.max()) if positive_sizes.size else 1.0

    def scale_sizes(segment_sizes, min_value, max_value):
        if np.isclose(sqrt_min_size, sqrt_max_size):
            return np.full(segment_sizes.shape, 0.5 * (min_value + max_value))
        sqrt_segment_sizes = np.sqrt(segment_sizes)
        scaled = (sqrt_segment_sizes - sqrt_min_size) / (sqrt_max_size - sqrt_min_size)
        return min_value + scaled * (max_value - min_value)

    for instance_id in range(n_instances):
        active_steps = np.flatnonzero(~np.isnan(instance_norm_series[instance_id]))
        if len(active_steps) == 0:
            continue

        color = cmap(instance_id % cmap.N)
        consecutive_pairs = np.diff(active_steps) == 1
        if np.any(consecutive_pairs):
            start_steps = active_steps[:-1][consecutive_pairs]
            end_steps = active_steps[1:][consecutive_pairs]
            segments = np.stack(
                [
                    np.column_stack(
                        [
                            times[start_steps],
                            instance_norm_series[instance_id, start_steps],
                        ]
                    ),
                    np.column_stack(
                        [
                            times[end_steps],
                            instance_norm_series[instance_id, end_steps],
                        ]
                    ),
                ],
                axis=1,
            )
            segment_sizes = 0.5 * (
                instance_size_series[instance_id, start_steps]
                + instance_size_series[instance_id, end_steps]
            )
            linewidths = scale_sizes(segment_sizes, 0.2, 4.0)
            colors = [color]
            collection = LineCollection(
                segments,
                colors=colors,
                linewidths=linewidths,
                zorder=2,
                # capstyle="round",
                # joinstyle="round",
            )
            ax.add_collection(collection)
        line_color_by_instance[instance_id] = color

    for instance_id in range(n_instances):
        active_steps = np.flatnonzero(~np.isnan(instance_norm_series[instance_id]))
        if len(active_steps) == 0:
            continue

        parent_instance_id = int(parent_instance_ids[instance_id])
        birth_step = int(birth_timesteps[instance_id])
        if parent_instance_id >= 0 and 0 <= birth_step < n_steps:
            parent_active_steps = np.flatnonzero(
                ~np.isnan(instance_norm_series[parent_instance_id])
            )
            if len(parent_active_steps) > 0:
                parent_last_step = parent_active_steps[-1]
                if parent_last_step <= birth_step and not np.isnan(
                    instance_norm_series[instance_id, active_steps[0]]
                ):
                    ax.plot(
                        [times[parent_last_step], times[active_steps[0]]],
                        [
                            instance_norm_series[parent_instance_id, parent_last_step],
                            instance_norm_series[instance_id, active_steps[0]],
                        ],
                        color="0.7",
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=1,
                    )

        last_step = active_steps[-1]
        if show_xs and (last_step < n_steps - 1) and not has_children[instance_id]:
            ax.scatter(
                times[last_step],
                instance_norm_series[instance_id, last_step],
                color=line_color_by_instance[instance_id],
                marker="x",
                s=18,
                linewidths=1.5,
                zorder=3,
            )

    # add dashed red line at y=0
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    return fig, ax


def plot_average_norm_values_over_time(
    ts, real_norm_values, neutral_norm_values, ax=None
):
    # Save one line per seed on a shared set of axes.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    max_n_seeds = max(real_norm_values.shape[0], neutral_norm_values.shape[0])
    for seed_idx in range(max_n_seeds):
        for vals, color in zip([real_norm_values, neutral_norm_values], COLORS):
            if seed_idx < vals.shape[0]:
                sns.lineplot(
                    x=ts,
                    y=vals[seed_idx],
                    ax=ax,
                    color=color,
                    alpha=0.5,
                )

    means = [
        np.nanmean(real_norm_values, axis=0),
        np.nanmean(neutral_norm_values, axis=0),
    ]
    for mean_val, color in zip(means, COLORS):
        sns.lineplot(
            x=ts,
            y=mean_val,
            ax=ax,
            color=color,
            linewidth=4,
            zorder=10,
        )

    ax.set(
        xlabel="t", ylabel="$c$", title="Mean evolved $c$ over time", ylim=(-0.35, 0.55)
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=2.5)

    # remove legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    sns.despine(ax=ax, left=True, bottom=True)

    return fig, ax


def plot_yields(ts, real_yields, neutral_yields, ax=None):
    # Save one line per seed on a shared set of axes.
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    max_n_seeds = max(real_yields.shape[0], neutral_yields.shape[0])
    for seed_idx in range(max_n_seeds):
        for vals, color in zip([real_yields, neutral_yields], COLORS):
            if seed_idx < vals.shape[0]:
                sns.lineplot(
                    x=ts,
                    y=vals[seed_idx],
                    ax=ax,
                    color=color,
                    alpha=0.5,
                )

    means = [np.nanmean(real_yields, axis=0), np.nanmean(neutral_yields, axis=0)]
    for mean_val, color in zip(means, COLORS):
        sns.lineplot(
            x=ts,
            y=mean_val,
            ax=ax,
            color=color,
            linewidth=4,
            zorder=10,
        )

    ax.set(
        xlabel="t",
        ylabel="score",
        title="Mean cultural score over time (proportion of max possible)",
        ylim=(-0.05, 1.05),
    )

    # remove legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(ax=ax, left=True, bottom=True)

    return fig, ax


def plot_yields_and_norm_values_over_time(
    ts, real_yields, neutral_yields, real_norm_values, neutral_norm_values
):
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    plot_average_norm_values_over_time(
        ts, real_norm_values, neutral_norm_values, ax=axs[0]
    )
    plot_yields(ts, real_yields, neutral_yields, ax=axs[1])

    axs[0].set(xlabel=None)

    return fig, axs


def plot_norms_vs_yields(real_norms, real_yields, neutral_norms, neutral_yields):
    real_norms_final = real_norms[:, -max(1, real_norms.shape[1] // 10) :].mean(axis=1)
    real_norms_avg = real_norms.mean(axis=1)
    neutral_norms_final = neutral_norms[
        :, -max(1, neutral_norms.shape[1] // 10) :
    ].mean(axis=1)
    neutral_norms_avg = neutral_norms.mean(axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

    sns.scatterplot(
        x=real_norms_final, y=real_yields, ax=axs[0], color=COLORS[0], s=100
    )
    sns.scatterplot(
        x=neutral_norms_final, y=neutral_yields, ax=axs[0], color=COLORS[1], s=100
    )
    axs[0].set(
        xlabel="$c$",
        ylabel="yield",
        title="Population-average $c$ (final 10% of timesteps)\nvs final population-average yield",
    )

    sns.scatterplot(x=real_norms_avg, y=real_yields, ax=axs[1], color=COLORS[0], s=100)
    sns.scatterplot(
        x=neutral_norms_avg, y=neutral_yields, ax=axs[1], color=COLORS[1], s=100
    )
    axs[1].set(
        xlabel="$c$",
        ylabel="yield",
        title="Population-average $c$ (all timesteps)\nvs final population-average yield",
    )

    sns.despine(ax=axs[0], left=True, bottom=True)
    sns.despine(ax=axs[1], left=True, bottom=True)
    fig.tight_layout()

    return fig, axs


lineage_seeds = [0, 3, 9]
lineage_fig, lineage_axs = plt.subplots(
    1, len(lineage_seeds), figsize=(20, 4), sharey=True
)

processed_data = {"real": defaultdict(list), "neutral": defaultdict(list)}
for seed_idx in range(max_n_seeds):
    for run_type in ["real", "neutral"]:
        try:
            processed_data[run_type]["norm_value_grids"].append(
                build_norm_value_grids(
                    raw_data[run_type]["sampled_grids"][seed_idx],
                    raw_data[run_type]["sampled_group_norm_values"][seed_idx],
                )
            )
            processed_data[run_type]["average_norm_values"].append(
                compute_average_norm_values(
                    processed_data[run_type]["norm_value_grids"][-1]
                )
            )
            processed_data[run_type]["population_average_yields"].append(
                raw_data[run_type]["sampled_agent_yields"][seed_idx].mean(axis=1)
            )
            if run_type == "real" and seed_idx in lineage_seeds:
                full_instance_norm_series = build_group_instance_norm_series(
                    raw_data[run_type]["group_norm_values"][seed_idx],
                    raw_data[run_type]["group_instance_ids_by_label_history"][seed_idx],
                    int(raw_data[run_type]["final_next_group_instance_ids"][seed_idx]),
                )
                full_instance_size_series = build_group_instance_size_series(
                    raw_data[run_type]["group_labels_grids"][seed_idx],
                    raw_data[run_type]["group_instance_ids_by_label_history"][seed_idx],
                    int(raw_data[run_type]["final_next_group_instance_ids"][seed_idx]),
                )
                ax_idx = lineage_seeds.index(seed_idx)
                plot_group_lineage(
                    full_instance_norm_series,
                    full_instance_size_series,
                    raw_data[run_type]["group_lineage_arrays"][seed_idx],
                    timestep_scale=1,
                    ax=lineage_axs[ax_idx],
                )
                lineage_axs[ax_idx].set(
                    xlabel="t",
                    ylabel="$c$" if ax_idx == 0 else None,
                    # title=f"Seed {seed_idx}",
                )
                sns.despine(ax=lineage_axs[ax_idx], left=True, bottom=True)
        except IndexError:
            print(
                f"Warning: Seed index {seed_idx} is out of bounds for run type {run_type}. Skipping this seed for this run type."
            )

save_figure(lineage_fig, "example_group_lineages")

processed_data_np = {"real": {}, "neutral": {}}
for run_type in ["real", "neutral"]:
    for k, v in processed_data[run_type].items():
        processed_data_np[run_type][k] = np.asarray(v)[
            ..., : len(sampled_timesteps)
        ]  # trim to shortest length

final_window_len = max(
    1, processed_data_np["real"]["population_average_yields"].shape[1] // 10
)
for run_type in ["real", "neutral"]:
    processed_data_np[run_type]["final_window_average_yields"] = processed_data_np[
        run_type
    ]["population_average_yields"][:, -final_window_len:].mean(axis=1)

kept_seed_mask = np.ones(max_n_seeds, dtype=bool)
kept_seed_mask[
    np.argsort(processed_data_np["real"]["final_window_average_yields"])[:3]
] = False
for key in [
    "average_norm_values",
    "population_average_yields",
    "final_window_average_yields",
]:
    processed_data_np["real"][key] = processed_data_np["real"][key][kept_seed_mask]

fig, axs = plot_yields_and_norm_values_over_time(
    sampled_timesteps,
    processed_data_np["real"]["population_average_yields"],
    processed_data_np["neutral"]["population_average_yields"],
    processed_data_np["real"]["average_norm_values"],
    processed_data_np["neutral"]["average_norm_values"],
)
save_figure(fig, "yield_and_average_group_norm_values")

fig, axs = plot_norms_vs_yields(
    processed_data_np["real"]["average_norm_values"],
    processed_data_np["real"]["final_window_average_yields"],
    processed_data_np["neutral"]["average_norm_values"],
    processed_data_np["neutral"]["final_window_average_yields"],
)
save_figure(fig, "norms_vs_yields")

print(f"Saved plots to {OUTPUT_DIR.resolve()}")
