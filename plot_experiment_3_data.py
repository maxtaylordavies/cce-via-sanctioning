from dataclasses import dataclass
from pathlib import Path
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from src.utils import save_fig, load_matching_outputs, get_scalar

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

RUN_TYPES = ("real", "neutral")
COLORS = {"real": "xkcd:turquoise", "neutral": "xkcd:periwinkle"}
VARIANTS = {"intrinsic": "Intrinsic motivation", "gift": "Deference gifting"}
RUN_TYPE_LABELS = {
    "real": "Norms active",
    "neutral": "Norms inactive (neutral drift baseline)",
}
REFERENCE_GAINS_PATH = Path("figures/experiment_1_reference_gains.csv")
MIN_SCATTER_SIZE = 20
MAX_SCATTER_SIZE = 240


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    title: str
    score_key: str
    score_normalizer: float | str
    role_source: str
    sample_interval: int


ENVIRONMENTS = (
    EnvironmentConfig(
        name="mesoudi_env",
        title="Binary trait environment",
        score_key="mean_traits_known",
        score_normalizer="max_total_l",
        role_source="agent_roles",
        sample_interval=50,
    ),
    EnvironmentConfig(
        name="miu_env",
        title="Refinement bandit environment",
        score_key="payoffs",
        score_normalizer=1.0,
        role_source="agent_roles",
        sample_interval=50,
    ),
    EnvironmentConfig(
        name="recipe_world_env",
        title="Recipe grammar environment",
        score_key="agent_yields",
        score_normalizer=10.0,
        role_source="agent_roles",
        sample_interval=50,
    ),
)

_save_fig = lambda fig, name: save_fig(fig, name, subfolder="experiment_3")


def get_data_dirs(config, variant):
    experiment_dir = Path(f"data/{config.name}/experiment_3")
    dirs = [experiment_dir / variant]
    if variant == "intrinsic":
        dirs.append(experiment_dir)
    return dirs


def find_data_dir(config, variant):
    for data_dir in get_data_dirs(config, variant):
        if any(data_dir.glob("real*.npz")) or any(data_dir.glob("neutral*.npz")):
            return data_dir
    return None


def load_reference_gains(path=REFERENCE_GAINS_PATH):
    if not path.exists():
        return {}

    reference_gains = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reference_gains[(row["environment"], row["variant"])] = float(
                row["reference_prestige_gain"]
            )
    return reference_gains


def build_norm_value_grids(grids, group_norm_values_history):
    return np.take_along_axis(
        group_norm_values_history[:, None, :],
        grids,
        axis=2,
    )


def compute_average_norm_values(norm_value_grids):
    return np.mean(norm_value_grids, axis=(1, 2))


def build_group_instance_norm_series(
    group_norm_values_history,
    group_instance_ids_by_label_history,
    next_group_instance_id,
):
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
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure

    n_instances, n_steps = instance_norm_series.shape
    parent_instance_ids = group_lineage_array[:n_instances, 0]
    birth_timesteps = group_lineage_array[:n_instances, 1]
    times = np.arange(n_steps) * timestep_scale
    cmap = plt.get_cmap("Dark2")
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
            collection = LineCollection(
                segments,
                colors=[cmap(instance_id % cmap.N)],
                linewidths=scale_sizes(segment_sizes, 0.2, 4.0),
                zorder=2,
            )
            ax.add_collection(collection)

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

    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    return fig, ax


def get_score_array(outputs, config):
    if config.score_key in outputs:
        score = np.asarray(outputs[config.score_key], dtype=np.float64)
    elif config.name == "mesoudi_env" and "traits_known" in outputs:
        score = np.asarray(outputs["traits_known"], dtype=np.float64).mean(axis=-1)
    else:
        raise KeyError(f"No score array found for {config.name}")

    if config.score_key == "agent_yields":
        score = score.mean(axis=-1)

    normalizer = config.score_normalizer
    if isinstance(normalizer, str):
        normalizer = float(get_scalar(outputs, normalizer, 1.0))
    return score / float(normalizer)


def get_p_innovate_array(outputs):
    role_innovate = int(get_scalar(outputs, "role_innovate", 0))
    roles = np.asarray(outputs["agent_roles"])
    return (roles == role_innovate).mean(axis=-1)


def trim_to_common_length(run_data):
    lengths = []
    for data in run_data.values():
        lengths.append(data["group_norm_values"].shape[1])
        lengths.append(data["group_labels_grids"].shape[1])
        lengths.append(data["score"].shape[1])
        t_main = get_scalar(data["outputs"], "T_main")
        if t_main is not None:
            lengths.append(int(t_main))
        if data["p_innovate"] is not None:
            lengths.append(data["p_innovate"].shape[1])
    return min(lengths)


def process_environment(config, variant):
    data_dir = find_data_dir(config, variant)
    if data_dir is None:
        return None

    raw = {}
    for run_type in RUN_TYPES:
        outputs = load_matching_outputs(data_dir, run_type)
        if outputs is None:
            continue

        required_keys = (
            "group_norm_values",
            "group_labels_grids",
            "group_instance_ids_by_label_history",
            "group_lineage_arrays",
            "final_next_group_instance_ids",
        )
        if any(key not in outputs for key in required_keys):
            continue

        raw[run_type] = {
            "outputs": outputs,
            "group_norm_values": np.asarray(outputs["group_norm_values"]),
            "group_labels_grids": np.asarray(outputs["group_labels_grids"]),
            "group_instance_ids_by_label_history": np.asarray(
                outputs["group_instance_ids_by_label_history"]
            ),
            "group_lineage_arrays": np.asarray(outputs["group_lineage_arrays"]),
            "final_next_group_instance_ids": np.asarray(
                outputs["final_next_group_instance_ids"]
            ),
            "score": get_score_array(outputs, config),
            "p_innovate": get_p_innovate_array(outputs),
        }

    if not raw:
        return None

    max_t = trim_to_common_length(raw)
    sampled_timesteps = np.arange(0, max_t, config.sample_interval)
    processed = {}

    for run_type, data in raw.items():
        sampled_norm_values = data["group_norm_values"][:, sampled_timesteps]
        sampled_grids = data["group_labels_grids"][:, sampled_timesteps]
        score = data["score"][:, sampled_timesteps]
        p_innovate = (
            data["p_innovate"][:, sampled_timesteps]
            if data["p_innovate"] is not None
            else None
        )

        average_norm_values = []
        for seed_idx in range(sampled_norm_values.shape[0]):
            norm_value_grids = build_norm_value_grids(
                sampled_grids[seed_idx],
                sampled_norm_values[seed_idx],
            )
            average_norm_values.append(compute_average_norm_values(norm_value_grids))

        processed[run_type] = {
            "average_norm_values": np.asarray(average_norm_values),
            "score": score,
            "p_innovate": p_innovate,
        }

    return {
        "data_dir": data_dir,
        "raw": raw,
        "processed": processed,
        "timesteps": sampled_timesteps,
    }


def build_processed_cache():
    reference_gains = load_reference_gains()
    cache = {}
    for variant in VARIANTS:
        for config in ENVIRONMENTS:
            panel_data = process_environment(config, variant)
            if panel_data is None:
                continue
            panel_data["reference_gain"] = reference_gains.get(
                (config.name, variant), 1.0
            )
            cache[(variant, config.name)] = panel_data
    return cache


def mark_missing(ax, title=None, xlabel=None, ylabel=None):
    ax.text(
        0.5,
        0.5,
        "No data",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="0.4",
    )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_seed_lines(
    ax, ts, values_by_run, xlabel=None, ylabel=None, title=None, ylim=None
):
    plotted = False
    for run_type in RUN_TYPES:
        values = values_by_run.get(run_type)
        if values is None:
            continue
        plotted = True
        for seed_series in values:
            ax.plot(ts, seed_series, color=COLORS[run_type], alpha=0.35, linewidth=1)
        ax.plot(
            ts,
            np.nanmean(values, axis=0),
            color=COLORS[run_type],
            linewidth=3,
            label=run_type,
            zorder=10,
        )

    if not plotted:
        mark_missing(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, ylim=ylim)
    sns.despine(ax=ax, left=True, bottom=True)


def plot_scatter(
    ax,
    processed,
    reference_gain,
    norm_window="all",
    title=None,
    xlabel=None,
    ylabel=None,
    show_regression=False,
):
    plotted = False
    reference_gain = reference_gain if reference_gain != 0 else 1.0

    for run_type in RUN_TYPES:
        if run_type not in processed:
            continue

        norms = processed[run_type]["average_norm_values"]
        scores = processed[run_type]["score"]
        p_innovate = processed[run_type]["p_innovate"]
        final_window_len = max(1, scores.shape[1] // 10)
        if norm_window == "all":
            x = norms.mean(axis=1) / reference_gain
        elif norm_window == "final":
            x = norms[:, -final_window_len:].mean(axis=1) / reference_gain
        else:
            raise ValueError(f"Unknown norm_window {norm_window!r}")
        y = scores[:, -final_window_len:].mean(axis=1)
        if p_innovate is None:
            sizes = np.full_like(x, 60.0, dtype=np.float64)
        else:
            mean_p_innovate = np.nanmean(p_innovate, axis=1)
            sizes = MIN_SCATTER_SIZE + np.clip(mean_p_innovate, 0, 1) * (
                MAX_SCATTER_SIZE - MIN_SCATTER_SIZE
            )

        plotted = True
        ax.scatter(
            x,
            y,
            color=COLORS[run_type],
            s=sizes,
            alpha=0.7,
            label=run_type,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.scatter(
            np.nanmean(x),
            np.nanmean(y),
            color=COLORS[run_type],
            marker="*",
            s=250,
            edgecolor="black",
            linewidth=1.0,
            zorder=5,
        )
        if show_regression:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                line_x,
                slope * line_x + intercept,
                color=COLORS[run_type],
                linewidth=2.5,
                alpha=0.5,
            )

    if not plotted:
        mark_missing(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    sns.despine(ax=ax, left=True, bottom=True)


def plot_main_figure(processed_cache, norm_window="all"):
    fig, axs = plt.subplots(
        len(VARIANTS),
        len(ENVIRONMENTS),
        figsize=(15, 6.5),
        squeeze=False,
    )

    for row_idx, variant in enumerate(VARIANTS):
        for col_idx, env_config in enumerate(ENVIRONMENTS):
            panel_data = processed_cache.get((variant, env_config.name))
            xlabel_detail = (
                "over all timesteps" if norm_window == "all" else "over final window"
            )
            title = env_config.title if row_idx == 0 else None
            xlabel = (
                f"Mean evolved norm $g/g_{{0.5}}$ ({xlabel_detail})"
                if row_idx == len(VARIANTS) - 1
                else None
            )
            ylabel = "Final mean payoff" if col_idx == 0 else None
            if panel_data is None:
                mark_missing(axs[row_idx, col_idx], title, xlabel, ylabel)
            else:
                plot_scatter(
                    axs[row_idx, col_idx],
                    panel_data["processed"],
                    panel_data["reference_gain"],
                    norm_window=norm_window,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

        axs[row_idx, 0].text(
            -0.2,
            0.5,
            VARIANTS[variant],
            transform=axs[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=13,
            fontweight="medium",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[run_type],
            marker="o",
            markersize=8,
            linestyle="",
            label=RUN_TYPE_LABELS[run_type],
        )
        for run_type in RUN_TYPES
    ]
    size_legend_values = (0.1, 0.3, 0.5)
    size_legend_handles = [
        Line2D(
            [0],
            [0],
            color="none",
            linestyle="",
            label="Innovator frequency (mean over all timesteps):",
        ),
        *[
            Line2D(
                [0],
                [0],
                color="0.35",
                marker="o",
                linestyle="",
                markerfacecolor="0.55",
                markeredgecolor="white",
                markersize=np.sqrt(
                    MIN_SCATTER_SIZE + value * (MAX_SCATTER_SIZE - MIN_SCATTER_SIZE)
                ),
                label=f"{value:g}",
            )
            for value in size_legend_values
        ],
    ]
    centroid_legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.55",
            marker="*",
            linestyle="",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=15,
            label="= mean over independent runs",
        )
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        handletextpad=0.0,
        columnspacing=0.8,
        bbox_to_anchor=(0.2, 1.05),
    )
    fig.legend(
        size_legend_handles,
        [handle.get_label() for handle in size_legend_handles],
        loc="upper center",
        ncol=4,
        frameon=False,
        handletextpad=0.0,
        columnspacing=0.8,
        bbox_to_anchor=(0.52, 1.05),
    )
    fig.legend(
        centroid_legend_handles,
        [handle.get_label() for handle in centroid_legend_handles],
        loc="upper center",
        ncol=1,
        frameon=False,
        handletextpad=0.0,
        bbox_to_anchor=(0.80, 1.05),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_timeseries_figure(processed_cache, metric):
    if metric not in {"norm", "score"}:
        raise ValueError(f"Unknown timeseries metric {metric!r}")

    fig, axs = plt.subplots(
        len(VARIANTS),
        len(ENVIRONMENTS),
        figsize=(13, 6),
        squeeze=False,
    )

    for row_idx, variant in enumerate(VARIANTS):
        for col_idx, env_config in enumerate(ENVIRONMENTS):
            panel_data = processed_cache.get((variant, env_config.name))
            title = env_config.title if row_idx == 0 else None
            xlabel = "$t$" if row_idx == len(VARIANTS) - 1 else None
            if metric == "norm":
                ylabel = "Mean evolved norm\n$g/g_{0.5}$" if col_idx == 0 else None
            else:
                ylabel = "Score" if col_idx == 0 else None

            if panel_data is None:
                mark_missing(axs[row_idx, col_idx], title, xlabel, ylabel)
                continue

            processed = panel_data["processed"]
            ts = panel_data["timesteps"]
            if metric == "norm":
                reference_gain = panel_data["reference_gain"] or 1.0
                values_by_run = {
                    run_type: (
                        processed[run_type]["average_norm_values"] / reference_gain
                    )
                    for run_type in RUN_TYPES
                    if run_type in processed
                }
            else:
                values_by_run = {
                    run_type: processed[run_type]["score"]
                    for run_type in RUN_TYPES
                    if run_type in processed
                }

            plot_seed_lines(
                axs[row_idx, col_idx],
                ts,
                values_by_run,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
            )

        axs[row_idx, 0].text(
            -0.32,
            0.5,
            VARIANTS[variant],
            transform=axs[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=13,
            fontweight="medium",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[run_type],
            linewidth=3,
            label=RUN_TYPE_LABELS[run_type],
        )
        for run_type in RUN_TYPES
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def plot_lineage_figures(processed_cache):
    for variant in VARIANTS:
        for config in ENVIRONMENTS:
            panel_data = processed_cache.get((variant, config.name))
            if panel_data is None or "real" not in panel_data["raw"]:
                continue

            real = panel_data["raw"]["real"]
            n_lineage_seeds = min(10, real["group_norm_values"].shape[0])
            fig, axs = plt.subplots(
                1,
                n_lineage_seeds,
                figsize=(4 * n_lineage_seeds, 3.5),
                sharey=True,
                squeeze=False,
            )
            axs = axs[0]

            for seed_idx in range(n_lineage_seeds):
                instance_norm_series = build_group_instance_norm_series(
                    real["group_norm_values"][seed_idx],
                    real["group_instance_ids_by_label_history"][seed_idx],
                    int(real["final_next_group_instance_ids"][seed_idx]),
                )
                instance_size_series = build_group_instance_size_series(
                    real["group_labels_grids"][seed_idx],
                    real["group_instance_ids_by_label_history"][seed_idx],
                    int(real["final_next_group_instance_ids"][seed_idx]),
                )
                plot_group_lineage(
                    instance_norm_series,
                    instance_size_series,
                    real["group_lineage_arrays"][seed_idx],
                    ax=axs[seed_idx],
                )
                axs[seed_idx].set(
                    title=f"Seed {seed_idx}",
                    xlabel="$t$",
                    ylabel="$g$" if seed_idx == 0 else None,
                )
                sns.despine(ax=axs[seed_idx], left=True, bottom=True)

            fig.suptitle(
                f"{config.title}: {VARIANTS[variant]} group lineages",
                y=1.02,
                fontweight="bold",
            )
            _save_fig(fig, f"lineages_{config.name}_{variant}")


def main():
    processed_cache = build_processed_cache()

    fig = plot_main_figure(processed_cache)
    _save_fig(fig, "main")
    # fig = plot_main_figure(processed_cache, norm_window="final")
    # _save_fig(fig, "main_final_window_norms")
    # fig = plot_timeseries_figure(processed_cache, "norm")
    # _save_fig(fig, "norm_timeseries")
    # fig = plot_timeseries_figure(processed_cache, "score")
    # _save_fig(fig, "score_timeseries")
    # plot_lineage_figures(processed_cache)


if __name__ == "__main__":
    main()
