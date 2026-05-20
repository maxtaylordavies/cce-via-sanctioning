from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib import animation
from matplotlib.collections import LineCollection
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sns.set_style("whitegrid")

DATA_DIR = Path("data/recipe_world_env/experiment_3")
OUTPUT_DIR = Path("figures/recipe_world_env/experiment_3")
LINEAGES_DIR = OUTPUT_DIR / "lineages"
VIDEOS_DIR = OUTPUT_DIR / "videos"
LINEAGES_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

T = int(1e4)
CGS_INTERVAL = 100
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
    raw_data[run_type]["agent_yields"] = outputs["agent_yields"]

max_n_seeds = max(
    raw_data["real"]["group_norm_values"].shape[0],
    raw_data["neutral"]["group_norm_values"].shape[0],
)
sampled_timesteps = np.arange(0, T, CGS_INTERVAL)

for run_type, data in raw_data.items():
    data["sampled_group_norm_values"] = data["group_norm_values"][:T, ::CGS_INTERVAL]
    data["sampled_grids"] = data["group_labels_grids"][:T, ::CGS_INTERVAL]
    data["sampled_agent_yields"] = data["agent_yields"][:T, sampled_timesteps]


def _configure_grid_axes(ax, grid_size, show_gridlines=False):
    # Draw cell boundaries but suppress all external axes decoration.
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    if show_gridlines:
        ax.grid(which="minor", color="black", linewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)


def build_boundary_masks(grids):
    # Precompute where group boundaries lie in each frame so the video renderer
    # can draw them without re-deriving them on every animation callback.
    horizontal_boundaries = grids[:, 1:, :] != grids[:, :-1, :]
    vertical_boundaries = grids[:, :, 1:] != grids[:, :, :-1]
    return horizontal_boundaries, vertical_boundaries


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


def build_group_labels_for_yield_history(group_labels_history, sampled_timesteps):
    # `agent_yields[t]` is computed before the CA/group update, while
    # `group_labels_grids[t]` is saved after that update. To recover the groups
    # that actually generated each yield vector, we therefore use the previous
    # recorded grid, with the all-zero initial grid standing in for t=0.
    aligned_group_labels = np.zeros(
        (len(sampled_timesteps),) + group_labels_history.shape[1:],
        dtype=group_labels_history.dtype,
    )
    has_previous_grid = sampled_timesteps > 0
    aligned_group_labels[has_previous_grid] = group_labels_history[
        sampled_timesteps[has_previous_grid] - 1
    ]
    return aligned_group_labels


def compute_group_average_yield_annotations(
    agent_yields_history,
    group_labels_for_yields,
    max_n_groups,
):
    # Build one text label per non-empty group per frame, positioned at the
    # occupied tile closest to the group's centroid and showing that group's
    # mean yield.
    n_steps = agent_yields_history.shape[0]
    flat_yields = agent_yields_history.reshape(n_steps, -1)
    flat_group_labels = group_labels_for_yields.reshape(n_steps, -1)
    rows, cols = np.indices(group_labels_for_yields.shape[1:])
    flat_rows = np.broadcast_to(rows.reshape(1, -1), flat_group_labels.shape)
    flat_cols = np.broadcast_to(cols.reshape(1, -1), flat_group_labels.shape)
    annotations = []

    for t in range(n_steps):
        counts = np.bincount(flat_group_labels[t], minlength=max_n_groups)
        total_yields = np.bincount(
            flat_group_labels[t],
            weights=flat_yields[t],
            minlength=max_n_groups,
        )
        row_totals = np.bincount(
            flat_group_labels[t],
            weights=flat_rows[t],
            minlength=max_n_groups,
        )
        col_totals = np.bincount(
            flat_group_labels[t],
            weights=flat_cols[t],
            minlength=max_n_groups,
        )
        frame_annotations = []
        for group_id in np.flatnonzero(counts > 0):
            group_mask = flat_group_labels[t] == group_id
            centroid_row = row_totals[group_id] / counts[group_id]
            centroid_col = col_totals[group_id] / counts[group_id]
            squared_distances = (flat_rows[t] - centroid_row) ** 2 + (
                flat_cols[t] - centroid_col
            ) ** 2
            label_tile_idx = np.argmin(np.where(group_mask, squared_distances, np.inf))
            frame_annotations.append(
                (
                    flat_cols[t, label_tile_idx],
                    flat_rows[t, label_tile_idx],
                    f"{total_yields[group_id] / counts[group_id]:.1f}",
                )
            )
        annotations.append(frame_annotations)

    return annotations


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
            print(segment_sizes)
            linewidths = scale_sizes(segment_sizes, 0.0, 8.0)
            colors = [color]
            collection = LineCollection(
                segments,
                colors=colors,
                linewidths=linewidths,
                zorder=2,
                capstyle="round",
                joinstyle="round",
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
                        linewidth=1.0,
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
                s=24,
                linewidths=1.5,
                zorder=3,
            )

    # add dashed red line at y=0
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)

    return fig, ax


def plot_average_norm_values_over_time(ts, real_norm_values, neutral_norm_values):
    # Save one line per seed on a shared set of axes.
    fig, ax = plt.subplots(figsize=(6, 4))

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

    ax.set(xlabel="t", ylabel="$c$", title="Population-average $c$ over time")
    ax.axhline(0, color="red", linestyle="--", linewidth=2.5)

    # remove legend
    ax.legend().remove()

    sns.despine(ax=ax, left=True, bottom=True)

    return fig, ax


def plot_yields(ts, real_yields, neutral_yields):
    # Save one line per seed on a shared set of axes.
    fig, ax = plt.subplots(figsize=(6, 4))

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
        ylabel="yield",
        title="Population-average yield over time",
    )

    # remove legend
    ax.legend().remove()
    sns.despine(ax=ax, left=True, bottom=True)

    return fig, ax


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


def _boundary_segments(horizontal_boundaries, vertical_boundaries):
    horizontal_segments = [
        ((col - 0.5, row + 0.5), (col + 0.5, row + 0.5))
        for row, col in np.argwhere(horizontal_boundaries)
    ]
    vertical_segments = [
        ((col + 0.5, row - 0.5), (col + 0.5, row + 0.5))
        for row, col in np.argwhere(vertical_boundaries)
    ]
    return horizontal_segments + vertical_segments


def _add_boundary_overlay(ax, horizontal_boundaries, vertical_boundaries):
    collection = LineCollection(
        _boundary_segments(horizontal_boundaries, vertical_boundaries),
        colors="limegreen",
        linewidths=1.0,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    ax.add_collection(collection)
    return collection


def render_grid_video(
    norm_value_grids,
    average_norm_values,
    output_path,
    fps=2,
    boundary_masks=None,
    group_yield_annotations=None,
):
    # Render a GIF from already-prepared per-frame norm_value grids plus an
    # average-norm_value time series for the annotation text.
    if fps <= 0:
        raise ValueError("fps must be positive")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        output_path = output_path.with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(
        norm_value_grids[0],
        cmap="vlag",
        vmin=-0.5,
        vmax=0.5,
        interpolation="nearest",
        animated=True,
    )
    _configure_grid_axes(ax, norm_value_grids.shape[1])
    boundary_overlay = None
    if boundary_masks is not None:
        boundary_overlay = _add_boundary_overlay(
            ax, boundary_masks[0][0], boundary_masks[1][0]
        )
    timestamp = ax.text(
        0.02,
        0.98,
        "t=0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        animated=True,
    )
    average_text = ax.text(
        0.02,
        0.90,
        f"avg p={average_norm_values[0]:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        animated=True,
    )
    group_yield_texts = []
    if group_yield_annotations is not None:
        max_labels = max(len(frame_labels) for frame_labels in group_yield_annotations)
        group_yield_texts = [
            ax.text(
                0,
                0,
                "",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 1,
                },
                animated=True,
                zorder=4,
            )
            for _ in range(max_labels)
        ]

    def update(frame_idx):
        # Update both the heatmap and the two annotation boxes each frame.
        image.set_data(norm_value_grids[frame_idx])
        if boundary_overlay is not None:
            boundary_overlay.set_segments(
                _boundary_segments(
                    boundary_masks[0][frame_idx], boundary_masks[1][frame_idx]
                )
            )
        timestamp.set_text(f"t={frame_idx * CGS_INTERVAL}")
        average_text.set_text(f"avg p={average_norm_values[frame_idx]:.3f}")
        if group_yield_annotations is not None:
            frame_labels = group_yield_annotations[frame_idx]
            for label_idx, text_artist in enumerate(group_yield_texts):
                if label_idx < len(frame_labels):
                    col, row, label = frame_labels[label_idx]
                    text_artist.set_position((col, row))
                    text_artist.set_text(label)
                    text_artist.set_visible(True)
                else:
                    text_artist.set_visible(False)
        if boundary_overlay is not None:
            return (
                image,
                boundary_overlay,
                timestamp,
                average_text,
                *group_yield_texts,
            )
        return image, timestamp, average_text, *group_yield_texts

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=norm_value_grids.shape[0],
        interval=1000 / fps,
        blit=True,
    )
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    return output_path


lineage_seeds = [0, 2, 11]
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
                    show_xs=True,
                )
                lineage_axs[ax_idx].set(
                    xlabel="t",
                    ylabel="$c$" if ax_idx == 0 else None,
                    # title=f"Seed {seed_idx}",
                )
                sns.despine(ax=lineage_axs[ax_idx], left=True, bottom=True)

                # boundary_masks = build_boundary_masks(
                #     raw_data[run_type]["sampled_grids"][seed_idx]
                # )
                # yields_group_labels = build_group_labels_for_yield_history(
                #     raw_data[run_type]["group_labels_grids"][seed_idx],
                #     sampled_timesteps,
                # )
                # group_yield_annotations = compute_group_average_yield_annotations(
                #     raw_data[run_type]["sampled_agent_yields"][seed_idx],
                #     yields_group_labels,
                #     max_n_groups=raw_data[run_type]["group_norm_values"].shape[2],
                # )
                # render_grid_video(
                #     processed_data[run_type]["norm_value_grids"][-1],
                #     processed_data[run_type]["average_norm_values"][-1],
                #     VIDEOS_DIR / f"seed_{seed_idx}.gif",
                #     fps=10,
                #     boundary_masks=boundary_masks,
                #     group_yield_annotations=group_yield_annotations,
                # )
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

fig, ax = plot_average_norm_values_over_time(
    sampled_timesteps,
    processed_data_np["real"]["average_norm_values"],
    processed_data_np["neutral"]["average_norm_values"],
)
save_figure(fig, "average_group_norm_values")

fig, ax = plot_yields(
    sampled_timesteps,
    processed_data_np["real"]["population_average_yields"],
    processed_data_np["neutral"]["population_average_yields"],
)
save_figure(fig, "population_average_yields")

fig, axs = plot_norms_vs_yields(
    processed_data_np["real"]["average_norm_values"],
    processed_data_np["real"]["final_window_average_yields"],
    processed_data_np["neutral"]["average_norm_values"],
    processed_data_np["neutral"]["final_window_average_yields"],
)
save_figure(fig, "norms_vs_yields")

print(f"Saved plots to {OUTPUT_DIR.resolve()}")
