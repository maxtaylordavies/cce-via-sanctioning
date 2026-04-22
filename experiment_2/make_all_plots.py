import numpy as np
from pathlib import Path
from matplotlib import animation
import matplotlib
from matplotlib.collections import LineCollection
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sns.set_style("whitegrid")

file_outputs = np.load("simulation_outputs_4-4.npz", allow_pickle=True)
# file_outputs = np.load("real.npz", allow_pickle=True)

full_group_norm_values = file_outputs["group_norm_values"]
interval = 25
output_dir = Path("tmp_plots")
output_dir.mkdir(parents=True, exist_ok=True)

full_grids = file_outputs["group_labels_grids"]
full_group_instance_ids_by_label_history = file_outputs[
    "group_instance_ids_by_label_history"
]
group_lineage_arrays = file_outputs["group_lineage_arrays"]
final_next_group_instance_ids = file_outputs["final_next_group_instance_ids"]
full_agent_yields = file_outputs["agent_yields"]
full_agent_roles = file_outputs["agent_roles"]
full_pop_role_rewards = file_outputs["pop_role_rewards"]
role_innovate = int(file_outputs["role_innovate"])
role_imitate = int(file_outputs["role_imitate"])

n_seeds = full_group_norm_values.shape[0]
sampled_timesteps = np.arange(0, full_agent_yields.shape[1], interval)
sampled_group_norm_values = full_group_norm_values[:, ::interval]
sampled_grids = full_grids[:, ::interval]
sampled_agent_yields = full_agent_yields[:, sampled_timesteps]
sampled_agent_roles = full_agent_roles[:, sampled_timesteps]
sampled_pop_role_rewards = full_pop_role_rewards[:, sampled_timesteps]

max_n_groups = full_group_norm_values.shape[2]


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


def compute_best_group_average_yields(
    agent_yields_history,
    group_labels_for_yields,
    max_n_groups,
):
    # For each timestep, compute the mean yield of each non-empty group and
    # return the best such group mean.
    n_steps = agent_yields_history.shape[0]
    flat_yields = agent_yields_history.reshape(n_steps, -1)
    flat_group_labels = group_labels_for_yields.reshape(n_steps, -1)
    best_group_average_yields = np.full(n_steps, np.nan, dtype=np.float64)

    for t in range(n_steps):
        counts = np.bincount(flat_group_labels[t], minlength=max_n_groups)
        total_yields = np.bincount(
            flat_group_labels[t],
            weights=flat_yields[t],
            minlength=max_n_groups,
        )
        non_empty_groups = counts > 0
        if np.any(non_empty_groups):
            group_mean_yields = (
                total_yields[non_empty_groups] / counts[non_empty_groups]
            )
            best_group_average_yields[t] = np.max(group_mean_yields)

    return best_group_average_yields


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


def plot_group_lineage(
    instance_norm_series,
    group_lineage_array,
    timestep_scale=1,
    ax=None,
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

    for instance_id in range(n_instances):
        active_steps = np.flatnonzero(~np.isnan(instance_norm_series[instance_id]))
        if len(active_steps) == 0:
            continue

        color = cmap(instance_id % cmap.N)
        ax.plot(
            times[active_steps],
            instance_norm_series[instance_id, active_steps],
            color=color,
            linewidth=2,
            alpha=1.0,
            zorder=2,
        )
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
                        linewidth=1.2,
                        alpha=0.9,
                        zorder=1,
                    )

        last_step = active_steps[-1]
        if last_step < n_steps - 1 and not has_children[instance_id]:
            ax.scatter(
                times[last_step],
                instance_norm_series[instance_id, last_step],
                color=line_color_by_instance[instance_id],
                marker="x",
                s=36,
                linewidths=1.5,
                zorder=3,
            )

    ax.set_xlabel("t")
    ax.set_ylabel("Group norm value")
    ax.set_title("Group Lineage Plot")
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    return fig, ax


def save_line_plot_by_seed(
    values_by_seed,
    timesteps,
    output_path,
    ylabel,
    title,
):
    # Save one line per seed on a shared set of axes.
    fig, ax = plt.subplots(figsize=(10, 6))
    for seed_idx, seed_values in enumerate(values_by_seed):
        sns.lineplot(
            x=timesteps,
            y=seed_values,
            ax=ax,
            label=f"seed {seed_idx}",
        )
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_role_fraction_history(agent_roles_history, role_id):
    # Summarise individual role choices as the population fraction taking a
    # given role at each timestep.
    return np.mean(agent_roles_history == role_id, axis=-1)


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
    norm_value_grids, average_norm_values, output_path, fps=2, boundary_masks=None
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

    def update(frame_idx):
        # Update both the heatmap and the two annotation boxes each frame.
        image.set_data(norm_value_grids[frame_idx])
        if boundary_overlay is not None:
            boundary_overlay.set_segments(
                _boundary_segments(
                    boundary_masks[0][frame_idx], boundary_masks[1][frame_idx]
                )
            )
        timestamp.set_text(f"t={frame_idx * interval}")
        average_text.set_text(f"avg p={average_norm_values[frame_idx]:.3f}")
        if boundary_overlay is not None:
            return image, boundary_overlay, timestamp, average_text
        return image, timestamp, average_text

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


average_norm_values_by_seed = []
population_average_yields_by_seed = []
best_group_average_yields_by_seed = []
innovator_fraction_by_seed = compute_role_fraction_history(
    sampled_agent_roles,
    role_innovate,
)
imitator_fraction_by_seed = compute_role_fraction_history(
    sampled_agent_roles,
    role_imitate,
)
innovator_reward_by_seed = sampled_pop_role_rewards[:, :, role_innovate]
imitator_reward_by_seed = sampled_pop_role_rewards[:, :, role_imitate]

for seed_idx in range(n_seeds):
    norm_value_grids = build_norm_value_grids(
        sampled_grids[seed_idx], sampled_group_norm_values[seed_idx]
    )
    average_norm_values = compute_average_norm_values(norm_value_grids)
    average_norm_values_by_seed.append(average_norm_values)
    boundary_masks = build_boundary_masks(sampled_grids[seed_idx])
    render_grid_video(
        norm_value_grids,
        average_norm_values,
        output_dir / f"group_grid_seed_{seed_idx}.gif",
        fps=10,
        boundary_masks=boundary_masks,
    )

    yields_group_labels = build_group_labels_for_yield_history(
        full_grids[seed_idx], sampled_timesteps
    )
    population_average_yields_by_seed.append(
        sampled_agent_yields[seed_idx].mean(axis=1)
    )
    best_group_average_yields_by_seed.append(
        compute_best_group_average_yields(
            sampled_agent_yields[seed_idx],
            yields_group_labels,
            max_n_groups,
        )
    )

    full_instance_norm_series = build_group_instance_norm_series(
        full_group_norm_values[seed_idx],
        full_group_instance_ids_by_label_history[seed_idx],
        int(final_next_group_instance_ids[seed_idx]),
    )
    fig, ax = plot_group_lineage(
        full_instance_norm_series,
        group_lineage_arrays[seed_idx],
        timestep_scale=1,
    )
    ax.set_title(f"Group Lineage Plot (Seed {seed_idx})")
    lineage_path = output_dir / f"group_lineage_seed_{seed_idx}.png"
    fig.savefig(lineage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

average_norm_values_by_seed = np.asarray(average_norm_values_by_seed)
population_average_yields_by_seed = np.asarray(population_average_yields_by_seed)
best_group_average_yields_by_seed = np.asarray(best_group_average_yields_by_seed)

save_line_plot_by_seed(
    average_norm_values_by_seed,
    sampled_timesteps,
    output_dir / "average_group_norm_value_by_seed.png",
    ylabel="Grid-weighted norm value",
    title="Grid-Weighted Group Norm Value by Seed",
)
save_line_plot_by_seed(
    population_average_yields_by_seed,
    sampled_timesteps,
    output_dir / "population_average_yield_by_seed.png",
    ylabel="Population average yield",
    title="Population Average Yield by Seed",
)
save_line_plot_by_seed(
    best_group_average_yields_by_seed,
    sampled_timesteps,
    output_dir / "best_group_average_yield_by_seed.png",
    ylabel="Best group average yield",
    title="Best Group Average Yield by Seed",
)
save_line_plot_by_seed(
    innovator_fraction_by_seed,
    sampled_timesteps,
    output_dir / "innovator_fraction_by_seed.png",
    ylabel="Fraction choosing innovate",
    title="Innovator Fraction by Seed",
)
save_line_plot_by_seed(
    imitator_fraction_by_seed,
    sampled_timesteps,
    output_dir / "imitator_fraction_by_seed.png",
    ylabel="Fraction choosing imitate",
    title="Imitator Fraction by Seed",
)
save_line_plot_by_seed(
    innovator_reward_by_seed,
    sampled_timesteps,
    output_dir / "innovator_reward_by_seed.png",
    ylabel="Average innovator reward",
    title="Average Innovator Reward by Seed",
)
save_line_plot_by_seed(
    imitator_reward_by_seed,
    sampled_timesteps,
    output_dir / "imitator_reward_by_seed.png",
    ylabel="Average imitator reward",
    title="Average Imitator Reward by Seed",
)

print(f"Saved plots to {output_dir.resolve()}")
