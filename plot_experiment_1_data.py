from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import save_fig

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

DEFAULT_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    column: str
    xlabel: str


PARAMETER_SPECS = {
    "value_capture_rates": ParameterSpec(
        key="value_capture_rates",
        column="value_capture_rate",
        xlabel=r"Value-capture rate ($\lambda$)",
    ),
    # These aliases make the loader tolerant of likely names used by the later
    # environment ports.
    "lambdas": ParameterSpec(
        key="lambdas",
        column="value_capture_rate",
        xlabel=r"Value-capture rate ($\lambda$)",
    ),
}
PARAMETER_KEYS = set(PARAMETER_SPECS)


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    title: str
    score_key: str
    score_normalizer: float | str
    final_window: int | str = 500
    data_dir: Path = DEFAULT_DATA_ROOT


ENVIRONMENTS = (
    EnvironmentConfig(
        name="mesoudi_env",
        title="Binary trait environment",
        score_key="mean_traits",
        score_normalizer="max_total_l",
        final_window=1000,
    ),
    EnvironmentConfig(
        name="miu_env",
        title="Refinement bandit environment",
        score_key="payoffs",
        score_normalizer=1.0,
        final_window=100,
    ),
    EnvironmentConfig(
        name="recipe_world_env",
        title="Recipe grammar environment",
        score_key="agent_yields",
        score_normalizer=10.0,
        final_window=100,
    ),
)


def get_experiment_data_path(config, data_root=DEFAULT_DATA_ROOT):
    return Path(data_root) / config.name / "experiment_1"


def contains_npz_outputs(data_dir):
    return Path(data_dir).is_dir() and any(Path(data_dir).glob("*.npz"))


def discover_available_datasets(data_root=DEFAULT_DATA_ROOT):
    """Return available datasets, preferring the new value-capture outputs.

    A new single-version output may be placed directly in an environment's
    ``experiment_1`` directory or in a ``value_capture`` subdirectory.  Once at
    least one such dataset exists, legacy prestige datasets are omitted so that
    old and new mechanisms are not mixed in the same figure.
    """
    value_capture_configs = []
    for config in ENVIRONMENTS:
        experiment_dir = get_experiment_data_path(config, data_root)

        # check path exists
        if Path(experiment_dir).exists():
            value_capture_configs.append(replace(config, data_dir=experiment_dir))

    return value_capture_configs


def load_npz_outputs(data_dir, array_keys, metadata_keys=()):
    outputs = {}
    concat_buffers = {}
    array_keys = set(array_keys)
    metadata_keys = set(metadata_keys)
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz outputs found in {Path(data_dir)!s}.")

    for path in paths:
        with np.load(path, allow_pickle=True) as file_outputs:
            for key in file_outputs.files:
                if key not in array_keys and key not in (
                    PARAMETER_KEYS | metadata_keys
                ):
                    continue

                value = file_outputs[key]
                if value.shape == ():
                    if key not in outputs:
                        outputs[key] = value
                    elif not np.array_equal(outputs[key], value):
                        raise ValueError(
                            f"Inconsistent scalar value for {key!r} in {path}"
                        )
                    continue

                if key in PARAMETER_KEYS:
                    if key not in outputs:
                        outputs[key] = value
                    elif not np.array_equal(outputs[key], value):
                        raise ValueError(
                            f"Inconsistent parameter grid for {key!r} in {path}"
                        )
                    continue

                concat_buffers.setdefault(key, []).append(value)

    for key, values in concat_buffers.items():
        outputs[key] = np.concatenate(values, axis=0)
    return outputs


def get_scalar(outputs, key, default=None):
    if key not in outputs:
        return default
    value = outputs[key]
    if getattr(value, "shape", None) == ():
        return value.item()
    return value


def get_parameter_values(outputs):
    for key, spec in PARAMETER_SPECS.items():
        if key in outputs:
            return np.asarray(outputs[key], dtype=np.float64), spec
    raise KeyError(
        "No supported experiment parameter found. Expected one of "
        f"{sorted(PARAMETER_KEYS)}."
    )


def load_environment_outputs(config, array_keys):
    metadata_keys = {"T", "grid_length", "role_innovate", "role_imitate"}
    if isinstance(config.score_normalizer, str):
        metadata_keys.add(config.score_normalizer)
    if isinstance(config.final_window, str):
        metadata_keys.add(config.final_window)
    return load_npz_outputs(config.data_dir, array_keys, metadata_keys)


def get_agent_roles(outputs):
    roles = np.asarray(outputs["agent_roles"])
    role_innovate = int(get_scalar(outputs, "role_innovate"))
    role_imitate = int(get_scalar(outputs, "role_imitate"))
    min_role = min(role_innovate, role_imitate)
    max_role = max(role_innovate, role_imitate)

    if roles.ndim != 4:
        raise ValueError(
            "Expected agent_roles to have shape (seed, parameter, time, agent), "
            f"but got {roles.shape}."
        )
    if roles.size and (roles.min() < min_role or roles.max() > max_role):
        raise ValueError(
            "agent_roles contains values outside the saved role codes: "
            f"expected values between {min_role} and {max_role}."
        )
    return roles


def get_score_ts(outputs, config):
    score = np.asarray(outputs[config.score_key])
    if config.score_key == "agent_yields":
        score = score.mean(axis=-1, dtype=np.float64)

    normalizer = config.score_normalizer
    if isinstance(normalizer, str):
        normalizer = float(get_scalar(outputs, normalizer, 1.0))
    return score / float(normalizer)


def get_final_window(outputs, config):
    if isinstance(config.final_window, str):
        return int(get_scalar(outputs, config.final_window, 500))
    return int(config.final_window)


def get_innovator_frequency_df(
    outputs, parameter_values, parameter_column, final_window
):
    role_innovate = int(get_scalar(outputs, "role_innovate"))
    roles = get_agent_roles(outputs)
    innovate = (roles == role_innovate)[:, :, -final_window:].mean(axis=(2, 3))

    rows = []
    for seed_idx in range(innovate.shape[0]):
        for parameter_idx, parameter_value in enumerate(parameter_values):
            rows.append(
                {
                    "seed": seed_idx,
                    parameter_column: parameter_value,
                    "innovator_frequency": innovate[seed_idx, parameter_idx],
                }
            )
    return pd.DataFrame(rows)


def get_final_score_df(score_ts, parameter_values, parameter_column, final_window):
    final_scores = score_ts[:, :, -final_window:].mean(axis=2)
    rows = []
    for seed_idx in range(final_scores.shape[0]):
        for parameter_idx, parameter_value in enumerate(parameter_values):
            rows.append(
                {
                    "seed": seed_idx,
                    parameter_column: parameter_value,
                    "score": final_scores[seed_idx, parameter_idx],
                }
            )
    return pd.DataFrame(rows)


def get_environment_summary_data(config):
    outputs = load_environment_outputs(config, {config.score_key, "agent_roles"})
    parameter_values, parameter_spec = get_parameter_values(outputs)
    score_ts = get_score_ts(outputs, config)
    final_window = get_final_window(outputs, config)

    final_score_df = get_final_score_df(
        score_ts, parameter_values, parameter_spec.column, final_window
    )
    innovator_frequency_df = get_innovator_frequency_df(
        outputs, parameter_values, parameter_spec.column, final_window
    )
    return (
        parameter_values,
        parameter_spec,
        final_score_df,
        innovator_frequency_df,
    )


def plot_environment_panel(
    config,
    ax,
    show_title=False,
    show_xlabel=False,
    show_yticks=False,
):
    (
        _,
        parameter_spec,
        final_score_df,
        innovator_frequency_df,
    ) = get_environment_summary_data(config)

    plot_column = parameter_spec.column

    mean_scores = final_score_df.groupby(plot_column)["score"].mean()
    peak_parameter = mean_scores.idxmax()
    sns.lineplot(
        final_score_df,
        x=plot_column,
        y="score",
        marker="o",
        color="black",
        err_style="bars",
        ax=ax,
        linewidth=2.5,
    )
    ax.axvline(
        peak_parameter,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.45,
        zorder=0,
    )
    sns.lineplot(
        innovator_frequency_df,
        x=plot_column,
        y="innovator_frequency",
        marker="s",
        color="xkcd:red",
        err_style="bars",
        ax=ax,
        linewidth=2.0,
        alpha=0.5,
    )

    plotted_parameters = np.asarray(
        final_score_df[plot_column].drop_duplicates(), dtype=float
    )
    parameter_span = plotted_parameters.max() - plotted_parameters.min()
    parameter_margin = 0.05 * parameter_span if parameter_span > 0 else 0.1
    ax.set(
        title=config.title if show_title else None,
        xlabel=parameter_spec.xlabel if show_xlabel else None,
        ylabel=None,
        xlim=(
            plotted_parameters.min() - parameter_margin,
            plotted_parameters.max() + parameter_margin,
        ),
        ylim=(-0.05, 1.05),
        # ylim=(0.7, 0.8),
    )
    # ax.set_yticks(np.linspace(0, 1, 6))
    ax.tick_params(axis="y", length=0)
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)
    sns.despine(ax=ax, left=True, bottom=True)


def plot_all_environment_summaries(data_root=DEFAULT_DATA_ROOT):
    fig, axs = plt.subplots(
        1,
        len(ENVIRONMENTS),
        figsize=(4 * len(ENVIRONMENTS), 3.5),
    )

    configs = discover_available_datasets(data_root)
    configs_by_name = {config.name: config for config in configs}

    for col_idx, environment in enumerate(ENVIRONMENTS):
        config = configs_by_name.get(environment.name)

        if config is None:
            axs[col_idx].set_visible(False)
            continue

        plot_environment_panel(
            config,
            axs[col_idx],
            show_title=True,
            show_xlabel=True,
            show_yticks=col_idx == 0,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linewidth=2,
            label="Final cultural performance (normalised)",
        ),
        Line2D(
            [0],
            [0],
            color="xkcd:red",
            alpha=0.5,
            marker="s",
            linewidth=2,
            label="Final innovator frequency",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def main():
    fig = plot_all_environment_summaries()
    save_fig(fig, "experiment_1_combined", tight=False)


if __name__ == "__main__":
    main()
