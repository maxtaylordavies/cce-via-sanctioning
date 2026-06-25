from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import save_fig

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

TARGET_INNOVATOR_FREQUENCY = 0.5
REFERENCE_GAINS_PATH = Path("figures/experiment_1_reference_gains.csv")


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    title: str
    score_key: str
    score_normalizer: float | str
    final_window: int | str = 500
    data_dir: Path = Path("data")


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
        # final_window="T_extra",
    ),
)


def get_data_path(config, variant):
    return Path(f"data/{config.name}/experiment_1/{variant}")


def load_npz_outputs(data_dir, array_keys, metadata_keys=()):
    outputs = {}
    concat_buffers = {}
    parameter_keys = {"fees", "prestige_gains"}
    array_keys = set(array_keys)
    metadata_keys = set(metadata_keys)

    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as file_outputs:
            for key in file_outputs.files:
                if key not in array_keys and key not in (
                    parameter_keys | metadata_keys
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

                if key in parameter_keys:
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


def get_prestige_gains(outputs):
    key = "prestige_gains" if "prestige_gains" in outputs else "fees"
    return np.asarray(outputs[key], dtype=np.float64)


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
            "Expected agent_roles to have shape (seed, prestige_gain, time, agent), "
            f"but got {roles.shape}."
        )

    if roles.min() < min_role or roles.max() > max_role:
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


def get_innovator_frequency_df(outputs, config, gains):
    role_innovate = int(get_scalar(outputs, "role_innovate"))
    roles = get_agent_roles(outputs)
    innovate = (roles == role_innovate).mean(axis=(2, 3))

    rows = []
    for seed_idx in range(innovate.shape[0]):
        for gain_idx, gain in enumerate(gains):
            rows.append(
                {
                    "seed": seed_idx,
                    "prestige_gain": gain,
                    "innovator_frequency": innovate[seed_idx, gain_idx],
                }
            )
    return pd.DataFrame(rows)


def get_any_innovation_attempt_df(outputs, config, gains):
    role_innovate = int(get_scalar(outputs, "role_innovate"))
    final_window = get_final_window(outputs, config)

    roles = get_agent_roles(outputs)
    any_innovation = (roles == role_innovate).any(axis=-1)
    probabilities = any_innovation[:, :, -final_window:].mean(axis=2)

    rows = []
    for seed_idx in range(probabilities.shape[0]):
        for gain_idx, gain in enumerate(gains):
            rows.append(
                {
                    "seed": seed_idx,
                    "prestige_gain": gain,
                    "any_innovation_probability": probabilities[seed_idx, gain_idx],
                }
            )
    return pd.DataFrame(rows)


def get_final_score_df(score_ts, gains, final_window):
    final_scores = score_ts[:, :, -final_window:].mean(axis=2)
    rows = []
    for seed_idx in range(final_scores.shape[0]):
        for gain_idx, gain in enumerate(gains):
            rows.append(
                {
                    "seed": seed_idx,
                    "prestige_gain": gain,
                    "score": final_scores[seed_idx, gain_idx],
                }
            )
    return pd.DataFrame(rows)


def get_reference_prestige_gain(innovator_frequency_df):
    mean_frequencies = (
        innovator_frequency_df.groupby("prestige_gain")["innovator_frequency"]
        .mean()
        .sort_index()
    )
    gains = mean_frequencies.index.to_numpy(dtype=float)
    frequencies = mean_frequencies.to_numpy(dtype=float)
    crossing_idxs = np.flatnonzero(frequencies >= TARGET_INNOVATOR_FREQUENCY)

    if crossing_idxs.size == 0:
        raise ValueError(
            f"Innovator frequency never reaches {TARGET_INNOVATOR_FREQUENCY}"
        )

    upper_idx = int(crossing_idxs[0])
    if upper_idx == 0:
        return gains[0]

    lower_idx = upper_idx - 1
    frequency_span = frequencies[upper_idx] - frequencies[lower_idx]
    if np.isclose(frequency_span, 0):
        return gains[upper_idx]

    crossing_fraction = (
        TARGET_INNOVATOR_FREQUENCY - frequencies[lower_idx]
    ) / frequency_span
    return gains[lower_idx] + crossing_fraction * (gains[upper_idx] - gains[lower_idx])


def get_environment_summary_data(config):
    outputs = load_environment_outputs(config, {config.score_key, "agent_roles"})
    gains = get_prestige_gains(outputs)
    score_ts = get_score_ts(outputs, config)
    final_window = get_final_window(outputs, config)

    final_score_df = get_final_score_df(score_ts, gains, final_window)
    innovator_frequency_df = get_innovator_frequency_df(outputs, config, gains)
    any_innovation_attempt_df = get_any_innovation_attempt_df(outputs, config, gains)

    return (
        gains,
        final_score_df,
        innovator_frequency_df,
        any_innovation_attempt_df,
    )


def plot_environment_panel(
    config,
    ax,
    show_title=False,
    show_xlabel=False,
    show_yticks=False,
    show_any_innovation_attempt=False,
):
    (
        gains,
        final_score_df,
        innovator_frequency_df,
        any_innovation_attempt_df,
    ) = get_environment_summary_data(config)
    reference_gain = get_reference_prestige_gain(innovator_frequency_df)
    final_score_df = final_score_df.assign(
        normalized_gain=final_score_df["prestige_gain"] / reference_gain
    )
    innovator_frequency_df = innovator_frequency_df.assign(
        normalized_gain=innovator_frequency_df["prestige_gain"] / reference_gain
    )
    any_innovation_attempt_df = any_innovation_attempt_df.assign(
        normalized_gain=(any_innovation_attempt_df["prestige_gain"] / reference_gain)
    )
    mean_scores = final_score_df.groupby("normalized_gain")["score"].mean()
    peak_gain = mean_scores.idxmax()

    sns.lineplot(
        final_score_df,
        x="normalized_gain",
        y="score",
        marker="o",
        color="black",
        err_style="bars",
        ax=ax,
        linewidth=2.5,
    )
    ax.axvline(
        peak_gain,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.45,
        zorder=0,
    )
    sns.lineplot(
        innovator_frequency_df,
        x="normalized_gain",
        y="innovator_frequency",
        marker="s",
        color="xkcd:red",
        err_style="bars",
        ax=ax,
        linewidth=2.0,
        alpha=0.5,
    )
    if show_any_innovation_attempt:
        sns.lineplot(
            any_innovation_attempt_df,
            x="normalized_gain",
            y="any_innovation_probability",
            marker="^",
            color="xkcd:pink",
            err_style="bars",
            ax=ax,
            linewidth=2.0,
            alpha=0.5,
        )

    normalized_gains = gains / reference_gain
    gain_span = normalized_gains.max() - normalized_gains.min()
    gain_margin = 0.05 * gain_span if gain_span > 0 else 0.1
    ax.set(
        title=config.title if show_title else None,
        xlabel=r"Normalised prestige gain ($g/g_{0.5}$)" if show_xlabel else None,
        ylabel=None,
        xlim=(
            normalized_gains.min() - gain_margin,
            normalized_gains.max() + gain_margin,
        ),
        ylim=(-0.05, 1.05),
    )
    shared_y_ticks = np.linspace(0, 1, 6)
    ax.set_yticks(shared_y_ticks)
    ax.tick_params(axis="y", length=0)
    if not show_yticks:
        ax.tick_params(axis="y", labelleft=False)

    sns.despine(ax=ax, left=True, bottom=True)
    return reference_gain


def plot_all_environment_summaries():
    variants = {"intrinsic": "Intrinsic motivation", "gift": "Deference gifting"}
    reference_gain_rows = []
    fig, axs = plt.subplots(
        len(variants),
        len(ENVIRONMENTS),
        figsize=(13, 6),
        squeeze=False,
    )

    for row_idx, variant in enumerate(list(variants.keys())):
        for col_idx, env_config in enumerate(ENVIRONMENTS):
            config_dict = env_config.__dict__.copy()
            config_dict["data_dir"] = get_data_path(env_config, variant)
            config = EnvironmentConfig(**config_dict)
            reference_gain = plot_environment_panel(
                config,
                axs[row_idx, col_idx],
                show_title=row_idx == 0,
                show_xlabel=row_idx == len(variants) - 1,
                show_yticks=col_idx == 0,
            )
            reference_gain_rows.append(
                {
                    "environment": env_config.name,
                    "environment_title": env_config.title,
                    "variant": variant,
                    "target_innovator_frequency": TARGET_INNOVATOR_FREQUENCY,
                    "reference_prestige_gain": reference_gain,
                }
            )

        axs[row_idx, 0].text(
            -0.1,
            0.5,
            variants[variant],
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
            color="black",
            marker="o",
            linewidth=2,
            label="Final average payoff (normalised)",
        ),
        Line2D(
            [0],
            [0],
            color="xkcd:red",
            alpha=0.5,
            marker="s",
            linewidth=2,
            label="Innovator frequency (mean over all timesteps)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )
    return fig, pd.DataFrame(reference_gain_rows)


def main():
    fig, reference_gain_df = plot_all_environment_summaries()
    REFERENCE_GAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference_gain_df.to_csv(REFERENCE_GAINS_PATH, index=False)
    save_fig(fig, "experiment_1_combined")


if __name__ == "__main__":
    main()
