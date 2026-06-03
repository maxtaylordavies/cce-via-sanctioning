from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import save_fig

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    title: str
    data_dir: Path
    score_key: str
    score_normalizer: float | str
    role_source: str
    final_window: int | str = 500
    role_window: int = 1000
    sample_interval: int = 20


ENVIRONMENTS = (
    EnvironmentConfig(
        name="mesoudi_env",
        title="Binary trait env",
        data_dir=Path("data/mesoudi_env/experiment_1/gift"),
        score_key="mean_traits",
        score_normalizer="max_total_l",
        role_source="role_probs",
        final_window=500,
        role_window=1000,
    ),
    EnvironmentConfig(
        name="miu_env",
        title="Refinement bandit env",
        data_dir=Path("data/miu_env/experiment_1/gift"),
        score_key="payoffs",
        score_normalizer=1.0,
        role_source="role_probs",
        final_window=500,
        role_window=1000,
    ),
    EnvironmentConfig(
        name="recipe_world_env",
        title="Recipe grammar env",
        data_dir=Path("data/recipe_world_env/experiment_1"),
        score_key="agent_yields",
        score_normalizer=10.0,
        role_source="agent_roles",
        final_window="T_extra",
        role_window=1000,
    ),
)


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
    metadata_keys = {"role_innovate", "role_imitate"}
    if isinstance(config.score_normalizer, str):
        metadata_keys.add(config.score_normalizer)
    if isinstance(config.final_window, str):
        metadata_keys.add(config.final_window)

    return load_npz_outputs(config.data_dir, array_keys, metadata_keys)


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


def get_role_split_df(outputs, config, gains):
    role_innovate = int(get_scalar(outputs, "role_innovate"))
    role_imitate = int(get_scalar(outputs, "role_imitate"))

    if config.role_source == "role_probs":
        role_probs = np.asarray(outputs["role_probs"], dtype=np.float64)
        role_window = role_probs[:, :, -config.role_window :, :]
        innovate = role_window[..., role_innovate].mean(axis=(0, 2))
        imitate = role_window[..., role_imitate].mean(axis=(0, 2))
    else:
        roles = np.asarray(outputs["agent_roles"])
        role_window = roles[:, :, -config.role_window :, :]
        innovate = (role_window == role_innovate).mean(axis=(0, 2, 3))
        imitate = (role_window == role_imitate).mean(axis=(0, 2, 3))

    return pd.DataFrame(
        {
            "prestige_gain": gains,
            "innovate": innovate,
            "imitate": imitate,
        }
    ).sort_values("prestige_gain")


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


def get_peak_gain(final_score_df):
    return (
        final_score_df.groupby("prestige_gain")["score"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )


def get_timeseries_df(score_ts, gains, peak_gain, sample_interval):
    baseline_idx = int(np.argmin(np.abs(gains - 0.0)))
    peak_idx = int(np.argmin(np.abs(gains - peak_gain)))
    selected = {
        baseline_idx: "baseline",
        peak_idx: f"peak ({gains[peak_idx]:g})",
    }

    rows = []
    for gain_idx, condition in selected.items():
        for seed_idx in range(score_ts.shape[0]):
            for t in range(0, score_ts.shape[2], sample_interval):
                rows.append(
                    {
                        "seed": seed_idx,
                        "t": t,
                        "score": score_ts[seed_idx, gain_idx, t],
                        "condition": condition,
                    }
                )
    return pd.DataFrame(rows)


def get_zero_p_innovate_ts(outputs, config):
    gains = get_prestige_gains(outputs)
    zero_idx = int(np.argmin(np.abs(gains - 0.0)))
    role_innovate = int(get_scalar(outputs, "role_innovate"))

    if config.role_source == "role_probs":
        return np.asarray(outputs["role_probs"])[:, zero_idx, :, role_innovate]

    agent_roles = np.asarray(outputs["agent_roles"])[:, zero_idx]
    return (agent_roles == role_innovate).mean(axis=2)


def get_environment_summary_data(config):
    outputs = load_environment_outputs(config, {config.score_key, config.role_source})
    gains = get_prestige_gains(outputs)
    score_ts = get_score_ts(outputs, config)
    final_window = get_final_window(outputs, config)

    final_score_df = get_final_score_df(score_ts, gains, final_window)
    peak_gain = float(get_peak_gain(final_score_df))
    role_split_df = get_role_split_df(outputs, config, gains)
    ts_df = get_timeseries_df(score_ts, gains, peak_gain, config.sample_interval)

    return gains, final_score_df, peak_gain, role_split_df, ts_df


def plot_environment_row(config, axs, show_column_titles=False, show_xlabels=False):
    gains, final_score_df, peak_gain, role_split_df, ts_df = (
        get_environment_summary_data(config)
    )

    panel_titles = (
        "Score over time (proportion of max possible)",
        "Final score as a function of prestige gain",
        "Time-averaged role frequencies",
    )

    axs[0].text(
        -0.24,
        0.5,
        config.title,
        transform=axs[0].transAxes,
        rotation=90,
        va="center",
        ha="right",
        fontsize=13,
        fontweight="bold",
    )
    sns.lineplot(
        ts_df,
        x="t",
        y="score",
        hue="condition",
        ax=axs[0],
        palette=["xkcd:periwinkle", "xkcd:turquoise"],
        legend=False,
        linewidth=2.0,
    )
    axs[0].set(
        title=panel_titles[0] if show_column_titles else None,
        xlabel="$t$" if show_xlabels else None,
        ylabel="Proportion",
        ylim=(0, 1),
    )
    sns.despine(ax=axs[0], left=True, bottom=True)

    sns.lineplot(
        final_score_df,
        x="prestige_gain",
        y="score",
        marker="o",
        color="black",
        err_style="bars",
        ax=axs[1],
        linewidth=2.0,
    )
    axs[1].axvline(peak_gain, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axs[1].set(
        title=panel_titles[1] if show_column_titles else None,
        xlabel="prestige gain" if show_xlabels else None,
        ylabel=None,
    )
    sns.despine(ax=axs[1], left=True, bottom=True)

    axs[2].stackplot(
        role_split_df["prestige_gain"],
        role_split_df["innovate"],
        role_split_df["imitate"],
        labels=["innovate", "imitate"],
        colors=["xkcd:burnt orange", "xkcd:salmon"],
        alpha=1.0,
    )
    axs[2].set(
        title=panel_titles[2] if show_column_titles else None,
        xlabel="prestige gain" if show_xlabels else None,
        ylabel=None,
        xlim=(gains.min(), gains.max()),
    )
    axs[2].grid(False)
    sns.despine(ax=axs[2], left=True, bottom=True)


def plot_environment_summary(config):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.5), sharey=True)
    fig.suptitle(config.title, y=1.03, fontsize=13, fontweight="bold")
    plot_environment_row(config, axs, show_column_titles=True, show_xlabels=True)
    fig.tight_layout()
    return fig


def plot_all_environment_summaries(configs=ENVIRONMENTS):
    fig, axs = plt.subplots(
        len(configs),
        3,
        figsize=(12, 8),
        sharey=True,
        squeeze=False,
    )

    for row_idx, config in enumerate(configs):
        plot_environment_row(
            config,
            axs[row_idx],
            show_column_titles=row_idx == 0,
            show_xlabels=row_idx == len(configs) - 1,
        )

    return fig


def plot_innovation_decay_panel(config, ax, show_ylabel=False):
    outputs = load_environment_outputs(config, {config.role_source})
    innov_prob_ts = get_zero_p_innovate_ts(outputs, config)
    t = np.arange(innov_prob_ts.shape[1])

    for seed_series in innov_prob_ts:
        ax.plot(t, seed_series, color="lightgray", alpha=0.7, linewidth=1)

    ax.plot(t, innov_prob_ts.mean(axis=0), color="black", linewidth=1.5)
    ax.set(
        title=config.title,
        xlabel="$t$",
        ylabel="Probability" if show_ylabel else None,
        xlim=(-5, 205),
        ylim=(0, 0.65),
    )
    sns.despine(ax=ax, left=True, bottom=True)


def plot_all_innovation_decay(configs=ENVIRONMENTS):
    fig, axs = plt.subplots(1, len(configs), figsize=(12, 3), sharey=True)
    fig.suptitle(
        "Initial probability of attempting innovation under zero prestige gain",
        y=1.05,
        fontsize=13,
        fontweight="bold",
    )

    for col_idx, config in enumerate(configs):
        plot_innovation_decay_panel(config, axs[col_idx], show_ylabel=col_idx == 0)

    fig.tight_layout()
    return fig


def main():
    fig = plot_all_environment_summaries()
    save_fig(fig, "experiment_1_combined")
    fig = plot_all_innovation_decay()
    save_fig(fig, "experiment_1_innovation_decay_combined")


if __name__ == "__main__":
    main()
