from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_fig

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

env_name, exp_num = "mesoudi_env", 1
DATA_DIR = Path(f"data/{env_name}/experiment_{exp_num}")
sampling_interval = 10

scalar_keys = {
    "T",
    "grid_length",
    "max_total_l",
    "prestige_decay",
    "prestige_baseline",
    "pool_share",
    "role_innovate",
    "role_imitate",
}
parameter_keys = {"prestige_gains", "fees"}

outputs = {}
concat_buffers = {}
for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".npz"):
        continue
    file_outputs = np.load(DATA_DIR / filename, allow_pickle=True)
    for key in file_outputs.files:
        value = file_outputs[key]

        if key in scalar_keys:
            if key not in outputs:
                outputs[key] = value
            elif outputs[key] != value:
                raise ValueError(f"Inconsistent scalar value for {key!r} in {filename}")
            continue

        if key in parameter_keys:
            if key not in outputs:
                outputs[key] = value
            elif not np.array_equal(outputs[key], value):
                raise ValueError(f"Inconsistent parameter grid for {key!r} in {filename}")
            continue

        if key not in concat_buffers:
            concat_buffers[key] = []
        concat_buffers[key].append(value)

for key, values in concat_buffers.items():
    outputs[key] = np.concatenate(values, axis=0)


def get_parameter_values(raw_outputs):
    if "prestige_gains" in raw_outputs:
        return "prestige_gain", np.asarray(raw_outputs["prestige_gains"], dtype=np.float64)
    return "fee", np.asarray(raw_outputs["fees"], dtype=np.float64)


parameter_name, parameter_values = get_parameter_values(outputs)


def plot_preliminary_innovation_decay(raw_outputs):
    _, values = get_parameter_values(raw_outputs)
    zero_idx = int(np.argmin(np.abs(values - 0.0)))
    role_innovate = int(raw_outputs["role_innovate"])
    innov_prob_ts = raw_outputs["role_probs"][:, zero_idx, :, role_innovate]

    t = np.arange(innov_prob_ts.shape[1])
    window = 5
    kernel = np.ones(window, dtype=np.float64) / window
    # smoothed = np.array(
    #     [np.convolve(seed_series, kernel, mode="same") for seed_series in innov_prob_ts]
    # )

    # smoothed = np.array(
    #     [
    #         pd.Series(seed_series)
    #         .rolling(window=window, min_periods=1, center=True)
    #         .mean()
    #         .to_numpy()
    #         for seed_series in innov_prob_ts
    #     ]
    # )
    smoothed = innov_prob_ts  # no smoothing for now

    fig, ax = plt.subplots(figsize=(6, 3))
    for seed_series in smoothed:
        ax.plot(t, seed_series, color="lightgray", alpha=0.7, linewidth=1)

    ax.plot(t, smoothed.mean(axis=0), color="black", linewidth=1.5)
    ax.set(
        xlabel="$t$",
        ylabel="Probability",
        title="Average probability of attempting innovation over time",
        xlim=(-5, 205),
        ylim=(0, 0.65),
    )
    sns.despine(ax=ax, left=True, bottom=True)
    plt.show()
    # save_fig(
    #     fig,
    #     "preliminary_innovation_decay_prestige_gain_0",
    #     subfolder=f"{env_name}/experiment_{exp_num}",
    # )


rows = []
for seed_idx, seed in enumerate(outputs["seeds"]):
    for parameter_idx, parameter_value in enumerate(parameter_values):
        for t in range(outputs["T"]):
            if t % sampling_interval != 0:
                continue
            rows.append(
                {
                    "seed": seed,
                    parameter_name: parameter_value,
                    "t": t,
                    "mean_traits": outputs["mean_traits"][seed_idx, parameter_idx, t],
                    "p_innovate": outputs["role_probs"][
                        seed_idx, parameter_idx, t, outputs["role_innovate"]
                    ],
                    "n_innovated": outputs["n_innovated"][seed_idx, parameter_idx, t],
                    "n_imitated": outputs["n_imitated"][seed_idx, parameter_idx, t],
                }
            )

df = pd.DataFrame(rows)

df = df[df[parameter_name] >= 0.0]

max_total_l = float(outputs.get("max_total_l", 5000))
df["mean_traits"] = df["mean_traits"] / max_total_l
for col in ["n_innovated", "n_imitated"]:
    if df[col].max() > 0:
        df[col] = df[col] / df[col].max()

x_lims = (df[parameter_name].min(), df[parameter_name].max())

final_df = df[df["t"] == df["t"].max()]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
titles = [
    "Mean cultural score (proportion of max possible)",
    "# successful innovation events (normalised)",
    "# successful transmission events (normalised)",
]
for i, metric in enumerate(["mean_traits", "n_innovated", "n_imitated"]):
    sns.lineplot(
        data=final_df,
        x=parameter_name,
        y=metric,
        ax=axs[i],
        marker="o",
        color="black",
        err_style="bars",
    )
    axs[i].set(
        xlabel="prestige gain" if parameter_name == "prestige_gain" else "$c$",
        xlim=x_lims,
        ylabel=None,
        title=titles[i],
    )
    sns.despine(ax=axs[i], left=True, bottom=True)

plt.show()

# save_fig(fig, "final_metrics_by_prestige_gain", subfolder=f"{env_name}/experiment_{exp_num}")
plot_preliminary_innovation_decay(outputs)
