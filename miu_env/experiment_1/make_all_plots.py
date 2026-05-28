from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_fig

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

env_name, exp_num = "miu_env", 1
DATA_DIR = Path(f"data/{env_name}/experiment_{exp_num}")
sampling_interval = 10

scalar_keys = {
    "T",
    "grid_length",
    "n_arms",
    "role_innovate",
    "role_imitate",
}

for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".npz"):
        continue
    file_outputs = np.load(DATA_DIR / filename, allow_pickle=True)

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

        if key == "fees":
            if key not in outputs:
                outputs[key] = value
            elif not np.array_equal(outputs[key], value):
                raise ValueError(f"Inconsistent fee grid in {filename}")
            continue

        if key not in concat_buffers:
            concat_buffers[key] = []
        concat_buffers[key].append(value)

for key, values in concat_buffers.items():
    outputs[key] = np.concatenate(values, axis=0)


def plot_preliminary_innovation_decay(raw_outputs):
    fees = np.asarray(raw_outputs["fees"], dtype=np.float64)
    fee_zero_idx = int(np.argmin(np.abs(fees - 0.0)))
    role_innovate = int(raw_outputs["role_innovate"])
    innov_prob_ts = raw_outputs["role_probs"][:, fee_zero_idx, :, role_innovate]

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
    save_fig(
        fig,
        "preliminary_innovation_decay_fee_0",
        subfolder=f"{env_name}/experiment_{exp_num}",
    )


rows = []
for seed_idx, seed in enumerate(outputs["seeds"]):
    for fee_idx, fee in enumerate(outputs["fees"]):
        for t in range(outputs["T"]):
            if t % sampling_interval != 0:
                continue
            rows.append(
                {
                    "seed": seed,
                    "fee": fee,
                    "t": t,
                    "payoff": outputs["payoffs"][seed_idx, fee_idx, t],
                    "avg_level": outputs["avg_levels"][seed_idx, fee_idx, t],
                    "max_level": outputs["max_levels"][seed_idx, fee_idx, t],
                    "n_innovated": outputs["n_innov"][seed_idx, fee_idx, t],
                    "n_imitated": outputs["n_imit"][seed_idx, fee_idx, t],
                }
            )

df = pd.DataFrame(rows)

for col in ["avg_level", "max_level"]:
    df[col] = df[col] / 100
for col in ["fee", "n_innovated", "n_imitated"]:
    df[col] = df[col] / df[col].max()

x_lims = (-1.05, 1.05)
lower_goldilocks, upper_goldilocks = 0.05, 0.45

final_df = df[df["t"] == df["t"].max()]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
titles = [
    "Mean cultural score (proportion of max possible)",
    "# successful innovation events (normalised)",
    "# successful transmission events (normalised)",
]
for i, metric in enumerate(["payoff", "n_innovated", "n_imitated"]):
    sns.lineplot(
        data=final_df,
        x="fee",
        y=metric,
        ax=axs[i],
        marker="o",
        color="black",
        err_style="bars",
    )
    axs[i].axvspan(x_lims[0], lower_goldilocks, color="#f4c7c3", alpha=0.3, zorder=0)
    axs[i].axvspan(
        lower_goldilocks, upper_goldilocks, color="#d8f0c8", alpha=0.3, zorder=0
    )
    axs[i].axvspan(upper_goldilocks, x_lims[1], color="#f4c7c3", alpha=0.3, zorder=0)
    axs[i].set(xlabel="$c$ (normalised)", xlim=x_lims, ylabel=None, title=titles[i])
    sns.despine(ax=axs[i], left=True, bottom=True)

trans = axs[0].get_xaxis_transform()
axs[0].text(
    x_lims[0] + 0.03,
    0.98,
    "too little\ninnovation\nand transmission",
    transform=trans,
    ha="left",
    va="top",
    fontsize=9,
    fontweight="bold",
    color="#831e17",
)
axs[0].text(
    lower_goldilocks + 0.2,
    0.98,
    '"goldilocks\nzone"',
    transform=trans,
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
    color="#0e4812",
)
axs[0].text(
    x_lims[1] - 0.03,
    0.98,
    "too little\ntransmission",
    transform=trans,
    ha="right",
    va="top",
    fontsize=9,
    fontweight="bold",
    color="#831e17",
)

save_fig(fig, "final_metrics", subfolder=f"{env_name}/experiment_{exp_num}")
plot_preliminary_innovation_decay(outputs)
