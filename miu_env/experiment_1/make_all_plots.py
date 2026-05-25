from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

DATA_DIR = Path("data/miu_env/experiment_1")
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
                    # "p_innovate": outputs["role_probs"][
                    #     seed_idx, fee_idx, t, outputs["role_innovate"]
                    # ],
                    "n_innovated": outputs["n_innov"][seed_idx, fee_idx, t],
                    "n_imitated": outputs["n_imit"][seed_idx, fee_idx, t],
                }
            )

df = pd.DataFrame(rows)

min_fee, max_fee = df["fee"].min(), df["fee"].max()
x_lims = (min_fee - 0.2, max_fee + 0.2)
# lower_goldilocks, upper_goldilocks = 0.5, 2.0

final_df = df[df["t"] == df["t"].max()]
fig, axs = plt.subplots(1, 5, figsize=(20, 4))
for i, metric in enumerate(
    ["payoff", "avg_level", "max_level", "n_innovated", "n_imitated"]
):
    sns.lineplot(
        data=final_df,
        x="fee",
        y=metric,
        ax=axs[i],
        marker="o",
        color="black",
        err_style="bars",
    )
    # axs[i].axvspan(x_lims[0], lower_goldilocks, color="#f4c7c3", alpha=0.3, zorder=0)
    # axs[i].axvspan(
    #     lower_goldilocks, upper_goldilocks, color="#d8f0c8", alpha=0.3, zorder=0
    # )
    # axs[i].axvspan(upper_goldilocks, x_lims[1], color="#f4c7c3", alpha=0.3, zorder=0)
    axs[i].set(xlabel="Fee", xlim=x_lims)
    sns.despine(ax=axs[i], left=True, bottom=True)

# trans = axs[0].get_xaxis_transform()
# axs[0].text(
#     x_lims[0] + 0.07,
#     0.98,
#     "too little\ninnovation\nand transmission",
#     transform=trans,
#     ha="left",
#     va="top",
#     fontsize=9,
#     fontweight="bold",
#     color="#831e17",
# )
# axs[0].text(
#     lower_goldilocks + 0.2,
#     0.98,
#     '"goldilocks\nzone"',
#     transform=trans,
#     ha="center",
#     va="top",
#     fontsize=9,
#     fontweight="bold",
#     color="#0e4812",
# )
# axs[0].text(
#     x_lims[1] - 0.07,
#     0.98,
#     "too little\ntransmission",
#     transform=trans,
#     ha="right",
#     va="top",
#     fontsize=9,
#     fontweight="bold",
#     color="#831e17",
# )

fig.tight_layout()
plt.show()
