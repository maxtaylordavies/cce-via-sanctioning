import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("whitegrid")

df = pd.read_csv("simulation_data.csv")
df = df[df["t"] % 10 == 0]

for hue in ["agent", None]:
    kwargs = {"legend": False}
    if hue:
        kwargs = {**kwargs, "hue": hue, "palette": "crest", "alpha": 0.5}
    else:
        kwargs = {**kwargs, "color": "black"}

    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    axs = axs.flatten()

    sns.lineplot(df, x="t", y="energy", ax=axs[0], **kwargs)
    axs[0].set_title("Energy")

    sns.lineplot(df, x="t", y="foraged_level", ax=axs[1], **kwargs)
    axs[1].set_title("Average level of foraged plants")

    sns.lineplot(df, x="t", y="yield", ax=axs[2], **kwargs)
    axs[2].set_title("Average yield")

    sns.lineplot(df, x="t", y="library_complexity", ax=axs[3], **kwargs)
    axs[3].set_title("Library complexity")

    sns.lineplot(df, x="t", y="p_innovate", ax=axs[4], **kwargs)
    axs[4].set_title("Probability of innovating")

    sns.lineplot(df, x="t", y="p_imitate", ax=axs[5], **kwargs)
    axs[5].set_title("Probability of imitating")

    for ax in axs:
        sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()
    fig.savefig(f"figures/simulation_plots_{'per_agent' if hue else 'pop_average'}.png")

fig, ax = plt.subplots()
sns.lineplot(df, x="t", y="diversity", ax=ax)
ax.set_title("Population diversity (mean pairwise Jaccard distance)")
sns.despine(ax=ax, left=True, bottom=True)
plt.tight_layout()
fig.savefig("figures/simulation_plots_diversity.png")
