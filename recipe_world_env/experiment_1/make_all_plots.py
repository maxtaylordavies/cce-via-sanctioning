from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from src.utils import save_fig

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

env_name, exp_num = "recipe_world_env", 1
DATA_DIR = Path(f"data/{env_name}/experiment_{exp_num}")


def load_raw_outputs():
    scalar_keys = {
        "T",
        "T_main",
        "T_extra",
        "grid_length",
        "num_rules_in_initial_library",
        "empty_recipe_id",
        "role_innovate",
        "role_imitate",
    }
    outputs = {}
    concat_buffers = {}

    for path in sorted(DATA_DIR.glob("*.npz")):
        file_outputs = np.load(path, allow_pickle=True)
        for key in file_outputs.files:
            value = file_outputs[key]
            if key in scalar_keys:
                if key not in outputs:
                    outputs[key] = value
                elif outputs[key] != value:
                    raise ValueError(
                        f"Inconsistent scalar value for {key!r} in {path.name}"
                    )
                continue

            if key == "fees":
                if key not in outputs:
                    outputs[key] = value
                elif not np.array_equal(outputs[key], value):
                    raise ValueError(f"Inconsistent fee grid in {path.name}")
                continue

            concat_buffers.setdefault(key, []).append(value)

    for key, values in concat_buffers.items():
        outputs[key] = np.concatenate(values, axis=0)

    return outputs


def load_data():
    return {
        "population": pd.read_csv(DATA_DIR / "population_data.csv"),
        "population_run_summary": pd.read_csv(DATA_DIR / "population_run_summary.csv"),
        "agent": pd.read_csv(DATA_DIR / "agent_data.csv"),
        "recipe_lineage": pd.read_csv(DATA_DIR / "recipe_lineage_data.csv"),
        "recipe_descendant": pd.read_csv(DATA_DIR / "recipe_descendant_data.csv"),
        "recipe_recombination": pd.read_csv(DATA_DIR / "recipe_recombination_data.csv"),
        "raw_outputs": load_raw_outputs(),
    }


def get_fee_axis_values(fees):
    fees = np.asarray(fees, dtype=np.float64)
    return np.round(fees, 6)


def get_fee_plotting_config(recipe_lineage_data):
    fee_key_order = np.sort(
        recipe_lineage_data["fee_axis_value"].unique()
        if "fee_axis_value" in recipe_lineage_data.columns
        else get_fee_axis_values(recipe_lineage_data["fee"].unique())
    )
    fee_label_map = {fee: f"{fee:g}" for fee in fee_key_order}
    fee_label_order = [fee_label_map[fee] for fee in fee_key_order]
    fee_palette = dict(
        zip(fee_label_order, sns.color_palette("plasma", n_colors=len(fee_label_order)))
    )
    return fee_label_map, fee_label_order, fee_palette


def add_fee_plot_column(df, fee_label_map, fee_label_order):
    if "fee_axis_value" not in df.columns:
        df["fee_axis_value"] = get_fee_axis_values(df["fee"])
    df["fee_plot"] = df["fee_axis_value"]
    df["fee_label"] = pd.Categorical(
        df["fee_axis_value"].map(fee_label_map),
        categories=fee_label_order,
        ordered=True,
    )


def format_p_value(p_value):
    if np.isnan(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def mutual_information_agent_to_prob(values):
    values = np.asarray(values)
    if values.size == 0:
        return np.nan
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    mi = float(-(probs * np.log2(probs)).sum())
    max_mi = np.log2(values.size)
    if max_mi <= 0:
        return np.nan
    return mi / max_mi


def get_recipe_lineage_stats(
    recipe_id,
    parent_1_ids,
    parent_2_ids,
    creator_agent_ids,
    num_rules_in_initial_library,
    empty_recipe_id,
    memo,
):
    recipe_id = int(recipe_id)
    if recipe_id in memo:
        return memo[recipe_id]

    if recipe_id < num_rules_in_initial_library:
        memo[recipe_id] = (frozenset(), frozenset())
        return memo[recipe_id]

    innovation_event_ids = {recipe_id}
    innovator_ids = set()
    creator_id = int(creator_agent_ids[recipe_id])
    if creator_id != empty_recipe_id:
        innovator_ids.add(creator_id)

    for parent_id in (int(parent_1_ids[recipe_id]), int(parent_2_ids[recipe_id])):
        if parent_id == empty_recipe_id:
            continue
        parent_event_ids, parent_innovators = get_recipe_lineage_stats(
            parent_id,
            parent_1_ids,
            parent_2_ids,
            creator_agent_ids,
            num_rules_in_initial_library,
            empty_recipe_id,
            memo,
        )
        innovation_event_ids.update(parent_event_ids)
        innovator_ids.update(parent_innovators)

    memo[recipe_id] = (frozenset(innovation_event_ids), frozenset(innovator_ids))
    return memo[recipe_id]


def get_recipe_age_stats(
    recipe_id,
    parent_1_ids,
    parent_2_ids,
    birth_timesteps,
    num_rules_in_initial_library,
    empty_recipe_id,
    final_timestep,
    memo,
):
    recipe_id = int(recipe_id)
    if recipe_id in memo:
        return memo[recipe_id]

    if recipe_id < num_rules_in_initial_library:
        memo[recipe_id] = (np.nan, np.nan, np.nan, np.nan)
        return memo[recipe_id]

    birth_timestep = float(birth_timesteps[recipe_id])
    if birth_timestep < 0:
        memo[recipe_id] = (np.nan, np.nan, np.nan, np.nan)
        return memo[recipe_id]

    earliest_ancestor_birth_timestep = birth_timestep
    for parent_id in (int(parent_1_ids[recipe_id]), int(parent_2_ids[recipe_id])):
        if parent_id == empty_recipe_id:
            continue
        _, parent_earliest_birth, _, _ = get_recipe_age_stats(
            parent_id,
            parent_1_ids,
            parent_2_ids,
            birth_timesteps,
            num_rules_in_initial_library,
            empty_recipe_id,
            final_timestep,
            memo,
        )
        if not np.isnan(parent_earliest_birth):
            earliest_ancestor_birth_timestep = min(
                earliest_ancestor_birth_timestep, parent_earliest_birth
            )

    recipe_age = float(final_timestep - birth_timestep)
    recipe_ancestor_age = float(final_timestep - earliest_ancestor_birth_timestep)
    memo[recipe_id] = (
        birth_timestep,
        earliest_ancestor_birth_timestep,
        recipe_age,
        recipe_ancestor_age,
    )
    return memo[recipe_id]


def fit_yield_models(df, x, y):
    fit_df = df[[x, y]].dropna().copy()
    if len(fit_df) < 3 or fit_df[x].nunique() < 2:
        return fit_df, None, None, None

    X_lin = sm.add_constant(fit_df[[x]])
    linear_model = sm.OLS(fit_df[y], X_lin).fit()

    quadratic_model = None
    mean_x = None
    if len(fit_df) >= 5 and fit_df[x].nunique() >= 3:
        mean_x = fit_df[x].mean()
        fit_df["x_centered"] = fit_df[x] - mean_x
        fit_df["x_centered_sq"] = fit_df["x_centered"] ** 2
        X_quad = sm.add_constant(fit_df[["x_centered", "x_centered_sq"]])
        quadratic_model = sm.OLS(fit_df[y], X_quad).fit()

    return fit_df, linear_model, quadratic_model, mean_x


def use_quadratic_model(linear_model, quadratic_model, x):
    if linear_model is None or quadratic_model is None:
        return False, np.nan, np.nan

    lin_p = linear_model.pvalues.get(x, np.nan)
    quad_p = quadratic_model.pvalues.get("x_centered_sq", np.nan)
    if np.isnan(quad_p):
        return False, lin_p, quad_p

    # if quad p is lower than linear p, and quad model has lower AIC by at least 2, then use quad model
    return (
        (quad_p < lin_p) and (quadratic_model.aic + 2 < linear_model.aic),
        lin_p,
        quad_p,
    )

    # return quad_p < 0.01 and quadratic_model.aic + 2 < linear_model.aic


def annotate_regression(ax, df, x, y):
    _, linear_model, quadratic_model, _ = fit_yield_models(df, x, y)
    if linear_model is None:
        text = "lin p = n/a\nquad p = n/a"
    else:
        lin_p = format_p_value(linear_model.pvalues[x])
        quad_p = (
            format_p_value(quadratic_model.pvalues["x_centered_sq"])
            if quadratic_model is not None
            else "n/a"
        )
        text = f"lin p = {lin_p}\nquad p = {quad_p}"

    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )


def standardized_ols_coef(df, outcome, predictor, controls=()):
    cols = [outcome, predictor, *controls]
    fit_df = df[cols].dropna().copy()
    if (
        len(fit_df) < 5
        or fit_df[predictor].nunique() < 2
        or fit_df[outcome].nunique() < 2
    ):
        return np.nan

    x_cols = []
    for col in [predictor, *controls]:
        std = fit_df[col].std(ddof=0)
        if std <= 0 or np.isnan(std):
            if col == predictor:
                return np.nan
            continue
        fit_df[col] = (fit_df[col] - fit_df[col].mean()) / std
        x_cols.append(col)

    if predictor not in x_cols:
        return np.nan

    X = sm.add_constant(fit_df[x_cols])
    model = sm.OLS(fit_df[outcome], X).fit()
    return float(model.params[predictor])


def build_recipe_plot_data(
    population_data,
    population_run_summary,
    recipe_lineage_data,
    recipe_recombination_data,
    fee_label_map,
    fee_label_order,
):
    for df in (
        recipe_lineage_data,
        recipe_recombination_data,
        population_data,
        population_run_summary,
    ):
        add_fee_plot_column(df, fee_label_map, fee_label_order)

    recipe_key_cols = ["seed", "fee_axis_value", "recipe_id"]
    extant_recipe_keys = recipe_lineage_data.loc[
        recipe_lineage_data["recipe_length"] >= 1, recipe_key_cols
    ].drop_duplicates()
    extant_recipe_lineage_data = recipe_lineage_data.merge(
        extant_recipe_keys, on=recipe_key_cols, how="inner"
    )

    recipe_summary = (
        extant_recipe_lineage_data.groupby(["seed", "fee_axis_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "mean_n_innovation_events": x["n_innovation_events"].mean(),
                    "mean_n_unique_innovators": x["n_unique_innovators"].mean(),
                    "mean_recipe_length": x["recipe_length"].mean(),
                    "mean_recipe_age": x["recipe_age"].mean(),
                    "mean_recipe_ancestor_age": x["recipe_ancestor_age"].mean(),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(recipe_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_data.groupby(["seed", "fee_axis_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    fee_outcome_summary = recipe_summary.merge(
        final_population_data,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
    )

    recombination_event_summary = (
        recipe_recombination_data.groupby(["seed", "fee_axis_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "n_recombination_v1": x["is_recombination_v1"].sum(),
                    "n_recombination_v2": x["is_recombination_v2"].sum(),
                    "mean_recomb_branch_distance_innov_only": x.loc[
                        x["is_recombination_v1"] == 1,
                        "recomb_branch_distance_innov_only",
                    ].mean(),
                    "mean_recomb_branch_distance_innov_only_v2": x.loc[
                        x["is_recombination_v2"] == 1,
                        "recomb_branch_distance_innov_only",
                    ].mean(),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(recombination_event_summary, fee_label_map, fee_label_order)
    recombination_summary = population_run_summary.merge(
        recombination_event_summary,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
        how="left",
    )
    add_fee_plot_column(recombination_summary, fee_label_map, fee_label_order)
    recombination_outcome_summary = recombination_summary.merge(
        final_population_data,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
    )

    return (
        extant_recipe_lineage_data,
        fee_outcome_summary,
        recombination_summary,
        recombination_outcome_summary,
    )


def build_specialisation_plot_data(
    agent_data, population_data, fee_label_map, fee_label_order
):
    agent_df = agent_data.copy()
    population_df = population_data.copy()
    add_fee_plot_column(agent_df, fee_label_map, fee_label_order)
    add_fee_plot_column(population_df, fee_label_map, fee_label_order)

    agent_df["innov_prob"] = 1 - agent_df["role"]
    specialisation_summary = (
        agent_df.groupby(["seed", "fee_axis_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "specialisation": mutual_information_agent_to_prob(
                        x["innov_prob"].to_numpy()
                    ),
                }
            )
        )
        .reset_index(drop=True)
    )
    add_fee_plot_column(specialisation_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_df.groupby(["seed", "fee_axis_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    specialisation_outcome_summary = specialisation_summary.merge(
        final_population_data,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
    )
    return specialisation_summary, specialisation_outcome_summary


def get_filtered_population_df(pop_df):
    plot_df = pop_df.copy()
    for metric in ["r_innov", "r_imit", "yield", "yield_gini"]:
        plot_df[metric] = plot_df.groupby(["fee", "seed"])[metric].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

    plot_df = plot_df[
        (plot_df["t"] < plot_df["t"].max() - 500) & (plot_df["t"] % 20 == 0)
    ]

    return plot_df


def plot_preliminary_innovation_decay(raw_outputs):
    fees = np.asarray(raw_outputs["fees"], dtype=np.float64)
    fee_zero_idx = int(np.argmin(np.abs(fees - 0.0)))
    role_innovate = int(raw_outputs["role_innovate"])
    agent_roles = raw_outputs["agent_roles"][:, fee_zero_idx]
    innov_prob_ts = (agent_roles == role_innovate).mean(axis=2)

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


def plot_final_performance_metrics(pop_df, population_run_summary):
    final_pop_df = (
        pop_df.groupby(["fee", "seed"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500].mean(numeric_only=True))
        .reset_index()
        .drop(columns=["t"])
    )
    final_pop_df = final_pop_df.merge(
        population_run_summary[
            ["fee", "seed", "n_innovation_events", "n_imitation_events"]
        ],
        on=["fee", "seed"],
        how="left",
    )

    final_pop_df["yield"] = final_pop_df["yield"] / 15.0
    for col in ["fee", "n_innovation_events", "n_imitation_events"]:
        final_pop_df[col] = final_pop_df[col] / final_pop_df[col].max()

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    axs = axs.flatten()

    titles = {
        "yield": "Mean cultural score (proportion of max possible)",
        "n_innovation_events": "# successful innovation events (normalised)",
        "n_imitation_events": "# successful transmission events (normalised)",
    }

    x_lims = (-1.05, 1.05)
    lower_goldilocks, upper_goldilocks = 0.1, 0.5

    for i, metric in enumerate(list(titles.keys())):
        axs[i].axvspan(
            x_lims[0], lower_goldilocks, color="#f4c7c3", alpha=0.3, zorder=0
        )
        axs[i].axvspan(
            lower_goldilocks, upper_goldilocks, color="#d8f0c8", alpha=0.3, zorder=0
        )
        axs[i].axvspan(
            upper_goldilocks, x_lims[1], color="#f4c7c3", alpha=0.3, zorder=0
        )

        sns.lineplot(
            final_pop_df,
            x="fee",
            y=metric,
            marker="o",
            color="black",
            err_style="bars",
            ax=axs[i],
        )
        axs[i].set(
            title=titles[metric],
            xlabel="$c$ (normalised)",
            ylabel=None,
            xlim=x_lims,
        )
        sns.despine(ax=axs[i], left=True, bottom=True)

    trans = axs[0].get_xaxis_transform()
    axs[0].text(
        x_lims[0] + 0.03,
        0.98,
        "too little\ninnovation",
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

    save_fig(
        fig, "final_performance_metrics", subfolder=f"{env_name}/experiment_{exp_num}"
    )


def main():
    data = load_data()
    filtered_pop_df = get_filtered_population_df(data["population"])
    plot_final_performance_metrics(
        filtered_pop_df, data["population_run_summary"].copy()
    )
    plot_preliminary_innovation_decay(data["raw_outputs"])


if __name__ == "__main__":
    main()
