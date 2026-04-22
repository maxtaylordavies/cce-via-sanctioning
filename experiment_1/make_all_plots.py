from collections import defaultdict
from pathlib import Path
from string import ascii_lowercase

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from src.utils import save_fig as save_fig_

save_fig = lambda fig, name, subfolder=None: save_fig_(
    fig,
    name,
    subfolder=(f"experiment_1/{subfolder}" if subfolder else "experiment_1"),
)

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

DATA_DIR = Path("data/experiment_1")
FORAGING_LEVEL_CMAP = LinearSegmentedColormap.from_list(
    "foraging_level", ["#d8f0c8", "#1d5c2f"]
)
LINE_COLOR = "#06B48B"


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
        "agent": pd.read_csv(DATA_DIR / "agent_data.csv"),
        "recipe_lineage": pd.read_csv(DATA_DIR / "recipe_lineage_data.csv"),
        "recipe_descendant": pd.read_csv(DATA_DIR / "recipe_descendant_data.csv"),
        "recipe_recombination": pd.read_csv(DATA_DIR / "recipe_recombination_data.csv"),
        "similarity_matrices": 1 - np.load(DATA_DIR / "jaccard_matrices.npy"),
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


def format_time_series_fee_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = []
    for label in labels:
        try:
            fee = float(label)
            formatted_labels.append(f"{fee:g}")
        except ValueError:
            formatted_labels.append(label)
    ax.legend(handles, formatted_labels, title="fee")


def format_fee_title(fee):
    return f"Fee = {fee:g}"


def box_and_strip_plot(
    df,
    y,
    ax,
    title,
    fee_label_order,
    fee_palette,
    y_jitter=None,
    strip_kws=None,
):
    plot_df = df.copy()
    y_ = y
    if y_jitter is not None:
        y_ = y + "_jittered"
        plot_df[y_] = plot_df[y] + np.random.normal(0, y_jitter, size=len(plot_df))

    default_strip_kws = {"alpha": 0.05, "size": 3, "dodge": False, "jitter": True}
    if strip_kws is None:
        strip_kws = {}
    strip_kws = {**default_strip_kws, **strip_kws}

    sns.stripplot(
        data=plot_df,
        x="fee_label",
        y=y_,
        order=fee_label_order,
        hue="fee_label",
        hue_order=fee_label_order,
        palette=fee_palette,
        ax=ax,
        **strip_kws,
    )
    sns.boxplot(
        data=plot_df,
        x="fee_label",
        y=y,
        hue="fee_label",
        order=fee_label_order,
        hue_order=fee_label_order,
        palette=fee_palette,
        showfliers=False,
        dodge=False,
        width=0.7,
        boxprops={"edgecolor": "black", "zorder": 3},
        whiskerprops={"color": "black", "zorder": 3},
        capprops={"color": "black", "zorder": 3},
        medianprops={"color": "black", "zorder": 3},
        ax=ax,
    )

    for patch, fee_label in zip(ax.patches[: len(fee_label_order)], fee_label_order):
        r, g, b = fee_palette[fee_label]
        patch.set_facecolor((r, g, b, 0.3))
        patch.set_edgecolor("black")

    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_title(title)
    ax.set_xlabel("fee")
    sns.despine(ax=ax, left=True, bottom=True)


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


def plot_yield_relationship(ax, df, x, fee_label_order, legend=False):
    sns.scatterplot(
        data=df,
        x=x,
        y="yield",
        hue="fee_label",
        hue_order=fee_label_order,
        palette="plasma",
        s=80,
        alpha=0.7,
        linewidth=0,
        ax=ax,
        legend=legend,
    )
    fit_df, linear_model, quadratic_model, mean_x = fit_yield_models(df, x, "yield")
    use_quad, lin_p, quad_p = use_quadratic_model(linear_model, quadratic_model, x)
    if linear_model is not None:
        x_grid = np.linspace(fit_df[x].min(), fit_df[x].max(), 200)
        if use_quad:
            pred_df = pd.DataFrame(
                {
                    "const": 1.0,
                    "x_centered": x_grid - mean_x,
                }
            )
            pred_df["x_centered_sq"] = pred_df["x_centered"] ** 2
            y_pred = quadratic_model.predict(pred_df)
        else:
            pred_df = pd.DataFrame({"const": 1.0, x: x_grid})
            y_pred = linear_model.predict(pred_df)
        ax.plot(x_grid, y_pred, color=LINE_COLOR, alpha=1.0, linewidth=3)
    annotate_regression(ax, df, x, "yield")
    sns.despine(ax=ax, left=True, bottom=True)

    model = "quadratic" if use_quad else "linear"
    p_val = quad_p if use_quad else lin_p
    return model, p_val


def build_recipe_plot_data(
    population_data,
    recipe_lineage_data,
    recipe_recombination_data,
    fee_label_map,
    fee_label_order,
):
    for df in (recipe_lineage_data, recipe_recombination_data, population_data):
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

    recombination_summary = (
        recipe_recombination_data.groupby(["seed", "fee_axis_value"], as_index=False)
        .apply(
            lambda x: pd.Series(
                {
                    "n_innovation_events": len(x),
                    "n_recombination_v1": x["is_recombination_v1"].sum(),
                    "mean_recomb_branch_distance_innov_only": x.loc[
                        x["is_recombination_v1"] == 1,
                        "recomb_branch_distance_innov_only",
                    ].mean(),
                    "mean_recomb_mrca_age_innov_only": x.loc[
                        x["is_recombination_v1"] == 1,
                        "recomb_mrca_age_innov_only",
                    ].mean(),
                }
            )
        )
        .reset_index(drop=True)
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


def build_spatial_structure_plot_data(
    similarity_matrices,
    population_data,
    raw_outputs,
    fee_label_map,
    fee_label_order,
):
    population_df = population_data.copy()
    add_fee_plot_column(population_df, fee_label_map, fee_label_order)

    grid_length = int(raw_outputs["grid_length"])
    n_agents = grid_length * grid_length
    positions = np.array(
        [np.unravel_index(i, (grid_length, grid_length)) for i in range(n_agents)]
    )

    distance_matrix = np.zeros((n_agents, n_agents), dtype=np.int32)
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue
            distance_matrix[i, j] = np.abs(positions[i] - positions[j]).sum()

    seeds = raw_outputs["seeds"]
    fees = raw_outputs["fees"]
    fee_axis_values = get_fee_axis_values(fees)
    valid_pair_mask = distance_matrix > 0
    rows = []
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            sim_mat = similarity_matrices[seed_idx, fee_idx]
            pair_distances = distance_matrix[valid_pair_mask]
            pair_similarities = sim_mat[valid_pair_mask]
            unique_distances = np.unique(pair_distances)
            distance_means = np.array(
                [
                    pair_similarities[pair_distances == distance].mean()
                    for distance in unique_distances
                ],
                dtype=np.float64,
            )
            if len(unique_distances) >= 2:
                spatial_slope = -float(
                    np.polyfit(unique_distances.astype(np.float64), distance_means, 1)[
                        0
                    ]
                )
            else:
                spatial_slope = np.nan
            rows.append(
                {
                    "seed": int(seed),
                    "fee": float(fee),
                    "fee_axis_value": float(fee_axis_values[fee_idx]),
                    "spatial_structure_slope": spatial_slope,
                }
            )

    spatial_summary = pd.DataFrame(rows)
    add_fee_plot_column(spatial_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_df.groupby(["seed", "fee_axis_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    spatial_outcome_summary = spatial_summary.merge(
        final_population_data,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
    )
    return spatial_summary, spatial_outcome_summary


def build_recipe_persistence_selectivity_plot_data(
    recipe_descendant_data,
    population_data,
    raw_outputs,
    fee_label_map,
    fee_label_order,
):
    descendant_df = recipe_descendant_data.copy()
    population_df = population_data.copy()
    add_fee_plot_column(descendant_df, fee_label_map, fee_label_order)
    add_fee_plot_column(population_df, fee_label_map, fee_label_order)
    seeds = raw_outputs["seeds"]
    fees = raw_outputs["fees"]
    final_next_recipe_ids = raw_outputs["final_next_recipe_ids"]
    recipe_lineage_arrays = raw_outputs["recipe_lineage_arrays"]
    num_rules_in_initial_library = int(raw_outputs["num_rules_in_initial_library"])
    empty_recipe_id = int(raw_outputs["empty_recipe_id"])
    final_timestep = int(raw_outputs["T"])

    seed_to_idx = {int(seed): idx for idx, seed in enumerate(seeds)}
    fee_axis_to_idx = {
        float(axis_value): idx
        for idx, axis_value in enumerate(get_fee_axis_values(fees))
    }

    summary_rows = []
    descendant_df = descendant_df[descendant_df["creator_agent_id"] != -1].copy()
    for (seed, fee, fee_axis_value), group in descendant_df.groupby(
        ["seed", "fee", "fee_axis_value"], sort=False
    ):
        seed_idx = seed_to_idx[int(seed)]
        fee_idx = fee_axis_to_idx[float(fee_axis_value)]
        valid_limit = int(final_next_recipe_ids[seed_idx, fee_idx])

        parent_1_ids = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 0]
        parent_2_ids = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 1]
        creator_agent_ids = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 2]
        birth_timesteps = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 3]

        lineage_memo = {}
        age_memo = {}
        run_rows = []
        for row in group.itertuples(index=False):
            recipe_id = int(row.recipe_id)
            if recipe_id >= valid_limit:
                continue
            innovation_event_ids, innovator_ids = get_recipe_lineage_stats(
                recipe_id,
                parent_1_ids,
                parent_2_ids,
                creator_agent_ids,
                num_rules_in_initial_library,
                empty_recipe_id,
                lineage_memo,
            )
            _, _, recipe_age, _ = get_recipe_age_stats(
                recipe_id,
                parent_1_ids,
                parent_2_ids,
                birth_timesteps,
                num_rules_in_initial_library,
                empty_recipe_id,
                final_timestep,
                age_memo,
            )
            run_rows.append(
                {
                    "is_extant": row.is_extant,
                    "has_extant_descendants": row.has_extant_descendants,
                    "n_innovation_events": len(innovation_event_ids),
                    "n_unique_innovators": len(innovator_ids),
                    "recipe_age": recipe_age,
                }
            )

        run_df = pd.DataFrame(run_rows)
        summary_rows.append(
            {
                "seed": int(seed),
                "fee": float(fee),
                "fee_axis_value": float(fee_axis_value),
                "beta_extant_on_depth_age_controlled": standardized_ols_coef(
                    run_df,
                    "is_extant",
                    "n_innovation_events",
                    controls=("recipe_age",),
                ),
                "beta_extant_on_innovators_age_controlled": standardized_ols_coef(
                    run_df,
                    "is_extant",
                    "n_unique_innovators",
                    controls=("recipe_age",),
                ),
                "beta_legacy_on_depth_age_controlled": standardized_ols_coef(
                    run_df,
                    "has_extant_descendants",
                    "n_innovation_events",
                    controls=("recipe_age",),
                ),
                "beta_legacy_on_innovators_age_controlled": standardized_ols_coef(
                    run_df,
                    "has_extant_descendants",
                    "n_unique_innovators",
                    controls=("recipe_age",),
                ),
            }
        )

    persistence_summary = pd.DataFrame(summary_rows)
    add_fee_plot_column(persistence_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_df.groupby(["seed", "fee_axis_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    persistence_outcome_summary = persistence_summary.merge(
        final_population_data,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
    )
    return persistence_summary, persistence_outcome_summary


def build_posthumous_contribution_plot_data(
    recipe_lineage_data,
    population_data,
    raw_outputs,
    fee_label_map,
    fee_label_order,
):
    lineage_df = recipe_lineage_data.copy()
    population_df = population_data.copy()
    add_fee_plot_column(lineage_df, fee_label_map, fee_label_order)
    add_fee_plot_column(population_df, fee_label_map, fee_label_order)

    seeds = raw_outputs["seeds"]
    fees = raw_outputs["fees"]
    final_agent_ids = raw_outputs["final_agent_ids"]
    final_next_recipe_ids = raw_outputs["final_next_recipe_ids"]
    recipe_lineage_arrays = raw_outputs["recipe_lineage_arrays"]
    num_rules_in_initial_library = int(raw_outputs["num_rules_in_initial_library"])
    empty_recipe_id = int(raw_outputs["empty_recipe_id"])

    seed_to_idx = {int(seed): idx for idx, seed in enumerate(seeds)}
    fee_axis_to_idx = {
        float(axis_value): idx
        for idx, axis_value in enumerate(get_fee_axis_values(fees))
    }

    extant_df = lineage_df[lineage_df["recipe_length"] >= 1].copy()
    rows = []
    for (seed, fee_axis_value), group in extant_df.groupby(
        ["seed", "fee_axis_value"], sort=False
    ):
        seed_idx = seed_to_idx[int(seed)]
        fee_idx = fee_axis_to_idx[float(fee_axis_value)]
        valid_limit = int(final_next_recipe_ids[seed_idx, fee_idx])
        parent_1_ids = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 0]
        parent_2_ids = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 1]
        creator_agent_ids = recipe_lineage_arrays[seed_idx, fee_idx, :valid_limit, 2]
        alive_agent_ids = set(final_agent_ids[seed_idx, fee_idx].tolist())
        lineage_memo = {}

        recipe_rows = []
        for row in group.itertuples(index=False):
            recipe_id = int(row.recipe_id)
            _, innovator_ids = get_recipe_lineage_stats(
                recipe_id,
                parent_1_ids,
                parent_2_ids,
                creator_agent_ids,
                num_rules_in_initial_library,
                empty_recipe_id,
                lineage_memo,
            )
            innovator_ids = {
                int(agent_id)
                for agent_id in innovator_ids
                if int(agent_id) != empty_recipe_id
            }
            if len(innovator_ids) == 0:
                continue
            n_dead = sum(agent_id not in alive_agent_ids for agent_id in innovator_ids)
            frac_dead = n_dead / len(innovator_ids)
            recipe_rows.append(
                {
                    "n_dead_innovators": n_dead,
                    "frac_dead_innovators": frac_dead,
                    "has_any_dead_innovator": int(n_dead > 0),
                    "all_innovators_dead": int(n_dead == len(innovator_ids)),
                    "n_copies": float(row.n_copies),
                }
            )

        if not recipe_rows:
            summary = {
                "mean_n_dead_innovators": np.nan,
                "mean_frac_dead_innovators": np.nan,
                "copy_weighted_frac_dead_innovators": np.nan,
                "prop_recipes_with_any_dead_innovator": np.nan,
                "prop_recipes_with_all_innovators_dead": np.nan,
            }
        else:
            recipe_metrics = pd.DataFrame(recipe_rows)
            copy_weights = recipe_metrics["n_copies"].to_numpy(dtype=np.float64)
            total_weight = copy_weights.sum()
            summary = {
                "mean_n_dead_innovators": float(
                    recipe_metrics["n_dead_innovators"].mean()
                ),
                "mean_frac_dead_innovators": float(
                    recipe_metrics["frac_dead_innovators"].mean()
                ),
                "copy_weighted_frac_dead_innovators": (
                    float(
                        np.average(
                            recipe_metrics["frac_dead_innovators"], weights=copy_weights
                        )
                    )
                    if total_weight > 0
                    else np.nan
                ),
                "prop_recipes_with_any_dead_innovator": float(
                    recipe_metrics["has_any_dead_innovator"].mean()
                ),
                "prop_recipes_with_all_innovators_dead": float(
                    recipe_metrics["all_innovators_dead"].mean()
                ),
            }

        rows.append(
            {
                "seed": int(seed),
                "fee_axis_value": float(fee_axis_value),
                **summary,
            }
        )

    posthumous_summary = pd.DataFrame(rows)
    add_fee_plot_column(posthumous_summary, fee_label_map, fee_label_order)

    final_population_data = (
        population_df.groupby(["seed", "fee_axis_value"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500, ["yield"]].mean())
        .reset_index()
    )
    add_fee_plot_column(final_population_data, fee_label_map, fee_label_order)

    posthumous_outcome_summary = posthumous_summary.merge(
        final_population_data,
        on=["seed", "fee_axis_value", "fee_plot", "fee_label"],
    )
    return posthumous_summary, posthumous_outcome_summary


def plot_performance_metrics(pop_df):
    plot_df = pop_df.copy()
    for metric in ["r_innov", "r_imit", "yield", "yield_gini"]:
        plot_df[metric] = plot_df.groupby(["fee", "seed"])[metric].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

    plot_df = plot_df[
        (plot_df["t"] < plot_df["t"].max() - 500)
        & (plot_df["t"] % 20 == 0)
        # & (plot_df["fee"] < plot_df["fee"].max())
    ]

    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    axs = axs.flatten()
    palette = sns.color_palette("plasma", n_colors=plot_df["fee"].nunique())
    titles = {
        "level": "Average level of foraged plants",
        "yield": "Average yield",
        "yield_gini": "Yield inequality (Gini)",
        "r_innov": "Average reward for innovating",
        "r_imit": "Average reward for imitating",
        "prop_innov": "Average proportion of innovators",
    }

    for i, metric in enumerate(titles):
        sns.lineplot(
            plot_df,
            x="t",
            y=metric,
            hue="fee",
            palette=palette,
            legend=i == 0,
            ax=axs[i],
        )
        axs[i].set_title(titles[metric])
        sns.despine(ax=axs[i], left=True, bottom=True)
        if i == 0:
            format_time_series_fee_legend(axs[i])

    save_fig(fig, "performance_metrics_over_time")
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

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed_series in smoothed:
        ax.plot(t, seed_series, color="lightgray", alpha=0.7, linewidth=1)

    ax.plot(t, smoothed.mean(axis=0), color="black", linewidth=1.5)
    ax.set(
        xlabel="$t$",
        ylabel="Probability",
        title="Average probability of attempting innovation over time",
        xlim=(-5, 205),
    )
    sns.despine(ax=ax, left=True, bottom=True)
    save_fig(fig, "preliminary_innovation_decay_fee_0")


def plot_final_performance_metrics(pop_df):
    final_pop_df = (
        pop_df.groupby(["fee", "seed"])
        .apply(lambda x: x.loc[x["t"] >= x["t"].max() - 500].mean(numeric_only=True))
        .reset_index()
        .drop(columns=["t"])
    )

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    axs = axs.flatten()
    titles = {
        # "level": "Average level of foraged plants",
        "yield": "Average yield",
        # "yield_gini": "Yield inequality (Gini)",
        "prop_innov": "Proportion of innovators",
    }
    labels = {
        "yield": "Yield",
        "prop_innov": "Proportion",
    }
    metric_groups = [
        # ("level",),
        ("yield",),
        # ("yield_gini",),
        # ("r_innov", "r_imit"),
        ("prop_innov",),
    ]

    for i, metric_group in enumerate(metric_groups):
        if len(metric_group) == 2:
            for metric, color in zip(metric_group, ["red", "blue"]):
                sns.lineplot(
                    final_pop_df,
                    x="fee",
                    y=metric,
                    marker="o",
                    color=color,
                    err_style="bars",
                    label=metric,
                    ax=axs[i],
                )
            axs[i].set(
                title="Final average reward by role",
                xlabel="$c$",
                ylabel="Average reward",
            )
            axs[i].legend()
        else:
            metric = metric_group[0]
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
                title=f"Final {titles[metric].lower()} vs $c$",
                xlabel="$c$",
                ylabel=labels[metric],
            )
        sns.despine(ax=axs[i], left=True, bottom=True)

    for ax in axs[len(metric_groups) :]:
        ax.set_visible(False)

    save_fig(fig, "final_performance_metrics")


def plot_agent_heatmaps(agent_df):
    plot_df = agent_df.copy()
    plot_df["row"] = plot_df["agent_idx"] // 10
    plot_df["col"] = plot_df["agent_idx"] % 10
    plot_df["innov_freq"] = 1 - plot_df["role"]

    per_agent_metrics = {
        "innov_freq": "Frequency of innovation attempts",
        "level": "Average foraging level",
        "yield": "Average yield",
    }
    fees = np.sort(plot_df["fee"].unique())

    for seed in np.sort(plot_df["seed"].unique()):
        seed_df = plot_df[plot_df["seed"] == seed]
        fig, axs = plt.subplots(
            len(per_agent_metrics),
            len(fees) + 1,
            figsize=(len(fees) * 3 + 1.0, 2.5 * len(per_agent_metrics)),
            gridspec_kw={"width_ratios": [1] * len(fees) + [0.08]},
        )
        heatmap_axs = axs[:, :-1]
        cbar_axs = axs[:, -1]

        metric_ranges = {
            metric: (seed_df[metric].min(), seed_df[metric].max())
            for metric in per_agent_metrics
        }

        for col, fee in enumerate(fees):
            fee_df = seed_df[seed_df["fee"] == fee]
            for row, metric in enumerate(per_agent_metrics):
                pivot = fee_df.pivot_table(
                    index="row", columns="col", values=metric, aggfunc="mean"
                )
                vmin, vmax = metric_ranges[metric]
                cmap = (
                    "coolwarm"
                    if metric == "innov_freq"
                    else FORAGING_LEVEL_CMAP if metric == "level" else "plasma"
                )
                sns.heatmap(
                    pivot,
                    ax=heatmap_axs[row, col],
                    cbar=col == len(fees) - 1,
                    cbar_ax=cbar_axs[row] if col == len(fees) - 1 else None,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                if row == 0:
                    heatmap_axs[row, col].set_title(format_fee_title(fee), fontsize=16)
                heatmap_axs[row, col].set_ylabel(
                    per_agent_metrics[metric] if col == 0 else None, fontsize=12
                )
                heatmap_axs[row, col].set_xlabel(None)
                heatmap_axs[row, col].set_xticks([])
                heatmap_axs[row, col].set_yticks([])

        save_fig(fig, f"seed_{seed}", subfolder="agent_heatmaps")


def plot_similarity(similarity_matrices, fees):
    mean_similarity = np.mean(similarity_matrices, axis=0)
    n_fees = mean_similarity.shape[0]
    grid_length = int(np.sqrt(mean_similarity.shape[1]))

    fig, axs = plt.subplots(2, n_fees, figsize=(n_fees * 4, 6), sharey="row")
    for fee_idx in range(n_fees):
        sim_mat = mean_similarity[fee_idx]
        distance_to_sims = defaultdict(list)
        for i in range(sim_mat.shape[0]):
            for j in range(sim_mat.shape[1]):
                if i == j:
                    continue
                pos_i = np.unravel_index(i, (grid_length, grid_length))
                pos_j = np.unravel_index(j, (grid_length, grid_length))
                dist = np.sum(np.abs(np.array(pos_i) - np.array(pos_j)))
                distance_to_sims[dist].append(sim_mat[i, j])

        dists = np.array(sorted(distance_to_sims))
        mean_sims = np.array([np.mean(distance_to_sims[d]) for d in dists])
        sns.lineplot(
            x=dists, y=mean_sims, marker="o", color="black", ax=axs[0, fee_idx]
        )
        sns.despine(ax=axs[0, fee_idx], left=True, bottom=True)
        axs[0, fee_idx].set(
            xlabel="Distance between agents",
            ylabel="Average library similarity",
            title=format_fee_title(fees[fee_idx]),
        )

        axs[1, fee_idx].imshow(sim_mat, vmin=0, vmax=1)
        axs[1, fee_idx].axis("off")

    save_fig(fig, "jaccard_similarity")


def plot_population_level_metrics(
    fee_outcome_summary,
    spatial_summary,
    spatial_outcome_summary,
    recombination_summary,
    recombination_outcome_summary,
    specialisation_summary,
    specialisation_outcome_summary,
    persistence_summary,
    persistence_outcome_summary,
    posthumous_summary,
    posthumous_outcome_summary,
    fee_label_order,
    fee_palette,
):
    metric_specs = [
        (
            "specialisation",
            "Population specialisation",
            "Normalised MI",
            specialisation_summary,
            specialisation_outcome_summary,
        ),
        (
            "n_innovation_events",
            "Total number of innovation events over population history",
            "Count",
            recombination_summary,
            recombination_outcome_summary,
        ),
        (
            "n_recombination_v1",
            "Total number of recombination events over population history",
            "Count",
            recombination_summary,
            recombination_outcome_summary,
        ),
        (
            "mean_recomb_branch_distance_innov_only",
            "Mean recombination branch distance",
            "Timesteps",
            recombination_summary,
            recombination_outcome_summary,
        ),
        (
            "mean_n_unique_innovators",
            "Mean number of unique innovators in final recipe histories",
            "Count",
            fee_outcome_summary,
            fee_outcome_summary,
        ),
        (
            "mean_recipe_age",
            "Mean age of final recipes",
            "Timesteps",
            fee_outcome_summary,
            fee_outcome_summary,
        ),
        (
            "mean_recipe_ancestor_age",
            "Mean age of final recipes' oldest ancestors",
            "Timesteps",
            fee_outcome_summary,
            fee_outcome_summary,
        ),
        (
            "beta_extant_on_depth_age_controlled",
            "Selectivity of recipe survival for lineage depth",
            "Standardised coefficient",
            persistence_summary,
            persistence_outcome_summary,
        ),
        (
            "beta_extant_on_innovators_age_controlled",
            "Selectivity of recipe survival for contributor count",
            "Standardised coefficient",
            persistence_summary,
            persistence_outcome_summary,
        ),
        (
            "mean_n_dead_innovators",
            "Mean number of dead innovators in final recipe ancestries",
            "Count",
            posthumous_summary,
            posthumous_outcome_summary,
        ),
        (
            "spatial_structure_slope",
            "Slope of library dissimilarity vs Manhattan distance",
            "Slope",
            spatial_summary,
            spatial_outcome_summary,
        ),
    ]
    n_metric_cols = 3
    n_metric_rows = int(np.ceil(len(metric_specs) / n_metric_cols))
    fig = plt.figure(figsize=(8 * n_metric_cols, 4 * n_metric_rows))
    outer = fig.add_gridspec(
        n_metric_rows,
        n_metric_cols,
        wspace=0.15,
        hspace=0.4,
    )

    for metric_idx, (metric, title, label, summary_df, outcome_df) in enumerate(
        metric_specs
    ):
        row = metric_idx // n_metric_cols
        col = metric_idx % n_metric_cols
        pair_grid = outer[row, col].subgridspec(1, 2, wspace=0.2)
        fee_ax = fig.add_subplot(pair_grid[0, 0])
        yield_ax = fig.add_subplot(pair_grid[0, 1])

        plot_df = (
            summary_df[["seed", "fee_plot", "fee_label", metric]]
            .drop_duplicates()
            .sort_values("fee_plot")
            .copy()
        )

        sns.scatterplot(
            data=plot_df,
            x="fee_plot",
            y=metric,
            hue="fee_label",
            hue_order=fee_label_order,
            palette=fee_palette,
            s=45,
            alpha=0.7,
            linewidth=0,
            legend=False,
            ax=fee_ax,
        )
        sns.lineplot(
            data=plot_df,
            x="fee_plot",
            y=metric,
            color=LINE_COLOR,
            linewidth=3,
            marker=None,
            errorbar=None,
            ax=fee_ax,
        )
        fee_ax.set(xlabel="$c$", ylabel=label, title="")
        sns.despine(ax=fee_ax, left=True, bottom=True)

        model_type, p_val = plot_yield_relationship(
            yield_ax, outcome_df, metric, fee_label_order, metric_idx == 0
        )
        yield_ax.set(xlabel=label, ylabel="Final population yield")

        bold = (model_type == "linear") and (p_val < 0.05)
        fee_ax.text(
            1.1,
            1.04,
            f"({ascii_lowercase[metric_idx]}) {title}",
            transform=fee_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold" if bold else "normal",
        )

    for metric_idx in range(len(metric_specs), n_metric_rows * n_metric_cols):
        row = metric_idx // n_metric_cols
        col = metric_idx % n_metric_cols
        empty_grid = outer[row, col].subgridspec(1, 2, wspace=0.12)
        fig.add_subplot(empty_grid[0, 0]).set_visible(False)
        fig.add_subplot(empty_grid[0, 1]).set_visible(False)

    fig.suptitle(
        "What population-level measures actually increase with performance?",
        fontsize=18,
    )
    save_fig(fig, "population_level_metrics")


def main():
    data = load_data()
    fee_label_map, fee_label_order, fee_palette = get_fee_plotting_config(
        data["recipe_lineage"]
    )

    filtered_pop_df = plot_performance_metrics(data["population"])
    plot_final_performance_metrics(filtered_pop_df)
    plot_preliminary_innovation_decay(data["raw_outputs"])
    plot_agent_heatmaps(data["agent"])
    plot_similarity(data["similarity_matrices"], np.sort(data["agent"]["fee"].unique()))

    (
        extant_recipe_lineage_data,
        fee_outcome_summary,
        recombination_summary,
        recombination_outcome_summary,
    ) = build_recipe_plot_data(
        data["population"].copy(),
        data["recipe_lineage"].copy(),
        data["recipe_recombination"].copy(),
        fee_label_map,
        fee_label_order,
    )
    (
        specialisation_summary,
        specialisation_outcome_summary,
    ) = build_specialisation_plot_data(
        data["agent"].copy(),
        data["population"].copy(),
        fee_label_map,
        fee_label_order,
    )
    (
        spatial_summary,
        spatial_outcome_summary,
    ) = build_spatial_structure_plot_data(
        data["similarity_matrices"],
        data["population"].copy(),
        data["raw_outputs"],
        fee_label_map,
        fee_label_order,
    )
    (
        persistence_summary,
        persistence_outcome_summary,
    ) = build_recipe_persistence_selectivity_plot_data(
        data["recipe_descendant"].copy(),
        data["population"].copy(),
        data["raw_outputs"],
        fee_label_map,
        fee_label_order,
    )
    (
        posthumous_summary,
        posthumous_outcome_summary,
    ) = build_posthumous_contribution_plot_data(
        data["recipe_lineage"].copy(),
        data["population"].copy(),
        data["raw_outputs"],
        fee_label_map,
        fee_label_order,
    )

    plot_population_level_metrics(
        fee_outcome_summary,
        spatial_summary,
        spatial_outcome_summary,
        recombination_summary,
        recombination_outcome_summary,
        specialisation_summary,
        specialisation_outcome_summary,
        persistence_summary,
        persistence_outcome_summary,
        posthumous_summary,
        posthumous_outcome_summary,
        fee_label_order,
        fee_palette,
    )


if __name__ == "__main__":
    main()
