from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from experiment_1.make_all_plots import (
    build_recipe_plot_data,
    build_specialisation_plot_data,
    get_fee_plotting_config,
    load_data,
)
from src.utils import save_fig as save_fig_

save_fig = lambda fig, name, subfolder=None: save_fig_(
    fig,
    name,
    subfolder=(
        f"experiment_1/recipe_world_env/{subfolder}"
        if subfolder
        else "experiment_1/recipe_world_env"
    ),
)

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")

DATA_DIR = Path("data/recipe_world_env/experiment_1")

_husl = sns.color_palette("husl", 6)
BAR_COLORS = {
    "baseline": "#B8B8B8",
    "innovation_events": _husl[0],
    "imitation_events": _husl[1],
    "innovation_imitation_events": _husl[2],
    "unique_contributors": _husl[3],
    "recombinations": _husl[4],
    "specialisation": _husl[5],
    # "recipe_age": _husl[6],
}

BASELINE_MODEL_SPECS = OrderedDict(
    [
        (
            "baseline",
            {
                "label": "Baseline:\n$c + c^2$",
                "predictors": ["fee_c", "fee_c_sq"],
            },
        ),
        (
            "innovations",
            {
                "label": "Baseline +\ninnovation events",
                "predictors": ["fee_c", "fee_c_sq", "z_n_innovation_events"],
            },
        ),
        (
            "imitations",
            {
                "label": "Baseline +\nimitation events",
                "predictors": ["fee_c", "fee_c_sq", "z_n_imitation_events"],
            },
        ),
        (
            "baseline_recomb_v2",
            {
                "label": "Baseline +\nrecombinations",
                "predictors": ["fee_c", "fee_c_sq", "z_n_recombination_v2"],
            },
        ),
        (
            "contributors",
            {
                "label": "Baseline +\nunique contributors",
                "predictors": ["fee_c", "fee_c_sq", "z_mean_n_unique_innovators"],
            },
        ),
        (
            "baseline_specialisation",
            {
                "label": "Baseline +\nspecialisation",
                "predictors": ["fee_c", "fee_c_sq", "z_specialisation"],
            },
        ),
        # (
        #     "baseline_recipe_age",
        #     {
        #         "label": "Baseline +\nrecipe age",
        #         "predictors": ["fee_c", "fee_c_sq", "z_mean_recipe_age"],
        #     },
        # ),
    ]
)

EVENT_BASE_MODEL_SPECS = OrderedDict(
    [
        (
            "innov_imit",
            {
                "label": "Baseline +\ninnovation + imitation events",
                "predictors": [
                    "fee_c",
                    "fee_c_sq",
                    "z_n_innovation_events",
                    "z_n_imitation_events",
                ],
            },
        ),
        (
            "innov_imit_plus_recomb_v2",
            {
                "label": "Baseline +\ninnovation + imitation + recombinations",
                "predictors": [
                    "fee_c",
                    "fee_c_sq",
                    "z_n_innovation_events",
                    "z_n_imitation_events",
                    "z_n_recombination_v2",
                ],
                "added_term": "z_n_recombination_v2",
            },
        ),
        (
            "innov_imit_plus_contributors",
            {
                "label": "Baseline +\ninnovation + imitation + contributors",
                "predictors": [
                    "fee_c",
                    "fee_c_sq",
                    "z_n_innovation_events",
                    "z_n_imitation_events",
                    "z_mean_n_unique_innovators",
                ],
                "added_term": "z_mean_n_unique_innovators",
            },
        ),
        (
            "innov_imit_plus_specialisation",
            {
                "label": "Baseline +\ninnovation + imitation + specialisation",
                "predictors": [
                    "fee_c",
                    "fee_c_sq",
                    "z_n_innovation_events",
                    "z_n_imitation_events",
                    "z_specialisation",
                ],
                "added_term": "z_specialisation",
            },
        ),
        # (
        #     "innov_imit_plus_recipe_age",
        #     {
        #         "label": "Baseline +\ninnovation + imitation + recipe age",
        #         "predictors": [
        #             "fee_c",
        #             "fee_c_sq",
        #             "z_n_innovation_events",
        #             "z_n_imitation_events",
        #             "z_mean_recipe_age",
        #         ],
        #         "added_term": "z_mean_recipe_age",
        #     },
        # ),
    ]
)

MODEL_SPECS = OrderedDict(
    [
        *BASELINE_MODEL_SPECS.items(),
        *EVENT_BASE_MODEL_SPECS.items(),
    ]
)

BASELINE_COMPARISONS = [
    ("baseline", "innovations", "Add innovation events\nto baseline"),
    ("baseline", "imitations", "Add imitation events\nto baseline"),
    ("baseline", "baseline_recomb_v2", "Add recombinations\nto baseline"),
    ("baseline", "contributors", "Add unique contributors\nto baseline"),
    ("baseline", "baseline_specialisation", "Add specialisation\nto baseline"),
    # ("baseline", "baseline_recipe_age", "Add recipe age\nto baseline"),
]

EVENT_BASE_COMPARISONS = [
    (
        "innovations",
        "innov_imit",
        "+ transmission events",
    ),
    (
        "imitations",
        "innov_imit",
        "+ innovation events",
    ),
    (
        "innov_imit",
        "innov_imit_plus_contributors",
        "+ unique\ncontributors",
    ),
    (
        "innov_imit",
        "innov_imit_plus_recomb_v2",
        "+ recombination\nevents",
    ),
    (
        "innov_imit",
        "innov_imit_plus_specialisation",
        "+ specialisation",
    ),
    # (
    #     "innov_imit",
    #     "innov_imit_plus_recipe_age",
    #     "Add recipe age\nbeyond innovation + imitation",
    # ),
]

TERM_LABELS = {
    "z_n_innovation_events": "Innovation events",
    "z_n_imitation_events": "Imitation events",
    "z_mean_n_unique_innovators": "Unique contributors",
    "z_n_recombination_v2": "Recombinations",
    "z_specialisation": "Specialisation",
    # "z_mean_recipe_age": "Recipe age",
}


def format_p_value(p_value):
    if np.isnan(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def build_regression_df():
    data = load_data()
    fee_label_map, fee_label_order, fee_palette = get_fee_plotting_config(
        data["recipe_lineage"]
    )

    _, fee_outcome_summary, recombination_summary, _ = build_recipe_plot_data(
        data["population"].copy(),
        data["population_run_summary"].copy(),
        data["recipe_lineage"].copy(),
        data["recipe_recombination"].copy(),
        fee_label_map,
        fee_label_order,
    )
    specialisation_summary, _ = build_specialisation_plot_data(
        data["agent"].copy(),
        data["population"].copy(),
        fee_label_map,
        fee_label_order,
    )
    df = (
        fee_outcome_summary[
            [
                "seed",
                "fee_axis_value",
                "fee_plot",
                "fee_label",
                "yield",
                "mean_n_unique_innovators",
                "mean_recipe_age",
            ]
        ]
        .merge(
            recombination_summary[
                [
                    "seed",
                    "fee_axis_value",
                    "n_innovation_events",
                    "n_imitation_events",
                    "n_recombination_v2",
                ]
            ],
            on=["seed", "fee_axis_value"],
        )
        .merge(
            specialisation_summary[["seed", "fee_axis_value", "specialisation"]],
            on=["seed", "fee_axis_value"],
        )
    )

    df["fee_c"] = df["fee_axis_value"] - df["fee_axis_value"].mean()
    df["fee_c_sq"] = df["fee_c"] ** 2

    for col in [
        "n_innovation_events",
        "n_imitation_events",
        "mean_n_unique_innovators",
        "n_recombination_v2",
        "specialisation",
        "mean_recipe_age",
    ]:
        std = df[col].std(ddof=0)
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std > 0 else np.nan

    return df, fee_label_order, fee_palette


def fit_models(df):
    fitted_models = {}
    model_rows = []
    coefficient_rows = []

    for model_id, spec in MODEL_SPECS.items():
        predictors = spec["predictors"]
        X = sm.add_constant(df[predictors])
        model = sm.OLS(df["yield"], X).fit()
        fitted_models[model_id] = model

        model_rows.append(
            {
                "model_id": model_id,
                "model_label": spec["label"],
                "n_predictors": len(predictors),
                "n_obs": int(model.nobs),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
            }
        )

        for term in predictors:
            ci_low, ci_high = model.conf_int().loc[term]
            coefficient_rows.append(
                {
                    "model_id": model_id,
                    "model_label": spec["label"],
                    "term": term,
                    "term_label": TERM_LABELS.get(term, term),
                    "coef": float(model.params[term]),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_value": float(model.pvalues[term]),
                }
            )

    model_summary = pd.DataFrame(model_rows)
    coefficient_summary = pd.DataFrame(coefficient_rows)
    return fitted_models, model_summary, coefficient_summary


def build_nested_test_summary(fitted_models, model_summary, comparisons):
    model_summary = model_summary.set_index("model_id")
    rows = []

    for lower_id, upper_id, label in comparisons:
        comparison = anova_lm(fitted_models[lower_id], fitted_models[upper_id])
        rows.append(
            {
                "lower_model_id": lower_id,
                "upper_model_id": upper_id,
                "comparison_label": label,
                "delta_adj_r2": float(
                    model_summary.loc[upper_id, "adj_r2"]
                    - model_summary.loc[lower_id, "adj_r2"]
                ),
                "delta_aic": float(
                    model_summary.loc[lower_id, "aic"]
                    - model_summary.loc[upper_id, "aic"]
                ),
                "f_stat": float(comparison["F"].iloc[1]),
                "p_value": float(comparison["Pr(>F)"].iloc[1]),
            }
        )

    return pd.DataFrame(rows)


def build_vif_summary(df):
    rows = []
    for model_id, spec in MODEL_SPECS.items():
        predictors = spec["predictors"]
        X = df[predictors]
        for idx, predictor in enumerate(predictors):
            rows.append(
                {
                    "model_id": model_id,
                    "predictor": predictor,
                    "term_label": TERM_LABELS.get(predictor, predictor),
                    "vif": float(variance_inflation_factor(X.values, idx)),
                }
            )
    return pd.DataFrame(rows)


def plot_family_summary(
    model_summary,
    nested_tests,
    family_models,
    display_label_map,
    model_color_keys,
    comparison_color_keys,
    output_name,
    title,
    comparison_title,
    width_ratios=(1, 1),
):
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.5),
        gridspec_kw={"width_ratios": width_ratios},
    )

    fit_plot_df = model_summary[model_summary["model_id"].isin(family_models)].copy()
    fit_plot_df["display_label"] = pd.Categorical(
        fit_plot_df["model_id"].map(display_label_map),
        categories=[display_label_map[model_id] for model_id in family_models],
        ordered=True,
    )
    fit_plot_df = fit_plot_df.sort_values("display_label")

    sns.barplot(
        data=fit_plot_df,
        x="display_label",
        y="adj_r2",
        color=BAR_COLORS["baseline"],
        ax=axs[0],
    )
    for patch, model_id in zip(axs[0].patches, family_models):
        patch.set_facecolor(BAR_COLORS[model_color_keys[model_id]])
    axs[0].set(xlabel="", ylabel="Adjusted $R^2$", title="Overall model fit")
    # axs[0].tick_params(axis="x", rotation=0)
    for patch, value in zip(axs[0].patches, fit_plot_df["adj_r2"]):
        axs[0].text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    sns.barplot(
        data=nested_tests,
        x="comparison_label",
        y="delta_adj_r2",
        color=BAR_COLORS["baseline"],
        ax=axs[1],
    )
    for patch, color_key in zip(axs[1].patches, comparison_color_keys):
        patch.set_facecolor(BAR_COLORS[color_key])
    axs[1].set(
        xlabel="",
        ylabel="Increase in adjusted $R^2$",
        title=comparison_title,
    )
    # axs[1].tick_params(axis="x", rotation=0)
    for patch, (_, row) in zip(axs[1].patches, nested_tests.iterrows()):
        axs[1].text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.0002,
            f"p = {format_p_value(row['p_value'])}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    sns.despine(ax=axs[0], left=True, bottom=True)
    sns.despine(ax=axs[1], left=True, bottom=True)

    axs[0].set_title("Overall model fit")
    fig.suptitle(title, y=1.03)
    save_fig(fig, output_name)


def plot_coefficient_summary(coefficient_summary):
    substantive_terms = ["z_n_innovation_events", "z_mean_n_unique_innovators"]
    plot_df = coefficient_summary[
        coefficient_summary["term"].isin(substantive_terms)
    ].copy()
    ordered_models = ["innovations", "contributors", "both"]
    plot_df = plot_df[plot_df["model_id"].isin(ordered_models)]

    fig, axs = plt.subplots(1, len(ordered_models), figsize=(12.5, 4.6), sharey=True)
    if len(ordered_models) == 1:
        axs = [axs]

    x_max = float(np.nanmax(np.abs(plot_df[["ci_low", "ci_high"]].to_numpy())))
    x_lim = max(0.25, x_max * 1.1)
    term_order = ["Innovation events", "Unique contributors"]
    y_positions = np.arange(len(term_order))[::-1]
    y_map = dict(zip(term_order, y_positions))

    for ax, model_id in zip(axs, ordered_models):
        model_df = plot_df[plot_df["model_id"] == model_id].copy()

        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        for _, row in model_df.iterrows():
            y_pos = y_map[row["term_label"]]
            ax.plot(
                [row["ci_low"], row["ci_high"]],
                [y_pos, y_pos],
                color="#4C78A8",
                linewidth=2,
            )
            ax.scatter(
                row["coef"],
                y_pos,
                color="#E45756",
                s=55,
                zorder=3,
            )
            ax.text(
                row["ci_high"] + 0.03 * x_lim,
                y_pos,
                f"p = {format_p_value(row['p_value'])}",
                va="center",
                ha="left",
                fontsize=8.5,
            )

        ax.set(
            title=MODEL_SPECS[model_id]["label"],
            xlabel="Coefficient (yield units per 1 SD predictor)",
            xlim=(-x_lim, x_lim),
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(term_order)
        sns.despine(ax=ax, left=False, bottom=True)

    axs[0].set_ylabel("Predictor")
    fig.suptitle(
        "Effects of innovation events and unique contributors\n(all models also control for fee and fee^2)",
        y=1.07,
    )
    save_fig(fig, "multivariate_regression_coefficients")


def plot_family_coefficients(
    coefficient_summary,
    family_model_terms,
    display_label_map,
    output_name,
    title,
):
    plot_rows = []
    for row_order, term_spec in enumerate(family_model_terms):
        if len(term_spec) == 2:
            model_id, term = term_spec
            custom_term_label = None
        else:
            model_id, term, custom_term_label = term_spec
        row = coefficient_summary[
            (coefficient_summary["model_id"] == model_id)
            & (coefficient_summary["term"] == term)
        ].copy()
        if row.empty:
            continue
        row["block_label"] = display_label_map[model_id]
        row["row_order"] = row_order
        if custom_term_label is not None:
            row["term_label"] = custom_term_label
        plot_rows.append(row)

    plot_df = pd.concat(plot_rows, ignore_index=True)
    plot_df = plot_df.sort_values("row_order")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    y_positions = np.arange(len(plot_df))[::-1]
    x_max = float(np.nanmax(np.abs(plot_df[["ci_low", "ci_high"]].to_numpy())))
    x_lim = max(0.25, x_max * 1.15)

    ax.axvline(0, color="black", linewidth=1, alpha=0.6)
    for y_pos, (_, row) in zip(y_positions, plot_df.iterrows()):
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [y_pos, y_pos],
            color="#4C78A8",
            linewidth=2,
        )
        ax.scatter(row["coef"], y_pos, color="#E45756", s=60, zorder=3)
        ax.text(
            row["ci_high"] + 0.03 * x_lim,
            y_pos,
            f"p = {format_p_value(row['p_value'])}",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set(
        yticks=y_positions,
        yticklabels=plot_df["term_label"],
        xlabel="Coefficient (yield units per 1 SD predictor)",
        ylabel="Added predictor",
        xlim=(-x_lim, x_lim),
        title=title,
    )
    sns.despine(ax=ax, left=False, bottom=True)
    save_fig(fig, output_name)


def main():
    regression_df, _, _ = build_regression_df()
    fitted_models, model_summary, coefficient_summary = fit_models(regression_df)
    baseline_nested_tests = build_nested_test_summary(
        fitted_models, model_summary, BASELINE_COMPARISONS
    )
    event_base_nested_tests = build_nested_test_summary(
        fitted_models, model_summary, EVENT_BASE_COMPARISONS
    )
    vif_summary = build_vif_summary(regression_df)

    regression_df.to_csv(DATA_DIR / "population_regression_data.csv", index=False)
    model_summary.to_csv(
        DATA_DIR / "population_regression_model_summary.csv", index=False
    )
    coefficient_summary.to_csv(
        DATA_DIR / "population_regression_coefficients.csv", index=False
    )
    baseline_nested_tests.to_csv(
        DATA_DIR / "population_regression_baseline_nested_tests.csv", index=False
    )
    event_base_nested_tests.to_csv(
        DATA_DIR / "population_regression_event_base_nested_tests.csv",
        index=False,
    )
    vif_summary.to_csv(DATA_DIR / "population_regression_vif.csv", index=False)

    baseline_nested_lookup = {
        (row["lower_model_id"], row["upper_model_id"]): row
        for _, row in baseline_nested_tests.iterrows()
    }
    event_base_lookup = {
        (row["lower_model_id"], row["upper_model_id"]): row
        for _, row in event_base_nested_tests.iterrows()
    }

    baseline_family_nested_tests = pd.DataFrame(
        [
            baseline_nested_lookup[("baseline", "innovations")],
            baseline_nested_lookup[("baseline", "imitations")],
            baseline_nested_lookup[("baseline", "baseline_recomb_v2")],
            baseline_nested_lookup[("baseline", "contributors")],
            baseline_nested_lookup[("baseline", "baseline_specialisation")],
            # baseline_nested_lookup[("baseline", "baseline_recipe_age")],
        ]
    )
    event_family_nested_tests = pd.DataFrame(
        [
            event_base_lookup[("innov_imit", "innov_imit_plus_recomb_v2")],
            event_base_lookup[("innov_imit", "innov_imit_plus_contributors")],
            event_base_lookup[("innov_imit", "innov_imit_plus_specialisation")],
            # event_base_lookup[("innov_imit", "innov_imit_plus_recipe_age")],
        ]
    )

    baseline_family_models = [
        "baseline",
        "innovations",
        "imitations",
        "baseline_recomb_v2",
        "contributors",
        "baseline_specialisation",
        # "baseline_recipe_age",
    ]
    baseline_family_labels = {
        "baseline": "Baseline:\n$c + c^2$",
        "innovations": "+ innovation events",
        "imitations": "+ transmission events",
        "baseline_recomb_v2": "+ recombinations",
        "contributors": "+ unique contributors",
        "baseline_specialisation": "+ specialisation",
        # "baseline_recipe_age": "Baseline +\nrecipe age",
    }
    baseline_family_terms = [
        (
            "innovations",
            "z_n_innovation_events",
            "Innovation events\n(added to fee baseline)",
        ),
        (
            "imitations",
            "z_n_imitation_events",
            "Transmission events\n(added to fee baseline)",
        ),
        (
            "baseline_recomb_v2",
            "z_n_recombination_v2",
            "Recombination events\n(added to fee baseline)",
        ),
        (
            "contributors",
            "z_mean_n_unique_innovators",
            "Unique contributors\n(added to fee baseline)",
        ),
        (
            "baseline_specialisation",
            "z_specialisation",
            "Specialisation\n(added to fee baseline)",
        ),
        # (
        #     "baseline_recipe_age",
        #     "z_mean_recipe_age",
        #     "Recipe age\n(added to fee baseline)",
        # ),
    ]
    baseline_family_model_color_keys = {
        "baseline": "baseline",
        "innovations": "innovation_events",
        "imitations": "imitation_events",
        "baseline_recomb_v2": "recombinations",
        "contributors": "unique_contributors",
        "baseline_specialisation": "specialisation",
        # "baseline_recipe_age": "recipe_age",
    }
    baseline_family_comparison_color_keys = [
        "innovation_events",
        "imitation_events",
        "recombinations",
        "unique_contributors",
        "specialisation",
        # "recipe_age",
    ]

    event_family_models = [
        "baseline",
        "innovations",
        "imitations",
        "innov_imit",
        "innov_imit_plus_recomb_v2",
        "innov_imit_plus_contributors",
        "innov_imit_plus_specialisation",
        # "innov_imit_plus_recipe_age",
    ]
    event_family_labels = {
        "baseline": "Baseline:\n$c + c^2$",
        "innovations": "+ innovation\nevents",
        "imitations": "+ transmission\nevents",
        "innov_imit": "+ innovation\n+ transmission",
        "innov_imit_plus_recomb_v2": "+ innovation\n+ transmission\n+ recombinations",
        "innov_imit_plus_contributors": "+ innovation\n+ transmission\n+ contributors",
        "innov_imit_plus_specialisation": "+ innovation\n+ transmission\n+ specialisation",
        # "innov_imit_plus_recipe_age": "Baseline +\ninnovation + transmission + recipe age",
    }
    event_family_terms = [
        (
            "innovations",
            "z_n_innovation_events",
            "Innovation events\n(in innovation-only model)",
        ),
        (
            "imitations",
            "z_n_imitation_events",
            "Transmission events\n(in transmission-only model)",
        ),
        (
            "innov_imit",
            "z_n_innovation_events",
            "Innovation events\n(added beyond transmission)",
        ),
        (
            "innov_imit",
            "z_n_imitation_events",
            "Transmission events\n(added beyond innovation)",
        ),
        ("innov_imit_plus_recomb_v2", "z_n_recombination_v2"),
        ("innov_imit_plus_contributors", "z_mean_n_unique_innovators"),
        ("innov_imit_plus_specialisation", "z_specialisation"),
        # ("innov_imit_plus_recipe_age", "z_mean_recipe_age"),
    ]
    event_family_model_color_keys = {
        "baseline": "baseline",
        "innovations": "innovation_events",
        "imitations": "imitation_events",
        "innov_imit": "innovation_imitation_events",
        "innov_imit_plus_recomb_v2": "recombinations",
        "innov_imit_plus_contributors": "unique_contributors",
        "innov_imit_plus_specialisation": "specialisation",
        # "innov_imit_plus_recipe_age": "recipe_age",
    }
    event_family_comparison_color_keys = [
        "recombinations",
        "unique_contributors",
        "specialisation",
        # "recipe_age",
    ]

    # plot_family_summary(
    #     model_summary,
    #     baseline_family_nested_tests,
    #     baseline_family_models,
    #     baseline_family_labels,
    #     baseline_family_model_color_keys,
    #     baseline_family_comparison_color_keys,
    #     "multivariate_regression_fee_baseline_summary",
    #     "Screening population-level predictors\nby adding each metric separately to fee controls",
    #     "Incremental value beyond fee controls",
    # )
    # plot_family_coefficients(
    #     coefficient_summary,
    #     baseline_family_terms,
    #     baseline_family_labels,
    #     "multivariate_regression_fee_baseline_coefficients",
    #     "Effects of added predictors\nwhen each metric is added separately to fee controls",
    # )
    plot_family_summary(
        model_summary,
        event_family_nested_tests,
        event_family_models,
        event_family_labels,
        event_family_model_color_keys,
        event_family_comparison_color_keys,
        "multivariate_regression_event_counts_base_summary",
        "Testing population-level predictors of final yield",
        "Incremental value beyond innovation and transmission event counts",
        width_ratios=(1.35, 0.75),
    )
    plot_family_coefficients(
        coefficient_summary,
        event_family_terms,
        event_family_labels,
        "multivariate_regression_event_counts_base_coefficients",
        "Effects of added predictors",
    )


if __name__ == "__main__":
    main()
