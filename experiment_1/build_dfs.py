from pathlib import Path
import os

import numpy as np
import pandas as pd

DATA_DIR = Path("data/experiment_1/5k")


def get_gini(values):
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    n = values.size
    mean_abs_diff = np.abs(values[:, None] - values[None, :]).mean()
    return float(mean_abs_diff * n / (2 * total))


def get_fee_axis_values(fees):
    fees = np.asarray(fees, dtype=np.float64)
    return np.round(fees, 6)


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


def get_recombination_distance_stats(
    parent_1_id,
    parent_2_id,
    child_id,
    parent_1_ids,
    parent_2_ids,
    creator_agent_ids,
    birth_timesteps,
    num_rules_in_initial_library,
    empty_recipe_id,
    lineage_memo,
):
    if parent_1_id == empty_recipe_id or parent_2_id == empty_recipe_id:
        return {
            "recomb_branch_distance_innov_only": np.nan,
            "recomb_mrca_age_innov_only": np.nan,
        }

    parent_1_birth = float(birth_timesteps[parent_1_id])
    parent_2_birth = float(birth_timesteps[parent_2_id])
    child_birth = float(birth_timesteps[child_id])

    if child_birth < 0:
        return {
            "recomb_branch_distance_innov_only": np.nan,
            "recomb_mrca_age_innov_only": np.nan,
        }

    ancestor_ids_1, _ = get_recipe_lineage_stats(
        parent_1_id,
        parent_1_ids,
        parent_2_ids,
        creator_agent_ids,
        num_rules_in_initial_library,
        empty_recipe_id,
        lineage_memo,
    )
    ancestor_ids_2, _ = get_recipe_lineage_stats(
        parent_2_id,
        parent_1_ids,
        parent_2_ids,
        creator_agent_ids,
        num_rules_in_initial_library,
        empty_recipe_id,
        lineage_memo,
    )

    common_ancestor_ids = set(ancestor_ids_1).intersection(ancestor_ids_2)
    common_births = [
        float(birth_timesteps[ancestor_id])
        for ancestor_id in common_ancestor_ids
        if birth_timesteps[ancestor_id] >= 0
    ]

    if common_births:
        mrca_birth_innov_only = max(common_births)
    else:
        mrca_birth_innov_only = np.nan

    recomb_branch_distance_innov_only = (
        np.nan
        if np.isnan(mrca_birth_innov_only)
        else (parent_1_birth - mrca_birth_innov_only)
        + (parent_2_birth - mrca_birth_innov_only)
    )

    recomb_mrca_age_innov_only = (
        np.nan if np.isnan(mrca_birth_innov_only) else child_birth - mrca_birth_innov_only
    )

    return {
        "recomb_branch_distance_innov_only": recomb_branch_distance_innov_only,
        "recomb_mrca_age_innov_only": recomb_mrca_age_innov_only,
    }


def summarise_recipe_lineages(
    seed,
    fee,
    libraries,
    recipe_ids,
    parent_1_ids,
    parent_2_ids,
    creator_agent_ids,
    birth_timesteps,
    next_recipe_id,
    num_rules_in_initial_library,
    empty_recipe_id,
    final_timestep,
):
    valid_recipe_ids = recipe_ids[recipe_ids != empty_recipe_id]
    if valid_recipe_ids.size == 0:
        return []

    lineage_rows = []
    memo = {}
    age_memo = {}
    valid_limit = int(next_recipe_id)
    parent_1_ids = parent_1_ids[:valid_limit]
    parent_2_ids = parent_2_ids[:valid_limit]
    creator_agent_ids = creator_agent_ids[:valid_limit]
    birth_timesteps = birth_timesteps[:valid_limit]

    unique_recipe_ids, copy_counts = np.unique(valid_recipe_ids, return_counts=True)
    for recipe_id, n_copies in zip(unique_recipe_ids.tolist(), copy_counts.tolist()):
        innovation_event_ids, innovator_ids = get_recipe_lineage_stats(
            recipe_id,
            parent_1_ids,
            parent_2_ids,
            creator_agent_ids,
            num_rules_in_initial_library,
            empty_recipe_id,
            memo,
        )
        (
            _recipe_birth_timestep,
            _earliest_ancestor_birth_timestep,
            recipe_age,
            recipe_ancestor_age,
        ) = get_recipe_age_stats(
            recipe_id,
            parent_1_ids,
            parent_2_ids,
            birth_timesteps,
            num_rules_in_initial_library,
            empty_recipe_id,
            final_timestep,
            age_memo,
        )
        holder_agent_idx, slot_idx = np.argwhere(recipe_ids == recipe_id)[0]
        recipe = libraries[holder_agent_idx, slot_idx]
        recipe_length = int(np.count_nonzero(recipe))
        lineage_rows.append(
            {
                "seed": seed,
                "fee": fee,
                "recipe_id": int(recipe_id),
                "n_innovation_events": len(innovation_event_ids),
                "n_unique_innovators": len(innovator_ids),
                "n_copies": int(n_copies),
                "recipe_length": recipe_length,
                "recipe_age": recipe_age,
                "recipe_ancestor_age": recipe_ancestor_age,
            }
        )

    return lineage_rows


def build_population_df(outputs):
    fees = outputs["fees"]
    seeds = outputs["seeds"]
    t_main = int(outputs["T_main"])
    agent_levels = outputs["agent_levels"]
    agent_yields = outputs["agent_yields"]
    pop_role_rewards = outputs["pop_role_rewards"]
    agent_roles = outputs["agent_roles"]
    role_innovate = int(outputs["role_innovate"])

    pop_levels = agent_levels.mean(axis=3)
    pop_yields = agent_yields.mean(axis=3)
    pop_prop_innovs = (agent_roles == role_innovate).mean(axis=3)
    pop_yield_ginis = np.apply_along_axis(get_gini, 3, agent_yields)

    rows = []
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            for t in range(t_main):
                if t % 10 != 0:
                    continue
                rows.append(
                    {
                        "fee": float(fee),
                        "t": t,
                        "level": float(pop_levels[seed_idx, fee_idx, t]),
                        "yield": float(pop_yields[seed_idx, fee_idx, t]),
                        "r_innov": float(
                            pop_role_rewards[seed_idx, fee_idx, t, role_innovate]
                        ),
                        "r_imit": float(
                            pop_role_rewards[seed_idx, fee_idx, t, 1 - role_innovate]
                        ),
                        "prop_innov": float(pop_prop_innovs[seed_idx, fee_idx, t]),
                        "yield_gini": float(pop_yield_ginis[seed_idx, fee_idx, t]),
                        "seed": int(seed),
                    }
                )
    return pd.DataFrame(rows)


def build_agent_df(outputs):
    fees = outputs["fees"]
    seeds = outputs["seeds"]
    t_extra = int(outputs["T_extra"])
    agent_levels = outputs["agent_levels"][:, :, -t_extra:].mean(axis=2)
    agent_yields = outputs["agent_yields"][:, :, -t_extra:].mean(axis=2)
    agent_roles = outputs["agent_roles"][:, :, -t_extra:].mean(axis=2)

    rows = []
    n_agents = agent_levels.shape[2]
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            for agent_idx in range(n_agents):
                rows.append(
                    {
                        "fee": float(fee),
                        "agent_idx": agent_idx,
                        "level": float(agent_levels[seed_idx, fee_idx, agent_idx]),
                        "yield": float(agent_yields[seed_idx, fee_idx, agent_idx]),
                        "role": float(agent_roles[seed_idx, fee_idx, agent_idx]),
                        "seed": int(seed),
                    }
                )
    return pd.DataFrame(rows)


def build_recipe_dfs(outputs):
    fees = outputs["fees"]
    seeds = outputs["seeds"]
    total_timesteps = int(outputs["T"])
    final_libraries = outputs["final_libraries"]
    final_recipe_ids = outputs["final_recipe_ids"]
    final_next_recipe_ids = outputs["final_next_recipe_ids"]
    recipe_lineage_arrays = outputs["recipe_lineage_arrays"]
    num_rules_in_initial_library = int(outputs["num_rules_in_initial_library"])
    empty_recipe_id = int(outputs["empty_recipe_id"])

    lineage_rows = []
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            recipe_parent_1_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 0]
            recipe_parent_2_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 1]
            recipe_creator_agent_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 2]
            if recipe_lineage_arrays.shape[-1] >= 4:
                recipe_birth_timesteps = recipe_lineage_arrays[seed_idx, fee_idx, :, 3]
            else:
                recipe_birth_timesteps = np.full_like(recipe_parent_1_ids, -1)
            fee_lineage_rows = summarise_recipe_lineages(
                seed=int(seed),
                fee=float(fee),
                libraries=final_libraries[seed_idx, fee_idx],
                recipe_ids=final_recipe_ids[seed_idx, fee_idx],
                parent_1_ids=recipe_parent_1_ids,
                parent_2_ids=recipe_parent_2_ids,
                creator_agent_ids=recipe_creator_agent_ids,
                birth_timesteps=recipe_birth_timesteps,
                next_recipe_id=int(final_next_recipe_ids[seed_idx, fee_idx]),
                num_rules_in_initial_library=num_rules_in_initial_library,
                empty_recipe_id=empty_recipe_id,
                final_timestep=total_timesteps - 1,
            )
            lineage_rows.extend(fee_lineage_rows)

    return pd.DataFrame(lineage_rows)


def build_recipe_descendant_df(outputs):
    fees = outputs["fees"]
    seeds = outputs["seeds"]
    final_recipe_ids = outputs["final_recipe_ids"]
    final_next_recipe_ids = outputs["final_next_recipe_ids"]
    recipe_lineage_arrays = outputs["recipe_lineage_arrays"]
    num_rules_in_initial_library = int(outputs["num_rules_in_initial_library"])
    empty_recipe_id = int(outputs["empty_recipe_id"])

    rows = []
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            recipe_ids = final_recipe_ids[seed_idx, fee_idx]
            valid_recipe_ids = recipe_ids[recipe_ids != empty_recipe_id]
            next_recipe_id = int(final_next_recipe_ids[seed_idx, fee_idx])
            parent_1_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 0][
                :next_recipe_id
            ]
            parent_2_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 1][
                :next_recipe_id
            ]
            creator_agent_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 2][
                :next_recipe_id
            ]

            extant_unique_recipe_ids, extant_copy_counts = np.unique(
                valid_recipe_ids, return_counts=True
            )
            extant_copy_count_map = dict(
                zip(extant_unique_recipe_ids.tolist(), extant_copy_counts.tolist())
            )

            memo = {}
            unique_descendants = {
                recipe_id: set()
                for recipe_id in range(num_rules_in_initial_library, next_recipe_id)
            }

            for focal_recipe_id in extant_copy_count_map:
                ancestor_event_ids, _ = get_recipe_lineage_stats(
                    focal_recipe_id,
                    parent_1_ids,
                    parent_2_ids,
                    creator_agent_ids,
                    num_rules_in_initial_library,
                    empty_recipe_id,
                    memo,
                )
                for ancestor_recipe_id in ancestor_event_ids:
                    if ancestor_recipe_id == focal_recipe_id:
                        continue
                    unique_descendants[ancestor_recipe_id].add(focal_recipe_id)

            for recipe_id in range(num_rules_in_initial_library, next_recipe_id):
                rows.append(
                    {
                        "seed": int(seed),
                        "fee": float(fee),
                        "recipe_id": recipe_id,
                        "creator_agent_id": int(creator_agent_ids[recipe_id]),
                        "is_extant": int(recipe_id in extant_copy_count_map),
                        "n_unique_extant_descendants": len(
                            unique_descendants[recipe_id]
                        ),
                        "has_extant_descendants": int(
                            len(unique_descendants[recipe_id]) > 0
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_recipe_recombination_df(outputs):
    fees = outputs["fees"]
    seeds = outputs["seeds"]
    final_next_recipe_ids = outputs["final_next_recipe_ids"]
    recipe_lineage_arrays = outputs["recipe_lineage_arrays"]
    num_rules_in_initial_library = int(outputs["num_rules_in_initial_library"])
    empty_recipe_id = int(outputs["empty_recipe_id"])

    rows = []
    for seed_idx, seed in enumerate(seeds):
        for fee_idx, fee in enumerate(fees):
            next_recipe_id = int(final_next_recipe_ids[seed_idx, fee_idx])
            parent_1_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 0][
                :next_recipe_id
            ]
            parent_2_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 1][
                :next_recipe_id
            ]
            creator_agent_ids = recipe_lineage_arrays[seed_idx, fee_idx, :, 2][
                :next_recipe_id
            ]
            if recipe_lineage_arrays.shape[-1] >= 4:
                birth_timesteps = recipe_lineage_arrays[seed_idx, fee_idx, :, 3][
                    :next_recipe_id
                ]
            else:
                birth_timesteps = np.full(next_recipe_id, -1, dtype=np.int32)
            lineage_memo = {}

            for recipe_id in range(num_rules_in_initial_library, next_recipe_id):
                parent_1_id = int(parent_1_ids[recipe_id])
                parent_2_id = int(parent_2_ids[recipe_id])
                is_combine_event = int(
                    parent_1_id != empty_recipe_id and parent_2_id != empty_recipe_id
                )

                parent_1_creator_id = empty_recipe_id
                if parent_1_id != empty_recipe_id:
                    parent_1_creator_id = int(creator_agent_ids[parent_1_id])

                parent_2_creator_id = empty_recipe_id
                if parent_2_id != empty_recipe_id:
                    parent_2_creator_id = int(creator_agent_ids[parent_2_id])

                has_valid_parent_creators = (
                    parent_1_creator_id != empty_recipe_id
                    and parent_2_creator_id != empty_recipe_id
                )
                is_recombination_v1 = int(
                    is_combine_event
                    and has_valid_parent_creators
                    and parent_1_creator_id != parent_2_creator_id
                )

                distance_stats = get_recombination_distance_stats(
                    parent_1_id,
                    parent_2_id,
                    recipe_id,
                    parent_1_ids,
                    parent_2_ids,
                    creator_agent_ids,
                    birth_timesteps,
                    num_rules_in_initial_library,
                    empty_recipe_id,
                    lineage_memo,
                )

                rows.append(
                    {
                        "seed": int(seed),
                        "fee": float(fee),
                        "recipe_id": recipe_id,
                        "is_recombination_v1": is_recombination_v1,
                        "recomb_branch_distance_innov_only": distance_stats[
                            "recomb_branch_distance_innov_only"
                        ],
                        "recomb_mrca_age_innov_only": distance_stats[
                            "recomb_mrca_age_innov_only"
                        ],
                    }
                )

    return pd.DataFrame(rows)


# load raw outputs
# there may be multiple outputs files corresponding to different seeds or fee conditions; load and concatenate them all
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

population_data = build_population_df(outputs)
agent_data = build_agent_df(outputs)
recipe_lineage_data = build_recipe_dfs(outputs)
recipe_descendant_data = build_recipe_descendant_df(outputs)
recipe_recombination_data = build_recipe_recombination_df(outputs)

for df in (
    recipe_lineage_data,
    recipe_descendant_data,
    recipe_recombination_data,
    population_data,
):
    df["fee_axis_value"] = get_fee_axis_values(df["fee"])
    df["fee_plot"] = df["fee_axis_value"]

population_data.to_csv(DATA_DIR / "population_data.csv", index=False)
agent_data.to_csv(DATA_DIR / "agent_data.csv", index=False)
recipe_lineage_data.to_csv(DATA_DIR / "recipe_lineage_data.csv", index=False)
recipe_descendant_data.to_csv(DATA_DIR / "recipe_descendant_data.csv", index=False)
recipe_recombination_data.to_csv(
    DATA_DIR / "recipe_recombination_data.csv", index=False
)
np.save(DATA_DIR / "jaccard_matrices.npy", outputs["jaccard_matrices"])
