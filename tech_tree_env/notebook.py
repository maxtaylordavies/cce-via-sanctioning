"""Value capture in an open-ended compositional technology tree.

The search space is a deep sequence of combinatorial technology levels.  Each
technology above level zero requires a different pair of technologies from the
preceding level.  The finite array bound is only a computational allocation:
the intended runs never approach it, so progress is governed by the population's
endogenous technological frontier rather than exhaustion of a small fixed DAG.

Agents choose between innovation and imitation with the same reinforcement-
learning mechanism used in the other experiment-1 environments.  Immediate
action rewards update the chosen role, while capture income received later is
credited to innovation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1
NO_CREATOR = -1

# A broad level permits multiple, complementary routes upward.  The deep array
# limit makes the realised population frontier effectively open-ended over the
# experiment's time horizon.
TECHNOLOGIES_PER_LEVEL = 16
MAX_TECH_LEVELS = 512
N_BASIC_TECHS = TECHNOLOGIES_PER_LEVEL
N_TECHS = TECHNOLOGIES_PER_LEVEL * MAX_TECH_LEVELS


def _build_open_ended_tree():
    """Build a deterministic deep lattice of pairwise recombinations."""
    prerequisites = [(NO_CREATOR, NO_CREATOR)] * TECHNOLOGIES_PER_LEVEL
    layers = [0] * TECHNOLOGIES_PER_LEVEL

    for level in range(1, MAX_TECH_LEVELS):
        previous_start = (level - 1) * TECHNOLOGIES_PER_LEVEL
        # Rotating the second parent prevents the lattice from decomposing into
        # independent vertical ladders while keeping every technology's recipe
        # to exactly two prerequisite technologies.
        offset = 1 + ((level - 1) % (TECHNOLOGIES_PER_LEVEL - 1))
        for branch in range(TECHNOLOGIES_PER_LEVEL):
            prerequisites.append(
                (
                    previous_start + branch,
                    previous_start + (branch + offset) % TECHNOLOGIES_PER_LEVEL,
                )
            )
            layers.append(level)

    return prerequisites, layers


_PREREQUISITES, _LAYERS = _build_open_ended_tree()
TECH_PREREQUISITES = jnp.asarray(_PREREQUISITES, dtype=jnp.int32)
TECH_LAYERS = jnp.asarray(_LAYERS, dtype=jnp.int32)


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters shared by local screens and the production GPU run."""

    n_agents: int = 100
    timesteps: int = 1100
    innovation_success: float = 0.25
    imitation_success: float = 1.0
    technology_value: float = 0.40
    innovation_cost: float = 0.15
    learning_rate: float = 0.10
    choice_beta: float = 0.10
    initial_q_value: float = 1.0
    turnover_rate: float = 0.001
    imitation_distance: int = 100


@jax.jit
def eligible_missing(known):
    """Return missing technologies whose two prerequisites are known."""
    safe_prerequisites = jnp.maximum(TECH_PREREQUISITES, 0)
    prerequisite_known = known[safe_prerequisites]
    prerequisite_known = jnp.where(
        TECH_PREREQUISITES == NO_CREATOR,
        True,
        prerequisite_known,
    )
    return (~known) & jnp.all(prerequisite_known, axis=1)


@jax.jit
def eligible_for_innovation(known, creators, agent_idx):
    """Require cumulative inventions to recombine some socially gained input."""
    eligible = eligible_missing(known)
    has_prerequisites = jnp.any(TECH_PREREQUISITES != NO_CREATOR, axis=1)
    safe_prerequisites = jnp.maximum(TECH_PREREQUISITES, 0)
    prerequisite_creators = creators[safe_prerequisites]
    has_social_input = jnp.any(
        (TECH_PREREQUISITES != NO_CREATOR) & (prerequisite_creators != agent_idx),
        axis=1,
    )
    return eligible & (~has_prerequisites | has_social_input)


@jax.jit
def update_role_values(q_values, roles, direct_rewards, capture_income, learning_rate):
    """Apply chosen-role RL and credit delayed capture income to innovation."""
    agent_indices = jnp.arange(roles.shape[0])
    innovating = roles == ROLE_INNOVATE
    chosen_rewards = direct_rewards + jnp.where(innovating, capture_income, 0.0)
    chosen_rpe = chosen_rewards - q_values[agent_indices, roles]
    updated_q_values = q_values.at[agent_indices, roles].add(learning_rate * chosen_rpe)

    delayed_capture = (~innovating) & (capture_income > 0.0)
    capture_rpe = capture_income - updated_q_values[:, ROLE_INNOVATE]
    return updated_q_values.at[:, ROLE_INNOVATE].add(
        learning_rate * jnp.where(delayed_capture, capture_rpe, 0.0)
    )


@jax.jit
def invalidate_dead_creators(creators, deaths):
    """Remove capture claims held by agents who die this period."""
    attributed = creators != NO_CREATOR
    safe_creators = jnp.where(attributed, creators, 0)
    creator_died = deaths[safe_creators]
    return jnp.where(attributed & creator_died, NO_CREATOR, creators)


@jax.jit
def _sample_from_mask(key, mask):
    """Sample uniformly from a mask, with a harmless empty-mask fallback."""
    has_options = jnp.any(mask)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    logits = jnp.where(has_options, logits, jnp.zeros_like(logits))
    return jax.random.categorical(key, logits), has_options


def make_neighbours_mask(grid_length, imitation_distance):
    """Pairwise toroidal Manhattan neighbourhood used for imitation."""
    n_agents = grid_length**2
    agent_indices = jnp.arange(n_agents)
    locations = jnp.stack(
        (agent_indices // grid_length, agent_indices % grid_length), axis=1
    )
    row_differences = jnp.abs(locations[:, None, 0] - locations[None, :, 0])
    column_differences = jnp.abs(locations[:, None, 1] - locations[None, :, 1])
    row_distances = jnp.minimum(row_differences, grid_length - row_differences)
    column_distances = jnp.minimum(column_differences, grid_length - column_differences)
    distances = row_distances + column_distances
    return (distances > 0) & (distances <= imitation_distance)


@jax.jit
def update_technologies(
    key,
    known,
    creators,
    roles,
    neighbours_mask,
    innovation_success,
    imitation_success,
    technology_value,
):
    """Perform simultaneous innovation or repertoire-imitation actions."""
    n_agents = known.shape[0]
    agent_indices = jnp.arange(n_agents)
    keys = jax.random.split(key, n_agents)

    def update_one(agent_key, agent_idx):
        innovate_key, demonstrator_key, copy_key = jax.random.split(agent_key, 3)
        own_known = known[agent_idx]
        own_creators = creators[agent_idx]

        def do_innovate(_):
            choice_key, success_key = jax.random.split(innovate_key)
            candidates = eligible_for_innovation(own_known, own_creators, agent_idx)
            technology, has_candidates = _sample_from_mask(choice_key, candidates)
            succeeds = has_candidates & jax.random.bernoulli(
                success_key, innovation_success
            )
            new_known = own_known.at[technology].set(own_known[technology] | succeeds)
            new_creators = own_creators.at[technology].set(
                jnp.where(succeeds, agent_idx, own_creators[technology])
            )
            return (
                new_known,
                new_creators,
                succeeds.astype(jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.zeros((N_TECHS,), dtype=jnp.bool_),
                jnp.full((N_TECHS,), NO_CREATOR, dtype=jnp.int32),
                succeeds.astype(jnp.float32) * technology_value,
            )

        def do_imitate(_):
            demonstrator, _ = _sample_from_mask(
                demonstrator_key, neighbours_mask[agent_idx]
            )
            ready_to_copy = eligible_missing(own_known) & known[demonstrator]
            copied = ready_to_copy & jax.random.bernoulli(
                copy_key,
                imitation_success,
                shape=(N_TECHS,),
            )
            new_known = own_known | copied
            new_creators = jnp.where(copied, creators[demonstrator], own_creators)
            copied_creator_ids = jnp.where(copied, creators[demonstrator], NO_CREATOR)
            n_copied = copied.sum(dtype=jnp.int32)
            return (
                new_known,
                new_creators,
                jnp.asarray(0, dtype=jnp.int32),
                n_copied,
                copied,
                copied_creator_ids,
                n_copied.astype(jnp.float32) * technology_value,
            )

        return jax.lax.cond(
            roles[agent_idx] == ROLE_INNOVATE,
            do_innovate,
            do_imitate,
            operand=None,
        )

    return jax.vmap(update_one)(keys, agent_indices)


@partial(
    jax.jit,
    static_argnames=("grid_length", "T", "imitation_distance"),
)
def run_simulation_loop(
    key,
    grid_length,
    T,
    value_capture_rate,
    innovation_success=0.25,
    imitation_success=1.0,
    technology_value=0.40,
    innovation_cost=0.15,
    turnover_rate=0.001,
    choice_beta=0.10,
    learning_rate=0.10,
    initial_q_value=1.0,
    imitation_distance=100,
):
    """Run one JAX-compiled open-ended technology-tree simulation."""
    n_agents = grid_length**2
    neighbours_mask = make_neighbours_mask(grid_length, imitation_distance)
    known = jnp.zeros((n_agents, N_TECHS), dtype=jnp.bool_)
    creators = jnp.full((n_agents, N_TECHS), NO_CREATOR, dtype=jnp.int32)
    q_values = jnp.full((n_agents, 2), initial_q_value, dtype=jnp.float32)

    def body_fn(carry, _):
        key_, known_, creators_, q_values_ = carry
        key_, death_key, role_key, update_key = jax.random.split(key_, 4)

        # Death creates a naive replacement and expires the deceased agent's
        # capture claims, matching the timing in the binary-trait environment.
        deaths = jax.random.bernoulli(death_key, turnover_rate, shape=(n_agents,))
        known_ = jnp.where(deaths[:, None], False, known_)
        creators_ = invalidate_dead_creators(creators_, deaths)
        creators_ = jnp.where(deaths[:, None], NO_CREATOR, creators_)
        q_values_ = jnp.where(deaths[:, None], initial_q_value, q_values_)

        role_probabilities = jax.nn.softmax(q_values_ / choice_beta, axis=1)
        roles = jax.random.categorical(role_key, jnp.log(role_probabilities), axis=1)
        (
            new_known,
            new_creators,
            innovated,
            imitated,
            copied,
            copied_creator_ids,
            gross_benefits,
        ) = update_technologies(
            update_key,
            known_,
            creators_,
            roles,
            neighbours_mask,
            innovation_success,
            imitation_success,
            technology_value,
        )

        capture_payments = (
            copied.astype(jnp.float32) * value_capture_rate * technology_value
        )
        attributed = copied & (copied_creator_ids != NO_CREATOR)
        safe_creators = jnp.where(attributed, copied_creator_ids, 0)
        capture_income = jnp.zeros((n_agents,), dtype=jnp.float32)
        capture_income = capture_income.at[safe_creators.reshape(-1)].add(
            jnp.where(attributed, capture_payments, 0.0).reshape(-1)
        )
        capture_paid = capture_payments.sum(axis=1)
        unclaimed_capture = jnp.where(copied & ~attributed, capture_payments, 0.0).sum()

        direct_rewards = gross_benefits - capture_paid
        direct_rewards -= jnp.where(roles == ROLE_INNOVATE, innovation_cost, 0.0)
        q_values_ = update_role_values(
            q_values_, roles, direct_rewards, capture_income, learning_rate
        )

        # Highest mastered level is the environment's cultural-complexity score.
        # Levels are one-indexed here so a population with basic technology has
        # score one and a completely naive agent has score zero.
        agent_levels = jnp.max(
            jnp.where(new_known, TECH_LAYERS[None, :] + 1, 0), axis=1
        )
        agent_scores = agent_levels.astype(jnp.float32)
        repertoire_sizes = new_known.sum(axis=1, dtype=jnp.int32)
        population_frontier = agent_levels.max()

        return (key_, new_known, new_creators, q_values_), (
            agent_scores,
            agent_levels,
            repertoire_sizes,
            roles,
            role_probabilities.mean(axis=0),
            innovated.sum(),
            imitated.sum(),
            capture_paid.sum(),
            capture_income.max(),
            unclaimed_capture,
            capture_income.mean(),
            population_frontier,
        )

    _, metrics = jax.lax.scan(
        body_fn,
        (key, known, creators, q_values),
        xs=None,
        length=T,
    )
    metrics = list(metrics)
    for metric_index in (5, 6, 7, 9):
        metrics[metric_index] = jnp.cumsum(metrics[metric_index])
    return tuple(metrics)


def run_simulation(seed, value_capture_rate, config=SimulationConfig()):
    """Convenience wrapper used by compact local screens and tests."""
    grid_length = int(np.sqrt(config.n_agents))
    if grid_length**2 != config.n_agents:
        raise ValueError("SimulationConfig.n_agents must be a perfect square.")
    metrics = run_simulation_loop(
        jax.random.PRNGKey(seed),
        grid_length,
        config.timesteps,
        value_capture_rate,
        innovation_success=config.innovation_success,
        imitation_success=config.imitation_success,
        technology_value=config.technology_value,
        innovation_cost=config.innovation_cost,
        turnover_rate=config.turnover_rate,
        choice_beta=config.choice_beta,
        learning_rate=config.learning_rate,
        initial_q_value=config.initial_q_value,
        imitation_distance=config.imitation_distance,
    )
    return {
        "mean_score": metrics[0].mean(axis=1),
        "innovator_frequency": (metrics[3] == ROLE_INNOVATE).mean(axis=1),
        "mean_tech_depth": metrics[1].mean(axis=1),
        "mean_repertoire_size": metrics[2].mean(axis=1),
        "population_frontier": metrics[11],
        "mean_capture_income": metrics[10],
    }


def run_grid(seeds, value_capture_rates, config=SimulationConfig()):
    """Run a lambda grid and return screen-friendly time-series arrays."""
    outputs = [
        [
            run_simulation(seed, value_capture_rate, config)
            for value_capture_rate in value_capture_rates
        ]
        for seed in seeds
    ]
    return {
        name: jnp.asarray(
            [[output[name] for output in seed_outputs] for seed_outputs in outputs]
        )
        for name in outputs[0][0]
    }


def main():
    """Run the production GPU sweep and save an analysis-ready NPZ archive."""
    seeds = list(range(3))
    grid_length, T_main, T_extra = 10, int(1e3), 100
    T = T_main + T_extra
    value_capture_rates = jnp.linspace(0.0, 1.0, 20)
    config = SimulationConfig(n_agents=grid_length**2, timesteps=T)

    def run_with_value_capture_rate(key, value_capture_rate):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                T,
                value_capture_rate,
                innovation_success=config.innovation_success,
                imitation_success=config.imitation_success,
                technology_value=config.technology_value,
                innovation_cost=config.innovation_cost,
                turnover_rate=config.turnover_rate,
                choice_beta=config.choice_beta,
                learning_rate=config.learning_rate,
                initial_q_value=config.initial_q_value,
                imitation_distance=config.imitation_distance,
            )
        )

    metric_names = (
        "agent_scores",
        "agent_levels",
        "agent_repertoire_sizes",
        "agent_roles",
        "role_probs",
        "n_innovated",
        "n_imitated",
        "total_value_captured",
        "max_capture_income",
        "total_unclaimed_capture",
        "mean_capture_income",
        "population_frontier",
    )
    all_metrics = {name: [] for name in metric_names}

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        seed_metrics = jax.vmap(run_with_value_capture_rate, in_axes=(None, 0))(
            key, value_capture_rates
        )
        for name, values in zip(metric_names, seed_metrics, strict=True):
            all_metrics[name].append(np.asarray(values))

    simulation_outputs = {
        "value_capture_rates": np.asarray(value_capture_rates),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "T_main": np.int32(T_main),
        "T_extra": np.int32(T_extra),
        "grid_length": np.int32(grid_length),
        "n_technologies": np.int32(N_TECHS),
        "technologies_per_level": np.int32(TECHNOLOGIES_PER_LEVEL),
        "max_tech_levels": np.int32(MAX_TECH_LEVELS),
        "max_portfolio_score": np.float32(MAX_TECH_LEVELS),
        "score_normalizer": np.float32(MAX_TECH_LEVELS),
        "innovation_success": np.float32(config.innovation_success),
        "imitation_success": np.float32(config.imitation_success),
        "technology_value": np.float32(config.technology_value),
        "innov_cost": np.float32(config.innovation_cost),
        "p_death": np.float32(config.turnover_rate),
        "init_q": np.float32(config.initial_q_value),
        "choice_beta": np.float32(config.choice_beta),
        "learning_rate": np.float32(config.learning_rate),
        "imit_dist_threshold": np.int32(config.imitation_distance),
        "model_variant": np.asarray("open_ended_compositional_tech_tree"),
        "creator_attribution": np.asarray("exact_technology_creator"),
        "score_definition": np.asarray("highest_technology_level_mastered"),
        "learning_rule": np.asarray(
            "chosen_role_plus_delayed_capture_income_to_innovate"
        ),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
    }
    simulation_outputs.update(
        {name: np.stack(values, axis=0) for name, values in all_metrics.items()}
    )
    np.savez(
        f"simulation_outputs_{seeds[0]}-{seeds[-1]}.npz",
        **simulation_outputs,
    )


if __name__ == "__main__":
    main()
