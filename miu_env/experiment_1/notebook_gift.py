from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

ROLE_INNOVATE, ROLE_IMITATE = 0, 1

p_max = 100
l_max = 100


def get_increment(l):
    tmp = 0
    for j in range(1, l + 1):
        tmp += 0.95 ** (l - j)
    tmp *= 0.05 / (1 - (0.95**l_max))
    return tmp * p_max


increments = jnp.array([0] + [get_increment(l) for l in range(1, l_max + 1)])


def exploit(agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    arm_idx = jnp.argmax(agent_arm_payoff_estimates)
    arm_level = agent_arm_levels[arm_idx]
    payoff = full_rewards[arm_idx, arm_level]
    return payoff, agent_arm_payoff_estimates.at[arm_idx].set(payoff)


def explore(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    unknown_mask = agent_arm_levels == 0
    n_unknown = unknown_mask.sum()

    def learn_new_arm():
        p_arm = unknown_mask / n_unknown
        arm_idx = jax.random.choice(key, full_rewards.shape[0], p=p_arm)
        payoff = full_rewards[arm_idx, 1]
        return (
            agent_arm_levels.at[arm_idx].set(1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            True,
        )

    def do_nothing():
        return agent_arm_levels, agent_arm_payoff_estimates, False

    return jax.lax.cond(n_unknown > 0, learn_new_arm, do_nothing)


def refine(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    eligible_mask = (agent_arm_levels > 0) & (agent_arm_levels < l_max)
    n_eligible = eligible_mask.sum()

    def refine_weighted_arm():
        arm_weights = jnp.where(
            eligible_mask, jnp.maximum(agent_arm_payoff_estimates, 0.0), 0.0
        )
        total_arm_weight = arm_weights.sum()

        def sample_weighted_arm():
            p_arm = arm_weights / total_arm_weight
            return jax.random.choice(key, full_rewards.shape[0], p=p_arm)

        def sample_uniform_eligible_arm():
            p_arm = eligible_mask / n_eligible
            return jax.random.choice(key, full_rewards.shape[0], p=p_arm)

        arm_idx = jax.lax.cond(
            total_arm_weight > 0,
            sample_weighted_arm,
            sample_uniform_eligible_arm,
        )

        arm_level = agent_arm_levels[arm_idx]
        payoff = full_rewards[arm_idx, arm_level + 1]
        return (
            agent_arm_levels.at[arm_idx].set(arm_level + 1),
            agent_arm_payoff_estimates.at[arm_idx].set(payoff),
            True,
        )

    def do_nothing():
        return agent_arm_levels, agent_arm_payoff_estimates, False

    return jax.lax.cond(n_eligible > 0, refine_weighted_arm, do_nothing)


def innovate(key, agent_arm_levels, agent_arm_payoff_estimates, full_rewards):
    keys = jax.random.split(key, 2)
    op = jax.random.bernoulli(keys[0])
    return jax.lax.cond(
        op,
        lambda: explore(
            keys[1], agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        ),
        lambda: refine(
            keys[1], agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        ),
    )


@jax.jit
def choose_demonstrator(
    key,
    neighbours_mask,
    prestiges,
    agent_idx,
    prestige_bias,
    demonstrator_prestige_baseline,
):
    candidate_mask = neighbours_mask[agent_idx]
    prestige_scores = (
        demonstrator_prestige_baseline + jnp.maximum(prestiges, 0.0)
    ) ** prestige_bias
    weights = jnp.where(candidate_mask, prestige_scores, 0.0)

    fallback_mask = jnp.arange(prestiges.shape[0]) != agent_idx
    fallback_weights = jnp.where(fallback_mask, 1.0, 0.0)
    weights = jnp.where(weights.sum() > 0, weights, fallback_weights)
    logits = jnp.where(weights > 0, jnp.log(weights), -jnp.inf)
    return jax.random.categorical(key, logits)


@jax.jit
def compute_gift(
    demonstrator_prestige,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    raw_gift = gift_base + gift_rate * (
        jnp.maximum(demonstrator_prestige, 0.0) ** gift_exponent
    )
    return jnp.minimum(raw_gift, gift_cap)


def imitate(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    neighbours_mask,
    prestiges,
    agent_idx,
    prestige_bias,
    demonstrator_prestige_baseline,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    demonstrator_key, arm_key = jax.random.split(key)
    demonstrator_idx = choose_demonstrator(
        demonstrator_key,
        neighbours_mask,
        prestiges,
        agent_idx,
        prestige_bias,
        demonstrator_prestige_baseline,
    )

    demonstrator_arm_levels = all_agent_arm_levels[demonstrator_idx]
    demonstrator_payoff_estimates = all_agent_arm_payoff_estimates[demonstrator_idx]
    arm_weights = jnp.where(
        demonstrator_arm_levels > 0,
        jnp.maximum(demonstrator_payoff_estimates, 0.0),
        0.0,
    )
    total_arm_weight = arm_weights.sum()

    def sample_weighted_arm():
        p_arm = arm_weights / total_arm_weight
        return jax.random.choice(arm_key, all_agent_arm_levels.shape[1], p=p_arm)

    def sample_uniform_arm():
        return jax.random.randint(arm_key, (), 0, all_agent_arm_levels.shape[1])

    arm_idx = jax.lax.cond(
        total_arm_weight > 0,
        sample_weighted_arm,
        sample_uniform_arm,
    )

    new_level = all_agent_arm_levels[demonstrator_idx, arm_idx]
    new_payoff = all_agent_arm_payoff_estimates[demonstrator_idx, arm_idx]
    current_level = all_agent_arm_levels[agent_idx, arm_idx]
    accept = new_level > current_level
    gift = compute_gift(
        prestiges[demonstrator_idx],
        gift_rate,
        gift_base,
        gift_exponent,
        gift_cap,
    )
    return jax.lax.cond(
        accept,
        lambda: (
            all_agent_arm_levels[agent_idx].at[arm_idx].set(new_level),
            all_agent_arm_payoff_estimates[agent_idx].at[arm_idx].set(new_payoff),
            True,
            demonstrator_idx,
            gift,
        ),
        lambda: (
            all_agent_arm_levels[agent_idx],
            all_agent_arm_payoff_estimates[agent_idx],
            False,
            demonstrator_idx,
            gift,
        ),
    )


@jax.jit
def update_arm_knowledge_and_gifts(
    key,
    all_agent_arm_levels,
    all_agent_arm_payoff_estimates,
    all_roles,
    neighbours_mask,
    full_rewards,
    prestiges,
    prestige_bias,
    demonstrator_prestige_baseline,
    gift_rate,
    gift_base,
    gift_exponent,
    gift_cap,
):
    def per_agent(key_, agent_idx):
        def do_innovate(_):
            new_levels, new_payoffs, success = innovate(
                key_,
                all_agent_arm_levels[agent_idx],
                all_agent_arm_payoff_estimates[agent_idx],
                full_rewards,
            )
            return (
                new_levels,
                new_payoffs,
                success.astype(int),
                0,
                agent_idx,
                jnp.asarray(0.0, dtype=jnp.float32),
            )

        def do_imitate(_):
            new_levels, new_payoffs, success, demonstrator_idx, gift = imitate(
                key_,
                all_agent_arm_levels,
                all_agent_arm_payoff_estimates,
                neighbours_mask,
                prestiges,
                agent_idx,
                prestige_bias,
                demonstrator_prestige_baseline,
                gift_rate,
                gift_base,
                gift_exponent,
                gift_cap,
            )
            return (
                new_levels,
                new_payoffs,
                0,
                success.astype(int),
                demonstrator_idx,
                gift,
            )

        return jax.lax.cond(
            all_roles[agent_idx] == ROLE_INNOVATE, do_innovate, do_imitate, operand=None
        )

    keys = jax.random.split(key, all_agent_arm_levels.shape[0])
    return jax.vmap(per_agent)(keys, jnp.arange(all_agent_arm_levels.shape[0]))


@partial(jax.jit, static_argnames=("n_arms",))
def sample_arm_rewards(key, n_arms):
    rewards = jax.random.exponential(key, shape=(n_arms,))
    rewards = jnp.ceil(rewards**2)

    full_rewards = jnp.zeros((n_arms, 1 + l_max))
    return full_rewards.at[:, 1:].set(rewards[:, None] + increments[:l_max][None, :])


@partial(jax.jit, static_argnames=("grid_length", "n_arms", "T"))
def run_simulation_loop(
    key,
    grid_length,
    n_arms,
    T,
    prestige_gain,
    innov_cost=2.0,
    p_death=0.001,
    p_change=0.01,
    choice_beta=0.1,
    imit_dist_threshold=1,
    learning_rate=0.1,
    init_q=1.0,
    prestige_decay=0.01,
    prestige_value=0.0,
    prestige_bias=1.0,
    demonstrator_prestige_baseline=1.0,
    gift_rate=0.01,
    gift_base=0.0,
    gift_exponent=1.0,
    gift_cap=jnp.inf,
    eligibility_trace_decay=0.5,
    eligibility_discount=1.0,
):
    n_agents = grid_length**2

    # Compute pairwise toroidal distances between agents for imitation.
    agent_idxs = jnp.arange(n_agents)
    agent_locs = jnp.stack(
        [
            agent_idxs // grid_length,  # row index
            agent_idxs % grid_length,  # column index
        ]
    ).T
    row_diffs = jnp.abs(agent_locs[:, None, 0] - agent_locs[None, :, 0])
    col_diffs = jnp.abs(agent_locs[:, None, 1] - agent_locs[None, :, 1])
    torus_row_dists = jnp.minimum(row_diffs, grid_length - row_diffs)
    torus_col_dists = jnp.minimum(col_diffs, grid_length - col_diffs)
    agent_dists = torus_row_dists + torus_col_dists
    neighbours_mask = ((agent_dists > 0) & (agent_dists <= imit_dist_threshold)).astype(
        jnp.bool_
    )

    vmapped_exploit = jax.vmap(exploit, in_axes=(0, 0, None))

    def body_fn(carry, t):
        (
            key,
            full_rewards,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            q_vals,
            prestiges,
            eligibilities,
        ) = carry

        # get new keys
        key, change_key, death_key, role_key, update_key = jax.random.split(key, 5)

        # maybe reset environment rewards
        full_rewards = jax.lax.cond(
            jax.random.bernoulli(change_key, p=p_change),
            lambda: sample_arm_rewards(key, n_arms),
            lambda: full_rewards,
        )

        # random deaths
        deaths = jax.random.bernoulli(death_key, p=p_death, shape=(n_agents,))
        agent_arm_levels = jnp.where(deaths[:, None], 0, agent_arm_levels)
        agent_arm_payoff_estimates = jnp.where(
            deaths[:, None], 0.0, agent_arm_payoff_estimates
        )
        prestiges = jnp.where(deaths, 0.0, prestiges)
        eligibilities = jnp.where(deaths[:, None], 0.0, eligibilities)
        prestiges = prestiges * (1.0 - prestige_decay)

        # compute payoffs from exploiting current knowledge
        curr_payoffs, agent_arm_payoff_estimates = vmapped_exploit(
            agent_arm_levels, agent_arm_payoff_estimates, full_rewards
        )

        # agents select roles
        role_probs = jax.nn.softmax(q_vals / choice_beta, axis=1)
        roles = jax.random.categorical(role_key, jnp.log(role_probs), axis=1)

        # update knowledge based on roles and compute transfers
        (
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            innovated,
            imitated,
            demonstrator_idxs,
            gifts_paid,
        ) = update_arm_knowledge_and_gifts(
            update_key,
            agent_arm_levels,
            agent_arm_payoff_estimates,
            roles,
            neighbours_mask,
            full_rewards,
            prestiges,
            prestige_bias,
            demonstrator_prestige_baseline,
            gift_rate,
            gift_base,
            gift_exponent,
            gift_cap,
        )
        new_payoffs, _ = vmapped_exploit(
            new_agent_arm_levels, new_agent_arm_payoff_estimates, full_rewards
        )
        prestige_changes = prestige_gain * innovated.astype(jnp.float32)
        new_prestiges = prestiges + prestige_changes

        incoming_gifts = (
            jnp.zeros((n_agents,), dtype=jnp.float32)
            .at[demonstrator_idxs]
            .add(gifts_paid)
        )
        transfer_rewards = incoming_gifts - gifts_paid

        # compute role rewards and costs
        rewards = new_payoffs - curr_payoffs
        rewards += transfer_rewards
        rewards -= jnp.where(roles == ROLE_INNOVATE, innov_cost, 0.0)
        rewards += prestige_value * prestige_changes

        # update q-values using an eligibility trace over recent role choices.
        rpe = rewards - q_vals[jnp.arange(n_agents), roles]
        decayed_eligibilities = (
            eligibility_discount * eligibility_trace_decay * eligibilities
        )
        role_eligibilities = jax.nn.one_hot(roles, 2, dtype=eligibilities.dtype)
        new_eligibilities = decayed_eligibilities + role_eligibilities
        new_all_q_vals = q_vals + learning_rate * (rpe[:, None] * new_eligibilities)

        # compute some metrics for logging
        mean_payoff = curr_payoffs.mean()
        mean_avg_level = agent_arm_levels.mean()
        mean_max_level = agent_arm_levels.max(axis=1).mean()
        mean_role_probs = role_probs.mean(axis=0)
        mean_prestige = new_prestiges.mean()
        max_prestige = new_prestiges.max()
        total_gifts = gifts_paid.sum()
        max_gift_income = incoming_gifts.max()

        return (
            key,
            full_rewards,
            new_agent_arm_levels,
            new_agent_arm_payoff_estimates,
            new_all_q_vals,
            new_prestiges,
            new_eligibilities,
        ), (
            mean_payoff,
            mean_avg_level,
            mean_max_level,
            innovated.sum(),
            imitated.sum(),
            mean_role_probs,
            mean_prestige,
            max_prestige,
            total_gifts,
            max_gift_income,
        )

    full_rewards = sample_arm_rewards(key, n_arms)
    carry = (
        key,
        full_rewards,
        jnp.zeros((n_agents, n_arms), dtype=jnp.int32),  # agent_arm_levels
        jnp.zeros((n_agents, n_arms), dtype=jnp.float32),  # agent_arm_payoff_estimates
        jnp.full((n_agents, 2), init_q, dtype=jnp.float32),  # q_vals
        jnp.zeros((n_agents,), dtype=jnp.float32),  # prestiges
        jnp.zeros((n_agents, 2), dtype=jnp.float32),  # eligibilities
    )

    _, metrics = jax.lax.scan(body_fn, carry, jnp.arange(T))
    metrics = list(metrics)
    metrics[0] /= full_rewards.max()
    metrics[3] = jnp.cumsum(metrics[3])  # cumulative number innovated
    metrics[4] = jnp.cumsum(metrics[4])  # cumulative number imitated
    metrics[8] = jnp.cumsum(metrics[8])  # cumulative gift transfers
    return metrics


def main():
    seeds = list(range(5))
    grid_length, n_arms, T = 10, 200, int(2e3)
    prestige_gain_vals = jnp.linspace(0.0, 10.0, 21)
    prestige_decay = 0.01
    prestige_value = 0.0
    prestige_bias = 1.0
    demonstrator_prestige_baseline = 1.0
    gift_rate = 0.01
    gift_base = 0.0
    gift_exponent = 1.0
    gift_cap = np.float32(np.inf)
    eligibility_trace_decay = 0.5
    eligibility_discount = 1.0

    def run_with_prestige_gain(key, prestige_gain):
        return jax.block_until_ready(
            run_simulation_loop(
                key,
                grid_length,
                n_arms,
                T,
                prestige_gain=prestige_gain,
                prestige_decay=prestige_decay,
                prestige_value=prestige_value,
                prestige_bias=prestige_bias,
                demonstrator_prestige_baseline=demonstrator_prestige_baseline,
                gift_rate=gift_rate,
                gift_base=gift_base,
                gift_exponent=gift_exponent,
                gift_cap=gift_cap,
                eligibility_trace_decay=eligibility_trace_decay,
                eligibility_discount=eligibility_discount,
            )
        )

    all_prop_payoffs = []
    all_mean_avg_levels = []
    all_mean_max_levels = []
    all_n_innovs = []
    all_n_imits = []
    all_mean_role_probs = []
    all_mean_prestige = []
    all_max_prestige = []
    all_total_gifts = []
    all_max_gift_income = []

    for seed in tqdm(seeds):
        key = jax.random.PRNGKey(seed)
        (
            prop_payoffs,
            mean_avg_levels,
            mean_max_levels,
            n_innovs,
            n_imits,
            mean_role_probs,
            mean_prestige,
            max_prestige,
            total_gifts,
            max_gift_income,
        ) = jax.vmap(run_with_prestige_gain, in_axes=(None, 0))(key, prestige_gain_vals)

        all_prop_payoffs.append(np.asarray(prop_payoffs))
        all_mean_avg_levels.append(np.asarray(mean_avg_levels))
        all_mean_max_levels.append(np.asarray(mean_max_levels))
        all_n_innovs.append(np.asarray(n_innovs))
        all_n_imits.append(np.asarray(n_imits))
        all_mean_role_probs.append(np.asarray(mean_role_probs))
        all_mean_prestige.append(np.asarray(mean_prestige))
        all_max_prestige.append(np.asarray(max_prestige))
        all_total_gifts.append(np.asarray(total_gifts))
        all_max_gift_income.append(np.asarray(max_gift_income))

    simulation_outputs = {
        "fees": np.asarray(prestige_gain_vals),
        "prestige_gains": np.asarray(prestige_gain_vals),
        "seeds": np.asarray(seeds),
        "T": np.int32(T),
        "grid_length": np.int32(grid_length),
        "n_arms": np.int32(n_arms),
        "prestige_decay": np.float32(prestige_decay),
        "prestige_value": np.float32(prestige_value),
        "prestige_bias": np.float32(prestige_bias),
        "demonstrator_prestige_baseline": np.float32(demonstrator_prestige_baseline),
        "gift_rate": np.float32(gift_rate),
        "gift_base": np.float32(gift_base),
        "gift_exponent": np.float32(gift_exponent),
        "gift_cap": np.float32(gift_cap),
        "eligibility_trace_decay": np.float32(eligibility_trace_decay),
        "eligibility_discount": np.float32(eligibility_discount),
        "role_innovate": np.int32(ROLE_INNOVATE),
        "role_imitate": np.int32(ROLE_IMITATE),
        "payoffs": np.stack(all_prop_payoffs, axis=0),
        "avg_levels": np.stack(all_mean_avg_levels, axis=0),
        "max_levels": np.stack(all_mean_max_levels, axis=0),
        "n_innov": np.stack(all_n_innovs, axis=0),
        "n_imit": np.stack(all_n_imits, axis=0),
        "role_probs": np.stack(all_mean_role_probs, axis=0),
        "mean_prestige": np.stack(all_mean_prestige, axis=0),
        "max_prestige": np.stack(all_max_prestige, axis=0),
        "total_gifts": np.stack(all_total_gifts, axis=0),
        "max_gift_income": np.stack(all_max_gift_income, axis=0),
    }

    np.savez(
        f"simulation_outputs_gift_{seeds[0]}-{seeds[-1]}.npz", **simulation_outputs
    )


if __name__ == "__main__":
    main()
