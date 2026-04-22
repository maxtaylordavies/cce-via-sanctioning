import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.grammar import (
    PAD,
    N,
    H,
    S,
    T,
    P,
    MAX_RECIPE_LEN,
    REVERSE_RULES,
    atomic_rules,
    generate_plant,
    pregenerate_plants,
)

TOKEN_TO_COLOR = {N: "#00A068", H: "#89D8D9", S: "#1C6897", T: "#C95B63", P: "#DF821E"}


def plant_to_colors(plant):
    # convert plant to list
    if isinstance(plant, jax.Array):
        plant = plant.tolist()
    return [TOKEN_TO_COLOR[token] for token in plant if token != PAD]


def render_tokens(
    colors,
    ax,
    unit_size=1,
    link_length=0.5,
    link_height=0.1,
    offset=0,
    x_shift=0.0,
    y_shift=0.0,
):
    x_offset = offset * (unit_size + link_length)
    for i, color in enumerate(colors):
        x = x_offset + x_shift + i * (unit_size + link_length)

        # draw link first, under squares
        if i > 0:
            prev_x = (
                x_shift
                + (i - 1 + offset) * (unit_size + link_length)
                + unit_size
                - 0.05
            )
            ax.add_patch(
                plt.Rectangle(
                    (prev_x, ((unit_size - link_height) / 2) + y_shift),
                    link_length + 0.1,
                    link_height,
                    color="grey",
                    zorder=1,
                    linewidth=0,
                )
            )

        # draw square on top
        ax.add_patch(
            plt.Circle(
                (x + unit_size / 2, (unit_size / 2) + y_shift),
                unit_size / 2,
                color=color,
                zorder=2,
                linewidth=0,
            )
        )


def visualise_single_plant(plant, unit_size=1, link_length=0.5, link_height=0.1):
    colors = plant_to_colors(plant)

    # we want to visualise the plant as a horizontal chain of colored squares
    # connected by lines to show the structure
    total_length = len(colors) * (unit_size + link_length) - link_length
    fig, ax = plt.subplots(figsize=(total_length, unit_size))

    render_tokens(colors, ax, unit_size, link_length, link_height)

    ax.set_xlim(0, len(colors) * (unit_size + link_length))
    ax.set_ylim(0, unit_size)
    ax.axis("off")

    return fig, ax


def visualise_multiple_plants(
    plants, unit_size=1, link_length=0.5, link_height=0.1, align=True
):
    # get actual length of longest plant
    max_length = max((plant != PAD).sum() for plant in plants)

    # compute offsets to align the plants' "N" tokens vertically
    n_indices = (plants == N).argmax(axis=1)
    offsets = n_indices[-1] - n_indices
    offsets = offsets if align else [0] * len(plants)

    # create figure and axes
    fig, axs = plt.subplots(
        len(plants),
        1,
        figsize=(max_length, len(plants) * unit_size * 0.85),
    )
    for plant, ax, offset in zip(plants, axs, offsets):
        colors = plant_to_colors(plant)
        render_tokens(
            colors,
            ax,
            offset=offset,
            unit_size=unit_size,
            link_length=link_length,
            link_height=link_height,
        )

        ax.set_xlim(0, max_length * (unit_size + link_length))
        ax.set_ylim(0, unit_size)
        ax.axis("off")

    fig.tight_layout()
    return fig, axs


def visualise_rules(
    reverse_rules, unit_size=1, link_length=0.5, link_height=0.1, flip=False
):
    rules = [
        (plant_to_colors(target), plant_to_colors(expansion))
        for target, expansions in reverse_rules.items()
        for expansion in expansions
    ]

    if flip:
        rules = [
            (expansion_colors, target_colors)
            for target_colors, expansion_colors in rules
        ]

    max_target_len = max(len(target_colors) for target_colors, _ in rules)
    max_expansion_len = max(len(expansion_colors) for _, expansion_colors in rules)
    arrow_gap = 1
    total_slots = max_target_len + arrow_gap + max_expansion_len

    fig, axs = plt.subplots(
        len(rules),
        1,
        figsize=(
            (total_slots - 1) * (unit_size + link_length),
            len(rules) * unit_size,
        ),
    )
    if len(rules) == 1:
        axs = [axs]

    arrow_start_x = max_target_len * (unit_size + link_length)
    arrow_dx = arrow_gap * (unit_size + link_length) - link_length

    for ax, (target_colors, expansion_colors) in zip(axs, rules):
        target_offset = max_target_len - len(target_colors)
        expansion_offset = max_target_len + arrow_gap

        render_tokens(
            target_colors,
            ax,
            offset=target_offset,
            unit_size=unit_size,
            link_length=link_length,
            link_height=link_height,
        )
        render_tokens(
            expansion_colors,
            ax,
            offset=expansion_offset,
            unit_size=unit_size,
            link_length=link_length,
            link_height=link_height,
        )

        ax.annotate(
            "",
            xy=(arrow_start_x + arrow_dx, unit_size / 2),
            xytext=(arrow_start_x, unit_size / 2),
            arrowprops={"arrowstyle": "<->", "linewidth": 3, "color": "black"},
        )
        ax.set_xlim(0, total_slots * (unit_size + link_length))
        ax.set_ylim(0, unit_size)
        ax.axis("off")

    # fig.tight_layout()
    return fig, axs


def visualise_recipe(recipe, unit_size=1, link_length=0.5, link_height=0.1):
    if isinstance(recipe, jax.Array):
        recipe = recipe.tolist()

    rule_ids = [rule_idx for rule_idx in recipe[:MAX_RECIPE_LEN] if rule_idx != PAD]
    rules = [
        (
            plant_to_colors(atomic_rules[rule_idx, 0]),
            plant_to_colors(atomic_rules[rule_idx, 1]),
        )
        for rule_idx in rule_ids
    ]

    # handle empty recipe
    if not rules:
        fig, ax = plt.subplots(figsize=(unit_size, unit_size))
        ax.set_xlim(0, unit_size)
        ax.set_ylim(0, unit_size)
        ax.axis("off")
        return fig, ax

    # max_target_len = max(len(target_colors) for target_colors, _ in rules)
    # max_expansion_len = max(len(expansion_colors) for _, expansion_colors in rules)
    arrow_gap = 1.0
    arrow_padding = 0.25
    box_padding = 0.25
    inter_box_gap = 0.75

    box_slot_counts = [len(tc) + len(ec) + arrow_gap for tc, ec in rules]
    box_widths = [
        (bsc * (unit_size + link_length)) + (2 * box_padding) - (2 * link_length)
        for bsc in box_slot_counts
    ]
    total_width = sum(box_widths) + ((len(box_widths) - 1) * inter_box_gap)

    fig, ax = plt.subplots(figsize=(total_width, unit_size + 2 * box_padding))

    # arrow_start_x = box_padding + max_target_len * (unit_size + link_length)
    arrow_dx = arrow_gap * (unit_size + link_length) - link_length
    y_mid = box_padding + unit_size / 2

    for idx, (target_colors, expansion_colors) in enumerate(rules):

        box_x = sum(box_widths[:idx]) + (idx * inter_box_gap)

        # box_x = idx * (box_width + inter_box_gap)
        # target_offset = max_target_len - len(target_colors)
        expansion_offset = len(target_colors) + arrow_gap

        render_tokens(
            target_colors,
            ax,
            # offset=target_offset,
            x_shift=box_x + box_padding,
            y_shift=box_padding,
            unit_size=unit_size,
            link_length=link_length,
            link_height=link_height,
        )
        render_tokens(
            expansion_colors,
            ax,
            offset=expansion_offset,
            x_shift=box_x + box_padding - link_length,
            y_shift=box_padding,
            unit_size=unit_size,
            link_length=link_length,
            link_height=link_height,
        )

        arrow_start_x = (
            box_padding
            + (len(target_colors) * (unit_size + link_length))
            - link_length
            + arrow_padding
        )
        ax.annotate(
            "",
            xy=(box_x + arrow_start_x + arrow_dx, y_mid),
            xytext=(box_x + arrow_start_x, y_mid),
            arrowprops={"arrowstyle": "->", "linewidth": 3, "color": "black"},
        )
        ax.add_patch(
            plt.Rectangle(
                (box_x, 0),
                box_widths[idx],
                unit_size + 2 * box_padding,
                fill=False,
                edgecolor="black",
                linewidth=2.5,
            )
        )

        if idx < len(rules) - 1:
            ax.plot(
                [box_x + box_widths[idx], box_x + box_widths[idx] + inter_box_gap],
                [y_mid, y_mid],
                color="black",
                linewidth=2.5,
            )

    ax.set_xlim(-0.1, total_width + 0.5)
    ax.set_ylim(-0.1, unit_size + 2 * box_padding + 0.1)
    ax.axis("off")
    # fig.tight_layout()
    return fig, ax


key = jax.random.PRNGKey(2)
max_level, num_per_level = 10, 10
plants = pregenerate_plants(key, num_per_level, max_level)

for level in range(1, max_level + 1):
    avg_length = (plants[level] != PAD).sum(axis=1).mean()
    print(f"Level {level}: average plant length = {avg_length:.2f}")
    fig, axs = visualise_multiple_plants(plants[level], align=False)
    fig.savefig(f"figures/plants/level-{level}.svg")

# for seed in range(5):
#     key = jax.random.PRNGKey(seed)
#     plant, intermediates = generate_plant(key, complexity_level=10)
#     fig, axs = visualise_multiple_plants(intermediates)
#     fig.savefig(f"figures/plants/generation-process-seed-{seed}.svg")

# fig, axs = visualise_rules(REVERSE_RULES)
# fig.savefig("figures/generating_rules.svg")

# recipe_ = jnp.array([1, 2, 3, 4, 5, 6], dtype=int)
# recipe = jnp.zeros(MAX_RECIPE_LEN, dtype=int).at[: recipe_.shape[0]].set(recipe_)

# fig, ax = visualise_recipe(recipe)
# fig.savefig("figures/recipe.pdf")
