from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


@dataclass(frozen=True)
class RenderConfig:
    rows: int
    cols: int
    max_foraging_level: int
    tile_width: float
    tile_height: float
    level_height: float
    dpi: int
    background: str
    show_agents: bool


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {value!r}")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in range(0, 6, 2))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    clipped = tuple(max(0.0, min(1.0, channel)) for channel in rgb)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in clipped)


def blend(color_a: str, color_b: str, weight: float) -> str:
    a = hex_to_rgb(color_a)
    b = hex_to_rgb(color_b)
    mixed = tuple(
        (1.0 - weight) * ca + weight * cb for ca, cb in zip(a, b, strict=True)
    )
    return rgb_to_hex(mixed)


def build_palette(start: str, end: str, steps: int) -> list[str]:
    if steps <= 0:
        raise ValueError("max_foraging_level must be at least 1")
    if steps == 1:
        return [start]
    return [blend(start, end, idx / (steps - 1)) for idx in range(steps)]


def project(x: float, y: float, z: float, cfg: RenderConfig) -> tuple[float, float]:
    return (
        (x - y) * cfg.tile_width / 2.0,
        (x + y) * cfg.tile_height / 2.0 - z * cfg.level_height,
    )


def projected_polygon(
    points: list[tuple[float, float, float]], cfg: RenderConfig
) -> list[tuple[float, float]]:
    return [project(x, y, z, cfg) for x, y, z in points]


def tile_polygon(x: int, y: int, cfg: RenderConfig) -> list[tuple[float, float]]:
    z = cfg.max_foraging_level
    corners = [
        (x, y, z),
        (x + 1, y, z),
        (x + 1, y + 1, z),
        (x, y + 1, z),
    ]
    return projected_polygon(corners, cfg)


def front_band_polygon(level_idx: int, cfg: RenderConfig) -> list[tuple[float, float]]:
    z_top = cfg.max_foraging_level - level_idx
    z_bottom = z_top - 1
    corners = [
        (0, cfg.rows, z_top),
        (cfg.cols, cfg.rows, z_top),
        (cfg.cols, cfg.rows, z_bottom),
        (0, cfg.rows, z_bottom),
    ]
    return projected_polygon(corners, cfg)


def right_band_polygon(level_idx: int, cfg: RenderConfig) -> list[tuple[float, float]]:
    z_top = cfg.max_foraging_level - level_idx
    z_bottom = z_top - 1
    corners = [
        (cfg.cols, 0, z_top),
        (cfg.cols, cfg.rows, z_top),
        (cfg.cols, cfg.rows, z_bottom),
        (cfg.cols, 0, z_bottom),
    ]
    return projected_polygon(corners, cfg)


def top_outline(cfg: RenderConfig) -> list[tuple[float, float]]:
    z = cfg.max_foraging_level
    corners = [
        (0, 0, z),
        (cfg.cols, 0, z),
        (cfg.cols, cfg.rows, z),
        (0, cfg.rows, z),
    ]
    return projected_polygon(corners, cfg)


def front_gridline(x: int, cfg: RenderConfig) -> list[tuple[float, float]]:
    return [
        project(x, cfg.rows, cfg.max_foraging_level, cfg),
        project(x, cfg.rows, 0, cfg),
    ]


def right_gridline(y: int, cfg: RenderConfig) -> list[tuple[float, float]]:
    return [
        project(cfg.cols, y, cfg.max_foraging_level, cfg),
        project(cfg.cols, y, 0, cfg),
    ]


def visible_bounds(cfg: RenderConfig) -> tuple[float, float, float, float]:
    corners = [
        project(x, y, z, cfg)
        for x in (0, cfg.cols)
        for y in (0, cfg.rows)
        for z in (0, cfg.max_foraging_level)
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), max(xs), min(ys), max(ys)


def role_color(role: int) -> str:
    palette = {
        0: "#d62828",
        1: "#1d4ed8",
        2: "#f59e0b",
        3: "#7c3aed",
        4: "#0f766e",
        5: "#db2777",
    }
    return palette.get(role, palette[role % len(palette)])


def load_roles_array(roles_path: Path, cfg: RenderConfig) -> np.ndarray:
    roles = np.asarray(np.load(roles_path))

    if roles.ndim == 2:
        expected_shape = (cfg.rows, cfg.cols)
        if roles.shape != expected_shape:
            raise ValueError(
                f"Expected a roles array of shape {expected_shape}, got {roles.shape}"
            )
        return roles.astype(int, copy=False)

    if roles.ndim != 1:
        raise ValueError("roles array must be 1D or 2D")

    expected_size = cfg.rows * cfg.cols
    if roles.size != expected_size:
        raise ValueError(f"Expected {expected_size} role entries, got {roles.size}")

    role_grid = np.empty((cfg.rows, cfg.cols), dtype=int)
    for agent_index, role in enumerate(roles.astype(int, copy=False)):
        row = agent_index // cfg.cols
        col = agent_index % cfg.cols
        role_grid[row, col] = role
    return role_grid


def render_environment(
    cfg: RenderConfig, output_path: Path, roles: np.ndarray | None = None
) -> None:
    palette = build_palette("#d8f0c8", "#1d5c2f", cfg.max_foraging_level)
    top_base = "#ffffff"
    grid_edge = "#000000"
    side_edge = "#000000"
    front_colors = palette
    right_colors = [blend(color, "#0f1f10", 0.18) for color in palette]

    min_x, max_x, min_y, max_y = visible_bounds(cfg)
    margin = max(cfg.tile_width, cfg.tile_height, cfg.level_height) * 1.2
    width = max_x - min_x + 2 * margin
    height = max_y - min_y + 2 * margin

    fig, ax = plt.subplots(figsize=(width * 0.75, height * 0.75), dpi=cfg.dpi)
    fig.patch.set_facecolor(cfg.background)
    ax.set_facecolor(cfg.background)

    for level_idx in reversed(range(cfg.max_foraging_level)):
        ax.add_patch(
            Polygon(
                front_band_polygon(level_idx, cfg),
                closed=True,
                facecolor=front_colors[level_idx],
                edgecolor=side_edge,
                linewidth=1.0,
                joinstyle="round",
            )
        )
        ax.add_patch(
            Polygon(
                right_band_polygon(level_idx, cfg),
                closed=True,
                facecolor=right_colors[level_idx],
                edgecolor=side_edge,
                linewidth=1.0,
                joinstyle="round",
            )
        )

    for x in range(1, cfg.cols):
        line = front_gridline(x, cfg)
        ax.plot(
            [point[0] for point in line],
            [point[1] for point in line],
            color=side_edge,
            linewidth=0.8,
            solid_capstyle="round",
        )

    for y in range(1, cfg.rows):
        line = right_gridline(y, cfg)
        ax.plot(
            [point[0] for point in line],
            [point[1] for point in line],
            color=side_edge,
            linewidth=0.8,
            solid_capstyle="round",
        )

    tile_order = sorted(
        ((x, y) for y in range(cfg.rows) for x in range(cfg.cols)),
        key=lambda point: (point[0] + point[1], point[1], point[0]),
    )
    for x, y in tile_order:
        ax.add_patch(
            Polygon(
                tile_polygon(x, y, cfg),
                closed=True,
                facecolor=top_base,
                edgecolor=grid_edge,
                linewidth=0.8,
                joinstyle="round",
            )
        )

    ax.add_patch(
        Polygon(
            top_outline(cfg),
            closed=True,
            facecolor="none",
            edgecolor="#000000",
            linewidth=1.4,
            joinstyle="round",
        )
    )

    if cfg.show_agents:
        for x, y in tile_order:
            agent_color = "#fffdf8" if roles is None else role_color(int(roles[y, x]))
            inset = 0.14
            ax.add_patch(
                Polygon(
                    projected_polygon(
                        [
                            (x + inset, y + inset, cfg.max_foraging_level),
                            (x + 1 - inset, y + inset, cfg.max_foraging_level),
                            (x + 1 - inset, y + 1 - inset, cfg.max_foraging_level),
                            (x + inset, y + 1 - inset, cfg.max_foraging_level),
                        ],
                        cfg,
                    ),
                    closed=True,
                    facecolor=agent_color,
                    edgecolor="none",
                    linewidth=0.0,
                    joinstyle="round",
                )
            )

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(max_y + margin, min_y - margin)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an isometric foraging-environment illustration as a layered cuboid "
            "with a tiled top face and one agent marker per tile."
        ),
        epilog=(
            "Examples:\n"
            "  python render_foraging_environment.py --rows 6 --cols 6 --max-foraging-level 5 "
            "--output figures/foraging_environment.svg\n"
            "  python render_foraging_environment.py --rows 8 --cols 5 --max-foraging-level 7 "
            "--output figures/foraging_environment.png --no-agents"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rows", type=int, default=20, help="Number of tile rows on the top face."
    )
    parser.add_argument(
        "--cols", type=int, default=20, help="Number of tile columns on the top face."
    )
    parser.add_argument(
        "--max-foraging-level",
        type=int,
        default=10,
        help=(
            "Number of stacked foraging levels to render. "
            "If your model uses zero-indexed levels, pass max_level + 1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/foraging_environment.png"),
        help="Output file path. The extension can be .svg, .png, .pdf, or another Matplotlib-supported format.",
    )
    parser.add_argument(
        "--tile-width",
        type=float,
        default=1.6,
        help="Width of a single tile in projected units.",
    )
    parser.add_argument(
        "--tile-height",
        type=float,
        default=0.82,
        help="Height of a single tile in projected units.",
    )
    parser.add_argument(
        "--level-height",
        type=float,
        default=0.82,
        help="Vertical size of each foraging level in projected units.",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Raster resolution used for PNG export."
    )
    parser.add_argument(
        "--background", default="#ffffff", help="Figure background color."
    )
    parser.add_argument(
        "--roles-path",
        type=Path,
        help=(
            "Optional path to a .npy role snapshot. "
            "If 1D, agent i is mapped to (i // cols, i %% cols); if 2D, shape must match rows x cols."
        ),
    )
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Hide the per-tile agent markers on the top face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 1 or args.cols < 1:
        raise ValueError("rows and cols must both be at least 1")
    if args.max_foraging_level < 1:
        raise ValueError("max_foraging_level must be at least 1")

    cfg = RenderConfig(
        rows=args.rows,
        cols=args.cols,
        max_foraging_level=args.max_foraging_level,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        level_height=args.level_height,
        dpi=args.dpi,
        background=args.background,
        show_agents=not args.no_agents,
    )
    roles = None if args.roles_path is None else load_roles_array(args.roles_path, cfg)
    render_environment(cfg, args.output, roles=roles)
    print(f"Saved foraging environment figure to {args.output}")


if __name__ == "__main__":
    main()
