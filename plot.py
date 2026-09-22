import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgb


# ============================================================
# 1. FILE
# ============================================================

DATA = Path(
    "/Users/yangshuhan/Desktop/coral-bleaching-heat-stress/"
    "data/noaa-bleaching-stress-extent-365d.csv"
)


# ============================================================
# 2. VISUAL SETTINGS
# ============================================================

BG = "#04151d"

# Much stronger colour contrast
DEEP_CORAL = np.array(to_rgb("#d7193f"))
BRIGHT_CORAL = np.array(to_rgb("#ff493d"))
ORANGE_CORAL = np.array(to_rgb("#ff8465"))
SOFT_PINK = np.array(to_rgb("#efb4aa"))
BONE_WHITE = np.array(to_rgb("#f5f1e8"))

np.random.seed(8)


# ============================================================
# 3. READ NOAA DATA
# ============================================================

def read_data(path):

    years = []
    global_values = []
    pacific_values = []
    atlantic_values = []
    indian_values = []

    with open(path, newline="", encoding="utf-8-sig") as f:

        reader = csv.reader(f)

        header = next(reader)
        header = [h.strip() for h in header]

        year_i = header.index("Year")
        global_i = header.index("Global %")
        pacific_i = header.index("Pacific %")
        atlantic_i = header.index("Atlantic %")
        indian_i = header.index("Indian %")

        for row in reader:

            if not row:
                continue

            try:

                year = int(row[year_i].strip())

                global_value = float(
                    row[global_i].strip()
                )

                pacific = float(
                    row[pacific_i].strip()
                )

                atlantic = float(
                    row[atlantic_i].strip()
                )

                indian = float(
                    row[indian_i].strip()
                )

            except (ValueError, IndexError):
                continue

            years.append(year)
            global_values.append(global_value)
            pacific_values.append(pacific)
            atlantic_values.append(atlantic)
            indian_values.append(indian)

    return (
        np.array(years),
        np.array(global_values),
        np.array(pacific_values),
        np.array(atlantic_values),
        np.array(indian_values),
    )


# ============================================================
# 4. YEARLY AVERAGE
# ============================================================

def yearly_average(years, values):

    unique_years = np.unique(years)

    averages = []

    for year in unique_years:

        mask = years == year

        averages.append(
            np.nanmean(values[mask])
        )

    return unique_years, np.array(averages)


# ============================================================
# 5. NORMALISE
# ============================================================

def normalize(values):

    values = np.asarray(values, dtype=float)

    # Percentiles stop one extreme year
    # from flattening all other differences.
    low = np.percentile(values, 3)
    high = np.percentile(values, 97)

    if high <= low:
        return np.zeros_like(values)

    result = (
        (values - low)
        / (high - low)
    )

    return np.clip(result, 0, 1)


# ============================================================
# 6. SMOOTH MORPH
# ============================================================

def ease(t):

    """
    Smoothstep interpolation.

    Instead of:
        A -> jump -> B

    we get:
        A -> gradual morph -> B
    """

    return t * t * (3 - 2 * t)


def mix(a, b, t):

    return (
        a * (1 - t)
        + b * t
    )


# ============================================================
# 7. STRONG BLEACHING PALETTE
# ============================================================

def bleaching_colour(stress):

    """
    Strong colour contrast:

    0.00  deep coral red
    0.25  bright coral
    0.50  orange coral
    0.72  soft pink
    1.00  bone white
    """

    stops = [
        (0.00, DEEP_CORAL),
        (0.25, BRIGHT_CORAL),
        (0.50, ORANGE_CORAL),
        (0.72, SOFT_PINK),
        (1.00, BONE_WHITE),
    ]

    for i in range(len(stops) - 1):

        p1, c1 = stops[i]
        p2, c2 = stops[i + 1]

        if p1 <= stress <= p2:

            local_t = (
                (stress - p1)
                / (p2 - p1)
            )

            return mix(
                c1,
                c2,
                local_t
            )

    return BONE_WHITE


# ============================================================
# 8. CREATE COMPLEX ORGANIC BODY
# ============================================================

def reef_geometry(
    stress,
    pacific,
    atlantic,
    indian,
    layer,
    motion
):

    theta = np.linspace(
        0,
        2 * np.pi,
        900
    )

    # --------------------------------------------------------
    # MAIN BODY
    # --------------------------------------------------------

    # Large folds
    large_fold = (
        0.25
        * np.sin(
            theta
            * (
                2.5
                + pacific * 2.5
            )
            + 0.8
        )
    )

    # Medium folds
    medium_fold = (
        0.15
        * np.sin(
            theta
            * (
                4.5
                + atlantic * 3.5
            )
            - 1.2
        )
    )

    # Fine folds
    fine_fold = (
        0.08
        * np.cos(
            theta
            * (
                8
                + indian * 5
            )
            + 0.5
        )
    )

    # Small organic irregularities
    micro_fold = (
        0.035
        * np.sin(
            theta * 13
            + layer * 4
        )
    )

    # --------------------------------------------------------
    # DATA-DRIVEN COLLAPSE
    # --------------------------------------------------------

    # Higher heat stress makes selected regions
    # collapse more strongly than others.

    collapse_pattern = (
        1
        - stress
        * (
            0.12
            + 0.16
            * (
                0.5
                + 0.5
                * np.sin(
                    theta * 3 - 0.8
                )
            )
        )
    )

    r = (
        1
        + large_fold
        + medium_fold
        + fine_fold
        + micro_fold
    )

    r *= collapse_pattern

    # --------------------------------------------------------
    # INNER / OUTER LAYER
    # --------------------------------------------------------

    scale = (
        0.16
        + layer * 0.84
    )

    r *= scale

    # --------------------------------------------------------
    # VERY SUBTLE CONTINUOUS MOTION
    # --------------------------------------------------------

    # This prevents the object feeling frozen,
    # but the main morph still comes from NOAA data.

    r += (
        0.008
        * np.sin(
            theta * 4
            + motion
        )
        * layer
    )

    # --------------------------------------------------------
    # XY
    # --------------------------------------------------------

    x = (
        np.cos(theta)
        * r
        * 1.12
    )

    y = (
        np.sin(theta)
        * r
        * 0.82
    )

    # Fold / fabric distortion
    x += (
        0.055
        * np.sin(
            y * 4
            + layer * 2.5
        )
        * (1 - stress * 0.35)
    )

    y += (
        0.035
        * np.sin(
            x * 5
            - layer * 3
        )
    )

    return theta, x, y


# ============================================================
# 9. LOCAL BLEACHING
# ============================================================

def local_colour(
    theta,
    global_stress,
    region_shift
):

    """
    The whole coral does NOT become white at once.

    Different areas bleach at slightly different rates.
    """

    local_pattern = (
        0.22
        * np.sin(
            theta * 2.2
            + region_shift
        )
        + 0.12
        * np.sin(
            theta * 5.3
            - 0.5
        )
    )

    local_stress = (
        global_stress
        + local_pattern
        * global_stress
    )

    return np.clip(
        local_stress,
        0,
        1
    )


# ============================================================
# 10. DRAW COMPLEX REEF
# ============================================================

def draw_reef(
    ax,
    stress,
    pacific,
    atlantic,
    indian,
    motion
):

    # High stress removes some visual density.
    layer_count = int(
        95 - stress * 28
    )

    # --------------------------------------------------------
    # CONTOUR LAYERS
    # --------------------------------------------------------

    for layer_index in range(
        layer_count
    ):

        layer = (
            layer_index
            / max(
                1,
                layer_count - 1
            )
        )

        theta, x, y = reef_geometry(
            stress,
            pacific,
            atlantic,
            indian,
            layer,
            motion
        )

        # Break each contour into segments
        # so different areas can bleach independently.

        local_stress = local_colour(
            theta,
            stress,
            layer * 3
        )

        # Draw in chunks
        chunk = 12

        for start in range(
            0,
            len(theta) - chunk,
            chunk
        ):

            end = start + chunk + 1

            chunk_stress = np.mean(
                local_stress[start:end]
            )

            colour = bleaching_colour(
                chunk_stress
            )

            # Severe stress also makes some areas
            # visually disappear.

            disappearance = (
                stress
                * chunk_stress
            )

            alpha = (
                0.025
                + layer * 0.22
            )

            alpha *= (
                1
                - disappearance * 0.45
            )

            ax.plot(
                x[start:end],
                y[start:end],
                color=colour,
                alpha=alpha,
                linewidth=(
                    0.28
                    + layer * 0.50
                ),
                solid_capstyle="round",
                solid_joinstyle="round",
            )


    # --------------------------------------------------------
    # FLOW LINES
    # --------------------------------------------------------

    # Additional lines give the form more depth
    # and stop it looking like simple concentric rings.

    flow_count = int(
        30 - stress * 8
    )

    for k in range(flow_count):

        t = (
            k
            / max(
                1,
                flow_count - 1
            )
        )

        theta = np.linspace(
            -2.3,
            2.3,
            450
        )

        radius = (
            0.22
            + t * 0.68
        )

        x = (
            radius
            * np.cos(theta)
        )

        y = (
            radius
            * np.sin(theta)
            * 0.72
        )

        x += (
            0.17
            * np.sin(
                theta * 2.1
                + t * 5
                + pacific
            )
        )

        y += (
            0.10
            * np.sin(
                theta * 3.2
                - t * 4
                + atlantic
            )
        )

        # rotate each family of lines
        angle = (
            0.5
            + t * 1.7
            + indian * 0.3
        )

        xr = (
            x * np.cos(angle)
            - y * np.sin(angle)
        )

        yr = (
            x * np.sin(angle)
            + y * np.cos(angle)
        )

        colour = bleaching_colour(
            np.clip(
                stress
                + 0.12
                * np.sin(t * np.pi),
                0,
                1
            )
        )

        ax.plot(
            xr,
            yr,
            color=colour,
            alpha=(
                0.08
                * (
                    1 - stress * 0.4
                )
            ),
            linewidth=0.5
        )


    # --------------------------------------------------------
    # OUTER SILHOUETTE
    # --------------------------------------------------------

    theta, x, y = reef_geometry(
        stress,
        pacific,
        atlantic,
        indian,
        1.0,
        motion
    )

    # Again use segments for local bleaching
    local_stress = local_colour(
        theta,
        stress,
        0
    )

    chunk = 10

    for start in range(
        0,
        len(theta) - chunk,
        chunk
    ):

        end = start + chunk + 1

        s = np.mean(
            local_stress[start:end]
        )

        ax.plot(
            x[start:end],
            y[start:end],
            color=bleaching_colour(s),
            alpha=0.9,
            linewidth=1.6,
            solid_capstyle="round"
        )


# ============================================================
# 11. LOAD NOAA DATA
# ============================================================

(
    raw_years,
    raw_global,
    raw_pacific,
    raw_atlantic,
    raw_indian,
) = read_data(DATA)


if len(raw_years) == 0:

    raise RuntimeError(
        "No NOAA data could be read."
    )


years, global_year = yearly_average(
    raw_years,
    raw_global
)

_, pacific_year = yearly_average(
    raw_years,
    raw_pacific
)

_, atlantic_year = yearly_average(
    raw_years,
    raw_atlantic
)

_, indian_year = yearly_average(
    raw_years,
    raw_indian
)


global_norm = normalize(
    global_year
)

pacific_norm = normalize(
    pacific_year
)

atlantic_norm = normalize(
    atlantic_year
)

indian_norm = normalize(
    indian_year
)


# ============================================================
# 12. ANIMATION SETTINGS
# ============================================================

# Number of transition frames BETWEEN years.
#
# 24 means:
# 1986 -> 1987 is not one jump.
# It contains 24 intermediate states.

TRANSITION_FRAMES = 24

total_frames = (
    (len(years) - 1)
    * TRANSITION_FRAMES
)


# ============================================================
# 13. FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 9),
    facecolor=BG
)

fig.canvas.manager.set_window_title(
    "A Reef Losing Its Color"
)


# ============================================================
# 14. ANIMATION UPDATE
# ============================================================

def update(frame):

    ax.clear()
    ax.set_facecolor(BG)

    # --------------------------------------------------------
    # WHICH TWO YEARS ARE WE BETWEEN?
    # --------------------------------------------------------

    year_index = (
        frame
        // TRANSITION_FRAMES
    )

    transition_frame = (
        frame
        % TRANSITION_FRAMES
    )

    raw_t = (
        transition_frame
        / TRANSITION_FRAMES
    )

    t = ease(raw_t)

    next_index = min(
        year_index + 1,
        len(years) - 1
    )


    # --------------------------------------------------------
    # INTERPOLATE NOAA DATA
    # --------------------------------------------------------

    stress = mix(
        global_norm[year_index],
        global_norm[next_index],
        t
    )

    pacific = mix(
        pacific_norm[year_index],
        pacific_norm[next_index],
        t
    )

    atlantic = mix(
        atlantic_norm[year_index],
        atlantic_norm[next_index],
        t
    )

    indian = mix(
        indian_norm[year_index],
        indian_norm[next_index],
        t
    )

    global_display = mix(
        global_year[year_index],
        global_year[next_index],
        t
    )

    pacific_display = mix(
        pacific_year[year_index],
        pacific_year[next_index],
        t
    )

    atlantic_display = mix(
        atlantic_year[year_index],
        atlantic_year[next_index],
        t
    )

    indian_display = mix(
        indian_year[year_index],
        indian_year[next_index],
        t
    )


    # --------------------------------------------------------
    # CURRENT YEAR DISPLAY
    # --------------------------------------------------------

    display_year = mix(
        years[year_index],
        years[next_index],
        t
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    ax.text(
        0.05,
        0.94,
        "A REEF LOSING ITS COLOR",
        transform=ax.transAxes,
        fontsize=25,
        fontweight="bold",
        color="#f5f1e8",
        va="top"
    )

    ax.text(
        0.05,
        0.885,
        "NOAA Coral Reef Watch · Global Bleaching Heat Stress",
        transform=ax.transAxes,
        fontsize=10,
        color="#8ba0a7"
    )


    # --------------------------------------------------------
    # DRAW REEF
    # --------------------------------------------------------

    draw_reef(
        ax,
        stress,
        pacific,
        atlantic,
        indian,
        frame * 0.018
    )


    # --------------------------------------------------------
    # YEAR
    # --------------------------------------------------------

    ax.text(
        0.92,
        0.92,
        f"{display_year:.1f}",
        transform=ax.transAxes,
        fontsize=27,
        fontweight="bold",
        color="#f5f1e8",
        ha="right"
    )


    # --------------------------------------------------------
    # CURRENT DATA
    # --------------------------------------------------------

    ax.text(
        0.92,
        0.84,
        f"GLOBAL     {global_display:.3f} %",
        transform=ax.transAxes,
        fontsize=9,
        color=bleaching_colour(stress),
        ha="right"
    )

    ax.text(
        0.92,
        0.805,
        f"PACIFIC    {pacific_display:.3f} %",
        transform=ax.transAxes,
        fontsize=8,
        color="#91a5ab",
        ha="right"
    )

    ax.text(
        0.92,
        0.775,
        f"ATLANTIC   {atlantic_display:.3f} %",
        transform=ax.transAxes,
        fontsize=8,
        color="#91a5ab",
        ha="right"
    )

    ax.text(
        0.92,
        0.745,
        f"INDIAN     {indian_display:.3f} %",
        transform=ax.transAxes,
        fontsize=8,
        color="#91a5ab",
        ha="right"
    )


    # --------------------------------------------------------
    # VISUAL KEY
    # --------------------------------------------------------

    ax.text(
        0.05,
        0.16,
        "GLOBAL %",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color="#f5f1e8"
    )

    ax.text(
        0.05,
        0.13,
        "bleaching · density · contraction",
        transform=ax.transAxes,
        fontsize=8,
        color="#82969c"
    )

    ax.text(
        0.32,
        0.16,
        "PACIFIC · ATLANTIC · INDIAN",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color="#f5f1e8"
    )

    ax.text(
        0.32,
        0.13,
        "large · medium · fine folds",
        transform=ax.transAxes,
        fontsize=8,
        color="#82969c"
    )


    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    x1 = 0.05
    x2 = 0.95
    line_y = 0.07

    ax.plot(
        [x1, x2],
        [line_y, line_y],
        transform=ax.transAxes,
        color="#70868d",
        linewidth=0.8,
        alpha=0.5
    )

    progress = (
        frame
        / max(
            1,
            total_frames - 1
        )
    )

    current_x = (
        x1
        + progress
        * (x2 - x1)
    )

    ax.scatter(
        current_x,
        line_y,
        transform=ax.transAxes,
        s=40,
        color=bleaching_colour(stress),
        edgecolors="none",
        zorder=20
    )

    ax.text(
        x1,
        0.04,
        str(years[0]),
        transform=ax.transAxes,
        fontsize=8,
        color="#82969c"
    )

    ax.text(
        x2,
        0.04,
        str(years[-1]),
        transform=ax.transAxes,
        fontsize=8,
        color="#82969c",
        ha="right"
    )


    # --------------------------------------------------------
    # CANVAS
    # --------------------------------------------------------

    ax.set_xlim(
        -1.55,
        1.55
    )

    ax.set_ylim(
        -1.12,
        1.12
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis(
        "off"
    )


# ============================================================
# 15. PLAY
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=40,
    repeat=True,
    cache_frame_data=False
)

plt.tight_layout()

plt.show()