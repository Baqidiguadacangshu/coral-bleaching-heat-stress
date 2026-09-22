from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


# ============================================================
# A REEF LOSING ITS COLOR
# NOAA Coral Reef Watch
# ============================================================


# ============================================================
# DATA FILE
# ============================================================

DATA = Path(
    "/Users/yangshuhan/Desktop/"
    "coral-bleaching-heat-stress/data/"
    "noaa-bleaching-stress-extent-365d.csv"
)


# ============================================================
# TIME
# ============================================================

START_YEAR = 1986
END_YEAR = 2026

FPS = 24
DURATION = 8

FRAMES = FPS * DURATION


# ============================================================
# VISUAL DENSITY
# ============================================================

N_STRANDS = 400
POINTS = 170
N_PARTICLES = 1400

np.random.seed(18)


# ============================================================
# COLORS
# ============================================================

BG = "#000000"
TEXT = "#F1EEE8"


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smoothstep(x):
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return a * (1.0 - t) + b * t


# ============================================================
# READ NOAA DATA
# ============================================================

def read_noaa_data(path):

    print("Reading NOAA data...")

    raw_years = []
    raw_values = []

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.reader(f)
        header = next(reader)

        year_index = None
        global_index = None

        for i, name in enumerate(header):

            clean = name.strip().lower()

            if clean == "year":
                year_index = i

            if clean == "global %":
                global_index = i

        if year_index is None:
            raise ValueError(
                "Could not find Year column."
            )

        if global_index is None:
            raise ValueError(
                "Could not find Global % column."
            )

        for row in reader:

            try:

                year = int(
                    float(row[year_index])
                )

                value = float(
                    row[global_index]
                )

            except (
                ValueError,
                TypeError,
                IndexError
            ):
                continue

            if START_YEAR <= year <= END_YEAR:

                raw_years.append(year)
                raw_values.append(value)

    raw_years = np.array(raw_years)
    raw_values = np.array(raw_values)

    if len(raw_years) == 0:

        raise ValueError(
            "No valid NOAA data found."
        )

    years = []
    values = []

    # Daily data -> yearly average

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        mask = raw_years == year

        if np.any(mask):

            years.append(year)

            values.append(
                np.nanmean(
                    raw_values[mask]
                )
            )

    years = np.array(
        years,
        dtype=float
    )

    values = np.array(
        values,
        dtype=float
    )

    print(
        "NOAA DATA OK:",
        int(years[0]),
        "→",
        int(years[-1])
    )

    print(
        "Years loaded:",
        len(years)
    )

    return years, values


# ============================================================
# NORMALIZE
# ============================================================

def normalize(values):

    lo = np.nanpercentile(
        values,
        5
    )

    hi = np.nanpercentile(
        values,
        95
    )

    if hi <= lo:
        return np.zeros_like(values)

    result = (
        values - lo
    ) / (
        hi - lo
    )

    return np.clip(
        result,
        0,
        1
    )


# ============================================================
# COLOR MAP
#
# IMPORTANT CHANGE:
#
# Low-stress colors now have much larger visual differences.
# Small NOAA changes therefore remain visible.
# ============================================================

COLOR_STOPS = [

    # stress, RGB

    (
        0.00,
        np.array([
            0.68,
            0.36,
            0.34
        ])
    ),

    # warmer coral

    (
        0.10,
        np.array([
            0.77,
            0.43,
            0.37
        ])
    ),

    # orange coral

    (
        0.20,
        np.array([
            0.85,
            0.51,
            0.42
        ])
    ),

    # light coral

    (
        0.30,
        np.array([
            0.91,
            0.61,
            0.51
        ])
    ),

    # pale pink

    (
        0.50,
        np.array([
            0.93,
            0.70,
            0.65
        ])
    ),

    # strongly bleached

    (
        0.70,
        np.array([
            0.95,
            0.81,
            0.76
        ])
    ),

    # warm ivory

    (
        1.00,
        np.array([
            0.97,
            0.93,
            0.87
        ])
    )
]


# ============================================================
# GET COLOR
# ============================================================

def get_color(stress):

    stress = clamp(stress)

    for i in range(
        len(COLOR_STOPS) - 1
    ):

        s1, c1 = COLOR_STOPS[i]
        s2, c2 = COLOR_STOPS[i + 1]

        if s1 <= stress <= s2:

            t = (
                stress - s1
            ) / (
                s2 - s1
            )

            t = smoothstep(t)

            return lerp(
                c1,
                c2,
                t
            )

    return COLOR_STOPS[-1][1]


# ============================================================
# PRECOMPUTED GEOMETRY
# ============================================================

theta = np.linspace(
    0,
    2 * np.pi,
    POINTS
)

layers = np.linspace(
    0,
    1,
    N_STRANDS
)

THETA = theta[None, :]
LAYER = layers[:, None]

FIBRE_PHASE = (
    np.arange(
        N_STRANDS
    )[:, None]
    * 0.081
)


# ============================================================
# IMPORTANT:
# Each strand receives a permanent individual identity.
#
# This lets different fibres bleach at slightly different
# moments rather than the whole coral changing at once.
# ============================================================

rng = np.random.default_rng(27)

strand_variation = rng.normal(
    0,
    0.055,
    N_STRANDS
)

strand_bleach_threshold = rng.uniform(
    0.25,
    0.90,
    N_STRANDS
)

strand_brightness = rng.uniform(
    0.88,
    1.12,
    N_STRANDS
)


# ============================================================
# CREATE CORAL FIBRES
# ============================================================

def create_all_strands(
    time,
    stress
):

    # --------------------------------------------------------
    # MAIN BODY
    # --------------------------------------------------------

    body = (

        1.0

        + 0.17
        * np.sin(
            5.0 * THETA
            +
            time * 0.75
        )

        + 0.095
        * np.sin(
            8.0 * THETA
            -
            time * 0.52
        )

        + 0.050
        * np.sin(
            13.0 * THETA
            +
            time * 0.65
        )

        + 0.025
        * np.sin(
            21.0 * THETA
            -
            time * 0.46
        )
    )


    # --------------------------------------------------------
    # LAYERS
    # --------------------------------------------------------

    layer_radius = (
        0.34
        +
        0.64 * LAYER
    )


    # --------------------------------------------------------
    # DATA-DRIVEN CONTRACTION
    # --------------------------------------------------------

    contraction = (
        1.0
        -
        0.22 * stress
    )

    radius = (
        layer_radius
        *
        body
        *
        contraction
    )


    # --------------------------------------------------------
    # BREATHING
    # --------------------------------------------------------

    breathing = (

        1.0

        + 0.030

        * np.sin(
            time * 2.4
            +
            THETA * 2.5
            +
            FIBRE_PHASE * 0.08
        )
    )

    radius *= breathing


    # --------------------------------------------------------
    # FINE FIBRE MOTION
    # --------------------------------------------------------

    fine_wave = (

        0.012

        * np.sin(
            THETA * 10.0
            +
            time * 1.8
            +
            FIBRE_PHASE
        )
    )

    radius += fine_wave


    # --------------------------------------------------------
    # HEAT STRESS DISTORTION
    # --------------------------------------------------------

    radius += (

        stress

        * 0.052

        * np.sin(
            THETA * 11
            +
            time * 2.0
            +
            FIBRE_PHASE * 0.15
        )
    )


    radius += (

        stress

        * 0.025

        * np.sin(
            THETA * 19
            -
            time * 1.5
            +
            FIBRE_PHASE * 0.10
        )
    )


    # --------------------------------------------------------
    # TWIST
    # --------------------------------------------------------

    twist = (

        (LAYER - 0.5)

        * (
            0.48

            + 0.10

            * np.sin(
                time * 0.9
            )
        )
    )


    angle = (

        THETA

        + twist

        * np.sin(
            THETA * 3
            +
            time * 0.85
        )
    )


    # --------------------------------------------------------
    # XY
    # --------------------------------------------------------

    x = (
        radius
        *
        np.cos(angle)
    )

    y = (
        radius
        *
        np.sin(angle)
    )


    # --------------------------------------------------------
    # ASYMMETRY
    # --------------------------------------------------------

    x += (

        0.065

        * np.sin(
            THETA * 2
            +
            time * 0.65
        )
    )


    y += (

        0.045

        * np.sin(
            THETA * 3
            -
            time * 0.55
        )
    )


    x *= 1.28
    y *= 0.88


    return x, y


# ============================================================
# PARTICLES
# ============================================================

particle_theta = np.random.uniform(
    0,
    2 * np.pi,
    N_PARTICLES
)

particle_radius = np.random.uniform(
    0.83,
    1.72,
    N_PARTICLES
)

particle_phase = np.random.uniform(
    0,
    2 * np.pi,
    N_PARTICLES
)

particle_speed = np.random.uniform(
    0.30,
    1.20,
    N_PARTICLES
)

particle_size = np.random.uniform(
    2.0,
    8.0,
    N_PARTICLES
)

particle_brightness = np.random.uniform(
    0.40,
    1.0,
    N_PARTICLES
)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # DATA
    # ========================================================

    years, raw_values = read_noaa_data(
        DATA
    )

    stress_values = normalize(
        raw_values
    )


    # --------------------------------------------------------
    # Print yearly values so you can see the data relationship
    # --------------------------------------------------------

    print("\nYEAR / STRESS")

    for y, raw, s in zip(
        years,
        raw_values,
        stress_values
    ):

        print(
            int(y),
            "raw:",
            round(raw, 4),
            "normalized:",
            round(s, 3)
        )


    # ========================================================
    # FRAME DATA
    # ========================================================

    frame_years = np.linspace(
        START_YEAR,
        END_YEAR,
        FRAMES
    )

    frame_stress = np.interp(
        frame_years,
        years,
        stress_values
    )

    frame_raw = np.interp(
        frame_years,
        years,
        raw_values
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(16, 9),
        facecolor=BG
    )

    ax = fig.add_axes([
        0,
        0,
        1,
        1
    ])

    ax.set_facecolor(BG)

    ax.set_xlim(
        -1.72,
        1.72
    )

    ax.set_ylim(
        -1.08,
        1.08
    )

    ax.axis("off")


    # ========================================================
    # TITLE
    # ========================================================

    ax.text(
        0.045,
        0.925,
        "A REEF LOSING ITS COLOR",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top"
    )


    ax.text(
        0.045,
        0.883,
        "NOAA Coral Reef Watch · Global Bleaching Heat Stress",
        transform=ax.transAxes,
        color="#777777",
        fontsize=9,
        ha="left",
        va="top"
    )


    ax.text(
        0.045,
        0.858,
        "1986 — 2026",
        transform=ax.transAxes,
        color="#555555",
        fontsize=7,
        ha="left",
        va="top"
    )


    # ========================================================
    # YEAR / DATA
    # ========================================================

    year_text = ax.text(
        0.955,
        0.925,
        "",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=30,
        fontweight="bold",
        ha="right",
        va="top"
    )


    stress_text = ax.text(
        0.955,
        0.878,
        "",
        transform=ax.transAxes,
        color="#777777",
        fontsize=8,
        ha="right",
        va="top"
    )


    # ========================================================
    # PARTICLES
    # ========================================================

    particles = ax.scatter(
        np.zeros(
            N_PARTICLES
        ),
        np.zeros(
            N_PARTICLES
        ),
        s=particle_size,
        linewidths=0,
        zorder=2
    )


    # ========================================================
    # FIBRES
    # ========================================================

    empty_segments = [

        np.zeros(
            (
                POINTS,
                2
            )
        )

        for _ in range(
            N_STRANDS
        )
    ]


    fibre_collection = LineCollection(
        empty_segments,
        linewidths=0.45,
        zorder=5
    )

    ax.add_collection(
        fibre_collection
    )


    # ========================================================
    # GLOW / OUTER
    # ========================================================

    glow, = ax.plot(
        [],
        [],
        linewidth=5.0,
        alpha=0.035,
        zorder=3
    )


    outer, = ax.plot(
        [],
        [],
        linewidth=1.1,
        alpha=0.78,
        zorder=7
    )


    # ========================================================
    # INFORMATION
    # ========================================================

    ax.text(
        0.045,
        0.105,
        "COLOR",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=7,
        fontweight="bold"
    )

    ax.text(
        0.045,
        0.085,
        "NOAA heat stress → fibre color",
        transform=ax.transAxes,
        color="#666666",
        fontsize=6
    )


    ax.text(
        0.215,
        0.105,
        "FORM",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=7,
        fontweight="bold"
    )

    ax.text(
        0.215,
        0.085,
        "400 flowing fibres",
        transform=ax.transAxes,
        color="#666666",
        fontsize=6
    )


    ax.text(
        0.365,
        0.105,
        "PARTICLES",
        transform=ax.transAxes,
        color=TEXT,
        fontsize=7,
        fontweight="bold"
    )

    ax.text(
        0.365,
        0.085,
        "Stress-driven dispersion",
        transform=ax.transAxes,
        color="#666666",
        fontsize=6
    )


    # ========================================================
    # COLOR LEGEND
    # ========================================================

    legend_x1 = 0.69
    legend_x2 = 0.955
    legend_y = 0.090

    legend_steps = 100

    for j in range(
        legend_steps
    ):

        s = (
            j
            /
            (
                legend_steps - 1
            )
        )

        c = get_color(s)

        x1 = (
            legend_x1
            +
            (
                legend_x2
                -
                legend_x1
            )
            *
            j
            /
            legend_steps
        )

        x2 = (
            legend_x1
            +
            (
                legend_x2
                -
                legend_x1
            )
            *
            (
                j + 1
            )
            /
            legend_steps
        )

        ax.plot(
            [x1, x2],
            [legend_y, legend_y],
            transform=ax.transAxes,
            color=c,
            linewidth=3
        )


    ax.text(
        legend_x1,
        0.108,
        "LOW STRESS",
        transform=ax.transAxes,
        color="#777777",
        fontsize=5,
        ha="left"
    )


    ax.text(
        legend_x2,
        0.108,
        "HIGH STRESS / BLEACHED",
        transform=ax.transAxes,
        color="#999999",
        fontsize=5,
        ha="right"
    )


    # ========================================================
    # TIMELINE
    # ========================================================

    timeline_y = 0.055


    ax.plot(
        [0.045, 0.955],
        [timeline_y, timeline_y],
        transform=ax.transAxes,
        color="#292929",
        linewidth=0.8
    )


    ax.text(
        0.045,
        0.027,
        "1986",
        transform=ax.transAxes,
        color="#555555",
        fontsize=6,
        ha="left"
    )


    ax.text(
        0.955,
        0.027,
        "2026",
        transform=ax.transAxes,
        color="#555555",
        fontsize=6,
        ha="right"
    )


    timeline_dot, = ax.plot(
        [],
        [],
        marker="o",
        markersize=4.5,
        linestyle="None",
        transform=ax.transAxes,
        zorder=20
    )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(frame):

        animation_time = (
            frame / FPS
        )


        current_year = (
            frame_years[frame]
        )


        current_stress = (
            frame_stress[frame]
        )


        current_raw = (
            frame_raw[frame]
        )


        # ====================================================
        # IMPORTANT COLOR RESPONSE
        #
        # Gamma < 1 expands differences in LOW stress.
        #
        # Data is NOT changed.
        # Only its visual mapping is nonlinear.
        # ====================================================

        visual_stress = (
            current_stress
            ** 0.62
        )


        visual_stress = clamp(
            visual_stress
        )


        base_color = get_color(
            visual_stress
        )


        # ====================================================
        # GEOMETRY
        # ====================================================

        x, y = create_all_strands(
            animation_time,
            current_stress
        )


        segments = np.stack(
            [
                x,
                y
            ],
            axis=2
        )


        fibre_collection.set_segments(
            segments
        )


        # ====================================================
        # INDIVIDUAL FIBRE COLORS
        # ====================================================

        fibre_colors = np.zeros(
            (
                N_STRANDS,
                4
            )
        )


        depth = (
            np.sin(
                layers * np.pi
            )
            ** 1.15
        )


        # ----------------------------------------------------
        # Each fibre gets a slightly different stress value.
        #
        # This creates multiple shades inside the coral.
        # ----------------------------------------------------

        individual_stress = (

            visual_stress

            + strand_variation
        )


        individual_stress = np.clip(
            individual_stress,
            0,
            1
        )


        # ----------------------------------------------------
        # Progressive bleaching
        #
        # As stress rises, more individual fibres are pushed
        # toward pale colors.
        # ----------------------------------------------------

        bleaching_amount = np.clip(

            (
                visual_stress
                -
                strand_bleach_threshold
            )
            * 2.2,

            0,
            1
        )


        individual_stress = np.clip(

            individual_stress

            + bleaching_amount
            * 0.25,

            0,
            1
        )


        # ----------------------------------------------------
        # Calculate every strand's own color
        # ----------------------------------------------------

        for i in range(
            N_STRANDS
        ):

            c = get_color(
                individual_stress[i]
            )


            # small brightness variation

            c = np.clip(

                c
                *
                strand_brightness[i],

                0,
                1
            )


            # preserve depth

            depth_brightness = (
                0.78
                +
                0.22
                * depth[i]
            )


            c *= (
                depth_brightness
            )


            fibre_colors[
                i,
                :3
            ] = np.clip(
                c,
                0,
                1
            )


        # ====================================================
        # ALPHA
        # ====================================================

        alpha = (
            0.11
            +
            0.28
            * depth
        )


        highlight = (
            np.arange(
                N_STRANDS
            )
            % 8
            == 0
        )


        alpha[
            highlight
        ] += 0.11


        fibre_colors[:, 3] = np.clip(
            alpha,
            0,
            1
        )


        fibre_collection.set_color(
            fibre_colors
        )


        # ====================================================
        # LINE WIDTH
        # ====================================================

        widths = (
            0.28
            +
            0.34
            * depth
        )


        widths[
            highlight
        ] += 0.18


        fibre_collection.set_linewidths(
            widths
        )


        # ====================================================
        # OUTER EDGE
        # ====================================================

        ox = x[-1]
        oy = y[-1]


        outer.set_data(
            ox,
            oy
        )


        outer.set_color(
            np.clip(
                base_color
                * 1.12,
                0,
                1
            )
        )


        glow.set_data(
            ox,
            oy
        )


        glow.set_color(
            base_color
        )


        # ====================================================
        # PARTICLES
        # ====================================================

        p_angle = (

            particle_theta

            + animation_time
            * particle_speed
            * 0.15
        )


        stress_drift = (

            current_stress

            * (
                0.15

                + 0.40

                * (
                    0.5

                    + 0.5

                    * np.sin(
                        particle_phase

                        + animation_time

                        * particle_speed

                        * 1.6
                    )
                )
            )
        )


        pulse = (

            0.05

            * np.sin(
                particle_phase

                + animation_time

                * particle_speed

                * 1.8
            )
        )


        pr = (

            particle_radius

            + stress_drift

            + pulse
        )


        px = (

            pr

            * np.cos(
                p_angle
            )

            * 1.25
        )


        py = (

            pr

            * np.sin(
                p_angle
            )

            * 0.80
        )


        px += (

            0.035

            * np.sin(
                animation_time

                * particle_speed

                * 1.4

                + particle_phase
            )
        )


        py += (

            0.030

            * np.cos(
                animation_time

                * particle_speed

                + particle_phase
            )
        )


        particles.set_offsets(
            np.column_stack(
                [
                    px,
                    py
                ]
            )
        )


        # ====================================================
        # PARTICLE COLOR
        # ====================================================

        particle_rgba = np.zeros(
            (
                N_PARTICLES,
                4
            )
        )


        particle_rgba[:, :3] = np.clip(

            base_color[None, :]

            * (
                0.80

                + 0.35

                * particle_brightness[
                    :,
                    None
                ]
            ),

            0,
            1
        )


        particle_rgba[:, 3] = (

            (
                0.20

                + 0.34

                * current_stress
            )

            * particle_brightness
        )


        particles.set_facecolors(
            particle_rgba
        )


        particles.set_sizes(

            particle_size

            * (
                0.90

                + 1.25

                * current_stress
            )
        )


        # ====================================================
        # TEXT
        # ====================================================

        year_text.set_text(
            f"{int(round(current_year))}"
        )


        stress_text.set_text(

            "GLOBAL HEAT STRESS   "

            f"{current_raw:.3f} %"
        )


        stress_text.set_color(
            np.clip(
                base_color * 1.08,
                0,
                1
            )
        )


        # ====================================================
        # TIMELINE
        # ====================================================

        progress = (

            current_year
            -
            START_YEAR

        ) / (

            END_YEAR
            -
            START_YEAR
        )


        dot_x = (

            0.045

            + progress

            * 0.91
        )


        timeline_dot.set_data(
            [dot_x],
            [timeline_y]
        )


        timeline_dot.set_color(
            base_color
        )


        return (
            fibre_collection,
            outer,
            glow,
            particles,
            year_text,
            stress_text,
            timeline_dot
        )


    # ========================================================
    # ANIMATION
    # ========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=FRAMES,
        interval=1000 / FPS,
        blit=False,
        repeat=True,
        cache_frame_data=False
    )
    gif_path = "out/coral-bleaching-animation.gif"

    print("Rendering NEW GIF...")

    animation.save(
        gif_path,
        writer="pillow",
        fps=FPS,
        dpi=80
    )

    print(f"Saved NEW GIF to: {gif_path}")

    plt.tight_layout()
    plt.show()



# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()