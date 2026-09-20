# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///

"""
Read the file in data/, make one picture, save it to out/.

    uv run plot.py

Three parts, and you will replace all three: rows() reads the file the way *your*
file needs reading, the loop in main() picks the numbers out of it, and the plot at
the bottom is the transformation you chose. Print before you plot.
"""

import csv
import random
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

FILE = "noaa-bleaching-stress-extent-365d.csv"  # CHANGE ME: the same name as in fetch.py
PICTURE = "plot.png"                           # what goes into out/, and into the README

HERE = Path(__file__).parent
DATA = HERE / "data" / FILE
OUT = HERE / "out"


def rows(path):
    """Read NOAA Coral Reef Watch bleaching heat stress extent data."""
    kept = []

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            kept.append(row)

    return kept

def draw_coral(ax, x, y, length, angle, depth, color):
    if depth == 0:
        return

    x2 = x + math.cos(angle) * length
    y2 = y + math.sin(angle) * length

    ax.plot([x, x2], [y, y2], color=color, linewidth=0.7)

    draw_coral(ax, x2, y2, length * 0.72, angle + 0.45, depth - 1, color)
    draw_coral(ax, x2, y2, length * 0.72, angle - 0.45, depth - 1, color)
def main():
    table = rows(DATA)
    print(f"{DATA.name}: {len(table)} rows. The first one: {table[0]}")

    dates, values = [], []

    for row in table:
        date = row[" Date"].strip()
        value = row[" Global %"].strip()

        if not value:
            continue

        dates.append(date)
        values.append(float(value))

    print(f"{len(values)} values")
    print(f"from {dates[0]} to {dates[-1]}")
    print(f"Global % range: {min(values)} to {max(values)}")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor("#071821")
    fig.patch.set_facecolor("#071821")
    ax.set_xlim(0, len(values))
    ax.set_ylim(0, 3500)
    random.seed(7)
    sample_step = 180
    for i in range(0, len(values), sample_step):
        stress = values[i]
        height = 500 + stress * 1800
        depth = 3 if stress > 0.25 else 4
        color = plt.cm.RdPu_r(min(stress / 0.8, 1))
        draw_coral(ax, i, 0.05, height, math.pi / 2, depth, color)
    ax.axis("off")
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / PICTURE, dpi=150)
    print(f"saved out/{PICTURE}")
    plt.show()


if __name__ == "__main__":
    main()
