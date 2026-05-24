"""
Final Season Episode Ratings — Grid Viz
========================================
Reads your IMDb CSV and plots episode-by-episode ratings
for each show's final season in a grid of subplots.

Requirements:
    pip install pandas matplotlib

Usage:
    python final_season_viz.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_CSV   = "episodes_with_ratings_2010_2018.csv"   # your scraped IMDb CSV
OUTPUT_IMG  = "final_season_ratings2010_2014.png"
COLS        = 4                    # panels per row
THRESHOLD   = 7.0                  # dashed reference line
DPI         = 180

# ── Colors ────────────────────────────────────────────────────────────────────

BLUE        = "#3266ad"
RED         = "#c0392b"
BLUE_FILL   = "#3266ad18"
RED_FILL    = "#c0392b18"
GRID_COLOR  = "#e8e8e8"
TEXT_DARK   = "#1a1a1a"
TEXT_MUTED  = "#888888"
THRESHOLD_C = "#bbbbbb"
BG          = "#ffffff"
PANEL_BG    = "#fafafa"

# ── Load & prep data ──────────────────────────────────────────────────────────

df = pd.read_csv("episodes_with_ratings_2010_2018.csv")

# Ensure correct types
df["Season"]  = pd.to_numeric(df["Season"],  errors="coerce")
df["Episode"] = pd.to_numeric(df["Episode"], errors="coerce")
df["Rating"]  = pd.to_numeric(df["Rating"],  errors="coerce")
df = df.dropna(subset=["Season", "Episode", "Rating"])

ERA = "2010-2014"   # change to "2014-2018" for the second batch

if ERA == "2010-2014":
    year_min, year_max = 2010, 2014
else:
    year_min, year_max = 2015, 2018

# Filter to shows whose final season aired in that range
final_years = df.groupby("Show")["Year"].max()
shows_in_era = final_years[(final_years >= year_min) & (final_years <= year_max)].index
df = df[df["Show"].isin(shows_in_era)]

# Find final season per show
final_season_idx = df.groupby("Show")["Season"].transform("max")
final_df = df[df["Season"] == final_season_idx].copy()

# Aggregate: average rating per episode (in case of duplicates)
final_df = (
    final_df.groupby(["Show", "Season", "Episode"], as_index=False)["Rating"]
    .mean()
    .sort_values(["Show", "Episode"])
)

# shows = sorted(final_df["Show"].unique())

final_year = (
    df[df["Season"] == final_season_idx]
    .groupby("Show")["Year"]
    .max()
    .reset_index()
    .sort_values("Year")
)
shows = final_year["Show"].tolist()

season_avgs = df.groupby(["Show", "Season"])["Rating"].mean().reset_index()
peak_avg = season_avgs.groupby("Show")["Rating"].max().to_dict()


# ── Decline detection ─────────────────────────────────────────────────────────

def is_declining(ratings):
    """True if second half avg is notably lower than first half avg."""
    mid = math.ceil(len(ratings) / 2)
    first_half = ratings[:mid]
    second_half = ratings[mid:]
    if len(second_half) == 0:
        return False
    return np.mean(second_half) < np.mean(first_half) - 0.2

# ── Layout ────────────────────────────────────────────────────────────────────

n      = len(shows)
rows   = math.ceil(n / COLS)
fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.8, rows * 2.8))
fig.patch.set_facecolor(BG)

# Flatten axes array for easy indexing
axes_flat = axes.flatten() if n > 1 else [axes]

for i, show in enumerate(shows):
    ax = axes_flat[i]
    show_data = final_df[final_df["Show"] == show].sort_values("Episode")
    episodes  = show_data["Episode"].tolist()
    ratings   = show_data["Rating"].tolist()
    season    = int(show_data["Season"].iloc[0])
    avg       = np.mean(ratings)
    decline   = is_declining(ratings)

    color      = RED   if decline else BLUE
    fill_color = RED_FILL if decline else BLUE_FILL
    # trend_label = "↓ declining" if decline else "↑ held strong"
    # trend_color = RED if decline else "#1a7a4a"

    # Panel background
    ax.set_facecolor(PANEL_BG)

    # # Threshold line
    # ax.axhline(THRESHOLD, color=THRESHOLD_C, linewidth=0.8, linestyle="--", zorder=1)
    # Peak season avg threshold line (per show)
    peak_threshold = peak_avg.get(show, THRESHOLD)
    ax.axhline(peak_threshold, color=THRESHOLD_C, linewidth=2, linestyle="--", zorder=1)

    # Fill under line
    ax.fill_between(episodes, ratings, alpha=0.12, color=color, zorder=2)

    # Main line
    ax.plot(episodes, ratings, color=color, linewidth=1.8,
            marker="o", markersize=3.5, markerfacecolor=color,
            markeredgewidth=0, zorder=3)

    # Y-axis range: tight around data but always show threshold
    y_min = min(min(ratings) - 0.5, THRESHOLD - 0.3)
    y_max = max(ratings) + 0.5
    # ax.set_ylim(max(1, y_min), min(10.2, y_max))
    ax.set_ylim(3, 10)

    # X-axis: integer episode ticks only
    ax.set_xlim(episodes[0] - 0.3, episodes[-1] + 0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))

    # Styling
    ax.tick_params(axis="both", labelsize=8, colors=TEXT_MUTED, length=2)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
        spine.set_linewidth(0.6)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, zorder=0)
    ax.grid(axis="x", visible=False)

    # Show title
    ax.set_title(show, fontsize=10, fontweight="bold", color=TEXT_DARK,
                 loc="left", pad=15)

    # Subtitle: season · avg · trend
    ax.text(0.0, 1.01,
            f"S{season}  ·  avg {avg:.1f}  ·  ",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, va="bottom")
    # Trend label in color
    x_offset = 0.0 + len(f"S{season}  ·  avg {avg:.1f}  ·  ") * 0.013
    # ax.text(x_offset, 1.01, trend_label,
    #         transform=ax.transAxes, fontsize=8, color=trend_color,
    #         fontweight="bold", va="bottom")

    # X label
    ax.set_xlabel("Episode", fontsize=8, color=TEXT_MUTED, labelpad=3)

# Hide unused panels
for j in range(n, len(axes_flat)):
    axes_flat[j].set_visible(False)

# ── Title & footer ────────────────────────────────────────────────────────────

fig.suptitle(
    "Are TV shows nailing their final season?",
    fontsize=15, fontweight="bold", color=TEXT_DARK,
    x=0.02, ha="left", y=1.01
)
# fig.text(
#     0.02, 0.995,
#     "Episode-by-episode IMDb ratings · final season only · dashed line = 7.0 threshold",
#     fontsize=9, color=TEXT_MUTED, ha="left"
# )
fig.text(
    0.02, 0.965,
    "Episode-by-episode IMDb ratings · final season only · dashed line = peak season avg",
    fontsize=9, color=TEXT_MUTED, ha="left"
)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(OUTPUT_IMG, dpi=DPI, bbox_inches="tight", facecolor=BG)
print(f"✅ Saved → {OUTPUT_IMG}")
plt.show()
