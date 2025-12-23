# -*- coding: utf-8 -*-
"""
Grooming Event Visualization Script

This script converts grooming event data into a graphical representation, where each event is represented
by a horizontal bar. The events are read from a tab-delimited text file, and the data is cleaned, processed,
and visualized with respect to a specified time window. The final visualization can be exported as an editable
vector graphic (SVG, PDF, EPS) or displayed on-screen.

Usage:
1) Specify the input data file, time window, and export options in the "User Parameters" section.
2) The script reads the data, processes it, and generates the corresponding grooming event visualization.
3) The output can be saved as a vector graphic or displayed.

Dependencies:
- pandas
- matplotlib
- opencv-python
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import BrokenBarHCollection

# ============== User Parameters (Modify as needed) ==============
FILE_PATH = r"input_file_path"
GROUP_NAME = r"GROUP_NAME"
GROUP_NAME = "1"  # Group name for the animal/group in the figure; leave empty to use the filename

# Time window (in seconds). Set to None to avoid clipping.
TMIN = 0  # e.g., 0
TMAX = 1800  # e.g., 1800

# Export settings: Choose one ('svg', 'pdf', 'eps', 'png')
EXPORT_FORMAT = "svg"
# Output file path; leave None to display the plot without saving
OUTFILE = r"outfile_path"  # or None
# ================================================================
def read_txt_robust(path: str) -> pd.DataFrame:
    """Read grooming event data from a text file, handling various encodings and whitespace delimiters."""
    last_err = None
    for enc in ("utf-8-sig", "gbk", "ansi"):
        try:
            df = pd.read_csv(
                path,
                sep=r"\s+",
                engine="python",
                header=None,
                dtype=str,  # Read as strings first, then convert to numeric
                names=["ID", "Grooming", "Start", "End", "Duration"],
                encoding=enc
            )
            return df
        except Exception as e:
            last_err = e
    raise last_err


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Clean and convert the series to numeric, removing unwanted characters."""
    s = s.astype(str).str.replace("\ufeff", "", regex=False)  # Remove BOM
    s = s.str.replace("\u3000", " ", regex=False)  # Remove full-width spaces
    s = s.str.strip()
    s = s.str.replace(r"[^\d\.\-]+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def clip_to_window(data: pd.DataFrame, tmin, tmax) -> pd.DataFrame:
    """Clip the events to the specified time window [tmin, tmax]. Events outside this range are discarded."""
    if tmin is None and tmax is None:
        return data.copy()

    start = data["Start"].copy()
    end = data["End"].copy()

    if tmin is not None:
        start = start.clip(lower=tmin)
    if tmax is not None:
        end = end.clip(upper=tmax)

    clipped = data.copy()
    clipped["Start"] = start
    clipped["End"] = end
    clipped["Duration"] = clipped["End"] - clipped["Start"]

    if tmin is not None:
        clipped = clipped[clipped["End"] > tmin]
    if tmax is not None:
        clipped = clipped[clipped["Start"] < tmax]
    clipped = clipped[clipped["Duration"] > 0]
    return clipped


def main():
    file_path = FILE_PATH
    group_name = GROUP_NAME or os.path.splitext(os.path.basename(file_path))[0]
    tmin_user = TMIN
    tmax_user = TMAX
    export_fmt = EXPORT_FORMAT.lower()
    outfile = OUTFILE

    # 1) Read the event data
    df = read_txt_robust(file_path)

    # 2) Clean and convert data to numeric
    for col in ["Grooming", "Start", "End", "Duration"]:
        df[col] = to_numeric_series(df[col])

    # Recalculate Duration if missing
    if df["Duration"].isna().any():
        df["Duration"] = df["End"] - df["Start"]

    # Drop rows with missing critical data
    df = df.dropna(subset=["Start", "End", "Duration", "Grooming"]).copy()

    # 3) Clip the data to the user-defined time window
    df = clip_to_window(df, tmin_user, tmax_user)
    if df.empty:
        print("No data found within the specified time window. Please adjust TMIN / TMAX.")
        return

    # 4) Add group name to the dataframe
    df["Group"] = group_name

    # 5) Define labels and colors
    label_map = {
        1: "Paw licking",
        2: "NoseFaceHead",
        3: "Body",
        4: "Leg",
        5: "TailGenetail",
    }
    color_map = {
        1: "#ff2600",
        2: "#ffaa00",
        3: "#fffa00",
        4: "#84dd26",
        5: "#0095ff",
    }

    # 6) Create the plot
    groups = df["Group"].unique()
    fig, ax = plt.subplots(figsize=(12, 3 + 0.6 * len(groups)))

    row_height = 8
    row_gap = 6
    legend_handles = {}

    xmin = (tmin_user if tmin_user is not None else float(df["Start"].min())) - 10
    xmax = (tmax_user if tmax_user is not None else float(df["End"].max())) + 10

    for i, g in enumerate(groups):
        sub = df[df["Group"] == g]
        y0 = i * (row_height + row_gap)

        for grcode, part in sub.groupby("Grooming"):
            grcode_int = int(grcode)
            xranges = list(zip(part["Start"], part["Duration"]))
            facecolor = color_map.get(grcode_int, None)

            coll = BrokenBarHCollection(
                xranges=xranges,
                yrange=(y0, row_height),
                facecolors=facecolor,
                edgecolors="none",
                alpha=1
            )
            ax.add_collection(coll)

            if grcode_int not in legend_handles:
                legend_handles[grcode_int] = Patch(
                    facecolor=facecolor,
                    edgecolor="none",
                    label=label_map.get(grcode_int, str(grcode_int))
                )

        # Add group labels on the left
        ax.text(xmin - 10, y0 + row_height / 2, g,
                va="center", ha="right", fontsize=10)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-2, len(groups) * (row_height + row_gap))
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])

    # Hide unnecessary spines
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    handles = [legend_handles[k] for k in sorted(legend_handles.keys())]
    ax.legend(handles=handles, loc="upper right", frameon=False, title="Behavior")

    plt.tight_layout()

    # 7) Save or display the plot
    if outfile:
        root, _ = os.path.splitext(outfile)
        outfile = f"{root}.{export_fmt}"
        plt.savefig(outfile, format=export_fmt, bbox_inches="tight")
        print(f"File saved: {outfile}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
