"""Project-wide Matplotlib styling and semantic colors."""

import matplotlib as mpl


# A restrained, colorblind-safe palette based on Paul Tol's bright scheme.
COLORS = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "gold": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "gray": "#7A7A7A",
}
PALETTE = tuple(COLORS.values())

# Semantic colors shared by every choice-conditioned trajectory plot.
CHOICE_COLORS = {
    -1: COLORS["blue"],
    1: COLORS["red"],
}

# A separate diverging palette keeps covariance sign distinct from choice.
COVARIANCE_COLORS = {
    "negative": "#542788",
    "neutral": "#F7F7F7",
    "positive": "#B35806",
}
COVARIANCE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "covariance",
    tuple(COVARIANCE_COLORS.values()),
)

# Signed task values use the same blue/red semantics as left/right choices.
SIGNED_VALUE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "signed_value",
    (CHOICE_COLORS[-1], "#F7F7F7", CHOICE_COLORS[1]),
)

# Residuals use colors that do not overlap with the blue/gold rank colors.
RESIDUAL_COLORS = {
    "negative": COLORS["purple"],
    "neutral": "#F7F7F7",
    "positive": COLORS["green"],
}
RESIDUAL_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "residual",
    tuple(RESIDUAL_COLORS.values()),
)
FREQUENCY_CMAP = mpl.colormaps["viridis"]

STIMULUS_WINDOW_STYLE = {
    "color": COLORS["gray"],
    "alpha": 0.12,
}
DECISION_WINDOW_STYLE = {
    "color": COLORS["green"],
    "alpha": 0.10,
}
REFERENCE_LINE_STYLE = {
    "color": COLORS["gray"],
    "linestyle": "--",
    "linewidth": 1,
}


STYLE = {
    # Canvas and typography
    "figure.dpi": 120,
    "figure.figsize": (6.4, 4.0),
    "figure.facecolor": "white",
    "figure.titlesize": 11,
    "figure.titleweight": "medium",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Helvetica Neue",
        "Helvetica",
        "Arial",
        "DejaVu Sans",
    ],
    "font.size": 10,
    "mathtext.fontset": "dejavusans",
    "text.color": "#27313B",
    # Axes
    "axes.axisbelow": True,
    "axes.edgecolor": "#4B5563",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.labelcolor": "#27313B",
    "axes.labelpad": 5,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlecolor": "#1F2933",
    "axes.titlepad": 8,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    # Lines, grids, and ticks
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "grid.alpha": 0.55,
    "grid.color": "#D8DEE7",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": "#4B5563",
    "xtick.direction": "out",
    "xtick.labelsize": 9,
    "xtick.major.size": 3.5,
    "xtick.major.width": 0.75,
    "ytick.color": "#4B5563",
    "ytick.direction": "out",
    "ytick.labelsize": 9,
    "ytick.major.size": 3.5,
    "ytick.major.width": 0.75,
    # Annotations and export
    "errorbar.capsize": 2.5,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.labelspacing": 0.4,
    "legend.title_fontsize": 9,
    "image.cmap": "viridis",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.05,
    "svg.fonttype": "none",
}


def set_plot_style() -> None:
    """Apply the project style to all subsequent Matplotlib figures."""
    mpl.rcParams.update(STYLE)
