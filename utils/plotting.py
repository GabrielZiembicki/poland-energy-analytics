"""Shared chart styling, color palette, and figure export helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

PALETTE = {
    "load": "#2B2D42",
    "residual": "#8D99AE",
    "wind": "#1B998B",
    "solar": "#F4A261",
    "price": "#E63946",
    "bess": "#6A4C93",
    "temp": "#457B9D",
    "grid": "#E5E5E5",
    "accent": "#FFB703",
    "muted": "#ADB5BD",
}

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

DEFAULT_SOURCE = "Source: PSE via Clarwise"

_TEMPLATE_NAME = "Base"
_PLOTLY_GRID_COLOR = "rgba(200,200,200,0.3)"


def _build_plotly_template() -> go.layout.Template:
    """Build the Base Plotly template by extending `simple_white`."""
    tpl = go.layout.Template(pio.templates["simple_white"])
    tpl.layout.update(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="sans-serif", size=12, color="#2B2D42"),
        margin=dict(t=100, b=80, l=60, r=20),
        colorway=[
            PALETTE["load"],
            PALETTE["solar"],
            PALETTE["wind"],
            PALETTE["price"],
            PALETTE["bess"],
            PALETTE["temp"],
            PALETTE["accent"],
            PALETTE["muted"],
        ],
        xaxis=dict(showgrid=True, gridcolor=_PLOTLY_GRID_COLOR, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=_PLOTLY_GRID_COLOR, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return tpl


pio.templates[_TEMPLATE_NAME] = _build_plotly_template()
pio.templates.default = _TEMPLATE_NAME


def set_style() -> None:
    """Apply the project's default matplotlib style."""
    mpl.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
        }
    )


def color(name: str) -> str:
    """Look up a project palette color by semantic name."""
    return PALETTE[name]


def save_fig(fig: plt.Figure, name: str, subdir: str | None = None) -> Path:
    """Save a figure under `figures/<subdir>/<name>.png`."""
    out_dir = FIG_DIR if subdir is None else FIG_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    return path


def annotate_source(ax: plt.Axes, text: str = DEFAULT_SOURCE) -> None:
    """Add a small attribution caption beneath a matplotlib plot."""
    ax.figure.text(
        0.99,
        -0.02,
        text,
        ha="right",
        va="top",
        fontsize=8,
        color=PALETTE["muted"],
    )


def _title_html(title: str, subtitle: str | None) -> str:
    head = f"<span style='font-size:16px;font-weight:bold'>{title}</span>"
    if not subtitle:
        return head
    return (
        f"{head}<br>"
        f"<span style='font-size:13px;color:gray'>{subtitle}</span>"
    )


def style_figure(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    source: str | None = DEFAULT_SOURCE,
    source_y: float = -0.08,
    axis_caption: str | None = None,
    axis_caption_y: float = -0.05,
    height: int | None = None,
) -> go.Figure:
    """Apply base title, subtitle, source annotation, and optional axis caption.

    Static styling (template, bg, grid, fonts, margins, colorway) is handled by the
    `base` Plotly template registered on import — no per-figure call needed.
    """
    fig.update_layout(title=dict(text=_title_html(title, subtitle)))
    if height is not None:
        fig.update_layout(height=height)
    if source:
        fig.add_annotation(
            text=source,
            xref="paper",
            yref="paper",
            x=0.0,
            y=source_y,
            xanchor="left",
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
    if axis_caption:
        fig.add_annotation(
            text=axis_caption,
            xref="paper",
            yref="paper",
            x=0.5,
            y=axis_caption_y,
            xanchor="center",
            yanchor="top",
            showarrow=False,
        )
    return fig
