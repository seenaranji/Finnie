"""One look for every chart: the same colours, fonts, spacing and hover style, so the app reads as one product. Colours match .streamlit/config.toml."""
import plotly.graph_objects as go

PALETTE = ["#0F766E", "#1D4ED8", "#B45309", "#7C3AED", "#BE185D", "#475569", "#15803D", "#0E7490"]   # teal, blue, amber, violet, rose, slate, green, cyan
POSITIVE, NEGATIVE, MUTED, INK, GRID = "#15803D", "#B91C1C", "#64748B", "#0F172A", "#E5EAF0"
SEQUENTIAL_RED_BLUE = [[0.0, "#B91C1C"], [0.5, "#F8FAFC"], [1.0, "#1D4ED8"]]   # correlation: -1 red, 0 near white, +1 blue


def style(figure: go.Figure, title: str | None = None, height: int = 360, legend_below: bool = False) -> go.Figure:
    """Apply the shared look. Data and traces are left exactly as the caller made them."""
    figure.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 15, "color": INK}} if title else None,
        font={"family": "Source Sans Pro, -apple-system, Segoe UI, Roboto, sans-serif", "size": 13, "color": INK},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=PALETTE, height=height,
        margin={"l": 10, "r": 10, "t": 50 if title else 16, "b": 10}, hoverlabel={"bgcolor": "white", "font_size": 13},
        legend={"orientation": "h", "y": -0.22, "x": 0} if legend_below else {})
    figure.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return figure


def donut(labels: list[str], values: list[float], title: str, height: int = 340) -> go.Figure:
    """A ring chart with the share inside each slice. Slices are coloured in a fixed order so the same kind keeps its colour across charts."""
    total = sum(values) or 1
    # The share goes in the LEGEND, not on the slices: when one slice is 96% the small ones are slivers and their labels pile on top of each other.
    figure = go.Figure(go.Pie(labels=[f"{label} · {value / total * 100:.1f}%" for label, value in zip(labels, values)], values=values, hole=0.55, sort=False, textinfo="none",
                              marker={"colors": PALETTE[:len(values)], "line": {"color": "white", "width": 2}}, hovertemplate="%{label}<extra></extra>"))
    figure.update_layout(showlegend=True)
    return style(figure, title, height).update_layout(legend={"orientation": "v", "x": 1.0, "y": 0.5})
