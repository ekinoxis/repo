"""
Gantt horizontal con Plotly Express (timeline): estable en Streamlit y fácil de mantener.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Tema oscuro: contraste alto en etiquetas Y (código + descripción)
_PAPER_DARK = "#0b1220"
_PLOT_DARK = "#111827"
_TEXT = "#e2e8f0"
_TEXT_MUTED = "#94a3b8"
_GRID = "rgba(148, 163, 184, 0.12)"
_AXIS_LINE = "#334155"


def _top_ruler_ticks(rng: list, span_days: int) -> list[pd.Timestamp]:
    """Fechas para marcas de la regla superior (no depende de xaxis2)."""
    a = pd.Timestamp(rng[0]).normalize()
    b = pd.Timestamp(rng[1]).normalize()
    if span_days <= 60:
        dr = pd.date_range(a, b, freq="7D")
    elif span_days <= 200:
        dr = pd.date_range(a, b, freq="14D")
    else:
        dr = pd.date_range(a, b, freq="MS")
    if len(dr) == 0:
        return [a, b]
    if len(dr) > 22:
        step = max(1, len(dr) // 22)
        dr = dr[::step]
    return [pd.Timestamp(x) for x in dr]


def _etiqueta(code: str, desc: str, nivel: str, profundidad: int) -> str:
    ind = "    " * min(max(0, int(profundidad)), 8)
    icono = {"Capítulo": "◆ ", "Subcapítulo": "▸ ", "Partida": "· "}.get(nivel, "")
    corta = (desc or "")[:72]
    if len(desc or "") > 72:
        corta += "…"
    return f"{ind}{icono}{code} — {corta}"


def _prof_val(r: pd.Series) -> int:
    try:
        p = r.get("Prof.")
        if p is None or pd.isna(p):
            return 0
        return int(float(p))
    except (TypeError, ValueError):
        return 0


def _tipo_fila(r: pd.Series) -> str:
    nv = str(r.get("Nivel", "")).strip()
    crit = str(r.get("Crítica", "")).strip().lower().startswith("s")
    if crit and nv == "Partida":
        return "Crítico"
    if nv in ("Capítulo", "Subcapítulo", "Partida"):
        return nv
    return "Partida"


def build_professional_gantt(plot_df: pd.DataFrame, chart_height: int, *, solo: bool = False) -> go.Figure:
    empty_layout = dict(
        margin=dict(l=40, r=24, t=56, b=40),
        paper_bgcolor=_PAPER_DARK,
        plot_bgcolor=_PLOT_DARK,
        font=dict(size=11, color=_TEXT),
    )
    if plot_df.empty:
        return go.Figure().update_layout(
            title=dict(text="Sin filas para el Gantt", font=dict(size=14, color=_TEXT)),
            height=min(chart_height, 400),
            **empty_layout,
        )

    df = plot_df.copy()
    df["_ini"] = pd.to_datetime(df["Inicio_lab"], errors="coerce")
    df["_fin"] = pd.to_datetime(df["Fin_lab"], errors="coerce") + pd.Timedelta(days=1)
    df = df.dropna(subset=["_ini", "_fin"])
    if df.empty:
        return go.Figure().update_layout(
            title=dict(text="Fechas inválidas: revisa Inicio_lab / Fin_lab", font=dict(size=14, color=_TEXT)),
            height=min(chart_height, 420),
            **empty_layout,
        )

    ys: list[str] = []
    tipos: list[str] = []
    for _, r in df.iterrows():
        ys.append(
            _etiqueta(
                str(r.get("Código", "")),
                str(r.get("Descripción", "")),
                str(r.get("Nivel", "")),
                _prof_val(r),
            )
        )
        tipos.append(_tipo_fila(r))
    df["_y"] = ys
    df["_tipo"] = tipos

    color_map = {
        "Crítico": "#f87171",
        "Capítulo": "#2dd4bf",
        "Subcapítulo": "#5eead4",
        "Partida": "#a5b4fc",
    }

    hover_cols = [
        "Código",
        "Nivel",
        "Días lab.",
        "Inicio_lab",
        "Fin_lab",
        "Holgura",
        "Predecesoras",
        "Crítica",
    ]
    hover_data = {c: True for c in hover_cols if c in df.columns}

    fig = px.timeline(
        df,
        x_start="_ini",
        x_end="_fin",
        y="_y",
        color="_tipo",
        color_discrete_map=color_map,
        category_orders={"_tipo": ["Capítulo", "Subcapítulo", "Partida", "Crítico"]},
        hover_data=hover_data if hover_data else None,
    )

    y_order = df["_y"].tolist()
    fig.update_yaxes(
        autorange="reversed",
        title=None,
        categoryorder="array",
        categoryarray=y_order,
        tickfont=dict(size=11, color=_TEXT),
        gridcolor=_GRID,
        showgrid=True,
        showline=True,
        linewidth=1,
        linecolor=_AXIS_LINE,
    )

    t0, t1 = df["_ini"].min(), df["_fin"].max()
    span = max(1, (pd.Timestamp(t1).normalize() - pd.Timestamp(t0).normalize()).days + 1)
    pad_days = max(2, min(14, span // 8 + 1))
    pad = pd.Timedelta(days=pad_days)
    rng = [pd.Timestamp(t0) - pad, pd.Timestamp(t1) + pad]

    if span <= 60:
        dtick = 7 * 86400000.0
        tickfmt = "%d %b"
    elif span <= 200:
        dtick = 14 * 86400000.0
        tickfmt = "%d %b"
    else:
        dtick = "M1"
        tickfmt = "%b %Y"

    max_label = max((len(s) for s in y_order), default=40)
    left_m = max(220, min(460, 7 * max_label // 10 + 180))

    fig.update_xaxes(
        type="date",
        range=rng,
        dtick=dtick,
        tickformat=tickfmt,
        showgrid=True,
        gridcolor=_GRID,
        showline=True,
        linewidth=1,
        linecolor=_AXIS_LINE,
        title=dict(text="Fecha", font=dict(size=11, color=_TEXT_MUTED)),
        tickfont=dict(size=10, color=_TEXT_MUTED),
        showticklabels=True,
    )

    top_extra = 28 if solo else 0
    # Regla superior: anotaciones (xaxis2 superpuesto con px.timeline suele no mostrar etiquetas en Streamlit).
    top_ticks = _top_ruler_ticks(rng, span)
    tick_lbl = "%d %b" if span <= 200 else "%b %Y"

    fig.update_layout(
        height=chart_height,
        barmode="overlay",
        bargap=0.2,
        paper_bgcolor=_PAPER_DARK,
        plot_bgcolor=_PLOT_DARK,
        margin=dict(l=left_m, r=28, t=108 + top_extra, b=64),
        font=dict(size=11, color=_TEXT),
        legend_title_text="",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.008,
            yref="paper",
            xanchor="right",
            x=0.99,
            xref="paper",
            font=dict(size=10, color=_TEXT),
            bgcolor="rgba(15, 23, 42, 0.55)",
            bordercolor=_AXIS_LINE,
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="#1e293b", font_size=11, font_color="#f8fafc", bordercolor="#475569"),
        hovermode="closest",
        showlegend=True,
        annotations=[
            dict(
                text="<b>Escala (arriba)</b>",
                xref="paper",
                yref="paper",
                x=0,
                xanchor="left",
                y=1.02,
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color="#e2e8f0"),
            ),
            *[
                dict(
                    x=pd.Timestamp(ts).isoformat(),
                    xref="x",
                    yref="paper",
                    y=1.0,
                    yanchor="bottom",
                    xanchor="center",
                    text=pd.Timestamp(ts).strftime(tick_lbl),
                    textangle=-90,
                    showarrow=False,
                    font=dict(size=9, color="#e2e8f0"),
                )
                for ts in top_ticks
            ],
        ],
    )

    fig.update_traces(marker_line_color="rgba(226, 232, 240, 0.35)", marker_line_width=1)

    hoy = pd.Timestamp(date.today())
    if rng[0] <= hoy <= rng[1]:
        xs = pd.Timestamp(hoy).isoformat()
        fig.add_shape(
            type="line",
            x0=xs,
            x1=xs,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#fb923c", width=2),
            layer="above",
        )
        fig.add_annotation(
            x=xs,
            y=1,
            xref="x",
            yref="paper",
            text=" Hoy",
            showarrow=False,
            yanchor="bottom",
            xanchor="left",
            font=dict(size=10, color="#fdba74"),
        )

    return fig
