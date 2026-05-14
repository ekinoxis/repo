"""
GanttMachine: BC3 (Presto), mediciones ~M, planificación L–V y CPM.
Edición en la aplicación; exportación opcional en CSV.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bc3_parser import parse_bc3
from gantt_figure import build_professional_gantt
from planning_engine import build_planning_rows, merge_tasks_from_planning_rows

BC3_FILE = Path(__file__).resolve().parent / "GaudiVENTA.bc3"
STATE_FILE = Path(__file__).resolve().parent / "schedule_state.json"
STATE_BACKUPS_DIR = Path(__file__).resolve().parent / "schedule_backups"
MAX_STATE_BACKUPS = 50
_BACKUP_GLOB = "schedule_state-*.json"


def _normalize_state_dict(data: dict) -> dict:
    out = {**data}
    if "tasks" not in out or not isinstance(out.get("tasks"), dict):
        out["tasks"] = {}
    if "project_start" not in out:
        out["project_start"] = date.today().isoformat()
    return out


def _load_state() -> dict:
    if not STATE_FILE.is_file():
        return _normalize_state_dict({"project_start": date.today().isoformat(), "tasks": {}})
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("estado no es un objeto")
        return _normalize_state_dict(data)
    except (json.JSONDecodeError, OSError, ValueError):
        return _normalize_state_dict({"project_start": date.today().isoformat(), "tasks": {}})


def _atomic_write_state(payload: str) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(STATE_FILE)


def _append_snapshot(payload: str) -> None:
    STATE_BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snap = STATE_BACKUPS_DIR / f"schedule_state-{stamp}.json"
    snap.write_text(payload, encoding="utf-8")


def _prune_old_backups() -> None:
    if not STATE_BACKUPS_DIR.is_dir():
        return
    paths = sorted(
        STATE_BACKUPS_DIR.glob(_BACKUP_GLOB),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in paths[MAX_STATE_BACKUPS:]:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


def _persist_state_payload(payload: str) -> None:
    _atomic_write_state(payload)
    _append_snapshot(payload)
    _prune_old_backups()


def _save_state(state: dict) -> None:
    payload = json.dumps(state, ensure_ascii=False, indent=2)
    _persist_state_payload(payload)


def _list_schedule_backups() -> list[dict]:
    if not STATE_BACKUPS_DIR.is_dir():
        return []
    rows: list[dict] = []
    for p in STATE_BACKUPS_DIR.glob(_BACKUP_GLOB):
        try:
            st_info = p.stat()
        except OSError:
            continue
        mtime = datetime.fromtimestamp(st_info.st_mtime)
        size_b = int(st_info.st_size)
        if size_b < 1024:
            size_s = f"{size_b} B"
        else:
            size_s = f"{(size_b + 1023) // 1024} KB"
        label = f"{mtime.strftime('%d/%m/%Y %H:%M:%S')} · {size_s} · {p.name}"
        rows.append({"path": p, "mtime": mtime, "size": size_b, "label": label})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def _restore_from_backup(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("El archivo no contiene un objeto JSON de planificación.")
    normalized = _normalize_state_dict(data)
    payload = json.dumps(normalized, ensure_ascii=False, indent=2)
    old_payload: str | None = None
    if STATE_FILE.is_file():
        try:
            old_payload = STATE_FILE.read_text(encoding="utf-8")
        except OSError:
            old_payload = None
    if old_payload is not None:
        _append_snapshot(old_payload)
        _prune_old_backups()
    _persist_state_payload(payload)


def _persist_signature(data: dict) -> str:
    """Compara de forma estable lo que se guarda en schedule_state.json."""
    return json.dumps(
        {
            "project_start": str(data.get("project_start", "")),
            "default_cuadrilla": float(data.get("default_cuadrilla", 3.0)),
            "tasks": dict(data.get("tasks") or {}),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


@st.cache_data(show_spinner=False)
def _cached_parse(path_str: str):
    return parse_bc3(Path(path_str))


def _etiqueta_gantt(code: str, desc: str, nivel: str, profundidad: int) -> str:
    ind = "    " * min(int(profundidad), 8)
    icono = {"Capítulo": "◆ ", "Subcapítulo": "▸ ", "Partida": "· "}.get(nivel, "")
    corta = (desc or "")[:72]
    if len(desc or "") > 72:
        corta += "…"
    return f"{ind}{icono}{code} — {corta}"


def _build_gantt_figure(plot_df: pd.DataFrame, chart_height: int, *, solo: bool = False) -> go.Figure:
    """Cronograma estilo herramienta de proyecto (barras claras, escala legible)."""
    return build_professional_gantt(plot_df, chart_height, solo=solo)


def _safe_pos_int(val) -> int | None:
    """Entero ≥1 para duración en días laborables, o None si no aplica."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        if pd.isna(val):
            return None
    except TypeError:
        pass
    try:
        return max(1, int(round(float(val))))
    except (TypeError, ValueError):
        return None


_EDITABLE_CMP_COLS = ("Cuadrilla", "Rend./día", "Días lab.", "Dur. partida", "Predecesoras")


def _norm_predecesoras_cmp(val) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        if pd.isna(val):
            return ""
    except TypeError:
        pass
    s = str(val).strip()
    if not s:
        return ""
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return ",".join(sorted(parts, key=str.lower))


def _editable_compare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Solo columnas editables, orden por código, valores normalizados (para detectar cambios al confirmar celda)."""
    if df is None or df.empty or "Código" not in df.columns:
        return pd.DataFrame()
    w = df.sort_values("Código", kind="stable").reset_index(drop=True)
    out = pd.DataFrame({"Código": w["Código"].astype(str)})
    for c in _EDITABLE_CMP_COLS:
        if c not in w.columns:
            continue
        if c == "Predecesoras":
            out[c] = w[c].map(_norm_predecesoras_cmp)
        else:
            out[c] = pd.to_numeric(w[c], errors="coerce")
    return out


def _planning_editable_equals(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        return _editable_compare_frame(a).equals(_editable_compare_frame(b))
    except Exception:
        return False


def _harmonize_partida_duration(edited: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Partidas: unifica Días lab. y Dur. partida según qué celda cambió respecto a la tabla base del editor."""
    if base_df is None or edited.shape[0] != base_df.shape[0]:
        return edited
    if "Código" not in base_df.columns or "Código" not in edited.columns:
        return edited
    out = edited.copy()
    base_by = base_df.set_index(base_df["Código"].astype(str), drop=False)
    for idx in out.index:
        if str(out.at[idx, "Nivel"]).strip() != "Partida":
            continue
        code = str(out.at[idx, "Código"])
        if code not in base_by.index:
            continue
        b = base_by.loc[code]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        dl = _safe_pos_int(out.at[idx, "Días lab."])
        du = _safe_pos_int(out.at[idx, "Dur. partida"])
        bdl = _safe_pos_int(b.get("Días lab."))
        bdu = _safe_pos_int(b.get("Dur. partida"))
        changed_l = dl is not None and bdl is not None and dl != bdl
        changed_u = du is not None and bdu is not None and du != bdu
        v: int | None = None
        if changed_l and not changed_u:
            v = dl
        elif changed_u and not changed_l:
            v = du
        elif changed_l and changed_u:
            v = dl
        elif dl is not None and du is not None and dl != du:
            v = du
        elif dl is not None and du is not None:
            v = dl
        else:
            v = dl or du or bdl or bdu or 1
        out.at[idx, "Días lab."] = int(v)
        if "Dur. partida" in out.columns:
            out.at[idx, "Dur. partida"] = int(v)
    return out


def _align_resumen_filas(edited: pd.DataFrame, df_gantt: pd.DataFrame) -> pd.DataFrame:
    """Resumen: Días lab. y fechas desde el motor. Partidas: Días lab. y Dur. partida coherentes con CPM + fechas."""
    mask_p = df_gantt["Nivel"].astype(str).str.strip() == "Partida"
    partida_codes = set(df_gantt.loc[mask_p, "Código"].astype(str))
    by_c = {str(row["Código"]): row for _, row in df_gantt.iterrows()}
    out = edited.copy(deep=True)
    for idx in out.index:
        code = str(out.at[idx, "Código"])
        gr = by_c.get(code)
        if gr is None:
            continue
        if code in partida_codes:
            dlab = int(gr["Días lab."])
            out.at[idx, "Días lab."] = dlab
            if "Dur. partida" in out.columns:
                out.at[idx, "Dur. partida"] = dlab
            out.at[idx, "Inicio_lab"] = gr["Inicio_lab"]
            out.at[idx, "Fin_lab"] = gr["Fin_lab"]
            if "Inicio" in out.columns:
                out.at[idx, "Inicio"] = gr["Inicio"]
            out.at[idx, "Días"] = float(dlab)
            continue
        out.at[idx, "Días lab."] = int(gr["Días lab."])
        if "Dur. partida" in out.columns:
            out.at[idx, "Dur. partida"] = pd.NA
        out.at[idx, "Inicio_lab"] = gr["Inicio_lab"]
        out.at[idx, "Fin_lab"] = gr["Fin_lab"]
        out.at[idx, "Días"] = float(gr["Días"])
        if "Inicio" in out.columns:
            out.at[idx, "Inicio"] = gr["Inicio"]
    return out


def main() -> None:
    st.set_page_config(page_title="GanttMachine", layout="wide")

    if not BC3_FILE.is_file():
        st.error(f"No se encuentra el BC3 en la raíz: `{BC3_FILE}`")
        st.stop()

    result = _cached_parse(str(BC3_FILE))
    state = _load_state()

    st.sidebar.header("Calendario y cuadrillas")
    project_start = st.sidebar.date_input(
        "Inicio de obra (primer día laborable)",
        value=date.fromisoformat(state.get("project_start", date.today().isoformat())),
    )
    default_crew = st.sidebar.number_input(
        "Cuadrilla base (oficios, por defecto)",
        min_value=0.5,
        max_value=50.0,
        value=float(state.get("default_cuadrilla", 3.0)),
        step=0.5,
    )
    state["default_cuadrilla"] = default_crew

    roots = result.roots
    n_med = len(result.mediciones)
    st.sidebar.caption(
        f"Capítulos: **{len(roots)}** · Conceptos: **{len(result.concepts)}** · "
        f"Partidas con medición ~M: **{n_med}**"
    )
    st.sidebar.markdown(
        "Las **mediciones** vienen del BC3 (~M). El **PDF** no está enlazado; si lo añades, "
        "se puede usar como documentación aparte."
    )

    selected = st.sidebar.multiselect(
        "Capítulos a incluir",
        options=roots,
        default=list(roots),
    )
    if not selected:
        st.warning("Selecciona al menos un capítulo.")
        st.stop()

    max_rows = st.sidebar.slider("Máx. filas en el Gantt interactivo", 30, 500, 200, 10)

    st.sidebar.divider()
    solo = st.sidebar.toggle(
        "Vista solo Gantt (cronograma ampliado)",
        key="solo_gantt_vista",
        help="Oculta la tabla y los botones de la página; el diagrama gana altura. Capítulos, filas y fecha siguen en esta barra.",
    )

    st.sidebar.divider()
    with st.sidebar.expander("Respaldos de planificación", expanded=False):
        st.caption(
            f"Cada guardado escribe `{STATE_FILE.name}` y deja una copia fechada en "
            f"`{STATE_BACKUPS_DIR.name}` (máx. **{MAX_STATE_BACKUPS}** archivos)."
        )
        backups = _list_schedule_backups()
        pending = st.session_state.get("_pending_restore_path")
        if pending:
            st.warning(
                f"Se **reemplazará** `{STATE_FILE.name}`. La versión actual se archiva antes en la carpeta de respaldos."
            )
            c_yes, c_no = st.columns(2)
            with c_yes:
                if st.button("Confirmar", type="primary", key="backup_restore_confirm"):
                    try:
                        _restore_from_backup(Path(pending))
                    except (json.JSONDecodeError, OSError, ValueError) as exc:
                        st.error(f"No se pudo restaurar: {exc}")
                        st.session_state.pop("_pending_restore_path", None)
                    else:
                        st.session_state.pop("_pending_restore_path", None)
                        st.session_state.pop("_edit_plan_cache", None)
                        st.cache_data.clear()
                        st.rerun()
            with c_no:
                if st.button("Cancelar", key="backup_restore_cancel"):
                    st.session_state.pop("_pending_restore_path", None)
                    st.rerun()
        elif not backups:
            st.info("Aún no hay respaldos; se crearán al guardar la planificación.")
        else:
            idx = st.selectbox(
                "Copias (más reciente arriba)",
                options=list(range(len(backups))),
                format_func=lambda i: backups[i]["label"],
                key="backup_select_idx",
            )
            if st.button("Restaurar copia seleccionada", key="backup_restore_btn"):
                st.session_state["_pending_restore_path"] = str(backups[idx]["path"])
                st.rerun()

    if solo:
        st.markdown(
            "<style>.block-container{padding-top:0.35rem;padding-bottom:0.15rem;max-width:100%;}</style>",
            unsafe_allow_html=True,
        )
        st.title("Cronograma de obra")
        st.caption(
            "Desactiva **Vista solo Gantt** en la barra lateral para volver a la tabla de planificación y a los botones."
        )
    else:
        st.title("GanttMachine — Planificación de obra (BC3 + L–V + CPM)")

    rows = build_planning_rows(
        result,
        state,
        selected,
        project_start,
        default_cuadrilla=default_crew,
    )
    df = pd.DataFrame(rows)

    if solo:
        cache = st.session_state.get("_edit_plan_cache")
        edited = pd.DataFrame(cache) if cache else df
        base_df = None
    else:
        st.subheader("Planificación — CPM en días laborables (L–V, 8 h)")
        st.caption(
            "Edita **Cuadrilla**, **Rend./día**, **Días lab.** o **Dur. partida** (solo partidas; ambas definen la misma duración) y **Predecesoras** (códigos separados por coma). "
            "En **capítulos/subcapítulos**, **Días lab.** se recalcula como suma de partidas (si lo cambias, se restaura al recalcular). "
            "Al **confirmar** una celda (**Enter**, **Tab** o clic fuera) con cambios en **Cuadrilla**, **Rend./día**, **Días lab.**, **Dur. partida** o **Predecesoras**, se **guarda** la planificación en `schedule_state.json`, se **limpia la caché** y se **actualiza toda la pantalla** (tabla alineada con CPM y Gantt), igual que **Guardar planificación**. "
        )
        cache = st.session_state.get("_edit_plan_cache")
        base_df = df
        if cache:
            try:
                cached_df = pd.DataFrame(cache)
                if list(cached_df.columns) == list(df.columns) and len(cached_df) == len(df):
                    base_df = cached_df
            except (TypeError, ValueError, KeyError):
                base_df = df
        edited = st.data_editor(
            base_df,
            column_config={
                "Código": st.column_config.TextColumn(disabled=True),
                "Capítulo": st.column_config.TextColumn(disabled=True),
                "Nivel": st.column_config.TextColumn(disabled=True),
                "Prof.": st.column_config.NumberColumn(disabled=True),
                "Unidad": st.column_config.TextColumn(disabled=True),
                "Descripción": st.column_config.TextColumn(disabled=True, width="large"),
                "Importe": st.column_config.TextColumn(disabled=True),
                "Cantidad": st.column_config.NumberColumn(disabled=True, format="%.4f"),
                "Cuadrilla": st.column_config.NumberColumn(min_value=0.5, max_value=50.0, step=0.5),
                "Rend./día": st.column_config.NumberColumn(min_value=0.1, max_value=5000.0, step=0.5),
                "Días lab.": st.column_config.NumberColumn(
                    min_value=1,
                    max_value=5000,
                    step=1,
                    help="Partidas: duración en días laborables (equivalente a Dur. partida). En capítulos/subcapítulos es agregado y se restaura al recalcular.",
                ),
                "Dur. partida": st.column_config.NumberColumn(
                    min_value=1,
                    max_value=5000,
                    step=1,
                    help="Solo partidas: misma duración que **Días lab.**; si editas una columna, la otra se alinea al guardar.",
                ),
                "Holgura": st.column_config.NumberColumn(disabled=True),
                "Crítica": st.column_config.TextColumn(disabled=True),
                "Predecesoras": st.column_config.TextColumn(width="medium"),
                "Inicio_lab": st.column_config.DateColumn(format="DD/MM/YYYY", disabled=True),
                "Fin_lab": st.column_config.DateColumn(format="DD/MM/YYYY", disabled=True),
                "Días": st.column_config.NumberColumn(disabled=True),
                "Inicio": st.column_config.DateColumn(disabled=True),
            },
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            height=min(560, 48 + 32 * min(len(df), 22)),
            key="editor_plan",
        )

    edited_for_merge = _harmonize_partida_duration(edited, base_df) if not solo else edited
    m_preview = merge_tasks_from_planning_rows(
        edited_for_merge.to_dict("records"),
        project_start,
        state.get("tasks"),
    )
    candidate_state = {**state, **m_preview, "default_cuadrilla": default_crew}
    plan_changed_on_disk = _persist_signature(candidate_state) != _persist_signature(state)
    table_edited = (not solo) and base_df is not None and (not _planning_editable_equals(edited, base_df))
    should_flush = plan_changed_on_disk or table_edited
    if should_flush:
        _save_state(candidate_state)
        state = candidate_state
        st.cache_data.clear()

    preview_state = {**state, "tasks": m_preview["tasks"], "project_start": m_preview["project_start"]}
    preview_state["default_cuadrilla"] = default_crew
    rows_gantt = build_planning_rows(
        result,
        preview_state,
        selected,
        project_start,
        default_cuadrilla=default_crew,
    )
    df_gantt = pd.DataFrame(rows_gantt)

    edited_aligned = _align_resumen_filas(edited, df_gantt)
    st.session_state["_edit_plan_cache"] = edited_aligned.to_dict("records")

    if should_flush:
        st.success(f"Guardado en `{STATE_FILE.name}`")
        st.rerun()

    if not solo:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Guardar planificación", type="primary"):
                new_state = {**state, **m_preview, "default_cuadrilla": default_crew}
                _save_state(new_state)
                st.success(f"Guardado en `{STATE_FILE.name}`")
                st.cache_data.clear()
                st.rerun()
        with c2:
            if st.button("Borrar duraciones guardadas (vista actual)"):
                tasks = dict(state.get("tasks", {}))
                for r in edited.to_dict("records"):
                    c = str(r["Código"])
                    if c in tasks:
                        tasks[c].pop("dias_lab", None)
                        tasks[c].pop("duration_days", None)
                _save_state(
                    {"project_start": project_start.isoformat(), "tasks": tasks, "default_cuadrilla": default_crew}
                )
                st.cache_data.clear()
                st.rerun()
        with c3:
            csv_bytes = df_gantt.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Descargar CSV (plan calculado)",
                csv_bytes,
                file_name="planificacion.csv",
                mime="text/csv",
            )

    if solo:
        csv_bytes = df_gantt.to_csv(index=False).encode("utf-8-sig")
        st.sidebar.download_button(
            label="Descargar CSV",
            data=csv_bytes,
            file_name="planificacion.csv",
            mime="text/csv",
            use_container_width=True,
        )

    gh_left, gh_right = st.columns([4, 2])
    with gh_left:
        if not solo:
            st.subheader("Diagrama de Gantt (fechas calendario; duración = días laborables)")
        else:
            st.markdown("#### Vista del cronograma")
    with gh_right:
        opc_v = st.radio(
            "Vista del Gantt",
            options=[
                "Todo (capítulos, subcapítulos y partidas)",
                "Solo capítulos",
                "Solo partidas",
            ],
            horizontal=True,
            key="gantt_vista_arbol",
            help="Resume el cronograma: solo filas de capítulo, solo partidas, o el árbol completo.",
        )
        if opc_v == "Solo partidas":
            modo_vista = "solo_partidas"
        elif opc_v == "Solo capítulos":
            modo_vista = "solo_capitulos"
        else:
            modo_vista = "completa"

    plot_src = df_gantt
    if modo_vista == "solo_capitulos":
        mask_c = plot_src["Nivel"].astype(str).str.strip() == "Capítulo"
        plot_src_f = plot_src.loc[mask_c].copy()
        if plot_src_f.empty:
            st.warning("No hay capítulos para mostrar en esta vista; se muestra la jerarquía completa.")
            plot_src_f = plot_src
        plot_src = plot_src_f
    elif modo_vista == "solo_partidas":
        mask_p = plot_src["Nivel"].astype(str).str.strip() == "Partida"
        plot_src_f = plot_src.loc[mask_p].copy()
        if plot_src_f.empty:
            st.warning("No hay partidas para mostrar en vista contraída; se muestra la jerarquía completa.")
            plot_src_f = plot_src
        plot_src = plot_src_f
    plot_df = plot_src.head(max_rows).copy() if len(plot_src) > max_rows else plot_src.copy()
    if len(plot_src) > max_rows:
        st.info(f"Gantt: **{max_rows}** de **{len(plot_src)}** filas. Sube el límite en la barra lateral.")

    chart_h = min(15000, max(900, 80 + 30 * len(plot_df))) if solo else min(2600, 160 + 24 * len(plot_df))
    fig = _build_gantt_figure(plot_df, chart_h, solo=solo)
    st.plotly_chart(fig, use_container_width=True, key="gantt_principal")

    if not solo:
        with st.expander("Cómo funciona el motor"):
            st.markdown(
                """
                - **Cantidad**: suma de ~M del BC3; si es **0**, se estima una cantidad representativa según **unidad** y **profundidad**
                  WBS (y el importe relativo) para autollenar **Días lab.** de forma aproximada pero coherente.
                - **Rend./día** y **Cuadrilla**: si no hay valores guardados para la partida, se **sugieren** a partir de la **descripción** del BC3 (oficios de obra, pintura, hormigón, fontanería, etc.), la **unidad** y el **importe relativo**. Puedes editarlos en la tabla; al guardarse en `schedule_state.json` dejan de aplicarse las sugerencias para esa partida.
                - **Días lab. / Dur. partida**: en **partidas** puedes editar **cualquiera de las dos** (misma duración en días laborables); al confirmar se alinean y se guarda en el estado. En capítulo/subcapítulo **Días lab.** es la **suma** de las partidas del tramo. Si no hay duración guardada, se calcula con cantidad/rendimiento/cuadrilla y un **ligero ajuste** según la descripción. Al **confirmar** la celda, se recalcula el CPM y se **autoguarda** si cambió el archivo.
                - **Capítulo / subcapítulo — Días lab.**: suma de los **Días lab.** de las **partidas** incluidas bajo ese nodo (esfuerzo total en jornadas). Las **fechas** de la fila siguen siendo el **periodo calendario** (del primer inicio al último fin) para la barra del Gantt.
                - **Predecesoras**: enlaces **FS** entre códigos de partida; si la celda está vacía, se usa la **secuencia
                  lineal** del presupuesto (orden DFS del BC3).
                - **CPM**: tempranos/tardíos en **días laborables**; **Holgura** y **Crítica** (holgura 0).
                - Puedes **descargar CSV** del plan ya calculado (mismo criterio que el Gantt).
                - **Vista solo Gantt**: amplía el cronograma; la última edición de la tabla se conserva en memoria al alternar la vista.
                - **Vista del gráfico** (junto al título del diagrama): **Todo** el árbol; **Solo capítulos** (resumen por capítulo); **Solo partidas** (detalle de partida).
                """
            )


if __name__ == "__main__":
    main()
