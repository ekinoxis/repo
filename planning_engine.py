"""
Planificación de obra: calendario laboral (L–V), duraciones por cantidad/rendimiento/cuadrilla,
importancia vía importe, predecesoras (FS) y camino crítico (CPM en días laborables).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from datetime import date, timedelta
from typing import Any

import pandas as pd

from bc3_parser import ParseResult, flatten_subtree
from descripcion_heuristica import inferencia_partida_desde_texto, tarea_tiene_valor_explicito


def normalize_unit(u: str) -> str:
    s = (u or "").strip().lower().replace("²", "2").replace("³", "3").replace(" ", "")
    s = s.replace("m2", "m²").replace("m^2", "m²")
    if s in ("m²", "mq"):
        return "m²"
    if s in ("ml", "m.l.", "mlineal", "mlineales"):
        return "ml"
    if s in ("m", "metro", "metros"):
        return "m"
    if s in ("ud", "u", "u.", "un", "unidad", "unidades"):
        return "ud"
    if s in ("kg",):
        return "kg"
    if s in ("t", "tn", "ton", "tonelada"):
        return "t"
    if s in ("m3", "m³"):
        return "m³"
    if s in ("h", "hora", "horas"):
        return "h"
    if s in ("p.a.", "pa", "p"):
        return "ud"
    return s or "ud"


RENDIMIENTO_POR_UNIDAD: dict[str, float] = {
    "m²": 25.0,
    "ml": 40.0,
    "m": 35.0,
    "ud": 8.0,
    "kg": 500.0,
    "t": 3.0,
    "m³": 12.0,
    "h": 8.0,
}


def default_rendimiento(unit_norm: str) -> float:
    return RENDIMIENTO_POR_UNIDAD.get(unit_norm, RENDIMIENTO_POR_UNIDAD["ud"])


def parse_importe(s: str) -> float:
    if not s:
        return 0.0
    t = str(s).strip().replace("€", "").replace("EUR", "").strip()
    try:
        t = t.replace(" ", "").replace(",", ".")
        return float(t)
    except ValueError:
        return 0.0


def factor_importancia(importe: float, importe_max: float) -> float:
    if importe_max <= 0:
        return 1.0
    r = importe / importe_max
    return 1.0 + min(0.6, 0.5 * math.sqrt(max(0.0, min(1.0, r))))


def duration_work_days(
    cantidad: float,
    rendimiento_dia: float,
    cuadrilla: float,
    importe: float,
    importe_max: float,
) -> int:
    if cantidad <= 0 or rendimiento_dia <= 0 or cuadrilla <= 0:
        return 1
    f_imp = factor_importancia(importe, importe_max)
    prod = rendimiento_dia * cuadrilla * f_imp
    return max(1, math.ceil(cantidad / prod))


def cantidad_heuristica_sin_medicion(
    unit_norm: str,
    profundidad: int,
    importe: float,
    importe_max: float,
) -> float:
    """
    Orden de magnitud de cantidad cuando no hay medición (~M = 0), según unidad y profundidad WBS.
    Partidas más profundas se asumen más desagregadas (menor volumen representativo medio).
    """
    prof_adj = max(0, min(int(profundidad), 10))
    bases: dict[str, float] = {
        "m²": 72.0,
        "m": 48.0,
        "ml": 60.0,
        "m³": 16.0,
        "ud": 10.0,
        "kg": 2000.0,
        "t": 3.5,
        "h": 32.0,
    }
    base = bases.get(unit_norm, 8.0)
    escala_prof = 1.0 / (1.0 + 0.16 * prof_adj)
    q = base * escala_prof
    if importe_max > 0 and importe > 0:
        rel = min(1.0, importe / importe_max)
        ref = 35.0 * (rel**0.55)
        q = max(q, ref)
    return max(0.5, q)


def dias_partida_autofill(
    cantidad_medicion: float,
    unit_norm: str,
    profundidad: int,
    rendimiento_dia: float,
    cuadrilla: float,
    importe: float,
    importe_max: float,
) -> int:
    """Días laborables: medición real si existe; si no, heurística unidad + profundidad + importe."""
    sin_med = cantidad_medicion <= 0
    q_use = (
        cantidad_medicion
        if not sin_med
        else cantidad_heuristica_sin_medicion(unit_norm, profundidad, importe, importe_max)
    )
    d = duration_work_days(q_use, rendimiento_dia, cuadrilla, importe, importe_max)
    if sin_med:
        piso_prof = 1 + min(5, max(0, int(profundidad)) // 2)
        piso_imp = 0
        if importe > 0:
            piso_imp = min(14, max(1, int(round(math.log1p(importe) / 3500.0))))
        d = max(d, piso_prof, piso_imp)
    return min(180, max(1, d))


def next_weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def workdays_from(project_start: date):
    d = next_weekday(project_start)
    while True:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def nth_workday(project_start: date, index: int) -> date:
    if index < 0:
        index = 0
    for i, d in enumerate(workdays_from(project_start)):
        if i == index:
            return d
    return next_weekday(project_start)


def last_workday_of_task(project_start: date, es_index: int, duration_wd: int) -> date:
    last_index = es_index + max(1, duration_wd) - 1
    return nth_workday(project_start, last_index)


def count_workdays_inclusive(a: date, b: date) -> int:
    if b < a:
        return 0
    n = 0
    d = a
    while d <= b:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


def default_linear_predecessors(partida_codes: list[str]) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    prev: str | None = None
    for c in partida_codes:
        preds[c] = [prev] if prev else []
        prev = c
    return preds


def parse_predecessors_str(s: str) -> list[str]:
    if not s or not str(s).strip():
        return []
    parts = re.split(r"[,;\s]+", str(s).strip())
    return [p for p in parts if p]


def topological_sort(nodes: list[str], preds: dict[str, list[str]]) -> list[str] | None:
    nodes_set = set(nodes)
    indeg: dict[str, int] = {n: 0 for n in nodes}
    for n in nodes:
        for p in preds.get(n, []):
            if p in nodes_set:
                indeg[n] += 1
    q = deque([n for n in nodes if indeg[n] == 0])
    out: list[str] = []
    while q:
        n = q.popleft()
        out.append(n)
        for m in nodes:
            if n in preds.get(m, []) and m in nodes_set:
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)
    if len(out) != len(nodes):
        return None
    return out


def validate_fs_graph(nodes: list[str], preds: dict[str, list[str]]) -> bool:
    """True si el grafo FS (predecesoras) es acíclico."""
    return topological_sort(nodes, preds) is not None


def build_successors(nodes: list[str], preds: dict[str, list[str]]) -> dict[str, list[str]]:
    succ: dict[str, list[str]] = defaultdict(list)
    nodes_set = set(nodes)
    for n in nodes:
        for p in preds.get(n, []):
            if p in nodes_set:
                succ[p].append(n)
    return dict(succ)


def run_cpm(
    nodes: list[str],
    duration: dict[str, int],
    preds: dict[str, list[str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, int], set[str]] | None:
    topo = topological_sort(nodes, preds)
    if topo is None:
        return None
    nodes_set = set(nodes)
    ES: dict[str, int] = {}
    EF: dict[str, int] = {}
    for n in topo:
        ps = [p for p in preds.get(n, []) if p in nodes_set and p in duration]
        es = max((EF[p] for p in ps), default=0)
        ES[n] = es
        EF[n] = es + duration.get(n, 1)

    succ = build_successors(nodes, preds)
    LS: dict[str, int] = {}
    LF: dict[str, int] = {}
    for n in reversed(topo):
        sc = [s for s in succ.get(n, []) if s in nodes_set]
        if not sc:
            LF[n] = EF[n]
        else:
            LF[n] = min(LS[s] for s in sc)
        LS[n] = LF[n] - duration.get(n, 1)

    slack = {n: LS[n] - ES[n] for n in nodes}
    crit = {n for n in nodes if slack[n] == 0}
    return ES, EF, LS, LF, slack, crit


def merge_task_state(base: dict[str, Any], code: str, defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    prev = base.get(code)
    if isinstance(prev, dict):
        for k, v in prev.items():
            if v is not None and v != "":
                out[k] = v
    return out


def _invert_parent_map(children: dict[str, list[str]]) -> dict[str, str]:
    pmap: dict[str, str] = {}
    for parent, chs in children.items():
        for c in chs:
            pmap[c] = parent
    return pmap


def _root_chapter(code: str, pmap: dict[str, str]) -> str:
    cur = code
    while cur and not re.fullmatch(r"\d{2}#", cur):
        cur = pmap.get(cur, "")
    return cur or ""


def _wbs_depth(code: str, pmap: dict[str, str]) -> int:
    d = 0
    cur = code
    while pmap.get(cur):
        d += 1
        cur = pmap[cur]
    return d


def _nivel_wbs(code: str) -> str:
    if re.fullmatch(r"\d{2}#", code):
        return "Capítulo"
    if code.endswith("#"):
        return "Subcapítulo"
    return "Partida"


def _descendant_partidas_codes(root: str, result: ParseResult, codes_set: set[str]) -> list[str]:
    """Partidas hoja bajo `root` (sin contar el propio root), respetando el subárbol seleccionado."""
    acc: list[str] = []
    stack = list(result.children.get(root, []) or [])
    while stack:
        ch = stack.pop()
        if ch not in codes_set:
            continue
        if _nivel_wbs(ch) == "Partida" and not ch.startswith("%"):
            acc.append(ch)
        else:
            stack.extend(result.children.get(ch, []) or [])
    return acc


def build_planning_rows(
    result: ParseResult,
    state: dict,
    selected_roots: list[str],
    project_start: date,
    default_cuadrilla: float = 3.0,
) -> list[dict[str, Any]]:
    pmap = _invert_parent_map(result.children)
    codes = flatten_subtree(result, selected_roots)
    codes_set = set(codes)
    tasks_st: dict[str, Any] = dict(state.get("tasks", {}))

    partida_codes = [c for c in codes if _nivel_wbs(c) == "Partida" and not c.startswith("%")]
    partida_set = set(partida_codes)
    importes = {c: parse_importe(result.concepts[c].amount) for c in partida_codes}
    imax = max(importes.values()) if importes else 1.0

    duration: dict[str, int] = {}
    rend_map: dict[str, float] = {}
    crew_map: dict[str, float] = {}

    for c in partida_codes:
        sv = tasks_st.get(c, {})
        con = result.concepts[c]
        qty = float(result.mediciones.get(c, 0.0))
        unit_n = normalize_unit(con.unit)
        base_rend = default_rendimiento(unit_n)
        imp_c = importes.get(c, 0.0)
        ir, ic, dm = inferencia_partida_desde_texto(
            con.description, unit_n, base_rend, default_cuadrilla, imp_c, imax
        )
        if tarea_tiene_valor_explicito(sv, "rendimiento"):
            rend = float(sv["rendimiento"])
        else:
            rend = ir
        if tarea_tiene_valor_explicito(sv, "cuadrilla"):
            crew = float(sv["cuadrilla"])
        else:
            crew = ic
        if rend <= 0:
            rend = base_rend
        if crew <= 0:
            crew = default_cuadrilla
        rend_map[c] = rend
        crew_map[c] = crew
        prof_c = _wbs_depth(c, pmap)

        if sv.get("dias_lab") is not None and str(sv.get("dias_lab", "")).strip() != "":
            duration[c] = max(1, int(float(sv["dias_lab"])))
        elif sv.get("duration_days") is not None:
            try:
                duration[c] = max(1, int(round(float(sv["duration_days"]))))
            except (TypeError, ValueError):
                d0 = dias_partida_autofill(qty, unit_n, prof_c, rend, crew, imp_c, imax)
                duration[c] = max(1, min(180, int(math.ceil(d0 * dm))))
        else:
            d0 = dias_partida_autofill(qty, unit_n, prof_c, rend, crew, imp_c, imax)
            duration[c] = max(1, min(180, int(math.ceil(d0 * dm))))

    linear = default_linear_predecessors(partida_codes)
    preds: dict[str, list[str]] = {}
    for c in partida_codes:
        user_p = parse_predecessors_str(str(tasks_st.get(c, {}).get("predecesoras", "")))
        user_p = [p for p in user_p if p in partida_set]
        preds[c] = user_p if user_p else linear.get(c, [])

    cpm_out = run_cpm(partida_codes, duration, preds)
    if cpm_out is None:
        preds = linear
        cpm_out = run_cpm(partida_codes, duration, preds)
    ES, EF, LS, LF, slack, crit = cpm_out if cpm_out else ({}, {}, {}, {}, {}, set())

    partida_start: dict[str, date] = {}
    partida_end: dict[str, date] = {}
    for c in partida_codes:
        es = ES.get(c, 0)
        du = duration.get(c, 1)
        partida_start[c] = nth_workday(project_start, es)
        partida_end[c] = last_workday_of_task(project_start, es, du)

    memo: dict[str, tuple[date, date]] = {}

    def rollup(code: str) -> tuple[date, date]:
        if code in memo:
            return memo[code]
        if _nivel_wbs(code) == "Partida":
            a, b = partida_start[code], partida_end[code]
            memo[code] = (a, b)
            return a, b
        chs = [ch for ch in result.children.get(code, []) if ch in codes_set]
        if not chs:
            d0 = next_weekday(project_start)
            memo[code] = (d0, d0)
            return d0, d0
        rs = [rollup(ch) for ch in chs]
        a = min(x[0] for x in rs)
        b = max(x[1] for x in rs)
        memo[code] = (a, b)
        return a, b

    rows: list[dict[str, Any]] = []
    for code in codes:
        con = result.concepts.get(code)
        if not con or code.startswith("%"):
            continue
        cap = _root_chapter(code, pmap) or code
        nv = _nivel_wbs(code)
        prof = _wbs_depth(code, pmap)
        desc = con.description[:240] + ("…" if len(con.description) > 240 else "")
        qty = float(result.mediciones.get(code, 0.0))
        imp = parse_importe(con.amount)
        s_lab, e_lab = rollup(code)
        if nv == "Partida":
            dias_lab = duration.get(code, 1)
        else:
            hojas = _descendant_partidas_codes(code, result, codes_set)
            dias_sum = sum(duration.get(c, 1) for c in hojas)
            dias_lab = (
                dias_sum
                if dias_sum > 0
                else max(1, count_workdays_inclusive(s_lab, e_lab))
            )

        pred_txt = ",".join(preds.get(code, [])) if code in preds else ""
        sl = slack.get(code, 0) if code in slack else 0
        cr = "Sí" if code in crit else "No"

        row = {
            "Código": code,
            "Capítulo": cap,
            "Nivel": nv,
            "Prof.": prof,
            "Unidad": con.unit,
            "Descripción": desc,
            "Importe": con.amount,
            "Cantidad": qty,
            "Cuadrilla": crew_map.get(code, default_cuadrilla) if nv == "Partida" else None,
            "Rend./día": rend_map.get(code) if nv == "Partida" else None,
            "Días lab.": dias_lab,
            "Dur. partida": int(dias_lab) if nv == "Partida" else None,
            "Holgura": sl if nv == "Partida" else None,
            "Crítica": cr if nv == "Partida" else "",
            "Predecesoras": pred_txt if nv == "Partida" else "",
            "Inicio_lab": s_lab,
            "Fin_lab": e_lab,
            "Días": float(dias_lab),
            "Inicio": s_lab,
            "_gantt_start": s_lab,
            "_gantt_end": e_lab,
        }
        rows.append(row)
    return rows


def _dias_lab_desde_fila(r: dict[str, Any]) -> int:
    """Duración en días laborables: prioriza la columna editable de partida, luego Días lab."""
    for k in (
        "Dur. partida",
        "Dur partida",
        "Días lab.",
        "Días lab",
        "Dias lab.",
        "dias_lab",
        "Días",
    ):
        if k not in r:
            continue
        dl = r[k]
        if dl is None:
            continue
        try:
            if pd.isna(dl):
                continue
        except TypeError:
            pass
        if isinstance(dl, float) and math.isnan(dl):
            continue
        try:
            return max(1, int(round(float(dl))))
        except (TypeError, ValueError):
            continue
    return 1


def task_entry_from_planning_row(r: dict[str, Any]) -> dict[str, Any]:
    def _blank(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str) and not str(v).strip():
            return True
        try:
            if pd.isna(v):
                return True
        except TypeError:
            pass
        if isinstance(v, float) and math.isnan(v):
            return True
        return False

    ini = r.get("Inicio_lab") or r.get("Inicio")
    if hasattr(ini, "isoformat"):
        start_s = ini.isoformat()
    else:
        start_s = str(ini)
    dl_int = _dias_lab_desde_fila(r)
    entry: dict[str, Any] = {
        "start_date": start_s,
        "duration_days": float(dl_int),
        "dias_lab": dl_int,
        "predecesoras": str(r.get("Predecesoras", "")),
    }
    if not _blank(r.get("Rend./día")):
        try:
            entry["rendimiento"] = float(r["Rend./día"])
        except (TypeError, ValueError):
            pass
    if not _blank(r.get("Cuadrilla")):
        try:
            entry["cuadrilla"] = float(r["Cuadrilla"])
        except (TypeError, ValueError):
            pass
    return entry


def merge_tasks_from_planning_rows(
    rows: list[dict[str, Any]],
    project_start: date,
    existing_tasks: dict[str, Any] | None,
) -> dict[str, Any]:
    tasks = dict(existing_tasks or {})
    for r in rows:
        if str(r.get("Nivel", "")).strip() != "Partida":
            continue
        code = str(r.get("Código", "")).strip()
        if not code or code.startswith("%"):
            continue
        prev = tasks.get(code, {})
        if not isinstance(prev, dict):
            prev = {}
        prev.update(task_entry_from_planning_row(r))
        tasks[code] = prev
    return {"project_start": project_start.isoformat(), "tasks": tasks}
