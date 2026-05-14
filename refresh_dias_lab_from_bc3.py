#!/usr/bin/env python3
"""
Recalcula dias_lab y duration_days en schedule_state.json para cada partida,
usando el mismo criterio que el motor (BC3 + descripción + cantidad + rend./cuadrilla guardados o sugeridos).
No modifica capítulos/subcapítulos en tasks ni predecesoras/rendimiento/cuadrilla.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from pathlib import Path

from bc3_parser import parse_bc3
from planning_engine import build_planning_rows

ROOT = Path(__file__).resolve().parent
BC3_FILE = ROOT / "GaudiVENTA.bc3"
STATE_FILE = ROOT / "schedule_state.json"


def _es_partida(code: str) -> bool:
    if re.fullmatch(r"\d{2}#", code):
        return False
    if code.endswith("#"):
        return False
    return True


def main() -> None:
    if not BC3_FILE.is_file():
        raise SystemExit(f"No existe {BC3_FILE}")
    if not STATE_FILE.is_file():
        raise SystemExit(f"No existe {STATE_FILE}")

    result = parse_bc3(BC3_FILE)
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    tasks = state.get("tasks") or {}
    if not isinstance(tasks, dict):
        raise SystemExit("tasks no es un objeto")

    state_tmp = copy.deepcopy(state)
    tmp_tasks = state_tmp.setdefault("tasks", {})
    for code, row in list(tmp_tasks.items()):
        if not isinstance(row, dict) or not _es_partida(str(code)):
            continue
        row.pop("dias_lab", None)
        row.pop("duration_days", None)

    try:
        project_start = date.fromisoformat(str(state.get("project_start", date.today().isoformat())))
    except ValueError:
        project_start = date.today()

    default_crew = float(state.get("default_cuadrilla", 3.0))
    roots = list(result.roots)

    rows = build_planning_rows(
        result,
        state_tmp,
        roots,
        project_start,
        default_cuadrilla=default_crew,
    )
    por_codigo = {str(r["Código"]): r for r in rows}

    n = 0
    for code, row in tasks.items():
        if not isinstance(row, dict) or not _es_partida(str(code)):
            continue
        gr = por_codigo.get(str(code))
        if not gr:
            continue
        dl = int(gr["Días lab."])
        row["dias_lab"] = dl
        row["duration_days"] = float(dl)
        n += 1

    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Actualizadas {n} partidas en {STATE_FILE.name}")


if __name__ == "__main__":
    main()
