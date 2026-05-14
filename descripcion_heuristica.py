"""
Heurísticas de cuadrilla, rendimiento/día y factor de duración a partir del texto
de la partida (presupuesto BC3, descripciones en español/catalán comunes).
No sustituye datos ya guardados en schedule_state.json para esa partida.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


def _fold(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


# Frases clave (texto ya normalizado sin acentos). Orden: de más específico a más general.
# Cada regla: (tupla de subcadenas a buscar), mult_rend, mult_cuadrilla, mult_dias (sobre duración ya calculada)
_REGLAS: list[tuple[tuple[str, ...], float, float, float]] = [
    (("demolicion", "demoler", "desescombro", "derribo", "bajada de", "retirada de escombros"), 0.65, 1.4, 1.12),
    (("excavacion", "movimiento de tierras", "terraplen", "zanja", "relleno compactado", "carga y acarreo"), 1.2, 1.15, 1.05),
    (("hormigon armado", "h a ", " e h a ", "encofrado", "armadura pasiva", "losas de forjado"), 0.72, 1.3, 1.08),
    (("hormigon", "hormigonado", "hormigon en masa", "masa de hm", "solera de hormigon"), 0.82, 1.2, 1.0),
    (("microcemento", "micro mort", "resina epoxi"), 1.15, 0.9, 1.05),
    (("pintura", "pintar", "enlucido", "alisado", "pintado", "imprimacion", "laca", "esmalte"), 1.38, 0.88, 1.0),
    (("impermeabilizacion", "bentonita", "geomembrana", "lamina asfaltica", "lamina betuminosa"), 0.95, 1.15, 1.06),
    (("aislamiento termico", "lana mineral", "lana de roca", "eps", "xps", "poliestireno extruido", "lana de vidrio"), 1.12, 1.05, 1.0),
    (("tabique", "ladrillo", "bloque de hormigon", "mamposteria", "tabiqueria", "tabica"), 0.58, 1.18, 1.0),
    (("mortero monocapa", "monocapa", "revoco", "revestimiento continuo"), 0.92, 1.05, 1.0),
    (("fontaneria", "fontanero", "saneamiento", "tuberia de evacuacion", "bajantes pvc", "suministro de agua"), 1.05, 1.12, 1.0),
    (("electricidad", "cableado", "iluminacion", "cuadro electrico", "canalizacion electrica", "ict"), 1.08, 1.05, 1.0),
    (("carpinteria", "carpinteria de madera", "puerta interior", "ventana pvc", "ventana de aluminio", "cerramiento"), 0.88, 1.08, 1.02),
    (("estructura metalica", "perfil metalico", "soldadura", "aceros", "chapa galvanizada", "cerramiento metalico"), 0.92, 1.22, 1.05),
    (("fachada ventilada", "trasdosado", "panel composite", "cortina de fachada", "revestimiento de fachada"), 0.78, 1.2, 1.1),
    (("cubierta plana", "impermeabilizacion de cubierta", "aislamiento de cubierta", "teja ceramica", "teja plana"), 0.9, 1.15, 1.05),
    (("solado", "pavimento", "baldosa", "gres porcelanico", "adoquin", "firme"), 0.95, 1.1, 1.0),
    (("cimentacion", "micropilote", "pilote", "pilotaje", "zapata corrida", "zapata aislada"), 0.85, 1.25, 1.08),
    (("tierra vegetal", "vegetal", "siembra", "cesped", "plantacion"), 1.1, 0.95, 1.0),
    (("andamio", "altura", "trabajo en vertical", "medidas de seguridad en altura"), 0.92, 1.15, 1.12),
]


def _match_reglas(texto_fold: str) -> tuple[float, float, float]:
    for frases, mr, mc, md in _REGLAS:
        if any(f in texto_fold for f in frases):
            return mr, mc, md
    return 1.0, 1.0, 1.0


def _boost_importe(importe: float, importe_max: float) -> tuple[float, float]:
    """Ligero incremento de cuadrilla y rendimiento en partidas de mayor importe relativo."""
    if importe_max <= 0 or importe <= 0:
        return 1.0, 1.0
    rel = min(1.0, importe / importe_max)
    r = math.sqrt(rel)
    crew_b = 1.0 + 0.22 * r
    rend_b = 1.0 + 0.12 * r
    return rend_b, crew_b


def _ajuste_unidad(unit_norm: str, mr: float, mc: float) -> tuple[float, float]:
    """Evita rendimientos absurdos en unidades poco habituales."""
    if unit_norm in ("h", "ud"):
        mr = (mr + 1.0) / 2.0
    if unit_norm in ("t", "kg"):
        mc = min(mc, 1.35)
        mr = max(0.85, mr * 0.95)
    return mr, mc


def inferencia_partida_desde_texto(
    descripcion: str,
    unit_norm: str,
    rend_base_unidad: float,
    cuadrilla_base: float,
    importe: float,
    importe_max: float,
) -> tuple[float, float, float]:
    """
    Devuelve (rendimiento_día sugerido, cuadrilla sugerida, multiplicador de días sobre la duración calculada).
    """
    t = _fold(descripcion)
    mr, mc, md = _match_reglas(t)
    br, bc = _boost_importe(importe, importe_max)
    mr, mc = _ajuste_unidad(unit_norm, mr * br, mc * bc)

    rend = max(0.15, min(4000.0, rend_base_unidad * mr))
    crew = max(0.5, min(14.0, cuadrilla_base * mc))
    crew = round(crew * 2.0) / 2.0
    rend = round(rend, 2)
    md = max(0.92, min(1.28, md))
    return rend, crew, md


def tarea_tiene_valor_explicito(sv: dict[str, Any], campo: str) -> bool:
    """True si en el estado guardado hay un valor explícito (no vacío) para ese campo."""
    if campo not in sv:
        return False
    v = sv[campo]
    if v is None:
        return False
    if isinstance(v, str) and not str(v).strip():
        return False
    try:
        if float(v) <= 0:
            return False
    except (TypeError, ValueError):
        return True
    return True
