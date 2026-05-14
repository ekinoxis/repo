"""
Lector mínimo FIEBDC-3 (Presto) para GanttMachine: conceptos (~C), jerarquía (~D),
textos largos (~T, ~A) y registros multilínea.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Concept:
    code: str
    unit: str
    description: str
    amount: str
    fecha: str


@dataclass
class ParseResult:
    concepts: dict[str, Concept] = field(default_factory=dict)
    """Código canónico (~C) -> concepto."""
    children: dict[str, list[str]] = field(default_factory=dict)
    """Padre canónico -> hijos canónicos (orden aparición en ~D)."""
    roots: list[str] = field(default_factory=list)
    """Capítulos de obra ^\\d{2}#$ en orden de aparición."""
    wbs_order: list[str] = field(default_factory=list)
    """DFS desde cada raíz; incluye repetidos si varios padres (dedupe en app)."""
    mediciones: dict[str, float] = field(default_factory=dict)
    """Cantidad total medida por código de partida (~M), si existe."""


def _iter_physical_lines(path: Path, encoding: str) -> Iterator[str]:
    with path.open(encoding=encoding, errors="replace") as f:
        for line in f:
            yield line.rstrip("\r\n")


def _iter_records(path: Path, encoding: str = "cp1252") -> Iterator[str]:
    """Une líneas de continuación (no empiezan por ~) al registro anterior."""
    buf: list[str] = []
    for line in _iter_physical_lines(path, encoding):
        if line.startswith("~"):
            if buf:
                yield "".join(buf)
            buf = [line]
        else:
            if buf:
                buf.append("\n")
                buf.append(line)
    if buf:
        yield "".join(buf)


def _split_record(record: str) -> tuple[str, list[str]]:
    parts = record.split("|")
    if not parts or not parts[0].startswith("~"):
        return "", parts
    rt = parts[0][1:]  # ~C -> C
    return rt, parts


def _is_wbs_child_code(code: str) -> bool:
    if not code:
        return False
    if code[0].isalpha():
        return False
    return code[0].isdigit()


def _canonical_code(raw: str, concept_codes: set[str]) -> str:
    if raw in concept_codes:
        return raw
    if raw + "#" in concept_codes:
        return raw + "#"
    return raw


def _parse_d_payload(payload: str) -> list[tuple[str, str, str]]:
    chunks = payload.split("\\")
    out: list[tuple[str, str, str]] = []
    for i in range(0, len(chunks) - 2, 3):
        a, b, c = chunks[i], chunks[i + 1], chunks[i + 2]
        if not a:
            continue
        out.append((a, b, c))
    return out


def _is_chapter_root(code: str) -> bool:
    return bool(re.fullmatch(r"\d{2}#", code))


def parse_bc3(path: Path, encoding: str = "cp1252") -> ParseResult:
    concepts: dict[str, Concept] = {}
    long_text: dict[str, str] = {}
    append_desc: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    roots_order: list[str] = []
    mediciones: dict[str, float] = {}

    for record in _iter_records(path, encoding):
        rt, parts = _split_record(record)
        if not rt:
            continue
        if rt == "C" and len(parts) >= 5:
            code = parts[1]
            unit = parts[2] if len(parts) > 2 else ""
            desc = parts[3] if len(parts) > 3 else ""
            amount = parts[4] if len(parts) > 4 else ""
            fecha = parts[5] if len(parts) > 5 else ""
            concepts[code] = Concept(
                code=code, unit=unit, description=desc, amount=amount, fecha=fecha
            )
            if _is_chapter_root(code) and code not in roots_order:
                roots_order.append(code)
        elif rt == "A" and len(parts) >= 3:
            code, frag = parts[1], parts[2]
            prev = append_desc.get(code, "")
            append_desc[code] = (prev + " " if prev else "") + frag.replace("\\", " ")
        elif rt == "T" and len(parts) >= 3:
            code = parts[1]
            body = "|".join(parts[2:])
            long_text[code] = long_text.get(code, "") + ("\n\n" if code in long_text else "") + body
        elif rt == "D" and len(parts) >= 3:
            parent = parts[1]
            payload = parts[2]
            concept_codes = set(concepts.keys())
            seen_local: set[str] = set()
            for raw, _coef, _qty in _parse_d_payload(payload):
                if not _is_wbs_child_code(raw):
                    continue
                child = _canonical_code(raw, concept_codes)
                if child in seen_local:
                    continue
                seen_local.add(child)
                children.setdefault(parent, []).append(child)
        elif rt == "M" and len(parts) >= 4:
            scope = parts[1]
            if "\\" in scope:
                _cap, sub = scope.split("\\", 1)
                concept_codes_live = set(concepts.keys())
                cod = _canonical_code(sub, concept_codes_live)
                try:
                    q = float(parts[3].replace(",", "."))
                except (ValueError, TypeError, IndexError):
                    q = 0.0
                if cod in concepts and q > 0:
                    mediciones[cod] = mediciones.get(cod, 0.0) + q

    concept_codes = set(concepts.keys())
    for code, frag in append_desc.items():
        if code in concepts:
            concepts[code].description = (concepts[code].description + " " + frag).strip()
    for code, txt in long_text.items():
        if code in concepts and len(txt.strip()) > len(concepts[code].description):
            concepts[code].description = txt.strip()

    wbs_order: list[str] = []
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node not in concept_codes:
            return
        if node not in visited:
            wbs_order.append(node)
            visited.add(node)
        for ch in children.get(node, []):
            dfs(ch)

    for r in roots_order:
        dfs(r)

    for code in concepts:
        if code.startswith("%"):
            continue
        if code not in visited and not _is_chapter_root(code):
            wbs_order.append(code)

    return ParseResult(
        concepts=concepts,
        children=children,
        roots=roots_order,
        wbs_order=wbs_order,
        mediciones=mediciones,
    )


def flatten_subtree(result: ParseResult, root_codes: list[str]) -> list[str]:
    """Orden DFS limitado a capítulos seleccionados (códigos ^\\d{2}#$)."""
    out: list[str] = []
    seen: set[str] = set()
    concept_codes = set(result.concepts.keys())

    def dfs(node: str) -> None:
        if node not in concept_codes or node.startswith("%"):
            return
        if node not in seen:
            out.append(node)
            seen.add(node)
        for ch in result.children.get(node, []):
            dfs(ch)

    for r in root_codes:
        dfs(r)
    return out
