"""KiCad board and schematic files to one design record.

Board frame as in registration's board_model and vision's pads: metres,
x = X_mm / 1000, y = -Y_mm / 1000, z out of the top face. Pad identity is
`<reference>.<pad number>`, e.g. R1.2, matching vision.pads. An untented via is `VIA<n>`,
n counting every via in file order from 1. Only copper reachable from the top is kept.
Nothing here comes from an image.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

_CIRCLE_POINTS = 24
_MPN_FIELDS = ("mpn", "manufacturer_part_number", "mfr part number", "mfr_pn", "part number")
_TESTPOINT = re.compile(r"^TP\d", re.IGNORECASE)


@dataclass(frozen=True)
class Pad:
    pad_id: str
    ref: str  # "" for a via
    number: str
    kind: str  # smd | thru_hole | via
    shape: str
    x: float
    y: float
    width: float
    height: float
    angle: float  # radians, board frame
    drill: float | None
    net: str
    pinfunction: str
    pintype: str
    mask_open: bool  # copper exposed on the top side
    outline: tuple  # ((x, y), ...) closed polygon, board frame


@dataclass
class Component:
    ref: str
    value: str = ""
    footprint: str = ""
    mpn: str = ""
    datasheet: str = ""
    description: str = ""
    layer: str = "F.Cu"
    x: float = 0.0
    y: float = 0.0
    rotation_deg: float = 0.0
    testpoint: bool = False
    dnp: bool = False
    fields: dict = field(default_factory=dict)


@dataclass
class Design:
    design_id: str  # hash of the board file; changes with any edit
    name: str
    pcb_path: str
    thickness_m: float | None
    components: dict  # ref -> Component
    pads: list
    schematic_paths: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def load_design(path: str | Path) -> Design:
    """A .kicad_pcb file, or a directory holding exactly one, plus every .kicad_sch beside it."""
    path = Path(path)
    if path.is_dir():
        boards = sorted(path.glob("*.kicad_pcb"))
        if len(boards) != 1:
            raise ValueError(f"expected one .kicad_pcb in {path}, found {len(boards)}")
        path = boards[0]
    if path.suffix != ".kicad_pcb":
        raise ValueError("expected a .kicad_pcb file or a KiCad project directory")
    raw = path.read_bytes()
    design = parse_pcb(raw.decode("utf-8"))
    design.design_id = hashlib.sha256(raw).hexdigest()[:16]
    design.name = path.stem
    design.pcb_path = str(path.resolve())
    for sheet in sorted(path.parent.glob("*.kicad_sch")):
        design.schematic_paths.append(str(sheet.resolve()))
        merge_schematic(design, sheet.read_text(encoding="utf-8"))
    if not design.schematic_paths:
        design.warnings.append("no_schematic: MPN and datasheet fields come from the board only")
    return design


def parse_pcb(text: str) -> Design:
    tree = parse_sexpr(text)
    root = next((n for n in tree if isinstance(n, list) and n and n[0] == "kicad_pcb"), None)
    if root is None:
        raise ValueError("not a kicad_pcb file")
    net_names = {item[1]: item[2] for item in children(root, "net") if len(item) > 2}
    design = Design("", "", "", _thickness(root), {}, [])
    default_tented = _tented(child(child(root, "setup") or [], "tenting"), True)

    vias = 0
    for item in root[1:]:
        if not isinstance(item, list) or not item:
            continue
        if item[0] in ("footprint", "module"):
            _footprint(item, net_names, design)
        elif item[0] == "via":
            vias += 1
            pad = _via(item, vias, net_names, default_tented)
            if pad is not None:
                design.pads.append(pad)
    return design


def merge_schematic(design: Design, text: str) -> None:
    """Copy MPN, datasheet and description fields onto the board's components by reference."""
    root = next((n for n in parse_sexpr(text) if isinstance(n, list) and n), None)
    if root is None or root[0] != "kicad_sch":
        design.warnings.append("schematic_unreadable")
        return
    for symbol in children(root, "symbol"):
        fields = _properties(symbol)
        references = [ref for ref in _instance_references(symbol) if ref]
        references = references or [fields.get("Reference", "")]
        for ref in references:
            component = design.components.get(ref)
            if component is None:
                continue
            component.fields.update({k: v for k, v in fields.items() if v and v != "~"})
            component.mpn = component.mpn or _mpn(fields)
            component.datasheet = component.datasheet or _clean(fields.get("Datasheet", ""))
            component.description = component.description or fields.get("Description", "")


# --- footprints and pads --------------------------------------------------------------


def _footprint(node: list, net_names: dict, design: Design) -> None:
    at = child(node, "at")
    fx, fy = float(at[1]), float(at[2])
    rotation = float(at[3]) if len(at) > 3 else 0.0
    fa = math.radians(rotation)
    fields = _properties(node)
    ref = fields.get("Reference", "?")
    attrs = set((child(node, "attr") or [])[1:])
    name = node[1] if len(node) > 1 and isinstance(node[1], str) else ""
    layer = (child(node, "layer") or [None, "F.Cu"])[1]
    component = Component(
        ref=ref,
        value=fields.get("Value", ""),
        footprint=name,
        mpn=_mpn(fields),
        datasheet=_clean(fields.get("Datasheet", "")),
        description=fields.get("Description", ""),
        layer=layer,
        x=fx / 1000.0,
        y=-fy / 1000.0,
        rotation_deg=rotation,
        testpoint=bool(_TESTPOINT.match(ref)) or "testpoint" in name.lower().replace("_", ""),
        dnp="dnp" in attrs or fields.get("DNP", "").lower() in ("1", "yes", "true", "dnp"),
        fields={k: v for k, v in fields.items() if v and v != "~"},
    )
    if ref in design.components:
        design.warnings.append(f"duplicate_reference:{ref}")
    design.components[ref] = component

    for pad in children(node, "pad"):
        number, kind, shape = pad[1], pad[2], pad[3]
        layers = (child(pad, "layers") or [None])[1:]
        if kind == "np_thru_hole" or not any(x in ("F.Cu", "*.Cu") for x in layers):
            continue
        pad_at = child(pad, "at")
        px, py = float(pad_at[1]), float(pad_at[2])
        # The pad's angle in the file already includes the footprint's rotation.
        pa = math.radians(float(pad_at[3])) if len(pad_at) > 3 else fa
        # KiCad's positive rotation is counter-clockwise on screen, in y-down millimetres.
        x = (fx + px * math.cos(fa) + py * math.sin(fa)) / 1000.0
        y = -(fy - px * math.sin(fa) + py * math.cos(fa)) / 1000.0
        size = child(pad, "size")
        w, h = float(size[1]) / 1000.0, float(size[2]) / 1000.0
        drill = child(pad, "drill")
        design.pads.append(
            Pad(
                pad_id=f"{ref}.{number}",
                ref=ref,
                number=str(number),
                kind="thru_hole" if kind == "thru_hole" else "smd",
                shape=shape,
                x=x,
                y=y,
                width=w,
                height=h,
                angle=pa,
                drill=_drill(drill),
                net=_net(child(pad, "net"), net_names),
                pinfunction=(child(pad, "pinfunction") or [None, ""])[1],
                pintype=(child(pad, "pintype") or [None, ""])[1],
                mask_open=any(x in ("F.Mask", "*.Mask") for x in layers),
                outline=_outline(shape, (x, y), (w, h), pa),
            )
        )


def _via(node: list, index: int, net_names: dict, default_tented: bool) -> Pad | None:
    layers = (child(node, "layers") or [None, "F.Cu", "B.Cu"])[1:]
    if "F.Cu" not in layers:
        return None  # blind or buried from the bottom
    if _tented(child(node, "tenting"), default_tented):
        return None
    at = child(node, "at")
    x, y = float(at[1]) / 1000.0, -float(at[2]) / 1000.0
    diameter = float(child(node, "size")[1]) / 1000.0
    return Pad(
        pad_id=f"VIA{index}",
        ref="",
        number=str(index),
        kind="via",
        shape="circle",
        x=x,
        y=y,
        width=diameter,
        height=diameter,
        angle=0.0,
        drill=_drill(child(node, "drill")),
        net=_net(child(node, "net"), net_names),
        pinfunction="",
        pintype="",
        mask_open=True,
        outline=_outline("circle", (x, y), (diameter, diameter), 0.0),
    )


def _tented(node, default: bool) -> bool:
    """Front-side tenting. KiCad <= 8 lists the tented sides, `(tenting front back)` or
    `(tenting none)`; KiCad 9 writes `(tenting (front yes) (back no))`."""
    if node is None:
        return default
    for item in node[1:]:
        if item == "front":
            return True
        if item == "none":
            return False
        if isinstance(item, list) and item and item[0] == "front":
            return len(item) < 2 or item[1] == "yes"
    return False if len(node) > 1 else default


def _outline(shape: str, center, size, angle: float) -> tuple:
    cx, cy = center
    w, h = size
    if shape == "circle":
        r = w / 2
        steps = [2 * math.pi * i / _CIRCLE_POINTS for i in range(_CIRCLE_POINTS)]
        return tuple((cx + r * math.cos(a), cy + r * math.sin(a)) for a in steps)
    if shape == "oval" and w != h:
        r, half = min(w, h) / 2, (max(w, h) - min(w, h)) / 2
        n = _CIRCLE_POINTS // 2
        arc = [-math.pi / 2 + math.pi * i / (n - 1) for i in range(n)]
        right = [(half + r * math.cos(a), r * math.sin(a)) for a in arc]
        local = right + [(-px, -py) for px, py in right]
        if h > w:
            local = [(-py, px) for px, py in local]
    else:  # rect, roundrect, trapezoid; a custom pad uses its anchor size
        local = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    c, s = math.cos(angle), math.sin(angle)
    return tuple((cx + px * c - py * s, cy + px * s + py * c) for px, py in local)


# --- fields -----------------------------------------------------------------------------


def _properties(node: list) -> dict:
    fields = {}
    for item in node[1:]:
        if not (isinstance(item, list) and len(item) > 2):
            continue
        if item[0] == "property":
            fields[str(item[1])] = str(item[2])
        elif item[0] == "fp_text" and item[1] in ("reference", "value"):
            fields.setdefault(item[1].capitalize(), str(item[2]))
    return fields


def _instance_references(symbol: list) -> list:
    """References from a symbol's `(instances (project ... (path ... (reference R1))))`."""
    found = []

    def walk(node):
        for item in node[1:]:
            if isinstance(item, list) and item:
                if item[0] == "reference" and len(item) > 1:
                    found.append(item[1])
                else:
                    walk(item)

    instances = child(symbol, "instances")
    if instances is not None:
        walk(instances)
    return found


def _mpn(fields: dict) -> str:
    for key, value in fields.items():
        if key.lower().replace("-", "_") in _MPN_FIELDS and value and value != "~":
            return value
    return ""


def _clean(value: str) -> str:
    return "" if value in ("", "~") else value


def _net(node, names: dict) -> str:
    if node is None or len(node) < 2:
        return ""
    # KiCad <= 9: (net 3 "/3V3"); newer files may drop the number: (net "/3V3").
    return node[2] if len(node) > 2 else names.get(node[1], node[1])


def _drill(node) -> float | None:
    if node is None:
        return None
    values = [v for v in node[1:] if isinstance(v, str) and _is_number(v)]
    return float(values[0]) / 1000.0 if values else None


def _thickness(root: list) -> float | None:
    general = child(root, "general")
    thickness = child(general, "thickness") if general else None
    return float(thickness[1]) / 1000.0 if thickness else None


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


# --- s-expressions ----------------------------------------------------------------------


def parse_sexpr(text: str) -> list:
    stack: list = [[]]
    for token in re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text):
        if token == "(":
            stack.append([])
        elif token == ")":
            node = stack.pop()
            stack[-1].append(node)
        else:
            stack[-1].append(token[1:-1].replace('\\"', '"') if token.startswith('"') else token)
    return stack[0]


def child(node: list, name: str):
    for item in node[1:]:
        if isinstance(item, list) and item and item[0] == name:
            return item
    return None


def children(node: list, name: str) -> list:
    return [item for item in node[1:] if isinstance(item, list) and item and item[0] == name]
