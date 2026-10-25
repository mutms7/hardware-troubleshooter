"""SQLite store of ingested designs, with exact lookups.

Coordinates, nets and pin maps come from here and nowhere else: never from text search.
One design is active at a time; ingesting makes the new design active.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .kicad import Component, Design, Pad

_SCHEMA = """
CREATE TABLE IF NOT EXISTS designs (
  design_id TEXT PRIMARY KEY, name TEXT, pcb_path TEXT, thickness_m REAL,
  schematics TEXT, warnings TEXT, ingested_at REAL, active INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS components (
  design_id TEXT, ref TEXT, value TEXT, footprint TEXT, mpn TEXT, datasheet TEXT,
  description TEXT, layer TEXT, x REAL, y REAL, rotation_deg REAL, testpoint INTEGER,
  dnp INTEGER, fields TEXT, PRIMARY KEY (design_id, ref)
);
CREATE TABLE IF NOT EXISTS pads (
  design_id TEXT, pad_id TEXT, ref TEXT, number TEXT, kind TEXT, shape TEXT,
  x REAL, y REAL, width REAL, height REAL, angle REAL, drill REAL, net TEXT,
  pinfunction TEXT, pintype TEXT, mask_open INTEGER, outline TEXT,
  PRIMARY KEY (design_id, pad_id)
);
CREATE INDEX IF NOT EXISTS pads_net ON pads (design_id, net);
CREATE INDEX IF NOT EXISTS pads_ref ON pads (design_id, ref);
CREATE TABLE IF NOT EXISTS pinouts (
  mpn TEXT, pin TEXT, name TEXT, function TEXT, notes TEXT, source TEXT,
  PRIMARY KEY (mpn, pin)
);
"""


class DesignStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- designs -------------------------------------------------------------------------

    def ingest(self, design: Design) -> dict:
        with self.db:
            for table in ("designs", "components", "pads"):
                self.db.execute(f"DELETE FROM {table} WHERE design_id = ?", (design.design_id,))
            self.db.execute("UPDATE designs SET active = 0")
            self.db.execute(
                "INSERT INTO designs VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    design.design_id,
                    design.name,
                    design.pcb_path,
                    design.thickness_m,
                    json.dumps(design.schematic_paths),
                    json.dumps(design.warnings),
                    time.time(),
                ),
            )
            self.db.executemany(
                "INSERT INTO components VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        design.design_id,
                        c.ref,
                        c.value,
                        c.footprint,
                        c.mpn,
                        c.datasheet,
                        c.description,
                        c.layer,
                        c.x,
                        c.y,
                        c.rotation_deg,
                        int(c.testpoint),
                        int(c.dnp),
                        json.dumps(c.fields),
                    )
                    for c in design.components.values()
                ],
            )
            self.db.executemany(
                "INSERT OR REPLACE INTO pads VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        design.design_id,
                        p.pad_id,
                        p.ref,
                        p.number,
                        p.kind,
                        p.shape,
                        p.x,
                        p.y,
                        p.width,
                        p.height,
                        p.angle,
                        p.drill,
                        p.net,
                        p.pinfunction,
                        p.pintype,
                        int(p.mask_open),
                        json.dumps(p.outline),
                    )
                    for p in design.pads
                ],
            )
        return self.summary(design.design_id)

    def summary(self, design_id: str) -> dict:
        row = self.db.execute("SELECT * FROM designs WHERE design_id = ?", (design_id,)).fetchone()
        if row is None:
            raise KeyError(design_id)
        count = lambda sql: self.db.execute(sql, (design_id,)).fetchone()[0]
        return {
            "design_id": row["design_id"],
            "name": row["name"],
            "pcb_path": row["pcb_path"],
            "thickness_m": row["thickness_m"],
            "schematics": json.loads(row["schematics"]),
            "active": bool(row["active"]),
            "components": count("SELECT COUNT(*) FROM components WHERE design_id = ?"),
            "pads": count("SELECT COUNT(*) FROM pads WHERE design_id = ? AND kind != 'via'"),
            "exposed_vias": count("SELECT COUNT(*) FROM pads WHERE design_id = ? AND kind = 'via'"),
            "nets": count("SELECT COUNT(DISTINCT net) FROM pads WHERE design_id = ? AND net != ''"),
            "warnings": json.loads(row["warnings"]),
        }

    def designs(self) -> list[dict]:
        rows = self.db.execute("SELECT design_id FROM designs ORDER BY ingested_at DESC")
        return [self.summary(r[0]) for r in rows.fetchall()]

    def active_id(self) -> str | None:
        row = self.db.execute("SELECT design_id FROM designs WHERE active = 1").fetchone()
        return row[0] if row else None

    def set_active(self, design_id: str) -> dict:
        summary = self.summary(design_id)
        with self.db:
            self.db.execute("UPDATE designs SET active = (design_id = ?)", (design_id,))
        summary["active"] = True
        return summary

    # --- lookups -------------------------------------------------------------------------

    def component(self, design_id: str, ref: str) -> Component | None:
        row = self.db.execute(
            "SELECT * FROM components WHERE design_id = ? AND ref = ?", (design_id, ref)
        ).fetchone()
        if row is None:
            return None
        values = {k: row[k] for k in row.keys() if k != "design_id"}  # noqa: SIM118 - Row iterates values
        values["fields"] = json.loads(values["fields"])
        values["testpoint"], values["dnp"] = bool(values["testpoint"]), bool(values["dnp"])
        return Component(**values)

    def testpoints(self, design_id: str) -> set[str]:
        rows = self.db.execute(
            "SELECT ref FROM components WHERE design_id = ? AND testpoint = 1", (design_id,)
        )
        return {r[0] for r in rows.fetchall()}

    def pads(self, design_id: str, *, ref: str | None = None, net: str | None = None) -> list[Pad]:
        sql, args = "SELECT * FROM pads WHERE design_id = ?", [design_id]
        if ref is not None:
            sql, args = sql + " AND ref = ?", args + [ref]
        if net is not None:
            sql, args = sql + " AND net = ?", args + [net]
        return [_pad(r) for r in self.db.execute(sql + " ORDER BY pad_id", args).fetchall()]

    def pad(self, design_id: str, pad_id: str) -> Pad | None:
        row = self.db.execute(
            "SELECT * FROM pads WHERE design_id = ? AND pad_id = ?", (design_id, pad_id)
        ).fetchone()
        return _pad(row) if row else None

    def nets(self, design_id: str) -> list[str]:
        rows = self.db.execute(
            "SELECT DISTINCT net FROM pads WHERE design_id = ? AND net != '' ORDER BY net",
            (design_id,),
        )
        return [r[0] for r in rows.fetchall()]

    def resolve_net(self, design_id: str, name: str) -> list[str]:
        """Exact net name, else nets whose last path segment matches case-insensitively:
        "3v3" finds "/3V3" and "/power/3V3". More than one match is ambiguous."""
        nets = self.nets(design_id)
        if name in nets:
            return [name]
        leaf = name.strip("/").split("/")[-1].lower()
        return [n for n in nets if n.strip("/").split("/")[-1].lower() == leaf]

    def components(self, design_id: str, mpn: str | None = None) -> list[Component]:
        sql, args = "SELECT ref FROM components WHERE design_id = ?", [design_id]
        if mpn:
            sql, args = sql + " AND mpn = ?", args + [mpn]
        refs = [r[0] for r in self.db.execute(sql + " ORDER BY ref", args).fetchall()]
        return [self.component(design_id, ref) for ref in refs]

    # --- pinouts -------------------------------------------------------------------------

    def record_pinout(self, mpn: str, pins: list[dict], source: str) -> int:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO pinouts VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        mpn,
                        str(p["pin"]),
                        p.get("name", ""),
                        p.get("function", ""),
                        p.get("notes", ""),
                        source,
                    )
                    for p in pins
                ],
            )
        return len(pins)

    def pinout(self, mpn: str) -> dict[str, dict]:
        rows = self.db.execute("SELECT * FROM pinouts WHERE mpn = ?", (mpn,)).fetchall()
        return {r["pin"]: {k: r[k] for k in ("name", "function", "notes", "source")} for r in rows}


def _pad(row) -> Pad:
    values = {k: row[k] for k in row.keys() if k != "design_id"}  # noqa: SIM118 - Row iterates values
    values["mask_open"] = bool(values["mask_open"])
    values["outline"] = tuple(tuple(p) for p in json.loads(values["outline"]))
    return Pad(**values)


def pad_dict(pad: Pad) -> dict:
    out = asdict(pad)
    out.pop("outline")
    return out
