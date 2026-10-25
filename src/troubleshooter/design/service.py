"""Design knowledge for the MCP tools: the design store, probe ranking and text search.

Everything here works without the rig. Nothing moves an arm, reads an instrument or
transforms into a camera frame; board-frame geometry from the PCB file is as far as it goes.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

from .docs import Chunk, DocsLibrary, LocalIndex, MossIndex, chunk_pages, pdf_pages
from .kicad import load_design
from .probe import rank
from .store import DesignStore, pad_dict

LOG = logging.getLogger(__name__)
_MAX_TIP_MM = 5.0
_MAX_UNCERTAINTY_MM = 5.0


def failed(reason: str, **extra) -> dict:
    return {"status": "failed", "failure_reason": reason, **extra}


class DesignService:
    def __init__(self, data_dir: str | Path, moss_project_id=None, moss_project_key=None):
        data_dir = Path(data_dir)
        self.store = DesignStore(data_dir / "design.sqlite3")
        moss = None
        if moss_project_id and moss_project_key:
            try:
                moss = MossIndex(moss_project_id, moss_project_key)
            except ImportError:
                LOG.error("Moss credentials are set but inferedge-moss isn't installed")
        self.docs = DocsLibrary(LocalIndex(data_dir / "docs.sqlite3"), moss)

    def close(self) -> None:
        self.store.close()
        self.docs.local.close()

    # --- designs -------------------------------------------------------------------------

    async def ingest_design(self, path: str) -> dict:
        try:
            design = load_design(path)
        except (OSError, ValueError, IndexError, TypeError, UnicodeDecodeError) as exc:
            return failed(f"unreadable_design:{exc}")
        summary = self.store.ingest(design)
        chunks = [
            Chunk(
                f"design:{design.design_id}:{ref}",
                f"design:{design.design_id}",
                "component",
                c.mpn,
                None,
                _component_text(c, self.store.pads(design.design_id, ref=ref)),
            )
            for ref, c in sorted(design.components.items())
        ]
        source = {
            "source_id": f"design:{design.design_id}",
            "kind": "design",
            "mpn": "",
            "title": design.name,
            "path": design.pcb_path,
            "pages": None,
        }
        summary["search_index"] = await self.docs.add(source, chunks)
        return {"status": "ok", **summary}

    def list_designs(self) -> dict:
        return {"status": "ok", "active": self.store.active_id(), "designs": self.store.designs()}

    def set_active_design(self, design_id: str) -> dict:
        try:
            return {"status": "ok", **self.store.set_active(design_id)}
        except KeyError:
            return failed(f"unknown_design:{design_id}")

    def query(self, ref=None, net=None, pad=None, mpn=None) -> dict:
        design_id = self.store.active_id()
        if design_id is None:
            return failed("no_design: ingest a .kicad_pcb first")
        if sum(x is not None for x in (ref, net, pad, mpn)) != 1:
            return failed("give exactly one of ref, net, pad, mpn")
        out = {"status": "ok", "design_id": design_id}
        if pad is not None:
            found = self.store.pad(design_id, pad)
            if found is None:
                return failed(f"unknown_pad:{pad}", design_id=design_id)
            out["pad"] = self._pad_view(design_id, found)
            if found.net:
                out["net_members"] = [p.pad_id for p in self.store.pads(design_id, net=found.net)]
        elif ref is not None:
            component = self.store.component(design_id, ref)
            if component is None:
                return failed(f"unknown_ref:{ref}", design_id=design_id)
            out["component"] = _component_view(component)
            out["pins"] = [
                self._pad_view(design_id, p) for p in self.store.pads(design_id, ref=ref)
            ]
        elif net is not None:
            nets = self.store.resolve_net(design_id, net)
            if len(nets) != 1:
                reason = "ambiguous_net" if nets else "unknown_net"
                return failed(f"{reason}:{net}", design_id=design_id, matches=nets)
            out["net"] = nets[0]
            out["members"] = [
                self._pad_view(design_id, p) for p in self.store.pads(design_id, net=nets[0])
            ]
        else:
            out["components"] = [_component_view(c) for c in self.store.components(design_id, mpn)]
            out["pinout"] = self.store.pinout(mpn)
        return out

    def find_probe_points(
        self,
        net: str | None,
        pad: str | None,
        tip_diameter_mm: float,
        position_uncertainty_mm: float,
        limit: int,
    ) -> dict:
        design_id = self.store.active_id()
        if design_id is None:
            return failed("no_design: ingest a .kicad_pcb first")
        if (net is None) == (pad is None):
            return failed("give exactly one of net, pad")
        for name, value, top in (
            ("tip_diameter_mm", tip_diameter_mm, _MAX_TIP_MM),
            ("position_uncertainty_mm", position_uncertainty_mm, _MAX_UNCERTAINTY_MM),
        ):
            if not (math.isfinite(value) and 0 <= value <= top):
                return failed(f"invalid:{name} must be within 0..{top}")
        if pad is not None:
            found = self.store.pad(design_id, pad)
            if found is None:
                return failed(f"unknown_pad:{pad}", design_id=design_id)
            targets, net_name = [found], found.net
        else:
            nets = self.store.resolve_net(design_id, net)
            if len(nets) != 1:
                reason = "ambiguous_net" if nets else "unknown_net"
                return failed(f"{reason}:{net}", design_id=design_id, matches=nets)
            net_name = nets[0]
            targets = self.store.pads(design_id, net=net_name)
        candidates = rank(
            targets,
            self.store.pads(design_id),
            self.store.testpoints(design_id),
            tip_diameter_mm / 1000.0,
            position_uncertainty_mm / 1000.0,
        )
        usable = [c for c in candidates if not c.refused]
        return {
            "status": "ok" if usable else "failed",
            "failure_reason": None if usable else "no_safe_pad",
            "design_id": design_id,
            "net": net_name,
            "tip_diameter_mm": tip_diameter_mm,
            "position_uncertainty_mm": position_uncertainty_mm,
            "covered_pads": sum(not p.mask_open for p in targets),
            "candidates": [
                {
                    "pad_id": c.pad.pad_id,
                    "category": c.category,
                    "refused": c.refused,
                    "reasons": list(c.reasons),
                    "clearance_mm": _mm(c.clearance_m),
                    "short_margin_mm": _mm(c.short_margin_m),
                    "landing_margin_mm": _mm(c.landing_margin_m),
                    "nearest_other_net_pad": c.nearest_other or None,
                    "pinfunction": c.pad.pinfunction,
                    "kind": c.pad.kind,
                    "center_board_m": [c.pad.x, c.pad.y],
                }
                for c in candidates[:limit]
            ],
            "note": (
                "Board-frame geometry only. Reach, tall parts and visibility aren't checked "
                "here; locate_on_image and the orchestrator own those."
            ),
        }

    def record_pinout(self, mpn: str, pins: list[dict], source: str) -> dict:
        if not mpn or not pins:
            return failed("mpn and pins are required")
        if not source:
            return failed("source is required: cite the datasheet and page")
        bad = [i for i, p in enumerate(pins) if not isinstance(p, dict) or "pin" not in p]
        if bad:
            return failed(f"invalid:pins[{bad[0]}] needs a pin number")
        return {"status": "ok", "mpn": mpn, "recorded": self.store.record_pinout(mpn, pins, source)}

    # --- documents -----------------------------------------------------------------------

    async def ingest_datasheet(self, path: str, mpn: str, title: str | None) -> dict:
        file = Path(path)
        try:
            raw = file.read_bytes()
            pages = pdf_pages(file) if file.suffix.lower() == ".pdf" else [raw.decode("utf-8")]
        except (OSError, UnicodeDecodeError, RuntimeError, ValueError) as exc:
            return failed(f"unreadable_document:{exc}")
        except Exception as exc:  # noqa: BLE001 - pypdf raises its own types on bad files
            return failed(f"unreadable_document:{type(exc).__name__}: {exc}")
        source_id = "ds:" + hashlib.sha256(raw).hexdigest()[:16]
        chunks = chunk_pages(source_id, "datasheet", mpn, pages)
        if not chunks:
            return failed("no_text: the PDF may be scanned images; OCR it first")
        source = {
            "source_id": source_id,
            "kind": "datasheet",
            "mpn": mpn,
            "title": title or file.stem,
            "path": str(file.resolve()),
            "pages": len(pages),
        }
        result = await self.docs.add(source, chunks)
        return {"status": "ok", "mpn": mpn, "pages": len(pages), **result}

    async def search(self, query: str, mpn: str | None, top_k: int) -> dict:
        if not query.strip():
            return failed("empty_query")
        found = await self.docs.search(query, mpn, max(1, min(top_k, 20)))
        sources = {s["source_id"]: s for s in self.docs.local.sources()}
        for hit in found["results"]:
            source = sources.get(hit["source_id"], {})
            hit["title"], hit["path"] = source.get("title"), source.get("path")
        return {"status": "ok", **found}

    def list_documents(self) -> dict:
        return {"status": "ok", "backend": self.docs.backend, "sources": self.docs.local.sources()}

    # --- views ---------------------------------------------------------------------------

    def _pad_view(self, design_id: str, pad) -> dict:
        view = pad_dict(pad)
        view["exposed"] = view.pop("mask_open")
        if pad.ref:
            component = self.store.component(design_id, pad.ref)
            if component and component.mpn:
                notes = self.store.pinout(component.mpn).get(pad.number)
                if notes:
                    view["datasheet_pin"] = notes
        return view


def _component_view(c) -> dict:
    return {
        "ref": c.ref,
        "value": c.value,
        "footprint": c.footprint,
        "mpn": c.mpn,
        "datasheet": c.datasheet,
        "description": c.description,
        "layer": c.layer,
        "center_board_m": [c.x, c.y],
        "rotation_deg": c.rotation_deg,
        "testpoint": c.testpoint,
        "dnp": c.dnp,
    }


def _component_text(c, pads) -> str:
    """One searchable paragraph per component, so "the buck converter" can find U3."""
    pins = "; ".join(
        f"{p.number}{' ' + p.pinfunction if p.pinfunction else ''} -> {p.net or 'unconnected'}"
        for p in pads
    )
    parts = [
        f"{c.ref} {c.value}",
        c.description,
        f"MPN {c.mpn}" if c.mpn else "",
        f"footprint {c.footprint}",
        "test point" if c.testpoint else "",
        "not populated (DNP)" if c.dnp else "",
        f"pins: {pins}" if pins else "",
    ]
    return ". ".join(p for p in parts if p)


def _mm(value_m: float) -> float | None:
    return round(value_m * 1000.0, 4) if math.isfinite(value_m) else None
