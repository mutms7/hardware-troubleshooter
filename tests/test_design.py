import asyncio
import math

import pytest

from troubleshooter.design.docs import LocalIndex, chunk_pages
from troubleshooter.design.kicad import load_design, parse_pcb
from troubleshooter.design.probe import rank
from troubleshooter.design.service import DesignService

# An imaginary board, test-only. KiCad 8 syntax; millimetres, y down.
BOARD = """
(kicad_pcb (version 20240108) (generator "pcbnew")
  (general (thickness 1.6))
  (setup (tenting front back))
  (net 0 "")
  (net 1 "/3V3")
  (net 2 "GND")
  (net 3 "/sensor/SDA")
  (footprint "TestPoint:TestPoint_Pad_D1.5mm" (layer "F.Cu") (at 10 10)
    (property "Reference" "TP1" (at 0 0 0)) (property "Value" "3V3")
    (attr smd)
    (pad "1" smd circle (at 0 0) (size 1.5 1.5) (layers "F.Cu" "F.Mask") (net 1 "/3V3")))
  (footprint "Resistor_SMD:R_0603_1608Metric" (layer "F.Cu") (at 20 10 90)
    (property "Reference" "R1") (property "Value" "4.7k")
    (pad "1" smd roundrect (at -0.8 0 90) (size 0.8 0.95)
      (layers "F.Cu" "F.Paste" "F.Mask") (net 1 "/3V3"))
    (pad "2" smd roundrect (at 0.8 0 90) (size 0.8 0.95)
      (layers "F.Cu" "F.Paste" "F.Mask") (net 3 "/sensor/SDA")))
  (footprint "Package_SO:TSSOP-8_3x3mm_P0.65mm" (layer "F.Cu") (at 30 20)
    (property "Reference" "U1") (property "Value" "TMP102")
    (pad "1" smd rect (at -2.8 -0.975) (size 1.5 0.4) (layers "F.Cu" "F.Mask")
      (net 3 "/sensor/SDA") (pinfunction "SDA") (pintype "bidirectional"))
    (pad "2" smd rect (at -2.8 -0.325) (size 1.5 0.4) (layers "F.Cu" "F.Mask") (net 2 "GND")
      (pinfunction "GND"))
    (pad "8" smd rect (at 2.8 -0.975) (size 1.5 0.4) (layers "F.Cu" "F.Mask") (net 1 "/3V3")
      (pinfunction "V+"))
    (pad "9" smd rect (at 0 0) (size 1.0 1.0) (layers "F.Cu") (net 2 "GND")))
  (footprint "MountingHole:MountingHole_3.2mm" (layer "F.Cu") (at 5 5)
    (property "Reference" "H1")
    (pad "" np_thru_hole circle (at 0 0) (size 3.2 3.2) (drill 3.2) (layers "*.Cu" "*.Mask")))
  (footprint "Connector:PinHeader_1x01" (layer "B.Cu") (at 50 30)
    (property "Reference" "J1")
    (pad "1" thru_hole circle (at 0 0) (size 1.7 1.7) (drill 1.0) (layers "*.Cu" "*.Mask")
      (net 2 "GND"))
    (pad "2" smd rect (at 0 3) (size 1 1) (layers "B.Cu" "B.Mask") (net 2 "GND")))
  (via (at 40 10) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 1))
  (via (at 42 10) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (tenting none) (net 1))
)
"""

SCHEMATIC = """
(kicad_sch (version 20231120) (generator "eeschema")
  (symbol (lib_id "Sensor_Temperature:TMP102") (at 100 100 0) (unit 1)
    (property "Reference" "U?" (at 0 0 0))
    (property "Value" "TMP102")
    (property "MPN" "TMP102AIDRLR")
    (property "Datasheet" "https://www.ti.com/lit/ds/symlink/tmp102.pdf")
    (property "Description" "Digital temperature sensor, I2C")
    (instances (project "demo" (path "/0001" (reference "U1") (unit 1)))))
)
"""


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "demo"
    folder.mkdir()
    (folder / "demo.kicad_pcb").write_text(BOARD, encoding="utf-8")
    (folder / "demo.kicad_sch").write_text(SCHEMATIC, encoding="utf-8")
    return folder


def pads_by_id(design):
    return {p.pad_id: p for p in design.pads}


def test_pads_land_in_the_board_frame():
    pads = pads_by_id(parse_pcb(BOARD))
    assert pads["TP1.1"].x == pytest.approx(0.010)
    assert pads["TP1.1"].y == pytest.approx(-0.010)
    # R1 is turned 90 degrees counter-clockwise on screen: local -x lands at screen +y.
    assert (pads["R1.1"].x, pads["R1.1"].y) == pytest.approx((0.020, -0.0108))
    assert (pads["R1.2"].x, pads["R1.2"].y) == pytest.approx((0.020, -0.0092))
    assert pads["U1.1"].net == "/sensor/SDA"
    assert pads["U1.1"].pinfunction == "SDA"


def test_only_top_reachable_copper_is_kept():
    pads = pads_by_id(parse_pcb(BOARD))
    assert "H1." not in pads  # non-plated hole
    assert "J1.1" in pads and pads["J1.1"].kind == "thru_hole"  # reaches the top
    assert "J1.2" not in pads  # bottom-only SMD
    assert not pads["U1.9"].mask_open  # no F.Mask opening
    assert pads["R1.1"].mask_open


def test_vias_follow_tenting():
    pads = pads_by_id(parse_pcb(BOARD))
    assert "VIA1" not in pads  # tented by the board default
    assert pads["VIA2"].kind == "via" and pads["VIA2"].net == "/3V3"


def test_schematic_fields_merge_by_instance_reference(project):
    design = load_design(project)
    u1 = design.components["U1"]
    assert u1.mpn == "TMP102AIDRLR"
    assert u1.datasheet.endswith("tmp102.pdf")
    assert design.components["TP1"].testpoint
    assert len(design.design_id) == 16


def test_fine_pitch_pin_is_refused_and_passive_preferred():
    design = parse_pcb(BOARD)
    sda = [p for p in design.pads if p.net == "/sensor/SDA"]
    ranked = rank(sda, design.pads, set(), tip_diameter_m=0.5e-3, uncertainty_m=0.25e-3)
    assert [c.pad.pad_id for c in ranked] == ["R1.2", "U1.1"]
    assert not ranked[0].refused
    u1 = ranked[1]
    assert u1.refused and u1.nearest_other == "U1.2"
    # Centre to U1.2's near edge: 0.65 mm pitch - 0.2 mm half height.
    assert u1.clearance_m == pytest.approx(0.45e-3)


def test_test_point_ranks_first_and_covered_copper_is_skipped():
    design = parse_pcb(BOARD)
    rail = [p for p in design.pads if p.net == "/3V3"]
    ranked = rank(rail, design.pads, {"TP1"}, 0.5e-3, 0.2e-3)
    ids = [c.pad.pad_id for c in ranked]
    assert ids[0] == "TP1.1"
    assert ids.index("R1.1") < ids.index("VIA2") < ids.index("U1.8")


def test_covered_pad_doesnt_count_against_clearance():
    design = parse_pcb(BOARD)
    # U1.9 (GND, covered) sits 1 mm from U1.1's row; only exposed copper can short.
    ranked = rank([p for p in design.pads if p.pad_id == "U1.8"], design.pads, set(), 0.1e-3, 0)
    assert ranked[0].nearest_other != "U1.9"


def test_chunks_overlap_and_keep_pages():
    text = ("The quick brown fox. " * 200).strip()
    chunks = chunk_pages("ds:x", "datasheet", "M1", [text, "page two"])
    assert chunks[-1].page == 2 and chunks[-1].text == "page two"
    assert all(len(c.text) <= 1200 for c in chunks)
    first, second = chunks[0], chunks[1]
    assert first.text[-50:].split()[-1] in second.text


def test_local_search_filters_by_mpn(tmp_path):
    index = LocalIndex(tmp_path / "docs.sqlite3")
    source = {
        "source_id": "s",
        "kind": "datasheet",
        "mpn": "A",
        "title": "",
        "path": "",
        "pages": 1,
    }
    index.replace(source, chunk_pages("s", "datasheet", "A", ["Absolute maximum supply 3.6 V"]))
    other = dict(source, source_id="t", mpn="B")
    index.replace(other, chunk_pages("t", "datasheet", "B", ["Supply range 1.4 V to 3.6 V"]))
    assert {h["mpn"] for h in index.search("supply voltage", None, 5)} == {"A", "B"}
    assert [h["mpn"] for h in index.search("supply", "B", 5)] == ["B"]
    assert index.search("!!!", None, 5) == []


def run(coro):
    return asyncio.run(coro)


def test_service_end_to_end(tmp_path, project):
    service = DesignService(tmp_path / "store")
    try:
        summary = run(service.ingest_design(str(project)))
        assert summary["status"] == "ok", summary
        assert summary["search_index"]["moss"] == "not_configured"
        assert summary["exposed_vias"] == 1

        net = service.query(net="sda")
        assert net["net"] == "/sensor/SDA"
        assert {m["pad_id"] for m in net["members"]} == {"R1.2", "U1.1"}

        assert service.query(net="nope")["failure_reason"] == "unknown_net:nope"
        assert service.query(ref="U1", net="x")["status"] == "failed"

        pins = service.record_pinout(
            "TMP102AIDRLR", [{"pin": "1", "name": "SDA", "notes": "open drain"}], "ds p3"
        )
        assert pins["recorded"] == 1
        pad = service.query(pad="U1.1")
        assert pad["pad"]["datasheet_pin"]["notes"] == "open drain"
        assert "R1.2" in pad["net_members"]

        probes = service.find_probe_points("SDA", None, 0.5, 0.25, 5)
        assert probes["status"] == "ok"
        assert probes["candidates"][0]["pad_id"] == "R1.2"
        assert probes["candidates"][1]["refused"]
        assert service.find_probe_points("SDA", None, math.nan, 0.2, 5)["status"] == "failed"

        tight = service.find_probe_points(None, "U1.1", 0.5, 0.25, 5)
        assert tight["status"] == "failed" and tight["failure_reason"] == "no_safe_pad"

        hits = run(service.search("temperature sensor I2C", None, 3))
        assert hits["backend"] == "local"
        assert hits["results"][0]["chunk_id"].endswith(":U1")
    finally:
        service.close()


def test_datasheet_text_ingest_and_reingest(tmp_path):
    service = DesignService(tmp_path / "store")
    try:
        sheet = tmp_path / "part.txt"
        sheet.write_text("Pin 3 is the enable input, active high.", encoding="utf-8")
        first = run(service.ingest_datasheet(str(sheet), "PART1", None))
        assert first["status"] == "ok" and first["chunks"] == 1
        run(service.ingest_datasheet(str(sheet), "PART1", None))
        assert len(service.list_documents()["sources"]) == 1
        hit = run(service.search("enable pin", "PART1", 5))["results"][0]
        assert hit["page"] == 1 and hit["title"] == "part"
        missing = run(service.ingest_datasheet(str(tmp_path / "none.pdf"), "X", None))
        assert missing["status"] == "failed"
    finally:
        service.close()


def test_no_design_fails_closed(tmp_path):
    service = DesignService(tmp_path / "store")
    try:
        assert service.query(ref="U1")["failure_reason"].startswith("no_design")
        assert service.find_probe_points("3V3", None, 0.5, 0.2, 5)["status"] == "failed"
    finally:
        service.close()


async def test_design_tools_over_mcp(tmp_path, project):
    import json

    from test_mcp import call, connection

    async with connection(tmp_path) as client:
        ingested = await call(client, "ingest_design", path=str(project))
        assert ingested["status"] == "ok" and ingested["components"] == 5
        probes = await call(client, "find_probe_points", net="3V3")
        assert probes["candidates"][0]["pad_id"] == "TP1.1"
        located = await client.call_tool(
            "locate_on_image", {"pad_ids": ["TP1.1", "R1.1"], "camera": "fixed"}
        )
        assert not located.isError
        body = json.loads(located.content[0].text)
        assert body["mode"] == "simulated"
        assert [p["pad_id"] for p in body["pads"]] == ["TP1.1", "R1.1"]
        hits = await call(client, "search_datasheets", query="temperature sensor")
        assert hits["backend"] == "local" and hits["results"]


async def test_locate_fails_closed_without_an_orchestrator(tmp_path):
    import json

    from test_mcp import connection

    async with connection(tmp_path, backend="none") as client:
        located = await client.call_tool("locate_on_image", {"pad_ids": ["TP1.1"]})
        body = json.loads(located.content[0].text)
        assert body["failure_reason"] == "orchestrator_unavailable"


class FakeMoss:
    def __init__(self, fail_search=False):
        self.docs, self.fail_search = {}, fail_search

    async def upsert(self, chunks):
        self.docs.update({c.chunk_id: c for c in chunks})

    async def delete(self, chunk_ids):
        for chunk_id in chunk_ids:
            self.docs.pop(chunk_id)

    async def search(self, query, mpn, top_k):
        if self.fail_search:
            raise ConnectionError("offline")
        return [{"chunk_id": c, "source_id": "", "page": 1, "text": ""} for c in self.docs][:top_k]


def test_moss_mirror_prunes_and_falls_back(tmp_path):
    from troubleshooter.design.docs import DocsLibrary

    moss = FakeMoss()
    library = DocsLibrary(LocalIndex(tmp_path / "docs.sqlite3"), moss)
    source = {
        "source_id": "s",
        "kind": "datasheet",
        "mpn": "A",
        "title": "",
        "path": "",
        "pages": 2,
    }
    long = run(library.add(source, chunk_pages("s", "datasheet", "A", ["one", "two"])))
    assert long["moss"] == "synced" and len(moss.docs) == 2
    run(library.add(source, chunk_pages("s", "datasheet", "A", ["one"])))
    assert list(moss.docs) == ["s:p1:0"]  # page two's chunk was deleted from Moss too
    assert run(library.search("one", None, 5))["backend"] == "moss"
    moss.fail_search = True
    fallback = run(library.search("one", None, 5))
    assert fallback["backend"] == "local" and fallback["fallback_reason"].startswith("moss_failed")
    assert fallback["results"][0]["chunk_id"] == "s:p1:0"
