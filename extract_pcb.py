"""Turn a .kicad_pcb into pcb_geometry.json: the board outline and every pad centre, in mm.

Board frame: origin at the board's top-left corner as KiCad draws it, X right, Y down (so Z points into the board).
Usage: python -I extract_pcb.py path/to/Controller.kicad_pcb [out.json]
"""
import json
import math
import re
import sys


def parse(text):
    tok = re.findall(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+', text)
    pos = 0

    def node():
        nonlocal pos
        out = []
        while pos < len(tok):
            t = tok[pos]
            pos += 1
            if t == "(":
                out.append(node())
            elif t == ")":
                return out
            else:
                out.append(t.strip('"'))
        return out

    return node()[0]


def find(n, key):
    return [c for c in n if isinstance(c, list) and c and c[0] == key]


tree = parse(open(sys.argv[1], encoding="utf-8").read())
xs, ys = [], []
for kind in ("gr_line", "gr_rect", "gr_arc", "gr_circle", "gr_poly"):
    for gr in find(tree, kind):
        layer = find(gr, "layer")
        if layer and layer[0][1] == "Edge.Cuts":
            for k in ("start", "end", "mid", "center"):
                for p in find(gr, k):
                    xs.append(float(p[1]))
                    ys.append(float(p[2]))
x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)

pads = []
for fp in find(tree, "footprint"):
    at = find(fp, "at")[0]
    fx, fy, th = float(at[1]), float(at[2]), math.radians(float(at[3]) if len(at) > 3 else 0.0)
    ref = next((pr[2] for pr in find(fp, "property") if pr[1] == "Reference"), "?")
    for pad in find(fp, "pad"):
        pat = find(pad, "at")[0]
        px, py = float(pat[1]), float(pat[2])
        gx = fx + px * math.cos(th) + py * math.sin(th)       # KiCad: y down, positive angle counter-clockwise on screen
        gy = fy - px * math.sin(th) + py * math.cos(th)
        net = find(pad, "net")
        pads.append({"ref": ref, "pad": pad[1], "x": round(gx - x0, 3), "y": round(gy - y0, 3),
                     "net": net[0][2] if net and len(net[0]) > 2 else ""})

out = {"source": sys.argv[1].replace("\\", "/").split("/")[-1], "units": "mm",
       "origin": "board top-left corner, X right, Y down",
       "width": round(x1 - x0, 3), "height": round(y1 - y0, 3), "pads": pads}
path = sys.argv[2] if len(sys.argv) > 2 else "pcb_geometry.json"
json.dump(out, open(path, "w"), indent=1)
print(f"board {out['width']} x {out['height']} mm, {len(pads)} pads -> {path}")
for ref in ("J1", "J2", "J3", "J4", "J5", "J6"):
    row = [p for p in pads if p["ref"] == ref]
    print(f"  {ref}: " + ", ".join(f"{p['pad']}=({p['x']:.2f},{p['y']:.2f}){'['+p['net']+']' if p['net'] else ''}" for p in row))
