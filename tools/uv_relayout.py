"""Give every triangle of a character GLB its own texture space on a new square sheet,
then copy the old paint across. Run inside Blender:
  blender -b --factory-startup --python-exit-code 1 --python uv_relayout.py -- in.glb outdir [size]

Keeps the old UV island cuts (seams where old UVs split), unfolds each island with
minimum stretch (so mirrored or stacked pieces get their own space), evens out texel
density, packs onto one sheet per texture, then copies paint per triangle.
Everything in the GLB except the mesh vertex/index data and the textures is kept."""
import bpy, bmesh, json, math, os, struct, sys, time
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = os.path.abspath(argv[0]), os.path.abspath(argv[1])
S = int(argv[2]) if len(argv) > 2 else 4096
MARGIN = 0.004
DILATE = 12
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

data = open(SRC, "rb").read()
jl = struct.unpack_from("<I", data, 12)[0]
J = json.loads(data[20:20 + jl])
bl = struct.unpack_from("<I", data, 20 + jl)[0]
BIN = data[20 + jl + 8:20 + jl + 8 + bl]
CT = {5120: ("i1", 1), 5121: ("u1", 1), 5122: ("i2", 2), 5123: ("u2", 2), 5125: ("u4", 4), 5126: ("f4", 4)}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def view_bytes(vi):
    v = J["bufferViews"][vi]
    o = v.get("byteOffset", 0)
    return BIN[o:o + v["byteLength"]]


def acc_rows(ai):
    a = J["accessors"][ai]
    assert "sparse" not in a
    _, cs = CT[a["componentType"]]
    es = cs * NC[a["type"]]
    v = J["bufferViews"][a["bufferView"]]
    stride = v.get("byteStride", es)
    raw = np.frombuffer(view_bytes(a["bufferView"]), np.uint8)
    idx = a.get("byteOffset", 0) + np.arange(a["count"])[:, None] * stride + np.arange(es)[None, :]
    return np.ascontiguousarray(raw[idx])


def rows_as(rows, ai):
    a = J["accessors"][ai]
    dt, _ = CT[a["componentType"]]
    return rows.view("<" + dt).reshape(len(rows), NC[a["type"]])


def acc_array(ai):
    return rows_as(acc_rows(ai), ai)


def load_image(ii):
    im = J["images"][ii]
    p = os.path.join(OUT, f"stock-{im.get('name', ii)}.png")
    open(p, "wb").write(view_bytes(im["bufferView"]))
    img = bpy.data.images.load(p)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[::-1].copy()


def save_png(arr, path):
    n = arr.shape[0]
    img = bpy.data.images.new(os.path.basename(path), n, n, alpha=True)
    img.colorspace_settings.name = "Non-Color"
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return open(path, "rb").read()


def raster(P, w, h, grow):
    """Pixel centers covered by triangle P (3x2, pixel units, centers at integers).
    grow > 0 also takes pixels within `grow` px of the edges (barycentrics clamped)."""
    x0 = max(int(math.floor(P[:, 0].min() - grow)), 0)
    x1 = min(int(math.ceil(P[:, 0].max() + grow)), w - 1)
    y0 = max(int(math.floor(P[:, 1].min() - grow)), 0)
    y1 = min(int(math.ceil(P[:, 1].max() + grow)), h - 1)
    if x1 < x0 or y1 < y0:
        return None
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    xs, ys = xs.ravel().astype(np.float64), ys.ravel().astype(np.float64)
    (ax, ay), (bx, by), (cx, cy) = P
    d = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(d) < 1e-9:
        return None
    b1 = ((xs - ax) * (cy - ay) - (ys - ay) * (cx - ax)) / d
    b2 = ((bx - ax) * (ys - ay) - (by - ay) * (xs - ax)) / d
    b = np.stack([1 - b1 - b2, b1, b2], 1)
    if grow > 0:
        el = np.array([math.hypot(cx - bx, cy - by), math.hypot(cx - ax, cy - ay), math.hypot(bx - ax, by - ay)])
        dist = b * (abs(d) / np.maximum(el, 1e-9))
        m = dist.min(1) >= -grow
        b = np.clip(b[m], 0, None)
        b /= b.sum(1, keepdims=True)
    else:
        m = b.min(1) >= -1e-7
        b = b[m]
    return xs[m].astype(np.int64), ys[m].astype(np.int64), b


def used_texels(old_uv_tris, w, h):
    used = np.zeros((h, w), bool)
    for T in old_uv_tris:
        P = T * [w, h] - 0.5
        off = np.floor(P.min(0) / [w, h]) * [w, h]  # bring wrapped triangles near the origin
        P = P - off
        r = raster(P, w * 3, h * 3, 0.71)
        if r:
            used[r[1] % h, r[0] % w] = True
    return used


def sample(img, used, uv):
    h, w, _ = img.shape
    x = uv[:, 0] * w - 0.5
    y = uv[:, 1] * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    xi, yi = x0.astype(np.int64), y0.astype(np.int64)
    acc = np.zeros((len(uv), 4), np.float32)
    wsum = np.zeros(len(uv), np.float32)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        cx, cy = (xi + dx) % w, (yi + dy) % h
        k = (wt * used[cy, cx]).astype(np.float32)
        acc += img[cy, cx] * k[:, None]
        wsum += k
    near = img[np.round(y).astype(np.int64) % h, np.round(x).astype(np.int64) % w]
    ok = wsum > 1e-6
    if not ok.all():  # no usable texel nearby: take the closest one in a 5x5 window
        bad = np.nonzero(~ok)[0]
        best = np.full(len(bad), np.inf)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cx = (np.round(x[bad]).astype(np.int64) + dx) % w
                cy = (np.round(y[bad]).astype(np.int64) + dy) % h
                dd = np.where(used[cy, cx], dx * dx + dy * dy, np.inf)
                better = dd < best
                near[bad[better]] = img[cy[better], cx[better]]
                best = np.minimum(best, dd)
    out = near.copy()
    out[ok] = acc[ok] / wsum[ok, None]
    return out


def dilate(img, filled, n):
    for _ in range(n):
        acc = np.zeros_like(img)
        cnt = np.zeros(filled.shape, np.float32)
        for sy, sx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            f = np.roll(filled, (sy, sx), (0, 1))
            acc += np.roll(img, (sy, sx), (0, 1)) * f[..., None]
            cnt += f
        grow = (~filled) & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow, None]
        filled = filled | grow
    return img


def uv_lines(tri_uv, tri_v, n):
    """Painter guide: faint lines on every triangle edge, bright 3 px lines on island borders,
    on a transparent sheet (lay it over the texture as a layer)."""
    img = np.zeros((n, n, 4), np.float32)
    count = {}
    for t in tri_v:
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            e = (min(a, b), max(a, b))
            count[e] = count.get(e, 0) + 1

    def draw(p, q, col, thick):
        steps = int(max(abs(q - p).max() * n * 2, 2))
        pts = p + (q - p) * np.linspace(0, 1, steps)[:, None]
        xs = np.round(pts[:, 0] * n - 0.5).astype(np.int64)
        ys = np.round(pts[:, 1] * n - 0.5).astype(np.int64)
        r = thick // 2
        for dy in range(-r, thick - r):
            for dx in range(-r, thick - r):
                img[np.clip(ys + dy, 0, n - 1), np.clip(xs + dx, 0, n - 1)] = col

    for border in (False, True):
        for t, u in zip(tri_v, tri_uv):
            for k in range(3):
                a, b = int(t[k]), int(t[(k + 1) % 3])
                if (count[(min(a, b), max(a, b))] == 1) == border:
                    draw(u[k], u[(k + 1) % 3], (1.0, 0.85, 0.2, 1.0) if border else (1.0, 1.0, 1.0, 0.35), 3 if border else 1)
    return img


def unwrap(pos, nor, uv, tris, name):
    """New UVs per triangle corner (T,3,2), glTF orientation, plus island stats."""
    key = np.concatenate([np.round(pos * 1e5), np.round(nor * 50)], 1).astype(np.int64)
    _, wid = np.unique(key, axis=0, return_inverse=True)
    wid = wid.reshape(-1)
    wpos = np.zeros((wid.max() + 1, 3))
    wpos[wid] = pos
    wt = wid[tris]
    good = (wt[:, 0] != wt[:, 1]) & (wt[:, 1] != wt[:, 2]) & (wt[:, 0] != wt[:, 2])
    gi = np.nonzero(good)[0]
    _, first = np.unique(np.sort(wt[good], 1), axis=0, return_index=True)
    keep = np.zeros(len(tris), bool)
    keep[gi[first]] = True
    kt = wt[keep]

    me = bpy.data.meshes.new(name)
    me.from_pydata(wpos.tolist(), [], kt.tolist())
    me.update()
    assert len(me.polygons) == len(kt), (len(me.polygons), len(kt))
    old = uv[tris[keep]].reshape(-1, 2).copy()
    old[:, 1] = 1 - old[:, 1]
    lay = me.uv_layers.new(name="old")
    lay.data.foreach_set("uv", old.astype(np.float32).ravel())

    # Seams where the old UVs split, plus non-manifold edges.
    bm = bmesh.new()
    bm.from_mesh(me)
    ul = bm.loops.layers.uv["old"]
    for e in bm.edges:
        ls = list(e.link_loops)
        if len(ls) != 2:
            e.seam = len(ls) > 2
            continue
        l1, l2 = ls
        a1, b1 = l1[ul].uv, l1.link_loop_next[ul].uv
        if l2.vert == l1.vert:
            a2, b2 = l2[ul].uv, l2.link_loop_next[ul].uv
        else:
            a2, b2 = l2.link_loop_next[ul].uv, l2[ul].uv
        e.seam = (a1 - a2).length > 1e-5 or (b1 - b2).length > 1e-5
    # Islands with no boundary at all cannot be flattened as they are: smart-project those.
    bm.faces.ensure_lookup_table()
    island = [-1] * len(bm.faces)
    closed = []
    for f in bm.faces:
        if island[f.index] >= 0:
            continue
        k = len(closed)
        stack, has_edge = [f], False
        island[f.index] = k
        while stack:
            g = stack.pop()
            for e in g.edges:
                if e.seam or len(e.link_faces) != 2:
                    has_edge = True
                    continue
                for h in e.link_faces:
                    if island[h.index] < 0:
                        island[h.index] = k
                        stack.append(h)
        closed.append(not has_edge)
    bm.to_mesh(me)
    bm.free()

    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    new = me.uv_layers.new(name="new")
    me.uv_layers.active = new
    bpy.context.scene.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")

    def select(pred):
        b = bmesh.from_edit_mesh(me)
        b.faces.ensure_lookup_table()
        for f in b.faces:
            f.select_set(pred(f.index))
        bmesh.update_edit_mesh(me)

    select(lambda i: not closed[island[i]])
    try:
        bpy.ops.uv.unwrap(method="MINIMUM_STRETCH", margin=0.0)
        method = "MINIMUM_STRETCH"
    except Exception:
        bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.0)
        method = "ANGLE_BASED"
    n_closed = sum(closed)
    if n_closed:
        select(lambda i: closed[island[i]])
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0, scale_to_bounds=False)
    select(lambda i: True)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    try:
        bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="AXIS_ALIGNED", scale=True,
                                merge_overlap=False, margin_method="FRACTION", margin=MARGIN, shape_method="CONCAVE")
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=MARGIN)
    bpy.ops.object.mode_set(mode="OBJECT")

    nl = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers["new"].data.foreach_get("uv", nl)
    nl = nl.reshape(-1, 3, 2).astype(np.float64)
    nl[..., 1] = 1 - nl[..., 1]

    out = np.zeros((len(tris), 3, 2))
    out[keep] = nl
    by_face = {}
    any_uv = {}
    for t, (fw, fu) in enumerate(zip(kt, nl)):
        by_face[tuple(sorted(fw))] = {int(v): u for v, u in zip(fw, fu)}
        for v, u in zip(fw, fu):
            any_uv.setdefault(int(v), u)
    for t in np.nonzero(~keep)[0]:
        f = by_face.get(tuple(sorted(wt[t])))
        for c in range(3):
            v = int(wt[t, c])
            out[t, c] = f[v] if f and v in f else any_uv.get(v, (0.0, 0.0))
    stats = {"triangles": int(len(tris)), "triangles_unwrapped": int(keep.sum()), "islands": len(closed),
             "closed_islands_smart_projected": int(n_closed), "unwrap": method,
             "uv_min": out.reshape(-1, 2).min(0).round(4).tolist(), "uv_max": out.reshape(-1, 2).max(0).round(4).tolist()}
    return out, stats


def density(pos3, uv2, size):
    """Per triangle: texels per unit length (sqrt of uv-px area / 3D area)."""
    a3 = np.linalg.norm(np.cross(pos3[:, 1] - pos3[:, 0], pos3[:, 2] - pos3[:, 0]), axis=1) / 2
    p = uv2 * size
    a2 = np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) / 2
    m = a3 > 1e-12
    return np.sqrt(a2[m] / a3[m]), a3[m]


def spread(d, w):
    """Area-weighted p5/p50/p95 of texel density."""
    o = np.argsort(d)
    c = np.cumsum(w[o]) / w.sum()
    return [float(d[o][np.searchsorted(c, q)]) for q in (0.05, 0.5, 0.95)]


report = {"source": os.path.basename(SRC), "size": S, "margin": MARGIN, "textures": []}
new_acc = {}      # accessor index -> (rows uint8, componentType, type)
new_img = {}      # image index -> png bytes
seen_img = set()
for mi, m in enumerate(J["meshes"]):
    for pi, prim in enumerate(m["primitives"]):
        assert prim.get("mode", 4) == 4 and "targets" not in prim
        A = prim["attributes"]
        pos = acc_array(A["POSITION"]).astype(np.float64)
        nor = acc_array(A["NORMAL"]).astype(np.float64) if "NORMAL" in A else np.zeros_like(pos)
        uv = acc_array(A["TEXCOORD_0"]).astype(np.float64)
        tris = acc_array(prim["indices"]).reshape(-1, 3).astype(np.int64)
        mat = J["materials"][prim["material"]]
        ii = J["textures"][mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]]["source"]
        assert ii not in seen_img, "two primitives share one image; pack them together first"
        seen_img.add(ii)
        iname = J["images"][ii].get("name", str(ii))
        print(f"[{time.time() - t0:5.1f}s] mesh {mi} prim {pi}: {len(pos)} verts, {len(tris)} tris, image {iname}", flush=True)

        nuv, st = unwrap(pos, nor, uv, tris, f"m{mi}p{pi}")
        old_uv = uv[tris]

        # New vertices: split only where the new layout needs it; never merge old vertices.
        corner_v = tris.reshape(-1)
        q = np.round(nuv.reshape(-1, 2) * (1 << 20)).astype(np.int64)
        _, first, inv = np.unique(np.stack([corner_v, q[:, 0], q[:, 1]], 1), axis=0, return_index=True, return_inverse=True)
        src = corner_v[first]
        new_uv = nuv.reshape(-1, 2)[first].astype(np.float32)
        new_idx = inv.reshape(-1)
        for k, ai in A.items():
            if k == "TEXCOORD_0":
                new_acc[ai] = (new_uv.view(np.uint8).reshape(len(new_uv), 8), 5126, "VEC2")
            else:
                a = J["accessors"][ai]
                new_acc[ai] = (acc_rows(ai)[src], a["componentType"], a["type"])
        it = np.uint16 if len(src) < 65536 else np.uint32
        ix = new_idx.astype(it)
        new_acc[prim["indices"]] = (ix.view(np.uint8).reshape(len(ix), ix.itemsize), 5123 if it == np.uint16 else 5125, "SCALAR")

        # Paint transfer.
        old = load_image(ii)
        oh, ow, _ = old.shape
        # Paint only from texels a triangle covers, and never from the magenta key background.
        key = (old[..., 0] > 0.75) & (old[..., 1] < 0.35) & (old[..., 2] > 0.6)
        used = used_texels(old_uv, ow, oh) & ~key
        img = np.zeros((S, S, 4), np.float32)
        filled = np.zeros((S, S), bool)
        cover = np.zeros((S, S), np.uint8)
        for grow in (1.0, 0.0):
            for t in range(len(tris)):
                r = raster(nuv[t] * S - 0.5, S, S, grow)
                if not r:
                    continue
                xs, ys, b = r
                ouv = b @ old_uv[t]
                img[ys, xs] = sample(old, used, ouv)
                filled[ys, xs] = True
                if grow == 0.0:
                    cover[ys, xs] = np.minimum(cover[ys, xs].astype(np.int32) + 1, 255)
        img = dilate(img, filled, DILATE)
        png = save_png(np.clip(img, 0, 1), os.path.join(OUT, f"{iname}.png"))
        new_img[ii] = png
        save_png(uv_lines(nuv, new_idx.reshape(-1, 3), S), os.path.join(OUT, f"{iname}-uv-lines.png"))

        # Old sheet: how much paint is shared between triangles.
        ocover = np.zeros((oh, ow), np.uint8)
        for T in old_uv:
            r = raster(T * [ow, oh] - 0.5, ow, oh, 0.0)
            if r:
                ocover[r[1], r[0]] = np.minimum(ocover[r[1], r[0]].astype(np.int32) + 1, 255)
        p3 = pos[tris]
        dn, wn = density(p3, nuv, S)
        do, wo = density(p3, old_uv, max(ow, oh))
        st.update({
            "image": iname, "old_size": [ow, oh], "new_vertices": int(len(src)), "old_vertices": int(len(pos)),
            "sheet_used": round(float((cover > 0).mean()), 4),
            "texels_painted_twice_new": int((cover > 1).sum()),
            "old_texels_used": int((ocover > 0).sum()), "old_texels_used_twice": int((ocover > 1).sum()),
            "density_old_p5_p50_p95": [round(x, 1) for x in spread(do, wo)],
            "density_new_p5_p50_p95": [round(x, 1) for x in spread(dn, wn)],
            "detail_gain_median": round(spread(dn, wn)[1] / max(spread(do, wo)[1], 1e-9), 1),
        })
        report["textures"].append(st)
        print(json.dumps(st), flush=True)

# Write the new GLB: same JSON and same bytes for every block we did not replace.
views = J["bufferViews"]
users = {}
for ai, a in enumerate(J["accessors"]):
    users.setdefault(a.get("bufferView"), []).append(ai)
for ai in new_acc:
    assert len(users[J["accessors"][ai]["bufferView"]]) == 1, f"accessor {ai} shares its block"
for ai, (rows, ct, ty) in new_acc.items():
    a = J["accessors"][ai]
    v = views[a["bufferView"]]
    a["componentType"], a["type"], a["count"] = ct, ty, len(rows)
    a.pop("byteOffset", None)
    a.pop("normalized", None) if ct == 5126 else None
    if "byteStride" in v:
        v["byteStride"] = rows.shape[1]
    if "min" in a or "max" in a:
        arr = rows_as(rows, ai)
        a["min"] = [x.item() for x in arr.min(0)]
        a["max"] = [x.item() for x in arr.max(0)]
replace = {J["accessors"][ai]["bufferView"]: rows.tobytes() for ai, (rows, _, _) in new_acc.items()}
for ii, png in new_img.items():
    replace[J["images"][ii]["bufferView"]] = png
    J["images"][ii]["mimeType"] = "image/png"
out_bin = bytearray()
for vi, v in enumerate(views):
    b = replace.get(vi, view_bytes(vi))
    out_bin += b"\0" * (-len(out_bin) % 4)
    v["byteOffset"] = len(out_bin)
    v["byteLength"] = len(b)
    out_bin += b
out_bin += b"\0" * (-len(out_bin) % 4)
J["buffers"][0]["byteLength"] = len(out_bin)
js = json.dumps(J, separators=(",", ":")).encode()
js += b" " * (-len(js) % 4)
glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(out_bin))
glb += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(out_bin), 0x004E4942) + bytes(out_bin)
stem = os.path.splitext(os.path.basename(SRC))[0].replace("-idle", "")
out_glb = os.path.join(OUT, f"{stem}-uv{S // 1024}k-v1.glb")
open(out_glb, "wb").write(glb)
report["output"] = os.path.basename(out_glb)
report["seconds"] = round(time.time() - t0, 1)
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2)
print(f"[{time.time() - t0:5.1f}s] wrote {out_glb} ({len(glb)} bytes)", flush=True)
