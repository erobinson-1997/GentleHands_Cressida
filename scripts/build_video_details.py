"""Stage 2: details taken from the walk-around videos (Reference Videos/Body.mp4).

Input : the stage-1 model (output of build_rear_damage.py, MD5 e02e7a7c...) -> config.STAGE1_KN5
Output: config.OUTPUT_KN5 (dist/)

    python scripts/build_video_details.py

Each step is a function below; switch steps on/off in STEPS. World coordinates: +x = car's LEFT, +y = up, +z = front.
"""
import sys, os, io; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter
import config
from kn5io import Kn5, Material, Node, world_pos, deform, split_mesh, update_bounds, unit, weld_normals

STEPS = ['wheels_black', 'mud_flaps', 'bar_rusty', 'plate', 'quarter_crease', 'mirror_primer', 'window_sticker', 'hood_fade', 'door_numbers', 'front_end', 'lower_panel', 'grille', 'clear_lamps', 'taillight_damage', 'trunk_face', 'pinstripe', 'paint_damage',
         'rear_seat_delete', 'bucket_seat', 'shifter', 'floor_mat', 'head_unit']
PLATE_TEXT = '8XKQ407'          # made-up number (not the real plate); set to None to keep the original mod's plate
PLATE_MOSAIC = True             # pixelate the number like a censored photo
FRONT_PLATE = True              # hangs below the front bumper
RACE_NUMBER = '888'             # door number panels; None for no panels
FONTS = r'C:\Windows\Fonts'


# ------------------------------------------------------------------ small helpers
def png_bytes(im):
    b = io.BytesIO(); im.save(b, format='PNG', optimize=True); return b.getvalue()

def font(names, size):
    for n in names:
        try: return ImageFont.truetype(os.path.join(FONTS, n), size)
        except OSError: pass
    return ImageFont.load_default()

def simple_material(name, texture, ambient, diffuse, spec, spec_exp, shader='ksPerPixel', alpha_ref=0.0):
    m = Material(name, shader)
    for k_, v in (('ksAmbient', ambient), ('ksDiffuse', diffuse), ('ksSpecular', spec), ('ksSpecularEXP', spec_exp),
                  ('ksEmissive', 0.0), ('ksAlphaRef', alpha_ref)):
        m.set(k_, v)
    m.tex = [['txDiffuse', 0, texture]]
    return m

def flat_texture(rgb, size=8):
    return png_bytes(Image.new('RGB', (size, size), rgb))

def noise(shape, sigma, seed):
    """smooth tileable noise, 0..1"""
    n = gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma, mode='wrap')
    return (n - n.min()) / (n.max() - n.min())

def fix_winding(V, F):
    F = np.asarray(F).reshape(-1, 3).copy(); p = V[:, :3]; n = V[:, 3:6]
    fn = np.cross(p[F[:, 1]] - p[F[:, 0]], p[F[:, 2]] - p[F[:, 0]])
    bad = (fn * n[F].mean(1)).sum(1) < 0; F[bad] = F[bad][:, [0, 2, 1]]
    return F

class Builder:
    """collects flat-shaded polygons into one V/F pair"""
    def __init__(s): s.V = []; s.F = []; s.n = 0
    def poly(s, pts, nrm, uv=None):
        pts = np.asarray(pts, float); nrm = np.asarray(nrm, float); nrm = nrm / np.linalg.norm(nrm)
        e = pts[1] - pts[0]; e = e - nrm * (e @ nrm); e /= max(np.linalg.norm(e), 1e-12)
        uv = np.zeros((len(pts), 2)) if uv is None else np.asarray(uv, float)
        for p, t in zip(pts, uv): s.V.append([*p, *nrm, *t, *e])
        for i in range(1, len(pts) - 1): s.F.append([s.n, s.n + i, s.n + i + 1])
        s.n += len(pts)
    def done(s):
        V = np.array(s.V, float); return V, fix_winding(V, np.array(s.F, np.int64))


# ------------------------------------------------------------------ 1. wheels: satin black, polished cap + nuts stay
def wheels_black(k):
    k.set_texture('gh_satin_black.png', flat_texture((30, 30, 32)))
    mid = k.add_material(simple_material('gh_wheel_black', 'gh_satin_black.png', 0.33, 0.36, 0.22, 22.0))
    out = []
    for name in ('SSR_MKIILF', 'SSR_MKII', 'SSR_MKIII', 'SSR_MKIIILR'):
        n, parent, M = k.find(name)
        r = np.hypot(n.V[:, 1], n.V[:, 2])                       # wheel-local: x is the axle
        polished = np.isclose(n.V[:, 6], 0.7959, atol=2e-3) & (r < 0.0545)   # centre cap and lug nuts
        tri_black = ~polished[n.F].all(1)
        V, F = split_mesh(n, tri_black)
        m = Node(2, name + '_BLACK'); m.V = V; m.F = F; m.mat = mid; update_bounds(m)
        parent.children.append(m); out.append((name, len(V), len(n.V)))
    return out


# ------------------------------------------------------------------ 2. mud flaps behind all four wheels
def flap(sx, z_top, sweep=0.030, thick=0.006):
    xi, xo, yt, yb, rc = 0.600, 0.858, 0.420, 0.065, 0.030
    out = [(xi, yt), (0.800, yt), (xo, 0.330), (xo, yb + rc)]
    out += [(xo - rc + rc * np.cos(a), yb + rc + rc * np.sin(a)) for a in np.linspace(0, -np.pi / 2, 6)[1:]]
    out += [(xi + rc + rc * np.cos(a), yb + rc + rc * np.sin(a)) for a in np.linspace(-np.pi / 2, -np.pi, 6)]
    out = np.array(out)
    def P(xy, dz):
        z = z_top - sweep * (yt - xy[:, 1]) / (yt - yb) + dz
        return np.c_[sx * xy[:, 0], xy[:, 1], z]
    nf = np.array([0, sweep / (yt - yb), 1.0])                   # front face normal (faces the tyre for the rear flap)
    b = Builder(); uv = (out - out.min(0)) / (out.max(0) - out.min(0))
    b.poly(P(out, +thick / 2), nf, uv); b.poly(P(out, -thick / 2), -nf, uv)
    c = out.mean(0)
    for i in range(len(out)):
        a, d = out[i], out[(i + 1) % len(out)]; mid = (a + d) / 2 - c
        q = np.array([a, d]); A = P(q, thick / 2); B = P(q, -thick / 2)
        b.poly([A[0], A[1], B[1], B[0]], [sx * mid[0], mid[1], 0.0])
    return b.done()

def mud_flaps(k):
    k.set_texture('gh_rubber.png', flat_texture((52, 52, 54)))
    mid = k.add_material(simple_material('gh_mudflap', 'gh_rubber.png', 0.36, 0.40, 0.08, 8.0))
    Vs = []; Fs = []; off = 0
    for sx in (1, -1):
        for z_top in (0.962, -1.692):                             # just behind the front / rear tyre
            V, F = flap(sx, z_top); Vs.append(V); Fs.append(F + off); off += len(V)
    k.add_mesh('MUD_FLAPS', np.vstack(Vs), np.vstack(Fs), mid)
    return off


# ------------------------------------------------------------------ 3. bash bar: bare steel with surface rust
def rust_texture(w=256, h=2048, seed=7):
    steel = np.array([148, 148, 146.0]); rust = np.array([122, 74, 50.0]); dark = np.array([78, 50, 38.0])
    fine = noise((h, w), 1.2, seed); blot = noise((h, w), 22, seed + 1); streak = noise((h, w), (60, 3), seed + 2)
    along = (np.arange(h) + 0.5) / h
    bias = 0.10 * np.exp(-((along - 0.42) / 0.22) ** 2)[:, None]  # the middle of the bar is rustiest
    m = np.clip((0.55 * blot + 0.45 * streak + bias - 0.57) / 0.07, 0, 1)
    img = steel * (0.86 + 0.28 * fine[..., None]) * (1 - m[..., None]) + (rust * (1 - fine[..., None]) + dark * fine[..., None]) * m[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def bar_rusty(k):
    k.set_texture('gh_bar_rust.png', png_bytes(rust_texture()))
    mid = k.add_material(simple_material('gh_bar_steel', 'gh_bar_rust.png', 0.32, 0.55, 0.35, 30.0))
    n, _, _ = k.find('BASH_BAR'); n.mat = mid
    n.V[:, 7] /= n.V[:, 7].max()                                  # one pass of the texture along the tube, no tiling
    return len(n.V)


# ------------------------------------------------------------------ 4. licence plate
def plate_image(text, w=390, h=192, ss=4):
    W, H = w * ss, h * ss
    im = Image.new('RGB', (W, H), (236, 236, 230)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([3 * ss, 3 * ss, W - 3 * ss, H - 3 * ss], radius=10 * ss, outline=(120, 120, 118), width=2 * ss)
    f = font(['arialbd.ttf'], 22 * ss); t = 'C A L I F O R N I A'
    d.text((W / 2, 24 * ss), t, font=f, fill=(196, 40, 44), anchor='mm')
    f = font(['bahnschrift.ttf', 'arialbd.ttf'], 118 * ss)
    tw = d.textlength(text, font=f)
    glyph = Image.new('L', (int(tw) + 8 * ss, 150 * ss), 0); ImageDraw.Draw(glyph).text((4 * ss, 75 * ss), text, font=f, fill=255, anchor='lm')
    glyph = glyph.crop(glyph.getbbox()).resize((int(W * 0.90), int(H * 0.50)), Image.LANCZOS)   # tall condensed plate digits
    im.paste(Image.new('RGB', glyph.size, (38, 44, 122)), ((W - glyph.size[0]) // 2, int(H * 0.30)), glyph)
    for x0, col in ((12, (70, 110, 190)), (w - 12 - 44, (60, 160, 90))):                         # month / year stickers
        d.rounded_rectangle([x0 * ss, 10 * ss, (x0 + 44) * ss, 34 * ss], radius=3 * ss, fill=col)
    for x0 in (60, w - 60):                                                                       # bolt heads
        d.ellipse([(x0 - 6) * ss, 172 * ss, (x0 + 6) * ss, 184 * ss], fill=(70, 70, 70))
    d.rectangle([40 * ss, 166 * ss, W - 40 * ss, 168 * ss], fill=(200, 200, 196))
    im = im.resize((w, h), Image.LANCZOS)
    if PLATE_MOSAIC:
        box = (int(w * 0.04), int(h * 0.27), int(w * 0.96), int(h * 0.83)); nx, ny = 23, 7
        tile = im.crop(box).resize((nx, ny), Image.BOX)
        a = np.asarray(tile).astype(float); a += np.random.default_rng(5).uniform(-22, 22, (ny, nx, 1))   # blocks never line up into digits
        tile = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB')
        im.paste(tile.resize((box[2] - box[0], box[3] - box[1]), Image.NEAREST), box[:2])
    dirt = noise((h, w), 6, 3)[..., None]
    return Image.fromarray(np.clip(np.asarray(im).astype(float) * (0.86 + 0.14 * dirt), 0, 255).astype(np.uint8), 'RGB')

def plate(k):
    info = {}
    if PLATE_TEXT:
        t = [t for t in k.textures if t[1] == 'LicensePlate.png'][0]
        old = np.asarray(Image.open(io.BytesIO(t[2])).convert('RGBA')).copy()
        p = np.asarray(plate_image(PLATE_TEXT))
        for y0 in (8, 210):
            old[y0:y0 + p.shape[0], 8:8 + p.shape[1], :3] = p     # alpha channel (0 = matte) is left as the mod had it
        b = io.BytesIO(); Image.fromarray(old, 'RGBA').save(b, format='DDS'); t[2] = b.getvalue()
        info['texture'] = len(t[2])
    if not FRONT_PLATE:
        n, parent, _ = k.find('Rear_Plate_002.001'); parent.children.remove(n); info['front_plate'] = 'removed'
    return info


# ------------------------------------------------------------------ 5. right rear quarter: deep crease ahead of the fuel door
def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

def crease(w):
    w = w.copy(); x, y, z = w[:, 0], w[:, 1], w[:, 2]
    side = ss(-0.70, -0.78, x)                                    # outer skin of the right side only
    zv = -1.455 - 0.42 * (y - 0.72)                               # valley line: leans rearward toward the top
    dz = z - zv
    prof = np.where(dz > 0, np.exp(-(dz / 0.040) ** 2), np.exp(-(dz / 0.105) ** 2))   # sharp wall toward the door, soft toward the filler door
    ridge = 0.30 * np.exp(-((dz - 0.075) / 0.028) ** 2)           # metal pushed up into a ridge next to the fold
    hy = ss(0.585, 0.66, y) * (1 - ss(0.795, 0.845, y))
    door = ss(-1.335, -1.375, z)                                  # fade out before the door shut line
    depth = 0.058 * (0.75 + 0.25 * np.sin(38 * y + 1.3))
    d = depth * (prof - ridge) * hy * side * door
    # shallow oil-canning of the panel between the fold and the filler door
    d += 0.010 * np.exp(-((z + 1.62) / 0.10) ** 2) * np.exp(-((y - 0.76) / 0.07) ** 2) * side
    w[:, 0] = x + d                                               # +x is inboard on the right side
    w[:, 1] = y - 0.25 * d * hy
    return w

def subdivide(n, M, box, levels=2, min_edge=0.012):
    """1->4 split of the triangles inside a world-space box (with conforming splits around it) so the fold has
    enough vertices to bend around. Vertices are matched by position so UV seams do not crack."""
    lo, hi = np.array(box[0]), np.array(box[1])
    for _ in range(levels):
        V = n.V.astype(np.float64); F = n.F; w = world_pos(n, M)
        key = np.round(V[:, :3] * 1e5).astype(np.int64); _, wid = np.unique(key, axis=0, return_inverse=True); wid = wid.ravel()
        c = w[F].mean(1); inside = ((c > lo) & (c < hi)).all(1)
        el = np.linalg.norm(w[F] - w[np.roll(F, -1, axis=1)], axis=2).max(1)
        marked = inside & (el > min_edge)
        if not marked.any(): break
        split = set()
        for t in F[marked]:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])): split.add((min(wid[a], wid[b]), max(wid[a], wid[b])))
        newV = []; cache = {}
        def midpoint(a, b):
            kk = (min(a, b), max(a, b))
            if kk not in cache:
                v = (V[a] + V[b]) / 2
                v[3:6] /= max(np.linalg.norm(v[3:6]), 1e-12); v[8:11] /= max(np.linalg.norm(v[8:11]), 1e-12)
                cache[kk] = len(V) + len(newV); newV.append(v)
            return cache[kk]
        out = []
        for t in F:
            e = [(min(wid[t[i]], wid[t[(i + 1) % 3]]), max(wid[t[i]], wid[t[(i + 1) % 3]])) in split for i in range(3)]
            m = [midpoint(t[i], t[(i + 1) % 3]) if e[i] else None for i in range(3)]
            ns = sum(e)
            if ns == 0: out.append(t)
            elif ns == 3: out += [[t[0], m[0], m[2]], [m[0], t[1], m[1]], [m[2], m[1], t[2]], [m[0], m[1], m[2]]]
            elif ns == 1:
                i = e.index(True); a, b, c_ = t[i], t[(i + 1) % 3], t[(i + 2) % 3]
                out += [[a, m[i], c_], [m[i], b, c_]]
            else:
                i = e.index(False); a, b, c_ = t[i], t[(i + 1) % 3], t[(i + 2) % 3]      # edge a-b is whole
                mb, mc = m[(i + 1) % 3], m[(i + 2) % 3]                                    # on b-c and c-a
                out += [[a, b, mb], [a, mb, mc], [mc, mb, c_]]
        n.V = np.vstack([V, np.array(newV)]).astype(np.float32); n.F = np.array(out, np.int64)
    update_bounds(n)

def quarter_crease(k):
    info = {}
    n, _, M = k.find('r_fenders'); v0 = len(n.V)
    subdivide(n, M, ((-0.90, 0.54, -1.80), (-0.68, 0.92, -1.30)))
    info['r_fenders_new_verts'] = len(n.V) - v0
    for name in ('r_fenders', 'body_part004'):                          # skin + the bright trim that sits on it
        n, _, M = k.find(name); info[name] = deform(n, M, crease, thr=1e-4)
    return info


# ------------------------------------------------------------------ 6. mirrors: dull grey shells with a worn-through tan patch
def mirror_texture(size=256, seed=11):
    """grey everywhere except one ragged, speckled tan blob centred at (0.55, 0.36)"""
    grey = np.array([104, 108, 114.0]); tan = np.array([204, 172, 104.0])
    yy, xx = np.mgrid[0:size, 0:size] / size
    edge = noise((size, size), 7, seed); fine = noise((size, size), 0.8, seed + 1); streak = noise((size, size), (1.0, 5.0), seed + 2)
    r = np.hypot((xx - 0.55) / 0.36, (yy - 0.36) / 0.21)
    m = np.clip((1.0 - r + 0.55 * (edge - 0.5)) / 0.30, 0, 1)
    m = m * (streak > 0.62 - 0.45 * m)                            # solid in the middle, breaking into flecks at the rim
    img = (grey * (1 - m[..., None]) + tan * m[..., None]) * (0.92 + 0.16 * fine[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def mirror_primer(k):
    k.set_texture('gh_mirror_primer.png', png_bytes(mirror_texture()))
    mid = k.add_material(simple_material('gh_mirror_primer', 'gh_mirror_primer.png', 0.38, 0.45, 0.14, 14.0))
    n, parent, M = k.find('mirrors'); n.mat = mid                 # both mirrors; the mesh sits in world space
    x = np.abs(n.V[:, 0]); y = n.V[:, 1]
    u = (x - 0.80) / 0.14; v = (0.907 - y) / 0.107                # housing seen from the front: u outboard, v down
    left = n.V[:, 0] > 0
    # driver side: the patch covers the upper middle of the shell. passenger side: a smaller patch near the top
    u = np.where(left, u, 0.55 + (u - 0.50) * 1.9); v = np.where(left, v, 0.36 + (v - 0.14) * 1.9)
    back = n.V[:, 5] < -0.3                                       # the mirror surround (faces rearward) stays plain
    n.V[:, 6] = np.where(back, 0.02, np.clip(u, 0.01, 0.99)); n.V[:, 7] = np.where(back, 0.98, np.clip(v, 0.01, 0.99))
    return len(n.V)

# ------------------------------------------------------------------ 7. "TIRE STREETS" sticker, left rear door glass
def sticker_image(w=1024, h=512):
    """green slanted block with TIRE knocked out in black, hand-brushed white STREETS underneath (see Reference Photos)"""
    def word(text, fonts, size, fill, box, stroke=0, stroke_fill=None, lean=0.0):
        f = font(fonts, size)
        im = Image.new('RGBA', (size * (len(text) + 2), size * 3), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((size, size), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        im = im.crop(im.getbbox()).resize(box, Image.LANCZOS)
        if lean:
            pad = int(lean * box[1]) + 2; c = Image.new('RGBA', (box[0] + pad, box[1]), (0, 0, 0, 0)); c.alpha_composite(im, (0, 0))
            im = c.transform(c.size, Image.AFFINE, (1, lean, -pad, 0, 1, 0), Image.BICUBIC)
        return im
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    top, bot, lean = 16, int(h * 0.56), 0.42
    x0, x1 = 60, w - 16
    d.polygon([(x0 + lean * (bot - top), top), (x1, top), (x1 - lean * (bot - top), bot), (x0, bot)], fill=(112, 222, 44, 255))
    tire = word('TIRE', ['impact.ttf', 'arialbd.ttf'], 300, (18, 22, 18, 255), (int(w * 0.50), int((bot - top) * 0.62)), lean=lean)
    im.alpha_composite(tire, ((w - tire.size[0]) // 2 + 20, top + int((bot - top) * 0.19)))
    # worn print: scratch some of the black back to green
    a = np.asarray(im).copy(); scr = noise((h, w), (0.7, 14), 31) > 0.80
    blk = (a[..., :3].sum(2) < 120) & (a[..., 3] > 0) & scr; a[blk, :3] = (112, 222, 44); im = Image.fromarray(a, 'RGBA')
    st = word('STREETS', ['segoeprb.ttf', 'Inkfree.ttf', 'comicz.ttf', 'impact.ttf'], 240, (236, 236, 240, 255),
              (int(w * 0.74), int(h * 0.40)), stroke=5, stroke_fill=(236, 236, 240, 255), lean=0.30)
    sh = Image.new('RGBA', st.size, (30, 32, 36, 255)); sh.putalpha(st.getchannel('A').filter(ImageFilter.MaxFilter(9)))
    y = int(h * 0.50)
    im.alpha_composite(sh, (8, y)); im.alpha_composite(st, (8, y))
    return im
def window_sticker(k):
    k.set_texture('gh_sticker.png', png_bytes(sticker_image()))
    m = simple_material('gh_sticker', 'gh_sticker.png', 0.45, 0.45, 0.05, 10.0, shader='ksPerPixelAT', alpha_ref=0.5)
    m.alpha_tested = 1; mid = k.add_material(m)
    n, _, M = k.find('window_ext'); w = world_pos(n, M)
    z0, z1, y0, y1 = -0.475, -0.915, 0.950, 1.170                  # front edge, rear edge, bottom, top (clear of the door frame)
    g = w[(w[:, 0] > 0.5) & (w[:, 2] < z0 + 0.05) & (w[:, 2] > z1 - 0.05) & (w[:, 1] > y0 - 0.05) & (w[:, 1] < y1 + 0.05)]
    A = np.c_[np.ones(len(g)), g[:, 1], g[:, 2]]; coef = np.linalg.lstsq(A, g[:, 0], rcond=None)[0]   # glass plane x(y,z)
    nrm = np.array([1.0, -coef[1], -coef[2]]); nrm /= np.linalg.norm(nrm)
    b = Builder(); nx, ny = 6, 4
    for off, sgn in ((0.004, 1), (-0.004, -1)):                    # outside and inside of the glass
        for i in range(nx):
            for j in range(ny):
                q = []
                for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    z = z0 + (z1 - z0) * u / nx; y = y1 + (y0 - y1) * v / ny
                    q.append(([coef[0] + coef[1] * y + coef[2] * z, y, z] + off * nrm, (u / nx, v / ny)))
                b.poly([p for p, _ in q], sgn * nrm, [t for _, t in q])
    V, F = b.done()
    k.add_mesh('WINDOW_STICKER', V, F, mid, shadows=0)
    return {'glass_points': len(g), 'plane': [round(float(c), 3) for c in coef]}


# ------------------------------------------------------------------ 8. hood: sun-faded, cloudy clear coat (painted into the body texture)
def dxt5_with_mips(rgb, alpha, header):
    """2048x2048 DXT5 with 12 mips; mips are built from colour only (this texture's alpha is all zero - see paint.py)"""
    def block(rgba):
        b = io.BytesIO(); rgba.save(b, format='DDS', pixel_format='DXT5'); return b.getvalue()[128:]
    levels = []; w = rgb.size[0]
    while w >= 1:
        if w >= 4:
            lv = rgb if w == rgb.size[0] else rgb.resize((w, w), Image.BOX if w >= 512 else Image.LANCZOS)
            al = alpha if w == rgb.size[0] else alpha.resize((w, w), Image.NEAREST)
            levels.append(block(Image.merge('RGBA', (*lv.split(), al))))
        else:
            px = rgb.resize((w, w), Image.BOX).resize((4, 4), Image.NEAREST)
            levels.append(block(Image.merge('RGBA', (*px.split(), Image.new('L', (4, 4), max(alpha.getextrema()))))))
        w //= 2
    return bytes(header) + b''.join(levels)

def hood_fade(k):
    t = [t for t in k.textures if t[1] == 'carpaint_diffuse.dds'][0]
    src = Image.open(io.BytesIO(t[2])).convert('RGBA'); a = np.asarray(src).astype(float); S = a.shape[0]
    n, _, M = k.find('bonnet'); w = world_pos(n, M); uv = n.V[:, 6:8].astype(float)
    cu = np.linalg.lstsq(np.c_[np.ones(len(w)), w[:, 0], w[:, 2]], uv[:, 0], rcond=None)[0]     # the hood is a flat top-down unwrap
    cv = np.linalg.lstsq(np.c_[np.ones(len(w)), w[:, 0], w[:, 2]], np.mod(uv[:, 1], 1.0), rcond=None)[0]
    px0, px1 = int(uv[:, 0].min() * S) - 2, int(uv[:, 0].max() * S) + 3
    py0, py1 = int(np.mod(uv[:, 1], 1.0).min() * S) - 2, int(np.mod(uv[:, 1], 1.0).max() * S) + 3
    yy, xx = np.mgrid[py0:py1, px0:px1]; U = (xx + 0.5) / S; Vv = (yy + 0.5) / S
    A = np.array([[cu[1], cu[2]], [cv[1], cv[2]]]); xz = np.linalg.solve(A, np.stack([U.ravel() - cu[0], Vv.ravel() - cv[0]]))
    X = xz[0].reshape(U.shape); Z = xz[1].reshape(U.shape)                                         # world position of every hood texel
    cloud = noise(U.shape, 38, 21); mott = noise(U.shape, 9, 22); fine = noise(U.shape, 1.0, 23)
    centre = np.exp(-(X / 0.62) ** 2) * ss(0.78, 1.00, Z) * (1 - ss(1.90, 2.04, Z))                # strongest down the middle, clean at the edges
    m = np.clip((0.90 * cloud + 0.10 * mott - 0.30) / 0.45, 0, 1) * centre          # soft, low-contrast haze
    dark = 0.0 * np.exp(-((X + 0.05) / 0.22) ** 2 - ((Z - 1.62) / 0.20) ** 2)                       # the darker blotch near the front centre
    chalk = np.array([150, 156, 166.0])
    reg = a[py0:py1, px0:px1, :3]
    out = reg * (1 - 0.16 * m[..., None]) + chalk * 0.16 * m[..., None]
    out *= (1 - 0.28 * dark[..., None]) * (0.99 + 0.02 * fine[..., None])
    a[py0:py1, px0:px1, :3] = np.where(reg.sum(2, keepdims=True) > 30, out, reg)                    # leave the black gutter between islands alone
    rgb = Image.fromarray(np.clip(a[..., :3], 0, 255).astype(np.uint8), 'RGB')
    dds = dxt5_with_mips(rgb, Image.fromarray(a[..., 3].astype(np.uint8)), t[2][:128])
    assert len(dds) == len(t[2]); t[2] = dds
    skin = config.PROJECT / 'gentlehands_toyota_cressida_mx83' / 'skins' / 'silver_176' / 'carpaint_diffuse.dds'
    if skin.exists(): skin.write_bytes(dds)                       # the skin's copy overrides the one inside the kn5
    return {'texels': [px0, py0, px1, py1], 'coverage': round(float((m > 0.3).mean()), 2), 'skin': skin.exists()}

# ------------------------------------------------------------------ 9. race number panels on the front doors
def number_panel_image(number, w=512, h=500, ss=2):
    W, H = w * ss, h * ss
    im = Image.new('RGB', (W, H), (238, 238, 234)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W - 1, H - 1], outline=(20, 20, 22), width=6 * ss)
    d.rectangle([14 * ss, 14 * ss, W - 14 * ss, 150 * ss], fill=(22, 22, 24))
    f = font(['arialbi.ttf', 'arialbd.ttf'], 100 * ss)
    d.text((W / 2, 82 * ss), 'ONGRID', font=f, fill=(238, 238, 234), anchor='mm', stroke_width=2 * ss, stroke_fill=(238, 238, 234))
    f = font(['arialbd.ttf', 'impact.ttf'], 330 * ss)
    g = Image.new('L', (W * 3, H), 0); ImageDraw.Draw(g).text((20, 0), number, font=f, fill=255)
    g = g.crop(g.getbbox()).resize((int(W * 0.78), int(H * 0.40)), Image.LANCZOS)
    im.paste(Image.new('RGB', g.size, (20, 20, 22)), ((W - g.size[0]) // 2, int(H * 0.39)), g)
    d.rectangle([60 * ss, int(H * 0.90), W - 60 * ss, int(H * 0.90) + 5 * ss], fill=(20, 20, 22))
    f = font(['arialbd.ttf'], 22 * ss); d.text((W / 2, H * 0.945), 'TRACK DAYS', font=f, fill=(20, 20, 22), anchor='mm')
    im = im.resize((w, h), Image.LANCZOS)
    dirt = noise((h, w), 10, 41)[..., None]
    return Image.fromarray(np.clip(np.asarray(im).astype(float) * (0.90 + 0.10 * dirt), 0, 255).astype(np.uint8), 'RGB')

def door_numbers(k):
    if not RACE_NUMBER: return None
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
    k.set_texture('gh_number_panel.png', png_bytes(number_panel_image(RACE_NUMBER)))
    mid = k.add_material(simple_material('gh_number_panel', 'gh_number_panel.png', 0.42, 0.50, 0.12, 20.0))
    zc, half_w, y0, y1 = 0.185, 0.140, 0.445, 0.708               # magnet panel between the side moulding and the pinstripe
    b = Builder(); nu, nv = 16, 16; info = {}
    for name, sx in (('lf_door', 1), ('rf_door', -1)):
        n, _, M = k.find(name); w = world_pos(n, M)
        skin = (n.V[:, 3] * sx > 0.5)                             # outward-facing door skin only
        lin = LinearNDInterpolator(w[skin][:, [2, 1]], w[skin][:, 0] * sx); near = NearestNDInterpolator(w[skin][:, [2, 1]], w[skin][:, 0] * sx)
        def surf(z, y):
            v = float(lin(z, y)); return v if np.isfinite(v) else float(near(z, y))
        def P(u, v):
            z = zc - sx * half_w * (2 * u - 1); y = y1 + (y0 - y1) * v     # u runs left->right as read from outside
            return np.array([sx * (surf(z, y) + 0.006), y, z])
        for i in range(nu):
            for j in range(nv):
                c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                pts = [P(a / nu, c_ / nv) for a, c_ in c]
                nrm = np.cross(pts[1] - pts[0], pts[3] - pts[0]); nrm *= np.sign(nrm[0] * sx)
                b.poly(pts, nrm, [(a / nu, c_ / nv) for a, c_ in c])
        info[name] = (int(skin.sum()), [round(float(v), 3) for v in (w[skin][:, 1].min(), w[skin][:, 1].max(), w[skin][:, 2].min(), w[skin][:, 2].max())])
    V, F = b.done(); assert np.isfinite(V).all(), 'panel left the door surface'
    k.add_mesh('DOOR_NUMBERS', V, F, mid, shadows=0)
    return info

# ------------------------------------------------------------------ 10. front end: US bumper with the lower valance taken off
def drop_tris(n, mask):
    n.F = n.F[~mask]; update_bounds(n); return int(mask.sum())

LAMP_X = (0.425, 0.585)           # the amber lamps under the bumper: inner / outer edge (m from the centre line)
NOTCH_Y = 0.312                   # height of the bumper's lower edge in the cut-outs above them

def bumper_path(k):
    """outline of the bumper rub strip seen from above, in polar form about (x 0, z 1.5):
    -> bin angles, radius per bin, (x, z) points from the right wheel arch round to the left, unit tangents, unit inward normals"""
    s, _, Ms = k.find('front_bumper_SUB1'); ws = world_pos(s, Ms); ws = ws[ws[:, 1] < 0.375]
    ang = np.arctan2(ws[:, 0], ws[:, 2] - 1.5); rad = np.hypot(ws[:, 0], ws[:, 2] - 1.5)
    edges = np.linspace(ang.min(), ang.max(), 141); ac = (edges[:-1] + edges[1:]) / 2
    r = np.array([rad[(ang >= a) & (ang <= b_)].max() if ((ang >= a) & (ang <= b_)).any() else np.nan for a, b_ in zip(edges[:-1], edges[1:])])
    ok = np.isfinite(r); r = np.interp(ac, ac[ok], r[ok]); r = gaussian_filter(r, 1.0, mode='nearest')
    path = np.c_[r * np.sin(ac), 1.5 + r * np.cos(ac)]
    def run_back(p, q):                                           # carry each end straight on to just ahead of the tyre
        d = (p - q) / np.linalg.norm(p - q); m = int(np.ceil(max(p[1] - 1.632, 0) / 0.012))
        return [p + d * (p[1] - 1.632) / max(-d[1], 0.5) * (j + 1) / m for j in range(m)]
    path = np.array(run_back(path[0], path[2])[::-1] + list(path) + run_back(path[-1], path[-3]))
    tg = np.gradient(path, axis=0); tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    inward = np.c_[-tg[:, 1], tg[:, 0]]; inward *= np.sign((inward * (np.array([0, 1.5]) - path)).sum(1))[:, None]
    return ac, r, path, tg, inward

def sweep(path, tg, inward, prof, uv, caps=False, both=False, shift=None, ytop=None):
    """sweep a (setback, height) profile along the outline; smooth normals pointing away from the car"""
    from kn5io import weld_normals
    V = []; F = []; np_ = len(prof); y0 = min(y for _, y in prof); y1 = max(y for _, y in prof)
    for i, (p, q) in enumerate(zip(path, inward)):
        for d, y in prof:
            if shift is not None: d = d + shift[i]                # per-station extra setback
            if ytop is not None and ytop[i] < y1: y = y0 + (y - y0) * (ytop[i] - y0) / (y1 - y0)   # per-station ceiling: squash, keep the bottom
            V.append([p[0] + q[0] * d, y, p[1] + q[1] * d, 0, 0, 0, uv[0], uv[1], tg[i, 0], 0, tg[i, 1]])
    for i in range(len(path) - 1):
        for j in range(np_ - 1):
            a = i * np_ + j; b_ = a + np_; F += [[a, b_, a + 1], [a + 1, b_, b_ + 1]]
    V = np.array(V, float); F = np.array(F, np.int64)
    nr = unit(weld_normals(V[:, :3], F)); mid = np.mean([y for _, y in prof])
    out = V[:, :3] - np.array([0, mid, 1.5]); out[:, 1] *= 6; nr *= np.sign((nr * out).sum(1))[:, None]
    V[:, 3:6] = nr
    if caps:                                                      # flat ends, facing along the outline
        for i, sgn in ((0, -1.0), (len(path) - 1, 1.0)):
            c = V[i * np_:(i + 1) * np_].copy(); c[:, 3:6] = [sgn * tg[i, 0], 0, sgn * tg[i, 1]]; o = len(V)
            V = np.vstack([V, c]); F = np.vstack([F, [[o, o + j, o + j + 1] for j in range(1, np_ - 1)]])
    F = fix_winding(V, F)
    if both:                                                      # visible from behind as well (no see-through backs)
        B_ = V.copy(); B_[:, :3] -= 0.0015 * B_[:, 3:6]; B_[:, 3:6] *= -1;   # 1.5 mm inside, so the two skins never z-fight
        F = np.vstack([F, F[:, [0, 2, 1]] + len(V)]); V = np.vstack([V, B_])
    return V, F

# ------------------------------------------------------------------ 10b. the panel under the bumper that the valance used to bolt to, with its two amber lamps
def lower_panel(k):
    """what is left under the bumper with the valance gone: a black beam, the two amber lamps hanging in notches of the
    bumper's lower edge, a narrow body-colour strip between them, and grey plastic flaps under the bumper corners"""
    ac, r, path, tg, inward = bumper_path(k)
    n, _, M = k.find('BUMPER_LOWER'); uv = n.V[0, 6:8]; black = k.mat_id('plastic_black1')
    x, z = path[:, 0], path[:, 1]; nose = (np.abs(x) < 0.60) & (z > 1.75)
    def part(name, sel, prof, mat, uv_=(0.5, 0.5), **kw):
        kw = {k_: v[sel] for k_, v in kw.items()}
        V, F = sweep(path[sel], tg[sel], inward[sel], prof, uv_, caps=True, both=True, **kw); k.add_mesh(name, V, F, mat)
    part('BUMPER_BEAM', (np.abs(x) < 0.72) & (z > 1.75), [(0.140, 0.305), (0.140, 0.250), (0.140, 0.195)], black)
    k.set_texture('gh_amber.png', flat_texture((214, 112, 26)))
    mid = k.add_material(simple_material('gh_front_marker', 'gh_amber.png', 0.45, 0.55, 0.45, 60.0))
    # the piece the valance bolted to is one part right round the nose: a body-colour lip along the bottom, and above it
    # (outboard of the lamps only) a grey plastic face. Nearly flush with the bumper along the sides, set back across the front.
    # Seen from the side face + lip are one smooth S: out from behind the bumper's lower edge (and behind the lamps), back
    # and down, forward again into a belly, then tucked under; the lip alone (centre, face cut away) is the lower half, a C.
    back = 0.040 * ss(1.72, 1.95, z)
    lip = [(0.110, 0.252), (0.086, 0.252), (0.080, 0.250), (0.078, 0.236), (0.068, 0.222), (0.062, 0.210), (0.062, 0.200), (0.068, 0.192), (0.082, 0.187), (0.120, 0.186)]
    part('BUMPER_PANEL', np.ones(len(x), bool), lip, n.mat, uv, shift=back)
    face = [(0.048, 0.318), (0.048, 0.296), (0.056, 0.280), (0.072, 0.264), (0.080, 0.250)]   # starts up behind the bumper edge and the lamps
    for side in (-1, 1):
        nm = 'RL'[side > 0]; xs = x * side
        lamp = nose & (xs > LAMP_X[0]) & (xs < LAMP_X[1]); pocket = nose & (xs > LAMP_X[0] - 0.012) & (xs < LAMP_X[1] + 0.012)
        part('FRONT_MARKER_' + nm, lamp, [(0.080, 0.304), (0.048, 0.303), (0.046, 0.300), (0.046, 0.266), (0.048, 0.263), (0.080, 0.262)], mid)
        part('FRONT_MARKER_HOUSING_' + nm, pocket, [(0.090, 0.310), (0.053, 0.309), (0.053, 0.262), (0.090, 0.261)], black)
        part('BUMPER_FLAP_' + nm, (xs > LAMP_X[0] - 0.03) | ((xs > 0) & ~nose), face, k.mat_id('gh_mudflap'), shift=back)
    return {'stations': len(path)}

def front_end(k):
    info = {}
    # the aero lip under the bumper goes (the side skirts and the pieces behind the front wheels stay)
    n, _, M = k.find('body_016'); c = world_pos(n, M)[n.F].mean(1); info['lip_tris'] = drop_tris(n, c[:, 2] > 1.55)
    # everything on the bumper below the rub strip (slotted lower face) goes too...
    n, _, M = k.find('f_bumper'); w = world_pos(n, M); c = w[n.F].mean(1)
    up = np.where((np.abs(w[:, 0] - 0.30) < 0.05) & (np.abs(w[:, 1] - 0.44) < 0.02) & (w[:, 2] > 2.0))[0]
    uv = n.V[up[0], 6:8].copy()                                   # a plain, evenly lit paint texel
    info['bumper_tris'] = drop_tris(n, (w[n.F][:, :, 1] < 0.372).all(1))
    lw = n.V[:, 1] < 0.366; n.V[lw, 1] = 0.366                    # straight lower edge instead of a saw-tooth (mesh is in world space)
    # ...and is replaced by a plain lower face that follows the strip around the nose and tucks under at the bottom
    ac, r, path, tg, inward = bumper_path(k)
    notch = (np.abs(np.abs(path[:, 0]) - np.mean(LAMP_X)) < np.ptp(LAMP_X) / 2 + 0.02) & (path[:, 1] > 1.75)   # cut-outs over the lamps
    prof = [(0.006, 0.370), (0.006, 0.348), (0.009, 0.326), (0.017, 0.307), (0.031, 0.294), (0.052, 0.287), (0.080, 0.285), (0.120, 0.288)]
    V = []; F = []; np_ = len(prof)
    for i, (p, q) in enumerate(zip(path, inward)):
        for j, (d, y) in enumerate(prof):
            V.append([p[0] + q[0] * d, max(y, NOTCH_Y) if notch[i] else y, p[1] + q[1] * d, 0, 0, 0, uv[0], uv[1], tg[i, 0], 0, tg[i, 1]])
    for i in range(len(path) - 1):
        for j in range(np_ - 1):
            a = i * np_ + j; b_ = a + np_; F += [[a, b_, a + 1], [a + 1, b_, b_ + 1]]
    V = np.array(V, float); F = np.array(F, np.int64)
    from kn5io import weld_normals
    nr = unit(weld_normals(V[:, :3], F)); out = V[:, :3] - np.array([0, 0.30, 1.5]); nr *= np.sign((nr * out).sum(1))[:, None]
    V[:, 3:6] = nr
    k.add_mesh('BUMPER_LOWER', V, fix_winding(V, F), n.mat)
    info['lower_face_verts'] = len(V)
    # the black underbody used to hide behind the lip; pull its nose back inside the new bumper
    c_, _, Mc = k.find('chasis')
    def tuck(w):
        w = w.copy(); a_ = np.arctan2(w[:, 0], w[:, 2] - 1.5); rr = np.hypot(w[:, 0], w[:, 2] - 1.5)
        lim = np.interp(a_, ac, r) - 0.13                        # stay inside the bumper shell
        m = (w[:, 1] < 0.36) & (w[:, 2] > 1.5) & (np.abs(a_) < ac.max()) & (rr > lim)
        s_ = np.where(m, lim / np.maximum(rr, 1e-9), 1.0)
        w[:, 0] *= s_; w[:, 2] = 1.5 + (w[:, 2] - 1.5) * s_
        lift = ss(1.64, 1.74, w[:, 2])                           # everything ahead of the front tyres
        w[:, 1] += lift * np.maximum(0.305 - w[:, 1], 0)
        return w
    info['underbody_verts'] = deform(c_, Mc, tuck)
    if FRONT_PLATE:
        pl, _, Mp = k.find('Rear_Plate_002.001'); deform(pl, Mp, lambda w: w + np.array([0, 0.025, 0.040]))
    return info

# ------------------------------------------------------------------ 11. grille: black with horizontal bright bars, old crest + TOYOTA badge
def components(p, F):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    key = np.round(p * 1e4).astype(np.int64); _, wid = np.unique(key, axis=0, return_inverse=True); wid = wid.ravel()
    e = np.r_[np.c_[wid[F[:, 0]], wid[F[:, 1]]], np.c_[wid[F[:, 1]], wid[F[:, 2]]]]
    nc, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(wid.max() + 1,) * 2), directed=False)
    return nc, lab[wid]

def badge_image(w=512, h=256):
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    silver, edge, field = (206, 210, 214, 255), (120, 124, 130, 255), (34, 36, 40, 255)
    def shield(cx, cy, sx, sy):
        return [(cx - sx, cy - sy), (cx + sx, cy - sy), (cx + sx, cy + 0.25 * sy), (cx, cy + sy), (cx - sx, cy + 0.25 * sy)]
    cx, cy = 80, 128
    d.polygon(shield(cx, cy, 62, 110), fill=edge); d.polygon(shield(cx, cy, 56, 102), fill=silver); d.polygon(shield(cx, cy, 44, 86), fill=field)
    d.polygon([(cx - 34, cy - 62), (cx + 34, cy - 62), (cx + 34, cy - 40), (cx + 10, cy - 40), (cx + 10, cy + 40), (cx - 10, cy + 40), (cx - 10, cy - 40), (cx - 34, cy - 40)], fill=silver)
    d.polygon([(cx - 34, cy - 22), (cx - 18, cy - 22), (cx - 18, cy + 22), (cx, cy + 58), (cx + 18, cy + 22), (cx + 18, cy - 22), (cx + 34, cy - 22), (cx + 34, cy + 28), (cx, cy + 74), (cx - 34, cy + 28)], fill=silver)
    f = font(['arialbd.ttf'], 150)
    g = Image.new('L', (1400, 300), 0); ImageDraw.Draw(g).text((10, 10), 'T O Y O T A', font=f, fill=255)
    g = g.crop(g.getbbox()).resize((306, 62), Image.LANCZOS)
    d.rounded_rectangle((160, 76, 506, 180), 10, fill=(20, 21, 24, 255), outline=silver, width=5)   # black plaque, bright rim
    im.alpha_composite(Image.new('RGBA', (1, 1), (0, 0, 0, 0)), (0, 0))
    sh = Image.new('RGBA', g.size, edge); sh.putalpha(g.filter(ImageFilter.MaxFilter(5))); im.alpha_composite(sh, (170, 93))
    tx = Image.new('RGBA', g.size, silver); tx.putalpha(g); im.alpha_composite(tx, (170, 93))
    return im

def grille(k):
    info = {}
    n, parent, M = k.find('radiator_grill'); w = world_pos(n, M); nc, lab = components(w, n.F)
    size = np.array([w[lab == c].max(0) - w[lab == c].min(0) for c in range(nc)])
    vert = np.where((size[:, 0] < 0.01) & (size[:, 1] > 0.10))[0]; horiz = np.where((size[:, 1] < 0.02) & (size[:, 0] > 0.6))[0]
    assert len(vert) == 19 and len(horiz) == 3, (len(vert), len(horiz))
    # egg-crate: twice as many bars each way (copies of the ones the mesh has, following the lean of the grille), all black
    order = horiz[np.argsort([w[lab == c][:, 1].min() for c in horiz])]
    vorder = vert[np.argsort([w[lab == c][:, 0].mean() for c in vert])]; vc = np.array([w[lab == c].mean(0) for c in vorder])
    copies = [(order[0], (0, -0.015, 0.001)), (order[0], (0, 0.015, -0.001)), (order[1], (0, 0.015, -0.005)), (order[2], (0, 0.015, -0.007)), (order[2], (0, 0.030, -0.015))]
    copies += [(c, (d[0] / 2, 0, d[2] / 2)) for c, d in zip(vorder[:-1], np.diff(vc, axis=0))]
    addV = []; addF = []; off = len(n.V); Mi = np.linalg.inv(M[:3, :3])
    for c, d in copies:
        vi = np.where(lab == c)[0]; remap = -np.ones(len(n.V), np.int64); remap[vi] = np.arange(len(vi)) + off
        V = n.V[vi].copy(); V[:, :3] += np.array(d) @ Mi                       # world-space offset (the mesh is in cm, rotated)
        tri = n.F[np.isin(n.F[:, 0], vi)]; addV.append(V); addF.append(remap[tri]); off += len(vi)
    is_bar = np.isin(lab, np.r_[vert, horiz])
    n.V = np.vstack([n.V] + addV); n.F = np.vstack([n.F] + addF)
    is_bar = np.r_[is_bar, np.ones(len(n.V) - len(is_bar), bool)]
    V, F = split_mesh(n, is_bar[n.F].all(1))                                  # only the surround stays bright
    m = Node(2, 'grille_uprights'); m.V = V; m.F = F; m.mat = k.mat_id('plastic_black1'); update_bounds(m); parent.children.append(m)
    info['uprights'] = len(V); info['bars_added'] = len(addV)
    # modern oval off, crest + lettering on
    for name in ('logo_part002', 'logo_part001'):
        nd, par, _ = k.find(name); par.children.remove(nd)
    k.set_texture('gh_grille_badges.png', png_bytes(badge_image()))
    mat = simple_material('gh_grille_badges', 'gh_grille_badges.png', 0.50, 0.50, 0.60, 40.0, shader='ksPerPixelAT', alpha_ref=0.5); mat.alpha_tested = 1
    mid = k.add_material(mat)
    w = np.vstack([world_pos(n, M), world_pos(m, M)])                          # surround + bars
    def front(x, y0, y1):
        s = w[(np.abs(w[:, 0] - x) < 0.03) & (w[:, 1] > y0) & (w[:, 1] < y1)]; return s[:, 2].max() + 0.004
    b = Builder()
    z = front(0.0, 0.49, 0.58)
    b.poly([(-0.030, 0.492, z), (0.030, 0.492, z), (0.030, 0.574, z - 0.004), (-0.030, 0.574, z - 0.004)], (0, 0, 1), [(0, 1), (0.3125, 1), (0.3125, 0), (0, 0)])
    xa, xb = 0.155, 0.305; za, zb = front(xa, 0.46, 0.50), front(xb, 0.46, 0.50)
    b.poly([(xa, 0.470, za), (xb, 0.470, zb), (xb, 0.497, zb), (xa, 0.497, za)], (0, 0, 1), [(0.305, 0.715), (0.995, 0.715), (0.995, 0.285), (0.305, 0.285)])
    V, F = b.done(); k.add_mesh('GRILLE_BADGES', V, F, mid, shadows=0)
    return info


# ------------------------------------------------------------------ 12. headlight + corner lenses: clear, slightly hazed, instead of smoked
def clear_lamps(k):
    t = [t for t in k.textures if t[1] == 'cressida_lights.dds'][0]
    a = np.asarray(Image.open(io.BytesIO(t[2])).convert('RGBA')).astype(float); S = a.shape[0]
    yy, xx = np.mgrid[0:S, 0:S] / S
    head = (xx > 0.685) & (yy > 0.245); corner = (xx < 0.343) & (yy > 0.598); reg = head | corner   # lens islands of the atlas
    lines = np.clip((a[..., :3].mean(2) - 24) * 3.0, 0, 60)[..., None]        # keep the moulded fluting, just lighter
    clear = np.array([196, 200, 198.0]) + lines
    a[..., :3] = np.where(reg[..., None], clear, a[..., :3])
    a[..., 3] = np.where(reg, 96, a[..., 3])
    # the corner lamps are clear too, with a small amber reflector toward the rear of the side face
    import cv2
    n, _, M = k.find('Object035'); w = world_pos(n, M); F = n.F[(w[n.F][:, :, 2] > 1.5).all(1)]; zr = w[F][:, :, 2].min()
    uv = np.mod(n.V[:, 6:8].astype(np.float64), 1)[F] * S                     # (t,3,2) texel coordinates of every lens triangle
    tid = np.zeros((S, S), np.int32)
    for i, p in enumerate(np.round(uv * 16).astype(np.int32)): cv2.fillConvexPoly(tid, p, i + 1, shift=4)
    A = np.concatenate([uv, np.ones((len(F), 3, 1))], 2); ok = np.abs(np.linalg.det(A)) > 1e-9
    co = np.zeros((len(F) + 1, 3, 2)); co[1:][ok] = np.linalg.solve(A[ok], w[F][ok][:, :, 1:3])   # texel -> (y, z), exact per triangle
    py, px = np.where(corner & (tid > 0)); c = co[tid[py, px]]
    yz = c[:, 0] * (px + 0.5)[:, None] + c[:, 1] * (py + 0.5)[:, None] + c[:, 2]
    amber = (yz[:, 0] > 0.500) & (yz[:, 0] < 0.562) & (yz[:, 1] > zr + 0.022) & (yz[:, 1] < zr + 0.066)
    a[py[amber], px[amber], :3] = np.array([236, 104, 14.0]) + 0.5 * lines[py[amber], px[amber]]
    a[py[amber], px[amber], 3] = 235
    b = io.BytesIO(); Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA').save(b, format='DDS', pixel_format='DXT5')
    new = bytes(t[2][:128]) + b.getvalue()[128:]
    assert len(new) == len(t[2]); t[2] = new
    # satin reflectors: mirror-chrome facets showed up as hard bright blocks behind the clear lens
    m = k.materials[k.mat_id('chrome')].copy('gh_lamp_reflector')
    for p, v in (('ksAmbient', 0.25), ('ksDiffuse', 0.30), ('ksSpecular', 1.2), ('ksSpecularEXP', 60.0), ('fresnelC', 0.10), ('fresnelMaxLevel', 0.30)): m.set(p, v)
    mid = k.add_material(m)
    for name in ('headlight1', 'headlights_part006'): k.find(name)[0].mat = mid
    return {'texels': int(reg.sum())}

# ------------------------------------------------------------------ 13. right tail light: outer end of the red lens broken away
# Decal space (s, t): s runs along the whole right lamp in the lens' own v coordinate (0 = inner end by the plate,
# 0.90 = the corner, 1 = end of the wrap-around; 1 px = 0.22 mm), t down it in u (0 = top of the red lens, 0.647 = red/amber
# joint, 1 = bottom of the amber lens). Shapes are traced from the owner's marked-up photo: px = 1.3165 * photo_x - 137.
TAIL_V = (-0.967, -0.024)
TAIL_U = (0.349, 0.618)
TW, TH, TRED = 2048, 512, 331
# The lamp is a smooth red outer sheet over a clear plastic block moulded with a mosaic of small optics. The red sheet
# is cracked; two pieces of it are missing (joined by a sliver along the top edge), and there the clear mosaic shows white.
TAIL_HOLE = [(1159, 4), (1765, 4), (1765, 235), (1719, 233), (1587, 55), (1581, 22), (1495, 25), (1508, 152), (1574, 309),
             (1548, 340), (1462, 331), (1403, 212), (1403, 55), (1324, 40), (1324, 15), (1159, 10)]                    # px
TAIL_CRACK = [(1126, 0), (1146, 122), (1113, 241), (969, 323), (784, 331), (574, 331), (520, 400), (480, 476)]   # across the red sheet, into the amber
def tail_px(pts):
    """points laid out on the earlier 1024-wide sheet -> this one"""
    return [(1126.5 + (x - 507.5) * 2.0545, y) for x, y in pts]
TAIL_DEPTH = 0.004              # how far behind the lens surface the exposed inside sits

def tail_st(uv):
    return np.c_[(uv[:, 1] - TAIL_V[0]) / (TAIL_V[1] - TAIL_V[0]), (uv[:, 0] - TAIL_U[0]) / (TAIL_U[1] - TAIL_U[0])]

def tail_mask(grow=0):
    """lookup: is a decal-space point inside the hole (optionally grown by `grow` px)"""
    S = 4; im = Image.new('L', (TW * S, TH * S), 0); ImageDraw.Draw(im).polygon([(x * S, y * S) for x, y in TAIL_HOLE], fill=255)
    if grow: im = im.filter(ImageFilter.MaxFilter(2 * grow * S + 1))
    a = np.asarray(im) > 0
    def inside(st):
        x = np.floor(st[:, 0] * TW * S).astype(int); y = np.floor(st[:, 1] * TH * S).astype(int)
        ok = (x >= 0) & (x < TW * S) & (y >= 0) & (y < TH * S); r = np.zeros(len(st), bool); r[ok] = a[y[ok], x[ok]]; return r
    return inside

def tail_inside_images():
    """what shows through the holes: the clear mosaic plastic behind the red sheet. Returns (diffuse RGB, normal map RGB)."""
    yy, xx = np.mgrid[0:TH, 0:TW].astype(float)
    px_ = (xx % 7.0) / 7.0 - 0.5; py_ = (yy % 6.0) / 6.0 - 0.5; pil = 1 - 4 * np.maximum(px_ ** 2, py_ ** 2)      # pillow optics
    a = np.array([226.0, 222, 224]) * (0.62 + 0.38 * pil[:, :, None]) * (0.92 + 0.16 * noise((TH, TW), 20, 3)[:, :, None])
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB'); d = ImageDraw.Draw(im)
    d.polygon(tail_px([(650, 318), (668, 300), (700, 296), (722, 306), (732, 345), (650, 345)]), fill=(10, 5, 6))             # dark gap at the foot of the hole
    # the edge of the red sheet throws a shadow onto the plastic along the top and inboard sides of each hole
    hole = Image.new('L', (TW, TH), 0); ImageDraw.Draw(hole).polygon(TAIL_HOLE, fill=255)
    lit = np.asarray(ImageChops.offset(hole, 10, 7).filter(ImageFilter.GaussianBlur(4))).astype(float) / 255
    a = np.asarray(im).astype(float) * (0.40 + 0.60 * lit[:, :, None])
    gy, gx = np.gradient(gaussian_filter(5.0 * pil, 0.6)); l = np.sqrt(gx ** 2 + gy ** 2 + 1)
    nm = np.dstack([-gx / l, gy / l, 1 / l]) * 0.5 + 0.5
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB'), Image.fromarray((nm * 255).astype(np.uint8), 'RGB')

def tail_crack_image():
    """lies on the red sheet and the amber lens; transparent except for the cracks and the bright broken edges"""
    rng = np.random.default_rng(84); im = Image.new('RGBA', (TW, TH), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    dark = (24, 3, 6, 255); lightc = (246, 214, 208, 255)
    side = [([(501, 249), (478, 240), (462, 258)], 3), ([(690, 342), (686, 392), (692, 432)], 4), ([(686, 392), (664, 410)], 3),
            ([(730, 340), (738, 420), (724, 512)], 4), ([(738, 420), (762, 450)], 3), ([(725, 309), (745, 318), (765, 300), (790, 270)], 4)]
    for line, w in [(TAIL_CRACK, 7), ([(784, 331), (700, 325), (610, 334)], 3)] + [(tail_px(l), w) for l, w in side]:
        j = [(x + rng.uniform(-2, 2), y + rng.uniform(-2, 2)) for x, y in line]; j[0] = line[0]
        d.line(j, fill=dark, width=w, joint='curve')
        d.line([(x + 1, y - w / 2 - 1) for x, y in j], fill=lightc, width=2 if w > 3 else 1, joint='curve')   # light catching the edge
    d.line(TAIL_HOLE + TAIL_HOLE[:1], fill=lightc, width=3, joint='curve')                           # broken edge of the red sheet
    return im

def tail_scuff_image():
    """soft white scrape marks (alpha-blended): on the piece of red sheet between the crack and the first hole, on the
    shard between the holes, and smeared along the red/amber joint below"""
    yy, xx = np.mgrid[0:TH, 0:TW].astype(float); al = np.zeros((TH, TW))
    def streaks(angle, seed, sig=(1.6, 22)):
        n = Image.fromarray((noise((TH * 2, TH * 2), sig, seed) * 255).astype(np.uint8)).rotate(angle, resample=Image.BICUBIC)
        n = np.asarray(n.crop((TH // 2, TH // 2, TH * 3 // 2, TH * 3 // 2))).astype(float) / 255
        n = np.tile(n, (1, TW // TH)); return (n - n.mean()) / n.std()
    for cx, cy, rx, ry, angle, seed, k_ in ((1290, 175, 125, 125, 24, 11, 0.95), (1330, 268, 110, 45, 12, 12, 1.0), (1546, 185, 34, 120, 68, 13, 1.0),
                                              (1400, 338, 190, 20, 4, 14, 0.9), (1730, 288, 42, 34, 60, 15, 0.8)):
        blob = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 1.3); s_ = streaks(angle, seed)
        al = np.maximum(al, k_ * blob * np.clip(0.35 + 0.55 * s_, 0, 1) ** 1.3 * 1.6)
    hm = Image.new('L', (TW, TH), 0); ImageDraw.Draw(hm).polygon(TAIL_HOLE, fill=255)
    al[np.asarray(hm.filter(ImageFilter.MaxFilter(5))) > 0] = 0
    cx_ = np.interp(yy[:, 0], [p_[1] for p_ in TAIL_CRACK[:4]], [p_[0] for p_ in TAIL_CRACK[:4]])        # nothing inboard of the crack
    al *= np.clip((xx - cx_[:, None] - 4) / 14, 0, 1) ** (yy < 300)
    al = gaussian_filter(np.clip(al, 0, 0.92), 0.8)
    rgb = np.dstack([np.full((TH, TW), v) for v in (242.0, 228, 230)])
    return Image.fromarray(np.dstack([rgb, al * 255]).astype(np.uint8), 'RGBA')

def subdivide_cut(V, F, st_of, inside, keep_inside=False, levels=4):
    """drop the triangles inside the hole (or keep only those), after splitting the ones that straddle its outline
    (1 -> 4, `levels` times). Returns V, F, number of triangles dropped."""
    V = V.astype(float); F = F.copy()
    for _ in range(levels):
        ins = inside(st_of(V)); f = ins[F]; mix = f.any(1) & ~f.all(1)
        if not mix.any(): break
        T = F[mix]; base = len(V)
        mids = [(V[T[:, i]] + V[T[:, (i + 1) % 3]]) / 2 for i in range(3)]
        for m in mids: m[:, 3:6] = unit(m[:, 3:6]); m[:, 8:11] = unit(m[:, 8:11])
        k_ = len(T); a, b, c = T.T; ab = base + np.arange(k_); bc = ab + k_; ca = bc + k_
        V = np.vstack([V] + mids)
        F = np.vstack([F[~mix], np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c], np.c_[ab, bc, ca]])
    cen = inside(st_of(V)[F].mean(1)) ^ keep_inside; F = F[~cen]
    used = np.unique(F); re = np.full(len(V), -1); re[used] = np.arange(len(used))
    return V[used], re[F], int(cen.sum())

def taillight_damage(k):
    inside = tail_mask(); near = tail_mask(grow=5)
    lens, _, M = k.find('Object036')                                             # red lens, both sides
    def st_of(M):                                                                # the left lamp shares the UVs: keep it out
        return lambda V: np.where(((V[:, :3] @ M[:3, :3])[:, 0] + M[3, 0] < 0)[:, None], tail_st(V[:, 6:8]), -1.0)
    st_n = st_of(M)
    # 1. what shows through: a copy of the lens surface, pushed in, with its own texture and normal map
    V, F, _ = subdivide_cut(lens.V, lens.F, st_n, near, keep_inside=True)
    V[:, :3] = V[:, :3] @ M[:3, :3] + M[3, :3]; nr = unit(V[:, 3:6] @ M[:3, :3]); nr *= np.sign(nr @ np.array([-0.3, 0.0, -1.0]) + 1e-9)[:, None]; V[:, 3:6] = nr
    V[:, :3] -= nr * TAIL_DEPTH; V[:, 6:8] = tail_st(V[:, 6:8])
    p = V[:, :3]; e1 = p[F[:, 1]] - p[F[:, 0]]; e2 = p[F[:, 2]] - p[F[:, 0]]            # tangent = direction of increasing s
    d1 = V[F[:, 1], 6:8] - V[F[:, 0], 6:8]; d2 = V[F[:, 2], 6:8] - V[F[:, 0], 6:8]; det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    tf = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / np.where(np.abs(det) < 1e-12, 1e-12, det)[:, None]; tg = np.zeros((len(V), 3))
    for i in range(3): np.add.at(tg, F[:, i], unit(tf))
    V[:, 8:11] = unit(tg)
    dif, nm = tail_inside_images()
    k.set_texture('gh_tail_inside.png', png_bytes(dif)); k.set_texture('gh_tail_inside_nm.png', png_bytes(nm))
    m = simple_material('gh_tail_inside', 'gh_tail_inside.png', 0.40, 0.45, 0.55, 45.0, shader='ksPerPixelNM')
    m.tex.append(['txNormal', 1, 'gh_tail_inside_nm.png'])
    k.add_mesh('TAILLIGHT_INSIDE_R', V, fix_winding(V, F), k.add_material(m), shadows=0)
    # 2. the hole itself, and the patterned layer that sat just behind the lens there
    info = {'inside_tris': len(F)}
    V, F, info['lens_tris_cut'] = subdivide_cut(lens.V, lens.F, st_n, inside)
    lens.V = V.astype(lens.V.dtype); lens.F = F.astype(lens.F.dtype); update_bounds(lens)
    bump, _, Mb = k.find('headlights_part010')
    info['layer_tris_cut'] = drop_tris(bump, inside(st_of(Mb)(bump.V.astype(float)))[bump.F].all(1))
    # 3. scrape marks (soft, alpha-blended) and, a hair further out, the cracks and bright broken edges (crisp, alpha-tested)
    k.set_texture('gh_taillight_scuffs.png', png_bytes(tail_scuff_image()))
    ms = simple_material('gh_taillight_scuffs', 'gh_taillight_scuffs.png', 0.50, 0.50, 0.15, 12.0, shader='ksPerPixelAT'); ms.blend = 1
    k.set_texture('gh_taillight_damage.png', png_bytes(tail_crack_image()))
    m = simple_material('gh_taillight_damage', 'gh_taillight_damage.png', 0.42, 0.45, 0.45, 35.0, shader='ksPerPixelAT', alpha_ref=0.5)
    m.alpha_tested = 1
    Vs = []; Fs = []; Ns = []; off = 0
    for name in ('Object036', 'signal_r_glass'):
        n, _, M = k.find(name); w = world_pos(n, M); st = tail_st(n.V[:, 6:8])
        nr = unit(n.V[:, 3:6] @ M[:3, :3]); nr *= np.sign(nr @ np.array([-0.3, 0.0, -1.0]) + 1e-9)[:, None]
        ok = (w[:, 0] < 0) & (st[:, 0] > -0.03) & (st[:, 0] < 1.0)
        F = n.F[ok[n.F].all(1)]; used = np.unique(F); re = np.full(len(w), -1); re[used] = np.arange(len(used))
        V = np.zeros((len(used), 11)); V[:, :3] = w[used]; V[:, 3:6] = nr[used]; V[:, 6:8] = st[used]; V[:, 8:11] = [-1, 0, 0]
        Vs.append(V); Fs.append(re[F] + off); off += len(V)
    V = np.vstack(Vs); F = np.vstack(Fs); F = fix_winding(V, F)
    for name, mat, out, tr in (('TAILLIGHT_SCUFFS_R', ms, 0.0010, 1), ('TAILLIGHT_DAMAGE_R', m, 0.0016, 0)):
        W = V.copy(); W[:, :3] += W[:, 3:6] * out; k.add_mesh(name, W, F, k.add_material(mat), transparent=tr, shadows=0)
    info['crack_tris'] = len(F); return info

# ------------------------------------------------------------------ 14. paint rubbed through above the right tail light
PD_BOX = (-0.55, -0.80, 0.82, 0.63)      # decal sheet seen from behind: x at its left / right edge, y at its top / bottom (m)
PD_W, PD_H = 1024, 768

def paint_damage_image():
    """trunk lid (left of px 492): three gouges through the paint (a bar on top, a curved stroke below it, a short bar by
    the lid's edge), each smeared away towards the car's centre. Body corner (right of px 492): scraped nearly black,
    pale brushed marks mostly in its lower half. Traced from the owner's marked-up photo: 1 photo px = 1.38 px here."""
    yy, xx = np.mgrid[0:PD_H, 0:PD_W].astype(float); rgb = np.zeros((PD_H, PD_W, 3)); al = np.zeros((PD_H, PD_W))
    def over(col, a):
        nonlocal rgb, al
        a = np.clip(a, 0, 1); out = a + al * (1 - a); w = np.where(out > 1e-6, a / np.maximum(out, 1e-6), 0)
        rgb = rgb * (1 - w[:, :, None]) + np.array(col, float) * w[:, :, None]; al = out
    def streaks(angle, seed, sig=(2.0, 26)):
        S = max(PD_W, PD_H) * 2
        n = Image.fromarray((noise((S, S), sig, seed) * 255).astype(np.uint8)).rotate(angle, resample=Image.BICUBIC)
        n = np.asarray(n.crop(((S - PD_W) // 2, (S - PD_H) // 2, (S + PD_W) // 2, (S + PD_H) // 2))).astype(float) / 255
        return (n - n.mean()) / n.std()
    grain = 0.85 + 0.30 * noise((PD_H, PD_W), 3, 21)
    primer = (188, 190, 196); metal = (74, 76, 82); lid = xx < 492
    # trunk lid, traced from the owner's close-up (1 photo px = 1.54 px here): a broad pale lobe on top with rust in it; from
    # its inboard end a band hooks down and back out along the bottom (a 'C' open towards the lid's edge) ending in a bright
    # spot; a darker rubbed patch inside the C with the scored line across it; a pale streak fading down the lid's edge.
    def soft(draw, blur):
        im_ = Image.new('L', (PD_W, PD_H), 0); draw(ImageDraw.Draw(im_))
        return np.asarray(im_.filter(ImageFilter.GaussianBlur(blur))).astype(float) / 255
    hstr = np.clip(0.62 + 0.42 * streaks(0, 32, (1.3, 30)), 0, 1)                                   # dragged sideways
    blot = np.clip(0.55 + 0.9 * noise((PD_H, PD_W), 14, 34), 0, 1)
    hook = [(215, 180), (190, 218), (184, 257), (202, 292), (264, 318), (341, 334), (418, 345), (446, 349)]
    band = soft(lambda d: [d.line(hook[i:i + 2], fill=255, width=w) for i, w in enumerate((40, 44, 48, 52, 54, 54, 50))]
                + [d.ellipse((x_ - 24, y_ - 24, x_ + 24, y_ + 24), fill=255) for x_, y_ in hook], 11)
    lobe = soft(lambda d: d.ellipse((206, 136, 476, 196), fill=255), 12)
    spot = soft(lambda d: d.ellipse((414, 322, 470, 374), fill=255), 9)
    tail = soft(lambda d: d.line([(457, 357), (467, 418), (472, 500)], fill=255, width=36, joint='curve'), 10) * np.clip((520 - yy) / 150, 0, 1)
    inner = soft(lambda d: d.ellipse((236, 196, 480, 300), fill=255), 16)
    L = 120; kern = np.exp(-np.arange(L) / 20.0); kern /= kern.sum()
    marks = np.clip(np.maximum(np.maximum(band, lobe), spot) * 1.25, 0, 1)
    smear = np.stack([np.convolve(row, kern, mode='full')[L - 1:] for row in marks]) * np.clip(xx / 90, 0, 1)   # trails off inboard
    over((40, 41, 47), 0.42 * inner * lid)
    over((170, 174, 182), 0.36 * np.clip(smear * 1.3, 0, 1) * hstr * lid)
    over((196, 199, 205), 0.90 * marks * (0.55 + 0.45 * hstr) * blot * lid)
    over((226, 228, 232), 0.75 * spot * lid)
    over((158, 120, 104), 0.50 * soft(lambda d: d.ellipse((330, 144, 452, 180), fill=255), 9) * blot * lid)       # rust in the top lobe
    over((184, 192, 208), 0.62 * tail * (0.45 + 0.55 * np.clip(0.5 + 0.6 * streaks(80, 33, (1.2, 22)), 0, 1)) * lid)
    # (what is left of the pinstripe here is the PINSTRIPE mesh itself, showing through this decal where the scrape is thin)
    # body corner: scraped nearly black, pale brush marks mostly low down
    m = Image.new('L', (PD_W, PD_H), 0)
    ImageDraw.Draw(m).polygon([(505, 330), (640, 215), (800, 290), (810, 520), (770, 655), (505, 665)], fill=255)
    m = np.asarray(m.filter(ImageFilter.GaussianBlur(16))).astype(float) / 255 * ~lid
    over((16, 16, 18), 0.98 * np.clip(m * 1.5, 0, 1))
    s1 = streaks(24, 23, (1.4, 30)); s2 = streaks(14, 24, (3.0, 40))
    sw = np.exp(-(((xx - 620) / 120) ** 2 + ((yy - 560) / 85) ** 2)) + 0.75 * np.exp(-(((xx - 745) / 70) ** 2 + ((yy - 470) / 60) ** 2)) \
        + 0.25 * np.exp(-(((xx - 700) / 90) ** 2 + ((yy - 320) / 50) ** 2))
    over((200, 205, 214), 0.90 * m * np.clip(sw, 0, 1) * np.clip(0.15 + 0.55 * s1, 0, 1) ** 1.2 * 1.3)
    over((120, 124, 132), 0.22 * m * np.clip(0.5 * s2, 0, 1))
    al = gaussian_filter(al, 0.6)
    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255), np.clip(al, 0, 1) * 255]).astype(np.uint8), 'RGBA')

def paint_damage(k):
    k.set_texture('gh_paint_damage.png', png_bytes(paint_damage_image()))
    m = simple_material('gh_paint_damage', 'gh_paint_damage.png', 0.45, 0.50, 0.12, 14.0, shader='ksPerPixelAT'); m.blend = 1
    x0, x1, y0, y1 = PD_BOX; Vs = []; Fs = []; off = 0
    for name in ('trunk', 'r_fenders'):                         # the decal follows the (dented) panels, 1.5 mm proud of them
        n, _, M = k.find(name); w = world_pos(n, M); nr = unit(n.V[:, 3:6] @ M[:3, :3]); c = w[n.F].mean(1)
        sel = (c[:, 0] < x0 + 0.02) & (c[:, 0] > x1 - 0.02) & (c[:, 1] < y0 + 0.02) & (c[:, 1] > y1 - 0.02) & (c[:, 2] < -2.2) & (nr[n.F].mean(1)[:, 2] < -0.15)
        F = n.F[sel]; used = np.unique(F); re = np.full(len(w), -1); re[used] = np.arange(len(used))
        V = np.zeros((len(used), 11)); V[:, :3] = w[used] + nr[used] * 0.0015; V[:, 3:6] = nr[used]
        V[:, 6] = (w[used, 0] - x0) / (x1 - x0); V[:, 7] = (w[used, 1] - y0) / (y1 - y0); V[:, 8:11] = [-1, 0, 0]
        Vs.append(V); Fs.append(re[F] + off); off += len(V)
    V = np.vstack(Vs); F = np.vstack(Fs)
    k.add_mesh('PAINT_DAMAGE_R', V, fix_winding(V, F), k.add_material(m), transparent=1, shadows=0)
    return {'tris': len(F)}

# ------------------------------------------------------------------ 15. double pinstripe along the beltline and across the trunk
# Height of the stripe's centre line along the car: three straight runs joined over gentle radii. Along the doors it runs
# parallel to the window sills (one straight line, y = 0.783 - 0.0222 z), 5.5 cm lower; behind the rear door it levels off at
# the height it crosses the trunk; ahead of the front door it follows the fender's shoulder down towards the headlight.
STRIPE_SILL = (0.783, -0.0222); STRIPE_DROP = 0.055; STRIPE_REAR_Y = 0.755; STRIPE_FRONT_Z = 1.70
STRIPE_FENDER = (0.900, 0.700, -0.0893)                    # z, y of a point on the fender run, and its slope
def _smin(a, b, r=0.004): return 0.5 * (a + b - np.sqrt((a - b) ** 2 + r * r))
def stripe_y(z):
    z = np.asarray(z, float)
    y = _smin(STRIPE_SILL[0] - STRIPE_DROP + STRIPE_SILL[1] * z, STRIPE_REAR_Y)
    return _smin(y, STRIPE_FENDER[1] + STRIPE_FENDER[2] * (z - STRIPE_FENDER[0]))

# The stripe is paint: on the crashed right rear corner it has to follow the metal, not the height above the ground. So it is
# laid out on the panels as they were BEFORE the crash. crash_field() is a copy of the displacement field stage 1 applied
# (build_rear_damage.py: total(), trunk_extra()); uncrashed() inverts it numerically.
def _ck(w, C, R):
    r2 = (((w - np.array(C)) / np.array(R)) ** 2).sum(1); return np.where(r2 < 1, (1 - r2) ** 2, 0.0)
def crash_field(w, trunk=False, calm=0.0):
    k1 = _ck(w, (-0.80, 0.60, -2.47), (0.55, 0.52, 0.78)); k2 = _ck(w, (-0.76, 0.34, -2.33), (0.40, 0.30, 0.50))
    s_ = w @ np.array([0.86, 0.15, -0.48]); q = np.array([0.35, 0.45, 0.82]); q /= np.linalg.norm(q)
    wr1 = 0.03 * np.sin(26 * s_ + 2.0 * np.sin(9 * w[:, 1] + 3 * w[:, 0])); wr2 = 0.045 * np.sin(34 * s_ + 3.0 * np.sin(13 * w[:, 0] + 7 * w[:, 1]))
    d = 0.20 * np.array([0.42, 0.0, 0.91])[None] * (k1 * (0.65 + 0.35 * np.sin(17 * w[:, 1] + 13 * w[:, 0])))[:, None]
    d = d + 0.13 * np.array([0.30, 0.35, 0.89])[None] * (k2 * (0.6 + 0.4 * np.sin(23 * w[:, 0] + 19 * w[:, 1])))[:, None]
    d = d + q[None] * (wr1 * k1 + wr2 * k2)[:, None]
    pv = np.array([-0.30, 0.60, -2.45]); e = w - pv; c, s1 = np.cos(np.radians(6.0)), np.sin(np.radians(6.0))    # lamp area: rigid tilt
    x = e[:, 0] * c - e[:, 1] * s1; y = e[:, 0] * s1 + e[:, 1] * c
    tl = pv + np.c_[x * c + e[:, 2] * s1, y, -x * s1 + e[:, 2] * c] + np.array([0.0, -0.004, 0.0])
    lo = np.array([-0.82, 0.46, -2.50]); hi = np.array([-0.28, 0.71, -2.20])
    wl = (1 - ss(0.02, 0.22, np.linalg.norm(np.maximum(np.maximum(lo - w, 0), np.maximum(w - hi, 0)), axis=1)))[:, None]
    w = w + wl * (tl - w) + (1 - wl) * d
    if trunk:
        k = _ck(w, (-0.66, 0.82, -2.40), (0.40, 0.28, 0.60)); up = ss(0.60, 0.80, w[:, 1])
        wr = 0.02 * np.sin(31 * (w[:, 0] * 0.8 + w[:, 2] * 0.6) + 2 * np.sin(11 * w[:, 0]))
        d = np.c_[0.045 * np.ones(len(w)), 0.075 * up, 0.030 * np.ones(len(w))] * (k * (0.7 + 0.3 * np.sin(15 * w[:, 0] + 9 * w[:, 2])))[:, None]
        d[:, 1] += wr * k * up; w = w + d * (1.0 - np.asarray(calm, float).reshape(-1, 1))
    return w

def trunk_face(k):
    """Stage 1 crumpled the whole right half of the trunk lid, its rear face included. On the real car only the top of the lid
    is crumpled at the corner; the rear face (pinstripe, badge) is still flat and just leans with the tail light. So the extra
    lid crumple is taken off the rear face again, fading out where the face rolls over into the top of the lid."""
    n, _, M = k.find('trunk')
    def fn(w):
        p0 = uncrashed(w, True); n.p0 = p0
        calm = (1 - ss(0.775, 0.805, p0[:, 1])) * (1 - ss(-2.40, -2.36, p0[:, 2]))
        return crash_field(p0, True, calm)
    return {'moved': deform(n, M, fn, thr=1e-4)}

def uncrashed(w, trunk=False):
    """where each point of the crashed body was before the crash (Newton iteration on crash_field)"""
    p = np.array(w, float); m = p[:, 2] < -1.0; q = p[m].copy(); tgt = p[m]; h = 1e-4
    for _ in range(30):
        r = crash_field(q, trunk) - tgt
        J = np.stack([(crash_field(q + e * h, trunk) - crash_field(q - e * h, trunk)) / (2 * h) for e in np.eye(3)], axis=2)
        st = np.linalg.solve(J + 1e-6 * np.eye(3)[None], r[:, :, None])[:, :, 0]
        ln = np.linalg.norm(st, axis=1, keepdims=True); q -= st * np.minimum(1.0, 0.04 / np.maximum(ln, 1e-12))
    p[m] = q
    return p
STRIPE_BANDS = [(0.0017, 0.0052), (-0.0055, -0.0035)]      # upper (3.5 mm) and lower (2 mm) line, relative to the centre line
STRIPE_RGB = (56, 24, 24)

def clip_band(V, F, f, lo, hi):
    """the part of each triangle where lo <= f <= hi (f given per vertex, linear inside a triangle)"""
    out_v = []; out_f = []; n = 0
    for tri in F[(f[F].max(1) > lo) & (f[F].min(1) < hi)]:
        poly = [(V[i], f[i]) for i in tri]
        for sgn, lim in ((1.0, lo), (-1.0, -hi)):                       # keep sgn * f >= lim
            nxt = []
            for i in range(len(poly)):
                (pa, fa), (pb, fb) = poly[i], poly[(i + 1) % len(poly)]; da = sgn * fa - lim; db = sgn * fb - lim
                if da >= 0: nxt.append((pa, fa))
                if (da >= 0) != (db >= 0):
                    u = da / (da - db); nxt.append((pa + (pb - pa) * u, fa + (fb - fa) * u))
            poly = nxt
            if len(poly) < 3: break
        if len(poly) < 3: continue
        out_v += [p_ for p_, _ in poly]; out_f += [[n, n + i, n + i + 1] for i in range(1, len(poly) - 1)]; n += len(poly)
    return (np.array(out_v), np.array(out_f, np.int64)) if out_v else (np.zeros((0, 11)), np.zeros((0, 3), np.int64))

def pinstripe(k):
    k.set_texture('gh_pinstripe.png', flat_texture(STRIPE_RGB))
    mid = k.add_material(simple_material('gh_pinstripe', 'gh_pinstripe.png', 0.40, 0.45, 0.30, 40.0))
    Vs = []; Fs = []; off = 0
    for name in ('f_fenders', 'lf_door', 'rf_door', 'rear_doors', 'r_fenders', 'trunk'):
        n, _, M = k.find(name); w = world_pos(n, M); nr = unit(n.V[:, 3:6] @ M[:3, :3])
        V = np.zeros((len(w), 11)); V[:, :3] = w; V[:, 3:6] = nr; V[:, 6:8] = 0.5; V[:, 8:11] = [0, 0, 1]
        p0 = n.p0 if hasattr(n, 'p0') else uncrashed(w, name == 'trunk')                               # the panel before the crash: the stripe is laid out there
        f = p0[:, 1] - stripe_y(p0[:, 2])
        tri = p0[n.F]; fn = unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])); c = tri.mean(1)
        fn *= np.where((fn * nr[n.F].mean(1)).sum(1) < 0, -1.0, 1.0)[:, None]
        side = (np.abs(c[:, 0]) > 0.70) & (fn[:, 0] * np.sign(c[:, 0]) > 0.35) & (c[:, 2] < STRIPE_FRONT_Z + 0.03)    # outer skin of the flanks
        back = (c[:, 2] < -2.30) & (fn[:, 2] < -0.35)                                                             # rear face of trunk and corners
        F = n.F[(side | back) & (np.abs(fn[:, 1]) < 0.85)]
        for lo, hi in STRIPE_BANDS:
            v_, f_ = clip_band(V, F, f, lo, hi)
            if len(v_):
                v_[:, 3:6] = unit(v_[:, 3:6]); v_[:, :3] += v_[:, 3:6] * 0.0008
                Vs.append(v_); Fs.append(f_ + off); off += len(v_)
    V = np.vstack(Vs); F = np.vstack(Fs)
    keep = (V[F][:, :, 2] <= STRIPE_FRONT_Z).all(1); F = F[keep]
    k.add_mesh('PINSTRIPE', V, fix_winding(V, F), mid, shadows=0)
    return {'tris': len(F)}

# ------------------------------------------------------------------ 16. interior (Reference Videos/Interrior.mp4)
# The real car: rear seat taken out (bare painted metal behind the front seats), a faded pink bucket seat with a harness for the
# driver (the passenger keeps the stock seat), a manual shifter with a black leather knob and a green cloth bag as its boot,
# and a black/grey checker mat in the driver's footwell.
SEAT_X = 0.38                     # centre line of the driver's seat (the car is left-hand drive: +x)
FLOOR_Y = 0.222                   # carpet height in the footwells
PAN_Z = (-0.66, -1.06, -1.50)     # bare rear floor: front edge of the seat pan, foot of the bulkhead, top of the bulkhead (parcel shelf)

def surf(P, UV, hint='out'):
    """a grid of points (m, n, 3) with UVs (m, n, 2) -> V, F with smooth normals, turned towards `hint` (a direction, or 'out')"""
    m, n = P.shape[:2]; p = P.reshape(-1, 3); idx = np.arange(m * n).reshape(m, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    F = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
    nr = unit(weld_normals(p, F))
    ref = (p - p.mean(0)) if isinstance(hint, str) else np.asarray(hint, float)[None]
    if (nr * ref).sum() < 0: nr = -nr
    tg = np.gradient(P, axis=1).reshape(-1, 3); tg = unit(tg - nr * (tg * nr).sum(1, keepdims=True))
    V = np.c_[p, nr, UV.reshape(-1, 2), tg]
    return V, fix_winding(V, F)

def join(*parts):
    Vs = []; Fs = []; off = 0
    for V, F in parts:
        Vs.append(V); Fs.append(np.asarray(F, np.int64) + off); off += len(V)
    return np.vstack(Vs), np.vstack(Fs)

def two_sided(V, F):
    V2 = V.copy(); V2[:, 3:6] *= -1
    return join((V, F), (V2, np.asarray(F)[:, ::-1]))

def ribbon(C, W, width, v0=0.0, rep=0.30):
    """a strap: centre line C (n, 3), width direction W (3,) or (n, 3); the texture repeats every `rep` metres along it"""
    C = np.asarray(C, float); W = unit(np.broadcast_to(np.asarray(W, float), C.shape).copy())
    ln = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))] / rep + v0
    P = np.stack([C - W * width / 2, C + W * width / 2], axis=1); UV = np.stack([np.c_[np.zeros(len(C)), ln], np.c_[np.ones(len(C)), ln]], axis=1)
    V, F = surf(P, UV, (0, 1, 0)); V2 = V.copy(); V2[:, 3:6] *= -1
    T = np.gradient(C, axis=0); hand = (np.cross(W, T) * V[::2, 3:6]).sum()          # which face would show the lettering mirrored
    (V if hand > 0 else V2)[:, 6] *= -1; (V if hand > 0 else V2)[:, 6] += 1
    return join((V, F), (V2, F[:, ::-1]))

def box(lo, hi, uv=(0.5, 0.5)):
    b = Builder(); (x0, y0, z0), (x1, y1, z1) = lo, hi; u = [uv] * 4
    b.poly([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0), u); b.poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0), u)
    b.poly([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0), u); b.poly([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0), u)
    b.poly([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1), u); b.poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1), u)
    return b.done()

def drop_components(k, name, pick):
    """remove the connected parts of a mesh for which pick(centre of its bounding box) is true"""
    n, _, M = k.find(name); w = world_pos(n, M); nc, lab = components(w, n.F)
    cen = np.array([(w[lab == c].min(0) + w[lab == c].max(0)) / 2 for c in range(nc)]); kill = np.array([bool(pick(c)) for c in cen])
    return drop_tris(n, kill[lab[n.F[:, 0]]])

# ---- rear seat delete
def pan_height(x, z):
    """the bare metal behind the front seats: a step up from the footwell, the seat pan with the tunnel, the bulkhead ramp, the wheel arches"""
    z0, z1, z2 = PAN_Z; t = np.clip((z1 - z) / (z1 - z2), 0, 1)
    y = FLOOR_Y + 0.095 * ss(z0 + 0.03, z0 - 0.03, z) + 0.02 * np.clip((z0 - z) / (z0 - z1), 0, 1)        # step and gently rising pan
    y = y + 0.055 * np.exp(-(x / 0.15) ** 2) * (1 - t) * ss(z0 + 0.03, z0 - 0.03, z)                      # transmission tunnel
    y = y + (0.845 - FLOOR_Y - 0.115) * t                                                                 # bulkhead up to the parcel shelf
    arch = FLOOR_Y + 0.115 + 0.30 * ss(0.47, 0.66, np.abs(x)) * ss(-0.93, -1.10, z)
    return np.maximum(y, arch)

def bare_metal_image(size=1024, seed=31):
    """U: across the car; V: from the front edge of the seat pan (top row) over the pan and up the bulkhead to the parcel shelf"""
    S = size; yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    n1 = noise((S, S), 40, seed); n2 = noise((S, S), 6, seed + 1); n3 = noise((S, S), 1.2, seed + 2)
    img = np.empty((S, S, 3)); img[:] = (112, 115, 122); img *= (0.80 + 0.30 * n1 + 0.10 * n2)[..., None]
    def over(rgb, a): img[:] = img * (1 - a[..., None]) + np.array(rgb, float) * a[..., None]
    def mask(fn, blur=0.0):
        im = Image.new('L', (S, S), 0); fn(ImageDraw.Draw(im))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)) if blur else im, np.float64) / 255
    pan = (yy > 0.045 * S) & (yy < 0.415 * S)
    # pressed ribs along the pan and the bulkhead: a light and a dark line side by side
    for x0 in (150, 235, 320, 405, 619, 704, 789, 874):
        over((205, 208, 214), 0.55 * mask(lambda d: d.line([(x0, 90), (x0, 385)], fill=255, width=5), 1.5))
        over((70, 72, 78), 0.60 * mask(lambda d: d.line([(x0 + 7, 90), (x0 + 7, 385)], fill=255, width=5), 1.5))
    # what is left of the sound deadening: black tar patches and brown dust, mostly on the pan
    tar = np.clip((noise((S, S), 22, seed + 3) - 0.56) * 9, 0, 1) * pan * (0.55 + 0.45 * n2); over((26, 26, 28), 0.88 * tar)
    over((92, 84, 72), 0.35 * np.clip((noise((S, S), 10, seed + 4) - 0.50) * 5, 0, 1) * (yy < 0.48 * S))
    # seam sealer and a little surface rust where the pan meets the bulkhead and at the step
    for row in (0.050 * S, 0.418 * S):
        pts = [(x, row + 5 * np.sin(x / 37.0) + 3 * np.sin(x / 11.0)) for x in range(0, S + 1, 8)]
        over((178, 176, 168), 0.75 * mask(lambda d: d.line(pts, fill=255, width=7), 1.2))
        over((112, 70, 44), 0.55 * np.clip((n2 - 0.52) * 6, 0, 1) * np.exp(-((yy - row - 10) / 14) ** 2))
    # bulkhead: two big openings into the trunk, rows of lightening holes in the beams above and below them
    hole = mask(lambda d: [d.rounded_rectangle(b_, radius=38, fill=255) for b_ in ((120, 560, 455, 800), (569, 560, 904, 800))], 2.0)
    over((9, 9, 11), 0.97 * hole)
    lip = mask(lambda d: [d.rounded_rectangle(b_, radius=38, outline=255, width=6) for b_ in ((120, 560, 455, 800), (569, 560, 904, 800))], 1.5)
    over((208, 211, 217), 0.5 * lip)
    def slots(d):
        for row, w_, h_, step in ((478, 46, 24, 96), (866, 40, 22, 84), (975, 52, 22, 112)):
            for x0 in range(70, S - 70, step): d.rounded_rectangle((x0, row - h_ // 2, x0 + w_, row + h_ // 2), radius=h_ // 2, fill=255)
    over((10, 10, 12), 0.95 * mask(slots, 1.0))
    over((60, 62, 68), 0.5 * mask(lambda d: [d.line([(0, r_), (S, r_)], fill=255, width=4) for r_ in (444, 520, 835, 905, 940)], 1.5))   # folds of the beams
    # shade: the corner under the bulkhead, the edges by the wheel arches, dirt in the low spots
    img *= (1 - 0.30 * np.exp(-((yy - 0.42 * S) / 40) ** 2) - 0.22 * np.exp(-((np.abs(xx - S / 2) - S / 2) / 60) ** 2))[..., None]
    img *= (0.94 + 0.12 * n3)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def rear_seat_delete(k):
    gone = sum(drop_components(k, nm, lambda c: c[2] < -0.75) for nm in ('seatleather1', 'seatleather2'))
    z0, z1, z2 = PAN_Z; xs = np.linspace(-0.745, 0.745, 61)
    zs = np.r_[np.linspace(z0 + 0.06, z0 - 0.06, 9), np.linspace(z0 - 0.08, z1, 14)[:-1], np.linspace(z1, z2, 30)]
    X, Z = np.meshgrid(xs, zs); Y = pan_height(X, Z); P = np.stack([X, Y, Z], axis=-1)
    prof = np.c_[zs, pan_height(np.full(len(zs), 0.30), zs)]; ln = np.r_[0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    UV = np.stack([(X + 0.745) / 1.49, np.broadcast_to((ln / ln[-1])[:, None], X.shape)], axis=-1)
    k.set_texture('gh_bare_metal.png', png_bytes(bare_metal_image()))
    k.add_mesh('REAR_BARE_FLOOR', *surf(P, UV, (0, 1, 0.3)), k.add_material(simple_material('gh_bare_metal', 'gh_bare_metal.png', 0.26, 0.40, 0.18, 28.0)))
    # the body sides behind the rear door cards, which the seat back used to cover
    im = np.empty((64, 64, 3)); im[:] = (100, 103, 110); im *= (0.85 + 0.3 * noise((64, 64), 5, 77))[..., None]
    k.set_texture('gh_bare_metal_side.png', png_bytes(Image.fromarray(np.clip(im, 0, 255).astype(np.uint8), 'RGB')))
    b = Builder()
    for sx in (-1, 1):
        zz = np.linspace(-1.10, z2 - 0.02, 8); x = 0.748 * sx
        for za, zb in zip(zz[:-1], zz[1:]):
            ya, yb = pan_height(np.array([x]), np.array([za]))[0] - 0.02, pan_height(np.array([x]), np.array([zb]))[0] - 0.02
            b.poly([(x, ya, za), (x, yb, zb), (x, 0.86, zb), (x, 0.86, za)], (-sx, 0, 0), [(za, ya), (zb, yb), (zb, 0.86), (za, 0.86)])
    k.add_mesh('REAR_BARE_SIDES', *b.done(), k.add_material(simple_material('gh_bare_metal_side', 'gh_bare_metal_side.png', 0.24, 0.36, 0.15, 24.0)))
    return {'seat_tris_removed': gone}

# ---- driver's bucket seat with harness
SEAT_PATH = np.array([   # z, y of the seat's centre line from the front of the cushion to the top of the backrest; half width; bolster height
    (0.130, 0.300, 0.205, 0.005), (0.125, 0.335, 0.220, 0.020), (0.070, 0.356, 0.235, 0.050), (-0.080, 0.338, 0.240, 0.075),
    (-0.230, 0.318, 0.235, 0.100), (-0.310, 0.326, 0.228, 0.125), (-0.348, 0.372, 0.225, 0.140), (-0.372, 0.470, 0.222, 0.135),
    (-0.400, 0.600, 0.235, 0.105), (-0.432, 0.740, 0.262, 0.090), (-0.462, 0.860, 0.230, 0.055), (-0.488, 0.960, 0.175, 0.030),
    (-0.500, 1.030, 0.120, 0.004)])
SEAT_RGB = (214, 168, 177)        # sun-faded red: dusty pink

def seat_frame(stations=57):
    from scipy.interpolate import PchipInterpolator
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(SEAT_PATH[:, :2], axis=0), axis=1))]; f = PchipInterpolator(d / d[-1], SEAT_PATH)
    S = np.linspace(0, 1, stations); q = f(S); t = unit(f.derivative()(S)[:, :2]); nrm = np.c_[t[:, 1], -t[:, 0]]     # (z, y): towards the occupant
    return S, q, t, nrm, d[-1]

def seat_point(q, nrm, i, u, lift=0.0, back=False):
    z, y, w, h = q[i]; off = -(0.020 + 0.040 * (1 - u * u)) - lift if back else h * ss(0.50, 0.98, abs(u)) + lift
    return np.array([SEAT_X + w * u, y + nrm[i, 1] * off, z + nrm[i, 0] * off])

def bucket_seat_image(S, q, i_sh, size=1024, seed=41):
    """left half: the front of the seat (U across, V from the front of the cushion to the top); right half: the back of the shell"""
    W = H = size; yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.empty((H, W, 3)); img[:] = SEAT_RGB
    img *= (0.86 + 0.20 * noise((H, W), 50, seed) + 0.08 * noise((H, W), 4, seed + 1))[..., None]             # uneven fading
    img *= (0.96 + 0.08 * noise((H, W), 0.8, seed + 2))[..., None]                                             # cloth
    def over(rgb, a): img[:] = img * (1 - a[..., None]) + np.array(rgb, float) * a[..., None]
    def mask(fn, blur=0.0):
        im = Image.new('L', (W, H), 0); fn(ImageDraw.Draw(im))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)) if blur else im, np.float64) / 255
    px = lambda u: (0.02 + 0.46 * (u + 1) / 2) * W; row = lambda s: (0.01 + 0.98 * s) * H
    corner = S[np.argmin(np.abs(q[:, 0] + 0.33) + (q[:, 1] > 0.40))]                                           # where cushion and backrest meet
    seams = [[(px(u_), row(0.03)), (px(u_), row(0.80))] for u_ in (-0.52, 0.52)]                               # along the bolsters
    seams += [[(px(-0.52), row(s_)), (px(0.52), row(s_))] for s_ in (corner * 0.48, corner, corner + 0.18)]    # across the centre panels
    seams += [[(px(0.0), row(corner * 0.48)), (px(0.0), row(corner))]]
    over((120, 86, 96), 0.70 * mask(lambda d: [d.line(s_, fill=255, width=4) for s_ in seams], 1.2))
    over((236, 200, 206), 0.35 * mask(lambda d: [d.line([(a[0] + 5, a[1] + 5), (b_[0] + 5, b_[1] + 5)], fill=255, width=4) for a, b_ in seams], 2.0))
    # harness slots at shoulder height, with their black trim
    slot = [(px(sg * 0.40) - 36, row(S[i_sh]) - 50, px(sg * 0.40) + 36, row(S[i_sh]) + 42) for sg in (-1, 1)]
    over((14, 14, 16), mask(lambda d: [d.rounded_rectangle(b_, radius=16, fill=255) for b_ in slot], 1.0))
    over((40, 40, 44), 0.8 * mask(lambda d: [d.rounded_rectangle(b_, radius=16, outline=255, width=7) for b_ in slot], 0.8))
    # the print on the headrest: a dark ring with a brushed band across it
    cx, cy = px(0.0), row(0.925)
    ring = mask(lambda d: (d.ellipse((cx - 50, cy - 46, cx + 50, cy + 46), outline=255, width=6), d.ellipse((cx - 36, cy - 33, cx + 36, cy + 33), outline=255, width=2)), 0.8)
    band = mask(lambda d: [d.line([(cx - 62 + 9 * j, cy + 6 + (j % 3) * 4 - 16 * (j % 2)), (cx - 56 + 9 * j, cy + 20 - (j % 4) * 3)], fill=255, width=6) for j in range(14)], 0.7)
    over((62, 30, 40), 0.80 * np.maximum(ring * 0.8, band))
    # grime: grey where shoulders, head and hands rub; darker towards the edges of the shell
    dirt = np.exp(-(((xx - px(0.78)) / 60) ** 2 + ((yy - row(0.80)) / 90) ** 2)) + np.exp(-(((xx - px(-0.78)) / 60) ** 2 + ((yy - row(0.78)) / 80) ** 2))
    dirt += 0.7 * np.exp(-(((xx - px(0.45)) / 70) ** 2 + ((yy - row(0.93)) / 40) ** 2)) + 0.5 * np.exp(-(((xx - px(0.0)) / 150) ** 2 + ((yy - row(0.10)) / 50) ** 2))
    over((120, 112, 112), np.clip(0.55 * dirt * (0.5 + noise((H, W), 9, seed + 3)), 0, 0.75))
    img[:, W // 2:] *= 0.90                                                                                    # back of the shell: a shade darker
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def harness_image(w=128, h=512):
    im = Image.new('RGB', (w, h), (17, 17, 19)); a = np.asarray(im, np.float64).copy()
    yy, xx = np.mgrid[0:h, 0:w]; a *= (0.85 + 0.30 * ((xx // 3 + yy // 2) % 2))[..., None]                     # webbing
    a[:, :6] *= 0.6; a[:, -6:] *= 0.6
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB')
    f = font(['arialbi.ttf', 'arialbd.ttf'], 76); t = Image.new('L', (330, 96), 0); ImageDraw.Draw(t).text((6, 2), 'sparco', font=f, fill=255)
    t = t.rotate(90, expand=True); im.paste((236, 236, 236), (int((w - t.width) / 2), 150), t)
    return im

def bucket_seat(k):
    gone = sum(drop_components(k, nm, lambda c: c[0] > 0.05 and c[2] >= -0.75) for nm in ('seatleather1', 'seatleather2'))
    S, q, t, nrm, length = seat_frame()
    uf = np.linspace(-1, 1, 21); ub = np.linspace(1, -1, 11)
    loop = [(u, False) for u in uf] + [(u, True) for u in ub] + [(-1.0, False)]
    U = np.r_[0.02 + 0.46 * (uf + 1) / 2, 0.52 + 0.46 * (1 - ub) / 2, 1.02]
    P = np.array([[seat_point(q, nrm, i, u, back=bk) for u, bk in loop] for i in range(len(S))])
    UV = np.stack([np.broadcast_to(U[None], P.shape[:2]), np.broadcast_to((0.01 + 0.98 * S)[:, None], P.shape[:2])], axis=-1)
    shell = surf(P, UV, 'out'); caps = Builder()
    for i, sgn in ((0, -1.0), (len(S) - 1, 1.0)):
        caps.poly(P[i, :-1], (0, sgn * t[i, 1], sgn * t[i, 0]), [(0.75, 0.01 + 0.98 * S[i])] * (P.shape[1] - 1))
    i_sh = int(np.argmin(np.abs(q[:, 1] - 0.80) + (q[:, 0] > -0.36))); i_lap = int(np.argmin(np.abs(q[:, 0] + 0.215) + (q[:, 1] > 0.36)))
    k.set_texture('gh_bucket_seat.png', png_bytes(bucket_seat_image(S, q, i_sh)))
    k.add_mesh('BUCKET_SEAT', *join(shell, caps.done()), k.add_material(simple_material('gh_bucket_seat', 'gh_bucket_seat.png', 0.42, 0.50, 0.04, 6.0)))
    # harness: shoulder straps down the backrest to the buckle, on through the slots to the rear floor; lap belt across the cushion
    straps = []
    for sg in (-1, 1):
        idx = range(i_lap, i_sh + 1); fr = [(i - i_lap) / (i_sh - i_lap) for i in idx]
        straps.append(ribbon([seat_point(q, nrm, i, sg * (0.10 + 0.30 * f_), lift=0.012) for i, f_ in zip(idx, fr)], (1, 0, 0), 0.072, v0=0.1))
        a = seat_point(q, nrm, i_sh, sg * 0.40, lift=0.004, back=True); b_ = np.array([SEAT_X + sg * 0.11, pan_height(np.array([SEAT_X]), np.array([-0.99]))[0] + 0.006, -0.99])
        f_ = np.linspace(0, 1, 12)[:, None]; straps.append(ribbon(a + (b_ - a) * f_ - np.array([0, 0.035, 0]) * np.sin(np.pi * f_), (1, 0, 0), 0.072, v0=0.55))
        straps.append(box(b_ - (0.022, 0.004, 0.022), b_ + (0.022, 0.010, 0.022), (0.5, 0.02)))                 # anchor plate bolted to the pan
    tg = np.array([0, t[i_lap, 1], t[i_lap, 0]])
    straps.append(ribbon([seat_point(q, nrm, i_lap, u, lift=0.014) for u in np.linspace(-1, 1, 17)], tg, 0.050, v0=0.6))
    c = seat_point(q, nrm, i_lap, 0.0, lift=0.016); straps.append(box(c - (0.038, 0.0, 0.032), c + (0.038, 0.020, 0.032), (0.5, 0.02)))   # buckle
    k.set_texture('gh_harness.png', png_bytes(harness_image()))
    k.add_mesh('HARNESS', *join(*straps), k.add_material(simple_material('gh_harness', 'gh_harness.png', 0.42, 0.45, 0.10, 12.0)), shadows=0)
    k.add_mesh('BUCKET_SEAT_MOUNT', *box((SEAT_X - 0.20, FLOOR_Y, -0.30), (SEAT_X + 0.20, 0.285, 0.09)), k.mat_id('plastic_black1'))
    return {'seat_tris_removed': gone}

# ---- manual shifter with the cloth bag as its boot
SHIFT_XZ = (-0.02, 0.215); SHIFT_Y = 0.333      # where the lever comes out of the console; height of the console there

def lathe(prof, v0, v1, seg=18, urep=1.0, tilt=0.0):
    """rings of radius r at height h around the shifter axis; r may be a function of the angle"""
    th = np.linspace(0, 2 * np.pi, seg + 1); P = []
    for r, h in prof:
        rr = r(th) if callable(r) else np.full(len(th), r)
        P.append(np.c_[SHIFT_XZ[0] + rr * np.cos(th), np.full(len(th), h), SHIFT_XZ[1] + rr * np.sin(th) - (h - SHIFT_Y) * np.tan(tilt)])
    P = np.array(P); hs = np.array([h for _, h in prof]); f = (hs - hs[0]) / (hs[-1] - hs[0])
    UV = np.stack([np.broadcast_to((th / (2 * np.pi) * urep)[None], P.shape[:2]), np.broadcast_to((v0 + (v1 - v0) * f)[:, None], P.shape[:2])], axis=-1)
    return surf(P, UV, 'out')

def shifter_image(size=256, seed=51):
    """top half: black leather with a stitched seam; bottom half: the green cloth bag (its gathered, cream-stitched hem at the top)"""
    S = size; h = S // 2; img = np.empty((S, S, 3)); yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    img[:h] = (24, 24, 26); img[:h] *= (0.8 + 0.5 * noise((h, S), 2, seed))[..., None]
    img[h:] = (64, 148, 104); folds = 0.5 + 0.5 * np.sin(xx[h:] / S * 2 * np.pi * 9 + 3 * noise((h, S), 12, seed + 1) * 6)
    img[h:] *= (0.62 + 0.38 * folds * (0.6 + 0.4 * noise((h, S), 5, seed + 2)))[..., None]
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB'); d = ImageDraw.Draw(im)
    for x0 in range(0, S, 10):
        d.line([(x0, 70), (x0 + 5, 73)], fill=(120, 120, 124), width=1)                                         # stitches round the knob
        d.line([(x0, h + 12), (x0 + 6, h + 12)], fill=(214, 206, 170), width=2); d.line([(x0 + 2, h + 20), (x0 + 8, h + 20)], fill=(214, 206, 170), width=2)
    d.rectangle((0, h, S, h + 5), fill=(30, 74, 52))
    return im

def shifter(k):
    x0, z0 = SHIFT_XZ; tilt = np.radians(7.0)
    def bag(t_):
        base = lambda th: 0.058 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th))) ** 0.75                    # rounded square at the console opening
        neck = 0.021 + 0.012 * max(0.0, (t_ - 0.82) / 0.18)                                                      # gathered round the lever, hem flaring out again
        m = ss(0.0, 0.85, t_) ** 0.8
        return lambda th: base(th) * (1 - m) + neck * m + 0.006 * np.sin(7 * th + 4 * t_) * np.sin(np.pi * min(t_ * 1.15, 1.0))
    ts = np.linspace(0, 1, 11)
    parts = [lathe([(bag(t_), SHIFT_Y - 0.004 + 0.082 * t_) for t_ in ts], 0.99, 0.51, seg=28, urep=2.0, tilt=tilt * 0.5),
             lathe([(0.009, SHIFT_Y + 0.045), (0.009, SHIFT_Y + 0.160)], 0.30, 0.45, seg=10, tilt=tilt),
             lathe([(0.011, SHIFT_Y + 0.150), (0.016, SHIFT_Y + 0.165), (0.023, SHIFT_Y + 0.190), (0.0275, SHIFT_Y + 0.215), (0.0265, SHIFT_Y + 0.232),
                    (0.019, SHIFT_Y + 0.245), (0.008, SHIFT_Y + 0.250), (0.0005, SHIFT_Y + 0.251)], 0.45, 0.02, seg=18, tilt=tilt)]
    k.set_texture('gh_shifter.png', png_bytes(shifter_image()))
    V, F = join(*parts)
    k.add_mesh('SHIFTER', V, F, k.add_material(simple_material('gh_shifter', 'gh_shifter.png', 0.42, 0.48, 0.12, 14.0)))
    return {'tris': len(F)}

# ---- checker mat in the driver's footwell
MAT_BOX = (0.275, 0.655, 0.185, 0.725)          # x from the tunnel to the sill, z from the seat to the pedals

def floor_mat_image(w=384, h=512, seed=61):
    yy, xx = np.mgrid[0:h, 0:w]; sq = 32; chk = ((xx // sq + yy // sq) % 2).astype(np.float64)
    img = (22 + 78 * chk)[..., None] * np.array([1.0, 1.0, 1.03]); img = img * (0.80 + 0.35 * noise((h, w), 1.0, seed))[..., None]
    img *= (0.75 + 0.35 * noise((h, w), 30, seed + 1))[..., None]                                               # dirt, heel wear
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)); img[edge < 9] = (14, 14, 15)     # bound edge
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def floor_mat(k):
    x0, x1, z0, z1 = MAT_BOX; y = FLOOR_Y + 0.005; b = Builder()
    b.poly([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], (0, 1, 0), [(0, 1), (1, 1), (1, 0), (0, 0)])
    k.set_texture('gh_floor_mat.png', png_bytes(floor_mat_image()))
    k.add_mesh('FLOOR_MAT', *b.done(), k.add_material(simple_material('gh_floor_mat', 'gh_floor_mat.png', 0.40, 0.45, 0.03, 5.0)), shadows=0)
    return {}

# ---- aftermarket single-DIN head unit with an open pocket below it, in place of the stock radio / cassette deck
HU_X = (0.077, -0.127); HU_Y = (0.598, 0.493)        # the double-DIN opening in the centre stack: left/right edge, top/bottom edge
HU_UNIT = (0.066, -0.116, 0.595, 0.547)              # the head unit's face inside it: left, right, top, bottom
HU_POCKET = (0.062, -0.112, 0.543, 0.499)

def hu_point(x, y, d):
    """a point d metres in front of the centre stack's face (which leans back towards the top)"""
    n = np.array([0.0, 0.196, -1.0]); n /= np.linalg.norm(n)
    return np.array([x, y, 0.435 + 0.196 * (y - 0.497)]) + n * d

def hu_uv(x, y):
    return ((HU_X[0] - x) / (HU_X[0] - HU_X[1]), 0.9 * (HU_Y[0] - y) / (HU_Y[0] - HU_Y[1]))

def head_unit_image(W=1024, H=512, seed=71):
    PH = int(H * 0.9); yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    img = np.empty((H, W, 3)); img[:] = (20, 20, 22); img *= (0.85 + 0.3 * noise((H, W), 1.0, seed))[..., None]       # satin black surround
    def over(rgb, a): img[:] = img * (1 - a[..., None]) + np.array(rgb, float) * a[..., None]
    def mask(fn, blur=0.0):
        im = Image.new('L', (W, H), 0); fn(ImageDraw.Draw(im))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)) if blur else im, np.float64) / 255
    def px(x, y): u, v = hu_uv(x, y); return u * W, v * H
    (ux0, uy0), (ux1, uy1) = px(HU_UNIT[0], HU_UNIT[2]), px(HU_UNIT[1], HU_UNIT[3]); uw, uh = ux1 - ux0, uy1 - uy0
    R = lambda a, b, c, d: (ux0 + a * uw, uy0 + b * uh, ux0 + c * uw, uy0 + d * uh)
    # the unit: gloss black face, a lighter bevel along its top edge, a reflection running across it
    over((7, 7, 9), mask(lambda d: d.rounded_rectangle(R(0, 0, 1, 1), radius=10, fill=255)))
    over((70, 72, 78), 0.8 * mask(lambda d: d.line([(ux0 + 8, uy0 + 4), (ux1 - 8, uy0 + 4)], fill=255, width=4), 1.5))
    over((60, 66, 80), 0.30 * mask(lambda d: d.polygon([(ux0 + 0.30 * uw, uy0), (ux0 + 0.62 * uw, uy0), (ux0 + 0.50 * uw, uy1), (ux0 + 0.18 * uw, uy1)], fill=255), 14) * mask(lambda d: d.rounded_rectangle(R(0, 0, 1, 1), radius=10, fill=255)))
    # display window with a dim read-out
    over((14, 17, 26), mask(lambda d: d.rounded_rectangle(R(0.30, 0.20, 0.80, 0.62), radius=6, fill=255)))
    f1 = font(['arialbd.ttf'], int(uh * 0.20)); f2 = font(['arial.ttf'], int(uh * 0.11)); f3 = font(['arialbd.ttf'], int(uh * 0.17))
    over((70, 120, 170), 0.55 * mask(lambda d: (d.text((ux0 + 0.34 * uw, uy0 + 0.30 * uh), 'BT AUDIO', font=f1, fill=255), d.text((ux0 + 0.66 * uw, uy0 + 0.30 * uh), '12:00', font=f1, fill=255)), 0.6))
    over((205, 205, 210), 0.9 * mask(lambda d: d.text((ux0 + 0.52 * uw, uy0 + 0.035 * uh), 'SONY', font=f3, fill=255), 0.4))
    # buttons: a row of small keys under the display, source/eject at the left, release key, aux jack and USB flap at the right
    keys = [R(0.30 + 0.085 * j, 0.72, 0.30 + 0.085 * j + 0.072, 0.92) for j in range(6)] + [R(0.03, 0.08, 0.10, 0.30), R(0.87, 0.10, 0.97, 0.40)]
    over((30, 31, 35), mask(lambda d: [d.rounded_rectangle(b_, radius=5, fill=255) for b_ in keys]))
    over((85, 87, 94), 0.7 * mask(lambda d: [d.line([(b_[0] + 4, b_[1] + 2), (b_[2] - 4, b_[1] + 2)], fill=255, width=2) for b_ in keys], 0.8))
    over((170, 170, 176), 0.8 * mask(lambda d: [d.text((b_[0] + 9, b_[1] + 0.045 * uh), str(j + 1), font=f2, fill=255) for j, b_ in enumerate(keys[:6])], 0.3))
    over((2, 2, 3), mask(lambda d: d.ellipse(R(0.845, 0.62, 0.845 + 0.20 * uh / uw, 0.82), fill=255), 0.8))
    over((120, 122, 128), 0.8 * mask(lambda d: d.ellipse(R(0.845, 0.62, 0.845 + 0.20 * uh / uw, 0.82), outline=255, width=3), 0.8))
    over((26, 27, 30), mask(lambda d: d.rounded_rectangle(R(0.90, 0.56, 0.975, 0.90), radius=4, fill=255)))
    # volume knob (the cap of the modelled knob uses this part of the picture)
    kx, ky, kr = ux0 + 0.20 * uw, uy0 + 0.55 * uh, 0.27 * uh
    dk = np.hypot(xx - kx, yy - ky); ang = np.arctan2(yy - ky, xx - kx)
    over((0, 0, 0), 0.7 * np.clip(1 - (dk - kr) / 14, 0, 1) * (dk > kr))                                              # shadow round it
    over((150, 152, 158), (dk <= kr) * 1.0); img[dk <= kr] *= (0.55 + 0.45 * np.cos(2 * ang[dk <= kr] + 0.6) ** 2)[..., None]   # brushed ring
    over((16, 16, 18), (dk <= kr * 0.70) * 1.0); over((90, 92, 100), 0.6 * np.exp(-(((xx - kx + 0.25 * kr) / (0.22 * kr)) ** 2 + ((yy - ky + 0.28 * kr) / (0.16 * kr)) ** 2)) * (dk <= kr * 0.70))
    # the open pocket: dark inside, its floor catching a little light, a rounded lip
    (qx0, qy0), (qx1, qy1) = px(HU_POCKET[0], HU_POCKET[2]), px(HU_POCKET[1], HU_POCKET[3])
    pocket = mask(lambda d: d.rounded_rectangle((qx0, qy0, qx1, qy1), radius=16, fill=255), 1.0)
    over((4, 4, 5), pocket); over((40, 41, 45), 0.75 * pocket * np.clip((yy - (qy0 + 0.62 * (qy1 - qy0))) / (0.38 * (qy1 - qy0)), 0, 1) ** 1.5)
    over((58, 60, 66), 0.8 * mask(lambda d: d.rounded_rectangle((qx0, qy0, qx1, qy1), radius=16, outline=255, width=4), 1.2))
    img[PH:, :W // 2] = (150, 152, 158); img[PH:, W // 2:] = (9, 9, 11)                                               # swatches: knob side, case sides
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def head_unit(k):
    inside = lambda c: HU_X[1] - 0.005 < c[0] < HU_X[0] + 0.005 and HU_Y[1] - 0.005 < c[1] < HU_Y[0] + 0.005
    gone = drop_components(k, 'Object022_002_002', inside) + drop_components(k, 'Cylinder001_002_002', inside) + drop_components(k, 'Object027_002_002', lambda c: True)
    n = np.array([0.0, 0.196, -1.0]); n /= np.linalg.norm(n); b = Builder(); black = (0.75, 0.95)
    def face(x0, x1, y0, y1, d): b.poly([hu_point(x0, y0, d), hu_point(x1, y0, d), hu_point(x1, y1, d), hu_point(x0, y1, d)], n, [hu_uv(x0, y0), hu_uv(x1, y0), hu_uv(x1, y1), hu_uv(x0, y1)])
    def rim(x0, x1, y0, y1, d0, d1):
        for (xa, ya), (xb, yb), nr in (((x0, y0), (x1, y0), (0, 1, 0)), ((x1, y0), (x1, y1), (-1, 0, 0)), ((x1, y1), (x0, y1), (0, -1, 0)), ((x0, y1), (x0, y0), (1, 0, 0))):
            b.poly([hu_point(xa, ya, d0), hu_point(xb, yb, d0), hu_point(xb, yb, d1), hu_point(xa, ya, d1)], nr, [black] * 4)
    face(HU_X[0], HU_X[1], HU_Y[0], HU_Y[1], 0.007); rim(HU_X[0], HU_X[1], HU_Y[0], HU_Y[1], -0.004, 0.007)          # surround plate with the pocket
    face(*HU_UNIT, 0.015); rim(*HU_UNIT, 0.007, 0.015)                                                                # the unit standing proud of it
    cx = HU_UNIT[0] + 0.20 * (HU_UNIT[1] - HU_UNIT[0]); cy = HU_UNIT[2] + 0.55 * (HU_UNIT[3] - HU_UNIT[2]); r = 0.27 * (HU_UNIT[2] - HU_UNIT[3])
    th = np.linspace(0, 2 * np.pi, 19)[:-1]; up = np.array([0.0, 1.0, 0.196]); up /= np.linalg.norm(up); ring = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in th]
    b.poly([hu_point(x, y, 0.027) for x, y in ring], n, [hu_uv(x, y) for x, y in ring])                              # volume knob: cap and side
    for (xa, ya), (xb, yb), a in zip(ring, ring[1:] + ring[:1], th + np.pi / 18):
        b.poly([hu_point(xa, ya, 0.015), hu_point(xb, yb, 0.015), hu_point(xb, yb, 0.027), hu_point(xa, ya, 0.027)], np.array([np.cos(a), 0, 0]) + up * np.sin(a), [(0.25, 0.95)] * 4)
    k.set_texture('gh_head_unit.png', png_bytes(head_unit_image()))
    V, F = b.done()
    k.add_mesh('HEAD_UNIT', V, F, k.add_material(simple_material('gh_head_unit', 'gh_head_unit.png', 0.40, 0.45, 0.55, 60.0)), shadows=0)
    return {'stock_radio_tris_removed': gone, 'tris': len(F)}

def build(src, out):
    k = Kn5.load(src); info = {}
    for step in STEPS: info[step] = globals()[step](k)
    k.save(out)
    return info

if __name__ == '__main__':
    import hashlib
    src = str(config.STAGE1_KN5); out = str(config.OUTPUT_KN5); config.OUTPUT_KN5.parent.mkdir(parents=True, exist_ok=True)
    if not config.STAGE1_KN5.exists(): raise SystemExit('Stage-1 model not found: %s (see README_SCRIPTS.md)' % src)
    print('stage 1 :', src); print('output  :', out)
    for step, res in build(src, out).items(): print('  %-16s %s' % (step, res))
    print('md5', hashlib.md5(open(out, 'rb').read()).hexdigest())
