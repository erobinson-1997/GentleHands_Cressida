"""Stage 2: details taken from the walk-around videos (Reference Videos/Body.mp4).

Input : the stage-1 model (output of build_rear_damage.py, MD5 e02e7a7c...) -> config.STAGE1_KN5
Output: config.OUTPUT_KN5 (dist/)

    python scripts/build_video_details.py

Each step is a function below; switch steps on/off in STEPS. World coordinates: +x = car's LEFT, +y = up, +z = front.
"""
import sys, os, io; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter
import config
from kn5io import Kn5, Material, Node, world_pos, deform, split_mesh, update_bounds, unit

STEPS = ['wheels_black', 'mud_flaps', 'bar_rusty', 'plate', 'quarter_crease', 'mirror_primer', 'window_sticker', 'hood_fade']
PLATE_TEXT = '8XKQ407'          # made-up number (not the real plate); set to None to keep the original mod's plate
PLATE_MOSAIC = True             # pixelate the number like a censored photo
FRONT_PLATE = False             # the real car runs no front plate
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
def sticker_image(w=1024, h=440):
    def word(text, size, fill, stroke, sw, box):
        f = font(['impact.ttf', 'arialbd.ttf'], size)
        im = Image.new('RGBA', (size * len(text), size * 2), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((sw + 4, sw + 4), text, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke)
        return im.crop(im.getbbox()).resize(box, Image.LANCZOS)
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); lean = 0.20; pad = int(lean * h / 2) + 4
    a = word('TIRE', 260, (120, 232, 46, 255), (16, 22, 16, 255), 12, (w - 2 * pad - 60, int(h * 0.54)))
    b = word('STREETS', 200, (232, 232, 236, 255), (40, 40, 46, 255), 9, (w - 2 * pad - 60, int(h * 0.38)))
    im.alpha_composite(a, (pad + 60, 6)); im.alpha_composite(b, (pad, h - b.size[1] - 6))
    return im.transform((w, h), Image.AFFINE, (1, lean, -lean * h / 2, 0, 1, 0), Image.BICUBIC)   # lean it like the logo
def window_sticker(k):
    k.set_texture('gh_sticker.png', png_bytes(sticker_image()))
    m = simple_material('gh_sticker', 'gh_sticker.png', 0.45, 0.45, 0.05, 10.0, shader='ksPerPixelAT', alpha_ref=0.5)
    m.alpha_tested = 1; mid = k.add_material(m)
    n, _, M = k.find('window_ext'); w = world_pos(n, M)
    z0, z1, y0, y1 = -0.40, -0.90, 0.950, 1.165                    # front edge, rear edge, bottom, top
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
    m = np.clip((0.62 * cloud + 0.38 * mott - 0.36) / 0.22, 0, 1) * (0.35 + 0.65 * centre)
    dark = 0.45 * np.exp(-((X + 0.05) / 0.22) ** 2 - ((Z - 1.62) / 0.20) ** 2)                       # the darker blotch near the front centre
    chalk = np.array([118, 124, 134.0])
    reg = a[py0:py1, px0:px1, :3]
    out = reg * (1 - 0.55 * m[..., None]) + chalk * 0.55 * m[..., None]
    out *= (1 - 0.28 * dark[..., None]) * (0.97 + 0.06 * fine[..., None])
    a[py0:py1, px0:px1, :3] = np.where(reg.sum(2, keepdims=True) > 30, out, reg)                    # leave the black gutter between islands alone
    rgb = Image.fromarray(np.clip(a[..., :3], 0, 255).astype(np.uint8), 'RGB')
    dds = dxt5_with_mips(rgb, Image.fromarray(a[..., 3].astype(np.uint8)), t[2][:128])
    assert len(dds) == len(t[2]); t[2] = dds
    skin = config.PROJECT / 'gentlehands_toyota_cressida_mx83' / 'skins' / 'silver_176' / 'carpaint_diffuse.dds'
    if skin.exists(): skin.write_bytes(dds)                       # the skin's copy overrides the one inside the kn5
    return {'texels': [px0, py0, px1, py1], 'coverage': round(float((m > 0.3).mean()), 2), 'skin': skin.exists()}

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
