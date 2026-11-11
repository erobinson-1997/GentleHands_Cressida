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
from kn5io import Kn5, Material, Node, world_pos, deform, split_mesh, update_bounds, unit

STEPS = ['wheels_black', 'mud_flaps', 'bar_rusty', 'plate', 'quarter_crease', 'mirror_primer', 'window_sticker', 'hood_fade', 'door_numbers', 'front_end', 'lower_panel', 'grille', 'clear_lamps', 'taillight_damage']
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
    zc, half_w, y0, y1 = 0.185, 0.160, 0.468, 0.770               # magnet panel between the side moulding and the belt line
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
# Decal space (s, t): s runs along the lamp in the lens' own v coordinate (0 = x -0.50, 0.84 = the corner, 1 = end of the
# wrap-around), t down it in u (0 = top of the red lens, 0.647 = red/amber joint, 1 = bottom of the amber lens).
TAIL_V = (-0.60, -0.024)
TAIL_U = (0.349, 0.618)
TW, TH, TRED = 1024, 512, 331
def tail_out(pts):
    """the lamp's rear face runs on further round the corner than its UVs suggest: stretch everything outboard of px 600"""
    return [(x if x < 600 else 600 + (x - 600) * 1.6, y) for x, y in pts]
TAIL_HOLE = tail_out([(520, 20), (506, 150), (470, 250), (400, 310), (333, 334), (600, 342), (848, 338), (852, 60), (800, 20)])   # px
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
    """what shows through the hole: the scuffed inner red lens, two bare patches of white reflector optics, a dark gap
    along the bottom. Returns (diffuse RGB, normal map RGB)."""
    rng = np.random.default_rng(83); yy, xx = np.mgrid[0:TH, 0:TW].astype(float)
    n1 = noise((TH, TW), 6, 1); n2 = noise((TH, TW), (2, 12), 2)
    a = np.array([118.0, 15, 27]) * (0.80 + 0.40 * n1[:, :, None]); hgt = 0.6 * n2                 # dull dark red
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB').convert('RGBA'); d = ImageDraw.Draw(im, 'RGBA')
    d.polygon(tail_out([(711, 20), (744, 20), (752, 150), (780, 245), (760, 320), (716, 320)]), fill=(168, 20, 38, 255))     # shard left standing
    sc = Image.new('RGBA', (TW, TH), (0, 0, 0, 0)); sd = ImageDraw.Draw(sc)                        # white scrape marks
    for cx, cy, sx, sy, cnt in ((585, 175, 42, 75, 700), (735, 200, 12, 70, 160), (640, 318, 60, 10, 140), (830, 250, 12, 50, 60)):
        for _ in range(cnt):
            x = tail_out([(rng.normal(cx, sx), 0)])[0][0]; y = rng.normal(cy, sy); L = rng.uniform(6, 34); an = rng.normal(-1.05, 0.22)
            sd.line([(x, y), (x + L * np.cos(an), y + L * np.sin(an))], fill=(244, 226, 228, int(rng.uniform(40, 150))), width=int(rng.integers(1, 3)))
    im.alpha_composite(sc.filter(ImageFilter.GaussianBlur(0.6)))
    grid = np.zeros((TH, TW), bool)
    for poly in ([(647, 58), (711, 60), (711, 300), (650, 297)], [(744, 52), (812, 60), (810, 185), (782, 240), (754, 150)]):
        g = Image.new('L', (TW, TH), 0); ImageDraw.Draw(g).polygon(tail_out(poly), fill=255); grid |= np.asarray(g) > 0
    a = np.asarray(im).astype(float)[:, :, :3]
    cell = 7.0; px_ = (xx % cell) / cell - 0.5; py_ = (yy % cell) / cell - 0.5; pil = 1 - 4 * np.maximum(px_ ** 2, py_ ** 2)   # pillow optics
    a[grid] = (np.array([232.0, 232, 236]) * (0.72 + 0.28 * pil[:, :, None]))[grid]; hgt = np.where(grid, 6.0 * pil, hgt)
    edge = Image.fromarray((grid * 255).astype(np.uint8)).filter(ImageFilter.FIND_EDGES); a[np.asarray(edge) > 0] = (60, 8, 14)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB'); d = ImageDraw.Draw(im)
    d.polygon(tail_out([(596, 322), (700, 314), (712, 296), (736, 300), (742, 318), (850, 322), (850, 345), (596, 345)]), fill=(10, 5, 6))   # gap at the bottom
    d.line(tail_out([(852, 60), (846, 322)]), fill=(40, 6, 10), width=4)
    # the lens edge throws a shadow onto the inside along the top and inboard sides of the hole
    hole = Image.new('L', (TW, TH), 0); ImageDraw.Draw(hole).polygon(TAIL_HOLE, fill=255)
    lit = np.asarray(ImageChops.offset(hole, 9, 12).filter(ImageFilter.GaussianBlur(5))).astype(float) / 255
    a = np.asarray(im).astype(float) * (0.38 + 0.62 * lit[:, :, None])
    gy, gx = np.gradient(gaussian_filter(hgt, 0.7)); nz = np.ones_like(gx); l = np.sqrt(gx ** 2 + gy ** 2 + 1)
    nm = np.dstack([-gx / l, gy / l, nz / l]) * 0.5 + 0.5
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB'), Image.fromarray((nm * 255).astype(np.uint8), 'RGB')

def tail_crack_image():
    """transparent except for the cracks that run on from the break, and the bright broken edge of the lens"""
    rng = np.random.default_rng(84); im = Image.new('RGBA', (TW, TH), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    dark = (40, 5, 9, 255); lightc = (240, 205, 200, 255)
    for line, w in (([(333, 334), (250, 336), (170, 338), (110, 372), (30, 420), (0, 432)], 3),      # along the joint, then down across the amber
                    ([(170, 338), (120, 330), (60, 334)], 1), ([(506, 150), (486, 120), (476, 60), (484, 0)], 2),
                    ([(470, 250), (430, 240), (396, 262)], 1), ([(690, 342), (684, 392), (694, 432)], 2),
                    ([(684, 392), (640, 410)], 1), ([(760, 340), (772, 420), (742, 512)], 2), ([(772, 420), (820, 450)], 1),
                    ([(600, 342), (560, 372), (500, 384)], 1)):
        line = tail_out(line); j = [(x + rng.uniform(-2, 2), y + rng.uniform(-2, 2)) for x, y in line]; j[0] = line[0]
        d.line(j, fill=dark, width=w, joint='curve')
        if w > 1: d.line([(x + 1, y - w) for x, y in j], fill=lightc, width=1)
    edge = TAIL_HOLE[:5]                                                                             # the curved break
    d.line([(x - 2, y - 1) for x, y in edge], fill=lightc, width=3, joint='curve'); d.line([(x - 5, y - 2) for x, y in edge], fill=dark, width=2, joint='curve')
    d.line(tail_out([(520, 17), (800, 17)]), fill=lightc, width=2)
    return im

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
    # 3. cracks and the bright broken edge, a hair outside the red and amber lenses
    k.set_texture('gh_taillight_damage.png', png_bytes(tail_crack_image()))
    m = simple_material('gh_taillight_damage', 'gh_taillight_damage.png', 0.42, 0.45, 0.45, 35.0, shader='ksPerPixelAT', alpha_ref=0.5)
    m.alpha_tested = 1; mid = k.add_material(m)
    Vs = []; Fs = []; off = 0
    for name in ('Object036', 'signal_r_glass'):
        n, _, M = k.find(name); w = world_pos(n, M); st = tail_st(n.V[:, 6:8])
        nr = unit(n.V[:, 3:6] @ M[:3, :3]); nr *= np.sign(nr @ np.array([-0.3, 0.0, -1.0]) + 1e-9)[:, None]
        ok = (w[:, 0] < 0) & (st[:, 0] > -0.03) & (st[:, 0] < 1.0)
        F = n.F[ok[n.F].all(1)]; used = np.unique(F); re = np.full(len(w), -1); re[used] = np.arange(len(used))
        V = np.zeros((len(used), 11)); V[:, :3] = w[used] + nr[used] * 0.0012; V[:, 3:6] = nr[used]; V[:, 6:8] = st[used]; V[:, 8:11] = [-1, 0, 0]
        Vs.append(V); Fs.append(re[F] + off); off += len(V)
    V = np.vstack(Vs); F = np.vstack(Fs)
    k.add_mesh('TAILLIGHT_DAMAGE_R', V, fix_winding(V, F), mid, shadows=0)
    info['crack_tris'] = len(F); return info

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
