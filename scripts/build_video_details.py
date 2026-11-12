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

STEPS = ['mud_flaps', 'bar_rusty', 'plate', 'quarter_crease', 'mirror_primer', 'window_sticker', 'hood_fade', 'door_numbers', 'front_end', 'lower_panel', 'grille', 'clear_lamps', 'taillight_damage', 'trunk_face', 'pinstripe', 'paint_damage',
         'rear_seat_delete', 'bucket_seat', 'shifter', 'floor_mat', 'head_unit', 'passenger_seat', 'interior_colors', 'bug_splatter', 'debris_layer']
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
    m = simple_material('gh_number_panel', 'gh_number_panel.png', 0.42, 0.50, 0.12, 20.0, shader='ksPerPixelAT', alpha_ref=0.5); m.alpha_tested = 1   # so a skin can hide it
    mid = k.add_material(m)
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
SLOT_HALF = (0.040, 0.055); SLOT_R = 0.016; STRAP_W = 0.056     # harness slots in the backrest; shoulder strap width

def seat_frame(stations=141):
    from scipy.interpolate import PchipInterpolator
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(SEAT_PATH[:, :2], axis=0), axis=1))]; f = PchipInterpolator(d / d[-1], SEAT_PATH)
    S = np.linspace(0, 1, stations); q = f(S); t = unit(f.derivative()(S)[:, :2]); nrm = np.c_[t[:, 1], -t[:, 0]]     # (z, y): towards the occupant
    return S, q, t, nrm, d[-1]

def seat_point(q, nrm, i, u, lift=0.0, back=False):
    z, y, w, h = q[i]; off = -(0.020 + 0.040 * (1 - u * u)) - lift if back else h * ss(0.50, 0.98, abs(u)) + lift
    return np.array([SEAT_X + w * u, y + nrm[i, 1] * off, z + nrm[i, 0] * off])

def seat_edge_point(q, nrm, i, u, phi):
    """the rounded outer edge of the shell, phi = 0 on the front face to pi on the back face"""
    z, y, w, h = q[i]; of, ob = h, -0.020; m, a = (of + ob) / 2, (of - ob) / 2; off = m + a * np.cos(phi); sg = np.sign(u)
    return np.array([SEAT_X + sg * (w + min(a, 0.022) * np.sin(phi)), y + nrm[i, 1] * off, z + nrm[i, 0] * off])

WRAP_U = 0.80                     # on the back, the cover wraps this far in from the sides (|u| beyond it is cloth) ...

def seat_logo(px_mm_x, px_mm_y, d_mm=100):
    """the NRG Prisma print: a dark ring, NR>>G across its top, 'Prisma' in red brush script over it with drips; drawn true to size, then
    squeezed to the texture's scale. Returns (rgb, alpha) arrays"""
    k_ = 4; Sz = int(d_mm * 1.5 * k_); c = Sz / 2; R = d_mm / 2 * k_
    col = Image.new('RGB', (Sz, Sz), (40, 26, 32)); a = Image.new('L', (Sz, Sz), 0); da = ImageDraw.Draw(a); dc = ImageDraw.Draw(col)
    da.ellipse((c - R, c - R, c + R, c + R), outline=255, width=int(2.4 * k_))
    da.ellipse((c - R * 0.86, c - R * 0.86, c + R * 0.86, c + R * 0.86), outline=255, width=int(1.0 * k_))
    f1 = font(['arialbd.ttf'], int(16 * k_)); da.text((c, c - R * 0.40), 'NR\u00bbG', font=f1, fill=255, anchor='mm')
    # 'Prisma' goes on in red, so it is cut out of the dark ring first
    pr = Image.new('L', (Sz, Sz), 0); dp = ImageDraw.Draw(pr); f2 = font(['Inkfree.ttf', 'segoescb.ttf'], int(30 * k_))
    dp.text((c, c + R * 0.10), 'Prisma', font=f2, fill=255, anchor='mm', stroke_width=int(1.2 * k_), stroke_fill=255)
    rng = np.random.default_rng(7); y0 = c + R * 0.36
    for x in np.linspace(c - R * 0.62, c + R * 0.62, 9):                                                         # drips under it
        L = rng.uniform(0.10, 0.30) * R; x += rng.normal(0, 2 * k_); dp.line([(x, y0), (x, y0 + L)], fill=255, width=int(1.1 * k_))
        dp.ellipse((x - 1.6 * k_, y0 + L - 1.6 * k_, x + 1.6 * k_, y0 + L + 1.6 * k_), fill=255)
    dp.line([(c - R * 0.70, y0), (c + R * 0.70, y0 - 2 * k_)], fill=255, width=int(1.2 * k_))
    pr = pr.rotate(6, resample=Image.BICUBIC, center=(c, c))
    a = ImageChops.subtract(a, pr.filter(ImageFilter.MaxFilter(9))); a = ImageChops.lighter(a, pr); col.paste((176, 34, 56), (0, 0), pr)
    size = (max(int(Sz / k_ * px_mm_x), 8), max(int(Sz / k_ * px_mm_y), 8))
    return np.asarray(col.resize(size, Image.LANCZOS), np.float64), np.asarray(a.resize(size, Image.LANCZOS), np.float64) / 255

def bucket_seat_image(S, q, i_sh, s_wrap, length, size=1024, seed=41):
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
    # the print on the headrest (Reference Photos/seat and shifter)
    sl = 0.925; wl = np.interp(sl, S, q[:, 2]); lc, la = seat_logo(0.46 * W / (2 * wl) / 1000, 0.98 * H / length / 1000)
    lc, la = lc[::-1], la[::-1]                                                                                 # the texture's rows run up the seat
    cx, cy = int(px(0.0)), int(row(sl)); h_, w_ = la.shape; y0_, x0_ = cy - h_ // 2, cx - w_ // 2
    reg = img[y0_:y0_ + h_, x0_:x0_ + w_]; aa = 0.88 * la[..., None]; reg[:] = reg * (1 - aa) + lc * aa
    # grime: grey where shoulders, head and hands rub; darker towards the edges of the shell
    dirt = np.exp(-(((xx - px(0.78)) / 60) ** 2 + ((yy - row(0.80)) / 90) ** 2)) + np.exp(-(((xx - px(-0.78)) / 60) ** 2 + ((yy - row(0.78)) / 80) ** 2))
    dirt += 0.7 * np.exp(-(((xx - px(0.45)) / 70) ** 2 + ((yy - row(0.93)) / 40) ** 2)) + 0.5 * np.exp(-(((xx - px(0.0)) / 150) ** 2 + ((yy - row(0.10)) / 50) ** 2))
    over((120, 112, 112), np.clip(0.55 * dirt * (0.5 + noise((H, W), 9, seed + 3)), 0, 0.75) * (xx < W // 2))
    # back of the shell: black gel coat full of prismatic glitter, inside the panel the cover leaves open
    rng = np.random.default_rng(seed + 4); bk = img[:, W // 2:]; bh, bw = bk.shape[:2]; cloth = bk.copy() * 0.93
    ub_ = 1 - 2 * ((np.arange(bw) + W // 2 + 0.5) / W - 0.52) / 0.46; sb_ = ((np.arange(bh) + 0.5) / H - 0.01) / 0.98
    shell = (np.abs(ub_)[None] < WRAP_U) & (sb_[:, None] < s_wrap)
    bk[:] = np.array([15, 15, 18.0]) * (0.85 + 0.3 * noise((bh, bw), 6, seed + 5))[..., None]
    n = int(0.10 * bh * bw); ys, xs = rng.integers(0, bh, n), rng.integers(0, bw, n); hue = rng.random(n)
    rgb = 0.5 + 0.5 * np.cos(2 * np.pi * (hue[:, None] + np.array([0, 1 / 3, 2 / 3]))); sat = rng.uniform(0.4, 0.9, n)[:, None]
    bk[ys, xs] = 255 * rng.uniform(0.25, 0.85, n)[:, None] * (1 - sat + sat * rgb)
    n = int(0.004 * bh * bw); ys, xs = rng.integers(0, bh - 1, n), rng.integers(0, bw - 1, n)
    for dy in (0, 1):
        for dx in (0, 1): bk[ys + dy, xs + dx] = 255 * rng.uniform(0.75, 1.0, n)[:, None] * np.array([1.0, 0.97, 1.0])
    from scipy.ndimage import distance_transform_edt
    edge = np.where(shell, distance_transform_edt(shell), distance_transform_edt(~shell))       # pixels from the cover's edge, either side
    bk[:] = np.where(shell[..., None], bk, cloth)
    bk *= (1 - 0.30 * np.exp(-(distance_transform_edt(shell) / 5.0) ** 2) * shell)[..., None]                   # a little shadow under the cover's edge
    bk[edge < 2.5] = (52, 40, 44)                                                                               # the turned edge of the cover
    st = (edge >= 3) & (edge < 5) & ~shell & ((np.arange(bw)[None] + np.arange(bh)[:, None]) // 5 % 2 == 0)
    bk[st] = (150, 116, 124)                                                                                    # its stitching
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
    S, q, t, nrm, length = seat_frame(141)
    uf = np.linspace(-1, 1, 61); ub = np.linspace(1, -1, 61); ph = np.linspace(0, np.pi, 9)[1:-1]            # front, rounded edge, back, rounded edge
    loop = [(u, 0, 0) for u in uf] + [(1.0, 2, p) for p in ph] + [(u, 1, 0) for u in ub] + [(-1.0, 2, p) for p in ph[::-1]] + [(-1.0, 0, 0)]
    U = np.r_[0.02 + 0.46 * (uf + 1) / 2, np.full(len(ph), 0.48), 0.52 + 0.46 * (1 - ub) / 2, np.full(len(ph), 1.02), 1.02]
    P = np.array([[seat_edge_point(q, nrm, i, u, p) if sd == 2 else seat_point(q, nrm, i, u, back=sd == 1) for u, sd, p in loop] for i in range(len(S))])
    UV = np.stack([np.broadcast_to(U[None], P.shape[:2]), np.broadcast_to((0.01 + 0.98 * S)[:, None], P.shape[:2])], axis=-1)
    i_sh = int(np.argmin(np.abs(q[:, 1] - 0.80) + (q[:, 0] > -0.36))); i_lap = int(np.argmin(np.abs(q[:, 0] + 0.215) + (q[:, 1] > 0.36)))
    # harness slots cut right through the backrest (both faces), lined with a black rubber grommet
    def at(s, u, lift=0.0, back=False):
        qi = np.array([np.interp(s, S, q[:, j]) for j in range(4)]); ni = unit(np.array([[np.interp(s, S, nrm[:, j]) for j in range(2)]]))
        return seat_point(qi[None], ni, 0, u, lift=lift, back=back)
    width = lambda s: np.interp(s, S, q[:, 2])
    slots = [(sg * 0.40 * width(S[i_sh]), S[i_sh] * length + 0.008) for sg in (-1, 1)]
    def sdf(a, b):
        out = np.full(np.shape(a), np.inf)
        for a0, b0 in slots:
            qa = np.abs(a - a0) - (SLOT_HALF[0] - SLOT_R); qb = np.abs(b - b0) - (SLOT_HALF[1] - SLOT_R)
            out = np.minimum(out, np.hypot(np.maximum(qa, 0), np.maximum(qb, 0)) + np.minimum(np.maximum(qa, qb), 0) - SLOT_R)
        return out
    us = np.array([u for u, _, _ in loop]); sds = np.array([sd for _, sd, _ in loop]); sc = (S[:-1] + S[1:]) / 2
    uc = (us[:-1] + us[1:]) / 2; same = (sds[:-1] == sds[1:]) & (sds[:-1] != 2)
    cut = (sdf(uc[None] * width(sc)[:, None], sc[:, None] * length) < 0.002) & same[None]
    Vs_, Fs_ = surf(P, UV, 'out'); Fs_ = Fs_[~np.r_[cut.ravel(), cut.ravel()]]
    def rrect(hx, hy, r, n=8):
        pts = []
        for cx, cy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
            for a in np.radians(np.linspace(a0, a0 + 90, n)): pts.append((cx * (hx - r) + r * np.cos(a), cy * (hy - r) + r * np.sin(a)))
        return np.array(pts + pts[:1])
    rims = []
    for a0, b0 in slots:
        rings = []
        for back, d, lift in ((False, 0.012, 0.0006), (False, 0.004, 0.0035), (False, -0.006, 0.0035), (True, -0.006, 0.0035), (True, 0.004, 0.0035), (True, 0.012, 0.0006)):
            o = rrect(SLOT_HALF[0] + d, SLOT_HALF[1] + d, SLOT_R + d); ss_ = (b0 + o[:, 1]) / length
            rings.append([at(s_, (a0 + da) / width(s_), lift, back) for (da, _), s_ in zip(o, ss_)])
        R = np.array(rings); rims.append(two_sided(*surf(R, np.full(R.shape[:2] + (2,), 0.5), 'out')))
    k.set_texture('gh_slot_trim.png', flat_texture((22, 22, 24)))
    k.add_mesh('BUCKET_SEAT_SLOTS', *join(*rims), k.add_material(simple_material('gh_slot_trim', 'gh_slot_trim.png', 0.40, 0.42, 0.20, 18.0)))
    caps = Builder()
    for i, sgn in ((0, -1.0), (len(S) - 1, 1.0)):
        caps.poly(P[i, :-1], (0, sgn * t[i, 1], sgn * t[i, 0]), [(0.75, 0.01 + 0.98 * S[i])] * (P.shape[1] - 1))
    top = (slots[0][1] + SLOT_HALF[1] + 0.065) / length; s_wrap = S[np.argmin(np.abs(S - top))]                # ... and over the top down to just above the slots
    k.set_texture('gh_bucket_seat.png', png_bytes(bucket_seat_image(S, q, i_sh, s_wrap, length)))
    V, F = join((Vs_, Fs_), caps.done()); uvc = V[F][:, :, 6:8].mean(1); ubc = 1 - 2 * (uvc[:, 0] - 0.52) / 0.46; scc = (uvc[:, 1] - 0.01) / 0.98
    shell = (uvc[:, 0] > 0.5) & (uvc[:, 0] < 1.0) & (np.abs(ubc) < WRAP_U) & (scc < s_wrap)                    # the open back of the shell: glossy black glitter gel coat
    def part(mask):
        used = np.unique(F[mask]); re = np.full(len(V), -1); re[used] = np.arange(len(used)); return V[used], re[F[mask]]
    k.add_mesh('BUCKET_SEAT', *part(~shell), k.add_material(simple_material('gh_bucket_seat', 'gh_bucket_seat.png', 0.42, 0.50, 0.04, 6.0)))
    m = simple_material('gh_seat_shell', 'gh_bucket_seat.png', 0.30, 0.40, 0.90, 90.0, shader='ksPerPixelReflection')
    for p_, v_ in (('fresnelC', 0.06), ('fresnelEXP', 3.0), ('isAdditive', 0.0), ('fresnelMaxLevel', 0.40)): m.set(p_, v_)
    k.add_mesh('BUCKET_SEAT_SHELL', *part(shell), k.add_material(m))
    # harness: shoulder straps down the backrest to the buckle, back through the slots to the rear floor; lap belt across the cushion
    straps = []
    for sg in (-1, 1):
        idx = range(i_lap, i_sh + 1); fr = [(i - i_lap) / (i_sh - i_lap) for i in idx]
        fe = [seat_point(q, nrm, i, sg * (0.10 + 0.30 * f_), lift=0.012) for i, f_ in zip(idx, fr)]
        straps.append(ribbon(fe, (1, 0, 0), STRAP_W, v0=0.1))
        a = seat_point(q, nrm, i_sh, sg * 0.40, lift=0.004, back=True)
        straps.append(ribbon([fe[-1], (fe[-1] + a) / 2, a], (1, 0, 0), STRAP_W, v0=0.5))                         # through the slot
        b_ = np.array([SEAT_X + sg * 0.11, pan_height(np.array([SEAT_X]), np.array([-0.99]))[0] + 0.006, -0.99])
        f_ = np.linspace(0, 1, 12)[:, None]; straps.append(ribbon(a + (b_ - a) * f_ - np.array([0, 0.035, 0]) * np.sin(np.pi * f_), (1, 0, 0), STRAP_W, v0=0.55))
        straps.append(box(b_ - (0.022, 0.004, 0.022), b_ + (0.022, 0.010, 0.022), (0.5, 0.02)))                 # anchor plate bolted to the pan
    tg = np.array([0, t[i_lap, 1], t[i_lap, 0]])
    straps.append(ribbon([seat_point(q, nrm, i_lap, u, lift=0.014) for u in np.linspace(-1, 1, 17)], tg, 0.050, v0=0.6))
    c = seat_point(q, nrm, i_lap, 0.0, lift=0.016); straps.append(box(c - (0.038, 0.0, 0.032), c + (0.038, 0.020, 0.032), (0.5, 0.02)))   # buckle
    k.set_texture('gh_harness.png', png_bytes(harness_image()))
    k.add_mesh('HARNESS', *join(*straps), k.add_material(simple_material('gh_harness', 'gh_harness.png', 0.42, 0.45, 0.10, 12.0)), shadows=0)
    k.add_mesh('BUCKET_SEAT_MOUNT', *box((SEAT_X - 0.20, FLOOR_Y, -0.30), (SEAT_X + 0.20, 0.285, 0.09)), k.mat_id('plastic_black1'))
    return {'seat_tris_removed': gone, 'slot_cells_cut': int(cut.sum())}

# ---- manual shifter with the cloth bag as its boot
SHIFT_XZ = (-0.02, 0.215); SHIFT_Y = 0.333      # where the lever comes out of the console; height of the console there

def lathe(prof, v0, v1, seg=18, urep=1.0, tilt=0.0, planar=None, u0=0.0):
    """rings of radius r at height h around the shifter axis; r may be a function of the angle.
    planar=(u, v, scale): map the texture straight down onto the part (image up = front of the car) instead of round it"""
    th = np.linspace(0, 2 * np.pi, seg + 1); P = []; R = []
    for r, h in prof:
        rr = r(th) if callable(r) else np.full(len(th), r); R.append(rr)
        P.append(np.c_[SHIFT_XZ[0] + rr * np.cos(th), np.full(len(th), h), SHIFT_XZ[1] + rr * np.sin(th) - (h - SHIFT_Y) * np.tan(tilt)])
    P = np.array(P); R = np.array(R); hs = np.array([h for _, h in prof]); f = (hs - hs[0]) / (hs[-1] - hs[0])
    if planar: UV = np.stack([planar[0] - R * np.cos(th) * planar[2], planar[1] - R * np.sin(th) * planar[2]], axis=-1)
    else: UV = np.stack([np.broadcast_to((u0 + th / (2 * np.pi) * urep)[None], P.shape[:2]), np.broadcast_to((v0 + (v1 - v0) * f)[:, None], P.shape[:2])], axis=-1)
    return surf(P, UV, 'out')

KNOB_BASE, KNOB_SEAM, KNOB_TOP = 0.130, 0.208, 0.2220       # heights above the console: bottom of the knob, its stitched ring, its crown
KNOB_R = 0.0250                                             # radius at the stitched ring

def shifter_image(size=1024, seed=51):
    """top half: the leather shift knob (TRD knob as fitted to the AE86) - its side with the badge and the baseball stitching, the
    crown with the shift pattern, a plain strip for the lever; bottom half: the green cloth bag (gathered, cream-stitched hem at the top)"""
    S = size; h = S // 2; k_ = S // 512; img = np.empty((S, S, 3))
    img[:h] = (30, 30, 32); img[:h] *= (0.78 + 0.44 * noise((h, S), 1.2 * k_, seed))[..., None]                  # leather grain
    yy, xx = np.mgrid[0:h, 0:S].astype(np.float64)
    img[h:] = (64, 148, 104); folds = 0.5 + 0.5 * np.sin(xx / S * 2 * np.pi * 9 + 3 * noise((h, S), 12 * k_, seed + 1) * 6)
    img[h:] *= (0.62 + 0.38 * folds * (0.6 + 0.4 * noise((h, S), 5 * k_, seed + 2)))[..., None]
    gath = (yy[:110] >= 0) * (0.74 + 0.26 * (0.5 + 0.5 * np.sin(xx[:110] / S * 2 * np.pi * 18 + 2 * noise((110, S), 4, seed + 6))))
    img[h + 8:h + 118] *= gath[..., None]                                                                        # the gathered hem
    img[330:430, 400:500] = (206, 196, 168); img[330:430, 400:500] *= (0.8 + 0.2 * (np.mgrid[0:100, 0:100].sum(0) // 6 % 2))[..., None]   # twisted cord
    img[330:430, 600:700] = (150, 152, 156)                                                                      # eyelets
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB'); d = ImageDraw.Draw(im)
    for x0 in range(0, S, 10 * k_):                                                                              # hem of the bag
        d.line([(x0, h + 12 * k_), (x0 + 6 * k_, h + 12 * k_)], fill=(214, 206, 170), width=2 * k_)
        d.line([(x0 + 2 * k_, h + 20 * k_), (x0 + 8 * k_, h + 20 * k_)], fill=(214, 206, 170), width=2 * k_)
    d.rectangle((0, h, S, h + 5 * k_), fill=(30, 74, 52))
    # side of the knob: rows 0..300 run from the stitched ring down to its base, the columns once round it
    thread = (78, 78, 82); sy = 16
    d.line([(0, sy), (S, sy)], fill=(12, 12, 13), width=3)                                                       # seam under the crown, baseball stitched
    for x0 in range(0, S, 22):
        d.line([(x0, sy - 9), (x0 + 9, sy)], fill=thread, width=3); d.line([(x0 + 11, sy + 9), (x0 + 20, sy)], fill=thread, width=3)
    sx = S // 2
    d.line([(sx, sy), (sx, 300)], fill=(12, 12, 13), width=4)                                                    # seam down its length
    for y0 in range(sy + 8, 296, 9):
        d.line([(sx - 20, y0 - 3), (sx, y0)], fill=thread, width=2); d.line([(sx + 20, y0 + 1), (sx, y0 + 4)], fill=thread, width=2)
    bw, bh = 600, 92; bd = Image.new('RGB', (bw, bh), (150, 152, 156)); e = ImageDraw.Draw(bd)                  # badge, drawn lying down
    e.rounded_rectangle((6, 6, bw - 7, bh - 7), radius=12, fill=(10, 10, 11))
    e.text((bw // 2, 8), 'TOYOTA MOTOR SPORTS', font=font(['arialbd.ttf'], 30), fill=(214, 30, 36), anchor='mt')
    for i, col in enumerate([(236, 236, 238), (236, 236, 238), (214, 30, 36)]):
        x1 = 24 + i * 52; e.polygon([(x1 + 24, 42), (x1 + 68, 42), (x1 + 44, bh - 10), (x1, bh - 10)], fill=col)
    f2 = font(['arialbi.ttf', 'arialbd.ttf'], 54); e.text((208, 34), 'TRD', font=f2, fill=(240, 240, 242)); e.text((210, 34), 'TRD', font=f2, fill=(240, 240, 242))
    # it sits upright on the driver's side of the knob, low down, reading from the bottom up
    im.paste(bd.rotate(90, expand=True).resize((104, 140), Image.LANCZOS), (S // 4 - 52, 146))
    im = im.resize((S, S))                                                                                       # (keeps everything above in one pass)
    # crown: the shift pattern pressed into the leather, 160 px square at (0, 310)
    d = ImageDraw.Draw(im); cx, cy = 80, 390; emb = (64, 64, 68); f3 = font(['arialbd.ttf'], 26)
    d.line([(cx - 32, cy), (cx + 32, cy)], fill=emb, width=4)
    for i, (a, b) in enumerate(['12', '34', '5R']):
        x1 = cx - 32 + 32 * i; d.line([(x1, cy - 20), (x1, cy + 20)], fill=emb, width=4)
        d.text((x1, cy - 22), a, font=f3, fill=emb, anchor='mb'); d.text((x1, cy + 22), b, font=f3, fill=emb, anchor='mt')
    # the bag: cream overlocked seams, and the gold script low down on the driver's side (the bag's texture runs clockwise, urep -1, so it reads the right way round)
    d = ImageDraw.Draw(im)
    for sx_ in (360, 870):
        for y0 in range(h + 30, S - 8, 9): d.line([(sx_ - 3, y0), (sx_ + 3, y0 + 5)], fill=(214, 206, 170), width=2)
    tx = Image.new('L', (330, 110), 0); ImageDraw.Draw(tx).text((165, 55), 'Crown Royal', font=font(['segoesc.ttf', 'georgiai.ttf', 'ariali.ttf'], 46), fill=255, anchor='mm')
    tx = tx.rotate(14, expand=True, resample=Image.BICUBIC)
    sh = tx.filter(ImageFilter.MaxFilter(3)); pos = (114 - tx.width // 2, 835 - tx.height // 2)
    im.paste((60, 70, 40), pos, sh); im.paste((206, 168, 82), pos, tx)
    return im

def tube(C, r, uvc, seg=8):
    """a round cord of radius r along the points C (n, 3); the texture is a small patch round uvc"""
    C = np.asarray(C, float); T = unit(np.gradient(C, axis=0)); ref = C.mean(0) - np.array([SHIFT_XZ[0], C[:, 1].mean(), SHIFT_XZ[1]]); ref[1] = 0
    N = unit(np.cross(T, unit(ref[None]))); B = np.cross(T, N); th = np.linspace(0, 2 * np.pi, seg + 1)
    P = C[:, None] + r * (np.cos(th)[None, :, None] * N[:, None] + np.sin(th)[None, :, None] * B[:, None])
    UV = np.stack([np.broadcast_to(uvc[0] + 0.03 * th[None] / (2 * np.pi), P.shape[:2]), np.broadcast_to(uvc[1] + 0.03 * np.linspace(0, 1, len(C))[:, None], P.shape[:2])], -1)
    return surf(P, UV, (P - C[:, None]).reshape(-1, 3))

def shifter(k):
    x0, z0 = SHIFT_XZ; tilt = np.radians(7.0); Y = SHIFT_Y; HB = 0.100
    phase = noise((1, 97), 3, 53)[0] * 2.0
    def bag(t_):
        base = lambda th: 0.060 / np.maximum(np.abs(np.cos(th)), np.abs(np.sin(th))) ** 0.75                    # rounded square at the console opening
        m = ss(0.05, 0.80, t_) ** 0.9; neck = 0.029 + 0.009 * max(0.0, (t_ - 0.80) / 0.20)                      # drawn in by the string, the ruffled hem opening out above it
        bulge = 0.007 * np.sin(np.pi * min(t_ / 0.6, 1.0))
        def r(th):
            folds = 0.006 * np.sin(7 * th + 4 * t_) * np.sin(np.pi * min(t_ * 1.15, 1.0))
            gathers = 0.0022 * ss(0.68, 0.84, t_) * np.sin(18 * th + np.interp(th, np.linspace(0, 2 * np.pi, 97), phase))
            return base(th) * (1 - m) + neck * m + bulge * (1 - m) + folds + gathers
        return r
    def bag_pt(t_, th, out=0.0):
        h = Y - 0.004 + HB * t_; r = bag(t_)(np.array([th]))[0] + out
        return np.array([x0 + r * np.cos(th), h, z0 + r * np.sin(th) - (h - Y) * np.tan(tilt * 0.5)])
    ts = np.linspace(0, 1, 21); B, Sm, T = Y + KNOB_BASE, Y + KNOB_SEAM, Y + KNOB_TOP
    # the knob: slim at the lever, swelling to a wide flat-topped head; the side is one leather panel, the crown another
    side = [(0.0080, B - 0.0005), (0.0135, B), (0.0150, B + 0.004), (0.0152, B + 0.018), (0.0162, B + 0.032), (0.0183, B + 0.046), (0.0212, B + 0.058),
            (0.0236, B + 0.068), (0.0248, B + 0.075), (KNOB_R, Sm)]
    crown = [(KNOB_R, Sm), (0.0244, Sm + 0.0045), (0.0222, Sm + 0.0085), (0.0180, Sm + 0.0115), (0.0120, Sm + 0.0132), (0.0060, Sm + 0.0139), (0.0004, T)]
    parts = [two_sided(*lathe([(bag(t_), Y - 0.004 + HB * t_) for t_ in ts], 0.99, 0.51, seg=144, urep=-1.0, tilt=tilt * 0.5)),   # seen from inside too, through the open hem
             lathe([(0.0110, Y + 0.045), (0.0125, Y + 0.080), (0.0118, Y + 0.105), (0.0108, B + 0.004)], 0.475, 0.495, seg=14, tilt=tilt),
             lathe(side, 300 / 1024, 16 / 1024, seg=24, urep=-1.0, u0=0.25, tilt=tilt),                          # badge towards the driver (+x), seam to the rear; urep -1 so it is not mirrored
             lathe(crown, 0, 0, seg=24, tilt=tilt, planar=(80 / 1024, 390 / 1024, 72 / 1024 / KNOB_R))]
    # the drawstring: two eyelets at the front of the hem, a loop of cream cord hanging between them and a knotted end from each
    cord, ring = (450 / 1024, 380 / 1024), (650 / 1024, 380 / 1024); te = 0.86; the = [np.pi / 2 + 0.24, np.pi / 2 - 0.24]
    for th in the:
        c = bag_pt(te, th, 0.0015); n_ = np.array([np.cos(th), 0, np.sin(th)]); e1 = np.array([-np.sin(th), 0, np.cos(th)]); e2 = np.array([0, 1.0, 0])
        a = np.linspace(0, 2 * np.pi, 13); parts.append(tube(c + 0.0035 * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2), 0.0011, ring, seg=6))
    s = np.linspace(0, 1, 15); thl = the[0] + (the[1] - the[0]) * s; tl = te - 0.42 * np.sin(np.pi * s)
    parts.append(tube([bag_pt(t_, th, 0.004 + 0.003 * np.sin(np.pi * f)) for t_, th, f in zip(tl, thl, s)], 0.0018, cord))
    for th, t_end, sw in ((the[0], 0.14, 0.30), (the[1], 0.30, -0.22)):
        s = np.linspace(0, 1, 10); pts = [bag_pt(te - (te - t_end) * f, th + sw * f, 0.005 + 0.004 * f) for f in s]
        parts.append(tube(pts, 0.0018, cord)); e = pts[-1]; dvec = unit((pts[-1] - pts[-2])[None])[0]
        parts.append(tube([e - dvec * 0.002, e + dvec * 0.004, e + dvec * 0.009], 0.0034, cord))               # knot, frayed end below it
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

# ---- passenger seat: the stock grey velour seat
PASS_X = -0.385
PASS_CUSHION = [(0.135, 0.310, 0.255, 0.000), (0.130, 0.385, 0.270, 0.010), (0.070, 0.415, 0.280, 0.022), (-0.100, 0.405, 0.285, 0.030),
                (-0.260, 0.390, 0.285, 0.030), (-0.360, 0.385, 0.280, 0.020)]
PASS_BACK = [(-0.335, 0.360, 0.262, 0.020), (-0.325, 0.450, 0.275, 0.045), (-0.345, 0.570, 0.275, 0.050), (-0.375, 0.680, 0.270, 0.040),
             (-0.400, 0.760, 0.250, 0.020), (-0.414, 0.805, 0.205, 0.004), (-0.420, 0.822, 0.140, 0.000)]
PASS_HEAD = [(-0.436, 0.842, 0.090, 0.000), (-0.430, 0.868, 0.132, 0.000), (-0.436, 0.950, 0.135, 0.000), (-0.446, 1.000, 0.122, 0.000),
             (-0.452, 1.018, 0.070, 0.000)]
VELOUR_RGB = (106, 110, 117)

def loft(path, xc, v0, v1, thick, stations=40, nu=27):
    """a cushion: `path` rows are z, y of the centre line of its face, half width, bolster height; thick = (at the edge, extra in the middle)"""
    from scipy.interpolate import PchipInterpolator
    path = np.asarray(path, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]; f = PchipInterpolator(d / d[-1], path)
    S = np.linspace(0, 1, stations); q = f(S); t = unit(f.derivative()(S)[:, :2]); nrm = np.c_[t[:, 1], -t[:, 0]]
    uf = np.linspace(-1, 1, nu); ub = np.linspace(1, -1, nu // 2 + 1)
    def pt(i, u, back):
        z, y, w, h = q[i]; off = -(thick[0] + thick[1] * (1 - u * u)) if back else h * ss(0.45, 0.95, abs(u)) + 0.012 * (1 - u ** 4)
        return [xc + w * u, y + nrm[i, 1] * off, z + nrm[i, 0] * off]
    loop = [(u, False) for u in uf] + [(u, True) for u in ub] + [(-1.0, False)]
    U = np.r_[0.02 + 0.46 * (uf + 1) / 2, 0.52 + 0.46 * (1 - ub) / 2, 1.02]
    P = np.array([[pt(i, u, bk) for u, bk in loop] for i in range(stations)])
    UV = np.stack([np.broadcast_to(U[None], P.shape[:2]), np.broadcast_to((v0 + (v1 - v0) * S)[:, None], P.shape[:2])], axis=-1)
    caps = Builder()
    for i, sgn in ((0, -1.0), (stations - 1, 1.0)):
        caps.poly(P[i, :-1], (0, sgn * t[i, 1], sgn * t[i, 0]), [(0.75, v0 + (v1 - v0) * S[i])] * (P.shape[1] - 1))
    return join(surf(P, UV, 'out'), caps.done())

def velour_image(size=1024, seed=81):
    """left half: the faces of cushion (top third), backrest (middle) and headrest (bottom quarter); right half: their backs and sides"""
    S = size; yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
    img = np.empty((S, S, 3)); img[:] = VELOUR_RGB
    img *= (0.88 + 0.16 * noise((S, S), 40, seed) + 0.10 * noise((S, S), 5, seed + 1))[..., None]             # the nap lying this way and that
    img *= (0.94 + 0.12 * noise((S, S), 0.7, seed + 2))[..., None]
    def over(rgb, a): img[:] = img * (1 - a[..., None]) + np.array(rgb, float) * a[..., None]
    def mask(fn, blur=0.0):
        im = Image.new('L', (S, S), 0); fn(ImageDraw.Draw(im))
        return np.asarray(im.filter(ImageFilter.GaussianBlur(blur)) if blur else im, np.float64) / 255
    px = lambda u: (0.02 + 0.46 * (u + 1) / 2) * S
    panels = [(px(-0.60), 0.035 * S, px(0.60), 0.300 * S), (px(-0.60), 0.385 * S, px(0.60), 0.690 * S)]       # centre inserts of cushion and backrest
    ins = mask(lambda d: [d.rounded_rectangle(b_, radius=34, fill=255) for b_ in panels], 1.0)
    over((122, 126, 133), 0.85 * ins); img *= (1 - 0.13 * ins * (0.5 + 0.5 * np.sin(xx * 2 * np.pi / 7.0)))[..., None]   # ribbed cloth, ribs running fore and aft
    over((66, 68, 74), 0.75 * mask(lambda d: [d.rounded_rectangle(b_, radius=34, outline=255, width=5) for b_ in panels], 1.3))
    over((66, 68, 74), 0.55 * mask(lambda d: [d.line([(px(-0.60), r_ * S), (px(0.60), r_ * S)], fill=255, width=3) for r_ in (0.17, 0.49, 0.59)], 1.2))
    img[:, S // 2:] *= 0.93
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGB')

def remove_mesh(k, name):
    n, parent, _ = k.find(name); parent.children.remove(n); return len(n.F)

def passenger_seat(k):
    gone = remove_mesh(k, 'seatleather1') + remove_mesh(k, 'seatleather2')                # by now only the passenger seat is left in them
    parts = [loft(PASS_CUSHION, PASS_X, 0.01, 0.32, (0.075, 0.035)), loft(PASS_BACK, PASS_X, 0.35, 0.73, (0.060, 0.045)),
             loft(PASS_HEAD, PASS_X, 0.77, 0.99, (0.050, 0.035), stations=24, nu=23)]
    k.set_texture('gh_velour.png', png_bytes(velour_image()))
    k.add_mesh('PASSENGER_SEAT', *join(*parts), k.add_material(simple_material('gh_velour', 'gh_velour.png', 0.42, 0.48, 0.03, 5.0)))
    posts = [box((PASS_X + sx * 0.055 - 0.006, 0.79, -0.462), (PASS_X + sx * 0.055 + 0.006, 0.86, -0.450)) for sx in (-1, 1)]
    posts.append(box((PASS_X - 0.24, FLOOR_Y, -0.33), (PASS_X + 0.24, 0.30, 0.11)))                             # seat base down to the floor
    k.add_mesh('PASSENGER_SEAT_BASE', *join(*posts), k.mat_id('plastic_black1'))
    return {'stock_seat_tris_removed': gone}

# ---- interior colours: charcoal upper dash and door cards, mid-grey lower dash and door inserts, light grey headliner and pillars
TRIM_CHARCOAL = (66, 68, 73); TRIM_GREY = (104, 107, 114); HEADLINER_GREY = (150, 153, 159); DOOR_CLOTH = (112, 116, 123)
HANDLE_RGB = (38, 39, 42)         # inside door handles: black plastic, not chrome

def grain_texture(rgb, seed, size=256, amount=0.10):
    a = np.empty((size, size, 3)); a[:] = rgb; a *= (1 - amount + 2 * amount * noise((size, size), 0.8, seed))[..., None]
    return png_bytes(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGB'))

def retone(k, base, name, rgb, seed, ambient=None, diffuse=None):
    """a copy of material `base` whose detail texture (the one that sets its colour) is a plain grain of the given colour"""
    m = k.materials[k.mat_id(base)].copy(name); k.set_texture(name + '.png', grain_texture(rgb, seed))
    for t in m.tex:
        if t[0] == 'txDetail': t[2] = name + '.png'
    if ambient is not None: m.set('ksAmbient', ambient); m.set('ksDiffuse', diffuse)
    return k.add_material(m)

def interior_colors(k):
    charcoal = retone(k, 'dashleather', 'gh_trim_charcoal', TRIM_CHARCOAL, 91); grey = retone(k, 'dashleather', 'gh_trim_grey', TRIM_GREY, 92)
    n, parent, M = k.find('dashboard'); w = world_pos(n, M); nc, lab = components(w, n.F)
    wide = np.array([np.ptp(w[lab == c][:, 0]) > 1.0 for c in range(nc)])                 # the dash itself spans the car; door cards and trims do not
    # the dash is built from separate panels (they do not share vertices); whole panels change colour, so the boundary follows their edges
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.r_[n.F[:, [0, 1]], n.F[:, [1, 2]]]; ni, isl = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(w),) * 2), directed=False)
    grey_isl = np.zeros(ni, bool); flank = np.zeros(len(w), bool)
    for i in np.unique(isl[n.F[wide[lab[n.F[:, 0]]]][:, 0]]):
        p = w[isl == i]; lo, hi = p.min(0), p.max(0)
        glove_box = hi[0] < -0.22 and hi[1] < 0.70 and lo[2] > 0.40; knee_panel = lo[0] > 0.12 and hi[1] < 0.63 and lo[2] > 0.44
        grey_isl[i] = glove_box or knee_panel
        if lo[0] > -0.23 and hi[0] < -0.13 and hi[1] < 0.70 and lo[2] > 0.40: flank |= isl == i
    # the flank between the centre stack and the glove box is a handful of loose triangles, each shaded on its own: shade them as one surface
    tri = flank[n.F[:, 0]]; nw = unit(weld_normals(n.V[:, :3].astype(np.float64), n.F[tri]))
    nw = np.where((nw * n.V[:, 3:6]).sum(1, keepdims=True) < 0, -nw, nw); n.V[flank, 3:6] = nw[flank]
    lower = grey_isl[isl[n.F[:, 0]]]
    Vt, Ft = split_mesh(n, lower); n.mat = charcoal
    m2 = Node(2, 'dashboard_lower'); m2.V, m2.F, m2.mat, m2.flags, m2.layer, m2.lod = Vt, Ft, grey, n.flags, n.layer, n.lod; update_bounds(m2); parent.children.append(m2)
    k.find('carpetroof')[0].mat = retone(k, 'carpet', 'gh_headliner', HEADLINER_GREY, 93, ambient=0.40, diffuse=0.36)
    k.find('carts.001')[0].mat = retone(k, 'seatleather2', 'gh_door_cloth', DOOR_CLOTH, 94)
    k.set_texture('gh_handle_plastic.png', grain_texture(HANDLE_RGB, 95, amount=0.06))
    k.find('chrome1')[0].mat = k.add_material(simple_material('gh_handle_plastic', 'gh_handle_plastic.png', 0.40, 0.45, 0.20, 25.0))
    return {'lower_dash_tris': int(lower.sum())}

# ------------------------------------------------------------------ 16b. highway grime on the front end: dead bugs and road film (Reference Photos/front-end)
BUG_BOX = (-0.88, 0.88, 0.17, 0.67)              # x0, x1, y0, y1 of the front-on texture
BUG_SRC = (('f_bumper', 1.0, 1), ('front_bumper_SUB1', 0.9, 1), ('front_bumper_SUB2', 0.5, 0), ('BUMPER_LOWER', 0.9, 1), ('BUMPER_PANEL', 0.75, 1), ('BUMPER_BEAM', 0.5, 1),
           ('grille_uprights', 0.35, 0), ('radiator_grill', 0.30, 0), ('bonnet', 0.55, 0), ('headlights_part001', 0.25, 0))   # mesh, how many bugs it catches, road film on it
_bugs = {}
BUG_CELL = 0.002

def bug_image(Dn, film, W=4096, H=1152, seed=111):
    """road film (a dusty brown haze, heavier low down) with dead bugs on it: dark specks, cream and yellow smears, the odd bigger splat"""
    x0, x1, y0, y1 = BUG_BOX; gh, gw = Dn.shape; rng = np.random.default_rng(seed); sx, sy = W / (x1 - x0), H / (y1 - y0)
    from scipy.ndimage import zoom
    up = lambda a: np.clip(zoom(a, (H / gh, W / gw), order=1), 0, None)
    f = up(film) * (0.55 + 0.9 * noise((H, W), 18, seed + 1)) * (0.75 + 0.5 * noise((H, W), 3, seed + 2))
    img = np.zeros((H, W, 4)); img[..., :3] = np.array([118, 108, 92]) * (0.85 + 0.3 * noise((H, W), 40, seed + 3))[..., None]; img[..., 3] = np.clip(f, 0, 0.55) * 255
    im = Image.fromarray(img.astype(np.uint8), 'RGBA'); d = ImageDraw.Draw(im)
    p = Dn.ravel() / Dn.sum(); n = int(9000 * Dn.sum() * BUG_CELL ** 2); i = rng.choice(Dn.size, n, p=p); r, c = np.divmod(i, gw)
    xs = x0 + (c + rng.random(n)) * BUG_CELL; ys = y1 - (r + rng.random(n)) * BUG_CELL
    P = lambda x, y: ((x - x0) * sx, (y1 - y) * sy)
    def blob(x, y, rad, col, a=255):
        cx, cy = P(x, y); d.ellipse((cx - rad * sx, cy - rad * sy, cx + rad * sx, cy + rad * sy), fill=tuple(int(v) for v in col) + (a,))
    tone = lambda col: np.clip(np.array(col) * rng.uniform(0.8, 1.15), 0, 255)
    for x, y in zip(xs, ys):
        t = rng.random()
        if t < 0.30: blob(x, y, rng.uniform(0.0003, 0.0009), tone((38, 32, 26)))                                 # dark specks
        elif t < 0.62: blob(x, y, rng.uniform(0.0003, 0.0008), tone((212, 204, 178)))                           # pale specks
        elif t < 0.96:                                                                                          # guts: a cream or yellow smear, swept outwards by the air
            col = tone((206, 196, 150) if rng.random() < 0.6 else (176, 164, 92)); rad = rng.uniform(0.0007, 0.0020)
            dx = np.sign(x) * rng.uniform(0.3, 1.0) + rng.normal(0, 0.3); dy = rng.uniform(0.2, 1.0); L = rng.uniform(1.0, 3.5) * rad; nn = np.hypot(dx, dy)
            for s in np.linspace(0, 1, 5): blob(x + dx / nn * L * s, y + dy / nn * L * s, rad * (1 - 0.6 * s), col, int(255 - 70 * s))
            if rng.random() < 0.4: blob(x, y, rad * 0.45, tone((52, 44, 32)))
        else:                                                                                                   # a bigger splat with droplets thrown round it
            rad = rng.uniform(0.0012, 0.0026); col = tone((190, 176, 120))
            for _ in range(rng.integers(3, 7)):                                                                 # an irregular lump, not a ring
                blob(x + rng.normal(0, rad * 0.6), y + rng.normal(0, rad * 0.6), rad * rng.uniform(0.3, 0.7), col)
            for _ in range(rng.integers(1, 4)):
                a = rng.uniform(0, 2 * np.pi); q = rng.uniform(1.4, 2.6) * rad; blob(x + q * np.cos(a), y + q * np.sin(a), rad * rng.uniform(0.1, 0.25), col)
            if rng.random() < 0.5: blob(x + rng.normal(0, rad * 0.4), y + rng.normal(0, rad * 0.4), rad * rng.uniform(0.2, 0.4), tone((60, 48, 34)))
    return im, n

def bug_splatter(k):
    x0, x1, y0, y1 = BUG_BOX; Vs = []; Fs = []; off = 0; rng = np.random.default_rng(109); P = []; Wt = []
    gw, gh = int(round((x1 - x0) / BUG_CELL)), int(round((y1 - y0) / BUG_CELL))
    for name, wt, fw in BUG_SRC:
        n, _, M = k.find(name); w = world_pos(n, M); nr = unit(n.V[:, 3:6].astype(np.float64) @ M[:3, :3]); c = w[n.F].mean(1); fn = nr[n.F].mean(1)
        sel = (fn[:, 2] > 0.20) & (c[:, 2] > 1.80) & (c[:, 1] > y0) & (c[:, 1] < y1)
        if not sel.any(): continue
        F = n.F[sel]; used = np.unique(F); re = np.full(len(w), -1); re[used] = np.arange(len(used))
        V = np.zeros((len(used), 11)); V[:, :3] = w[used] + nr[used] * 0.0012; V[:, 3:6] = nr[used]
        V[:, 6] = (w[used, 0] - x0) / (x1 - x0); V[:, 7] = (y1 - w[used, 1]) / (y1 - y0); V[:, 8:11] = [1, 0, 0]
        Vs.append(V); Fs.append(re[F] + off); off += len(V)
        t = w[F]; area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1); cnt = max(int(area.sum() / 2 / BUG_CELL ** 2 * 6), 1)
        i = rng.choice(len(F), cnt, p=area / area.sum()); a, b = rng.random(cnt), rng.random(cnt); fl = a + b > 1; a[fl] = 1 - a[fl]; b[fl] = 1 - b[fl]
        pts = t[i, 0] + (t[i, 1] - t[i, 0]) * a[:, None] + (t[i, 2] - t[i, 0]) * b[:, None]
        P.append(pts); Wt.append(np.c_[np.full(cnt, wt), np.clip(fn[sel][i, 2], 0, 1), np.full(cnt, fw)])
    V = np.vstack(Vs); F = np.vstack(Fs); P = np.vstack(P); Wt = np.vstack(Wt)
    ix = np.clip(((P[:, 0] - x0) / BUG_CELL).astype(int), 0, gw - 1); iy = np.clip(((y1 - P[:, 1]) / BUG_CELL).astype(int), 0, gh - 1)
    o = np.argsort(P[:, 2]); G = np.zeros((gh, gw, 3)); G[iy[o], ix[o]] = Wt[o]                                   # the frontmost surface wins
    G = np.stack([gaussian_filter(G[..., j], 1.0) for j in range(3)], -1)
    Dn = G[..., 0] * G[..., 1] ** 1.5                                                                             # surfaces square to the wind catch the most
    low = np.clip((0.50 - (y1 - (np.arange(gh) + 0.5) * BUG_CELL)) / 0.30, 0, 1)[:, None]                        # spray from the road dirties the lower bumper most
    film = gaussian_filter(G[..., 2] * ss(0.25, 0.60, G[..., 1]), 4) * (0.12 + 0.30 * low)          # fades out where the panel turns away, so no hard edge
    im, n = bug_image(Dn, film)
    _bugs['img'] = im; k.set_texture('gh_bugs.png', png_bytes(Image.new('RGBA', (8, 8), (118, 108, 92, 0))))   # clear here: only the pine skin is dirty
    m = simple_material('gh_bugs', 'gh_bugs.png', 0.45, 0.50, 0.10, 12.0, shader='ksPerPixelAT'); m.blend = 1
    k.add_mesh('BUG_SPLATTER', V, fix_winding(V, F), k.add_material(m), transparent=1, shadows=0)
    return {'tris': len(F), 'bugs': n}

# ------------------------------------------------------------------ 17. skins: stickers / no stickers / pine needles and sap
# A skin can only swap textures, so everything a skin switches lives in a texture of its own:
#   stickers       gh_sticker.png, gh_number_panel.png (alpha tested: a clear texture hides them)
#   wheels         MKII_Grey.png / MKIII_Grey.png (colour), SSR_MKII_Map.png / SSR_MKIII_Map.png (shine)
#   pine debris    gh_debris.png on DEBRIS_LAYER, a shell 2.5 mm above the top of the car; clear in the model itself
SKINS = config.PROJECT / 'gentlehands_toyota_cressida_mx83' / 'skins'; BASE_SKIN = 'silver_176'
DEBRIS_BOX = (-0.96, 0.96, -2.50, 2.20)         # x0, x1, z0, z1 covered by the top-down debris texture
DEBRIS_SRC = (('bonnet', 0.60), ('roof', 0.60), ('trunk', 0.60), ('r_fenders', 0.60), ('f_fenders', 0.60), ('rear_doors', 0.60), ('lf_door', 0.60),
              ('rf_door', 0.60), ('wiper_part001', 0.45), ('window_ext', 0.60), ('f_bumper', 0.60), ('front_bumper_SUB1', 0.60))      # mesh, how far its normal must point up
DEBRIS_CELL = 0.004
_debris = {}

def debris_layer(k):
    x0, x1, z0, z1 = DEBRIS_BOX; Vs = []; Fs = []; off = 0; rng = np.random.default_rng(101); P = []; K = []
    for name, thr in DEBRIS_SRC:
        n, _, M = k.find(name); w = world_pos(n, M); nr = unit(n.V[:, 3:6].astype(np.float64) @ M[:3, :3]); c = w[n.F].mean(1)
        sel = (nr[n.F].mean(1)[:, 1] > thr) & (c[:, 1] > (0.38 if 'bumper' in name else 0.55))
        F = n.F[sel]; used = np.unique(F); re = np.full(len(w), -1); re[used] = np.arange(len(used))
        V = np.zeros((len(used), 11)); V[:, :3] = w[used] + nr[used] * 0.0025; V[:, 3:6] = nr[used]
        V[:, 6] = (w[used, 0] - x0) / (x1 - x0); V[:, 7] = (w[used, 2] - z0) / (z1 - z0); V[:, 8:11] = [1, 0, 0]
        Vs.append(V); Fs.append(re[F] + off); off += len(V)
        # points scattered over the surface: what lies where, for laying out the debris
        t = w[F]; area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1); cnt = max(int(area.sum() / 2 / DEBRIS_CELL ** 2 * 6), 1)
        i = rng.choice(len(F), cnt, p=area / area.sum()); a, b = rng.random(cnt), rng.random(cnt); fl = a + b > 1; a[fl] = 1 - a[fl]; b[fl] = 1 - b[fl]
        P.append(t[i, 0] + (t[i, 1] - t[i, 0]) * a[:, None] + (t[i, 2] - t[i, 0]) * b[:, None]); K.append(np.full(cnt, 2 if name == 'window_ext' else 3 if name == 'wiper_part001' else 1))
    V = np.vstack(Vs); F = np.vstack(Fs); P = np.vstack(P); K = np.concatenate(K)
    gw, gh = int(round((x1 - x0) / DEBRIS_CELL)), int(round((z1 - z0) / DEBRIS_CELL))
    ix = np.clip(((P[:, 0] - x0) / DEBRIS_CELL).astype(int), 0, gw - 1); iz = np.clip(((P[:, 2] - z0) / DEBRIS_CELL).astype(int), 0, gh - 1)
    o = np.argsort(P[:, 1]); H = np.zeros((gh, gw)); kind = np.zeros((gh, gw), np.int64); H[iz[o], ix[o]] = P[o, 1]; kind[iz[o], ix[o]] = K[o]      # the topmost surface wins
    _debris['H'] = H; _debris['kind'] = kind
    k.set_texture('gh_debris.png', png_bytes(Image.new('RGBA', (8, 8), (150, 100, 50, 0))))
    m = simple_material('gh_debris', 'gh_debris.png', 0.36, 0.40, 0.03, 8.0, shader='ksPerPixelAT', alpha_ref=0.25); m.alpha_tested = 1
    k.add_mesh('DEBRIS_LAYER', V, fix_winding(V, F), k.add_material(m), shadows=0)
    return {'tris': len(F), 'covered_m2': round(float((kind > 0).sum() * DEBRIS_CELL ** 2), 2)}

def debris_density(H, kind):
    """needles per square metre: a few strays on the panels, fewer on the glass they slide off, and a thick mat where they pool - in the
    cowl under the windscreen, and a little along the foot of each window"""
    cov = kind > 0; gh = len(H); zrow = DEBRIS_BOX[2] + (np.arange(gh) + 0.5) * DEBRIS_CELL
    Hs = gaussian_filter(np.where(cov, H, 0.0), 6) / np.maximum(gaussian_filter(cov.astype(float), 6), 1e-3)
    D = np.where(kind == 1, 22.0, 0.0) + np.where(kind == 2, 6.0, 0.0)
    D += 60.0 * np.clip((Hs - H) / 0.006, 0, 1) * (kind == 1)                               # panel gaps, gutters
    # it gathers in the corners and thins out towards the middle, more on one side than the other, in clumps
    xc = DEBRIS_BOX[0] + (np.arange(H.shape[1]) + 0.5) * DEBRIS_CELL; lump = noise((1, H.shape[1]), 9, 107)[0]
    side = (0.10 + 1.25 * np.clip(np.abs(xc + 0.06) / 0.66, 0, 1) ** 1.8) * np.where(xc > 0, 0.74, 0.60) * (0.35 + 1.3 * lump)
    pool = (kind == 3).astype(float)                                                        # the cowl itself is all but hidden under the hood's rear edge,
    for rows, heap in ((zrow > -0.5, 1.0), (zrow < -0.5, 0.0)):                             # so the mat lies along the foot of the windscreen, round the wipers
        g = (kind == 2) & rows[:, None]; foot = np.where(g, H, np.inf).min(0); foot = np.where(np.isfinite(foot), foot, 0.0)
        lift = np.where(g, H - foot[None], 1.0)
        pool += heap * side[None] * np.exp(-(lift / (0.012 + 0.016 * np.minimum(side, 1.3))[None]) ** 2); D += np.where(g, 220.0 * np.exp(-lift / 0.012), 0.0)
    return D * cov, 15000.0 * pool * cov

def debris_image(W=2048, Hpx=4096, seed=103):
    H, kind = _debris['H'], _debris['kind']; gh, gw = H.shape; rng = np.random.default_rng(seed); x0, x1, z0, z1 = DEBRIS_BOX
    S = 2; sx, sz = W * S / (x1 - x0), Hpx * S / (z1 - z0)                                  # drawn at twice the size, then reduced
    im = Image.new('RGBA', (W * S, Hpx * S), (104, 84, 60, 0)); d = ImageDraw.Draw(im)
    def scatter(D, per_m2=1.0):
        count = int(per_m2 * D.sum() * DEBRIS_CELL ** 2)
        if count == 0: return np.zeros(0), np.zeros(0)
        i = rng.choice(D.size, count, p=D.ravel() / D.sum()); r, c = np.divmod(i, gw)
        return x0 + (c + rng.random(count)) * DEBRIS_CELL, z0 + (r + rng.random(count)) * DEBRIS_CELL
    px = lambda x, z: ((x - x0) * sx, (z - z0) * sz)
    def blob(x, z, r, col):
        cx, cy = px(x, z); d.ellipse((cx - r * sx, cy - r * sz, cx + r * sx, cy + r * sz), fill=col)
    def needle(x, z, L, col, wd):
        a = rng.uniform(0, 2 * np.pi); bend = rng.normal(0, 0.12); s = np.linspace(0, 1, 4)
        d.line([px(x + L * t * np.cos(a + bend * t), z + L * t * np.sin(a + bend * t)) for t in s], fill=col + (255,), width=wd, joint='curve')
    tone = lambda col, f: tuple(int(min(255, c * f)) for c in col)
    cov = (kind > 0).astype(float); D, pool = debris_density(H, kind); thin = max(int(round(0.0013 * sx)), 2)
    # sap: a fine speckle of dull dried drops, and the odd tan blotch a couple of centimetres across
    for x, z in zip(*scatter(cov, 160)):
        blob(x, z, rng.uniform(0.0008, 0.0020), tone((150, 144, 128) if rng.random() < 0.6 else (112, 96, 70), rng.uniform(0.85, 1.1)) + (255,))
    for x, z in zip(*scatter(cov, 3.0)):
        r = rng.uniform(0.004, 0.009); col = tone((168, 140, 86), rng.uniform(0.85, 1.1)) + (255,)
        for _ in range(rng.integers(3, 7)): blob(x + rng.normal(0, r * 0.7), z + rng.normal(0, r * 0.7), r * rng.uniform(0.35, 0.8), col)
    # stray needles and bits of twig: short, thin, weathered grey-brown
    xs, zs = scatter(D); n = len(xs)
    for x, z in zip(xs, zs):
        col = tone((112, 88, 62) if rng.random() < 0.7 else (78, 64, 50), rng.uniform(0.8, 1.15)); needle(x, z, rng.uniform(0.02, 0.06), col, thin)
        if rng.random() < 0.3: needle(x, z, rng.uniform(0.02, 0.05), col, thin)
    # the mat that collects in the cowl: dark litter underneath, dry tan needles on top
    for x, z in zip(*scatter(pool, 0.6)): blob(x, z, rng.uniform(0.002, 0.005), tone((70, 56, 42), rng.uniform(0.7, 1.1)) + (255,))
    xs, zs = scatter(pool); n += len(xs)
    for x, z in zip(xs, zs):
        t = rng.random(); col = tone((150, 122, 82) if t < 0.5 else (122, 94, 62) if t < 0.85 else (92, 70, 48), rng.uniform(0.8, 1.15))
        needle(x, z, rng.uniform(0.025, 0.06), col, thin + 1)
    return im.resize((W, Hpx), Image.LANCZOS), n

def write_skins():
    """the two skins derived from the base one; previews and livery icons are only seeded, never overwritten (Content Manager regenerates them)"""
    import shutil, json
    base = SKINS / BASE_SKIN
    if not base.exists(): return {'skipped': 'no ' + str(base)}
    clear = png_bytes(Image.new('RGBA', (8, 8), (0, 0, 0, 0))); pine, needles = debris_image()
    wheel_black = flat_texture((15, 15, 16)); wheel_matte = flat_texture((30, 40, 6))        # maps: little specular, broad highlight, next to no reflection
    variants = {
        'silver_176_no_stickers': ('Silver 176 - no stickers', 1, {'gh_sticker.png': clear, 'gh_number_panel.png': clear}),
        'silver_176_pine': ('Silver 176 - pine needles and sap', 2, {'gh_sticker.png': clear, 'gh_number_panel.png': clear, 'gh_debris.png': png_bytes(pine), 'gh_bugs.png': png_bytes(_bugs['img']), 'MKII_Grey.png': wheel_black, 'MKIII_Grey.png': wheel_black,
                                                                    'SSR_MKII_Map.png': wheel_matte, 'SSR_MKIII_Map.png': wheel_matte}),
    }
    out = {'needle_bundles': needles}
    for name, (title, prio, tex) in variants.items():
        dst = SKINS / name; dst.mkdir(exist_ok=True)
        for f in base.iterdir():
            if f.name == 'ui_skin.json' or (f.name in ('preview.jpg', 'livery.png') and (dst / f.name).exists()): continue
            shutil.copy2(f, dst / f.name)
        ui = json.loads((base / 'ui_skin.json').read_text(encoding='utf-8-sig')); ui['skinname'] = title; ui['priority'] = prio
        (dst / 'ui_skin.json').write_text(json.dumps(ui, indent=2), encoding='utf-8')
        for tn, data in tex.items(): (dst / tn).write_bytes(data)
        out[name] = sorted(tex)
    return out

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
    print('  %-16s %s' % ('skins', write_skins()))
    print('md5', hashlib.md5(open(out, 'rb').read()).hexdigest())
