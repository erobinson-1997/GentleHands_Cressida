"""Quick software preview of a kn5 (no game needed): painter's-algorithm render with per-triangle texture colour.

    python scripts/preview.py <kn5> <out_dir> [view ...]

Views are named in VIEWS below; with none given, all are rendered. Coordinates: +x = car's left, +y = up, +z = front.
This is only a geometry/colour sanity check - the showroom is the real test.
"""
import sys, os, io, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from PIL import Image
from kn5 import R, parse_nodes

VIEWS = {
    # name: (eye, target, fov_deg)
    'rear':        ((0.0, 0.75, -6.0), (0.0, 0.55, -2.3), 32),
    'rear_low':    ((-0.6, 0.25, -4.6), (-0.1, 0.5, -2.3), 38),
    'rear_right':  ((-3.2, 1.0, -5.2), (-0.3, 0.55, -1.6), 32),
    'rear_left':   ((3.2, 1.0, -5.2), (0.3, 0.55, -1.6), 32),
    'right':       ((-6.5, 0.9, 0.0), (0.0, 0.6, 0.0), 36),
    'left':        ((6.5, 0.9, 0.0), (0.0, 0.6, 0.0), 36),
    'front':       ((0.0, 0.9, 6.0), (0.0, 0.55, 2.0), 32),
    'front_right': ((-3.2, 1.2, 5.2), (-0.2, 0.55, 1.4), 32),
    'front_left':  ((3.2, 1.2, 5.2), (0.2, 0.55, 1.4), 32),
    'top':         ((0.0, 7.5, -0.2), (0.0, 0.5, -0.1), 36),
    'hood':        ((-1.6, 2.2, 3.6), (0.0, 0.7, 1.3), 40),
    'rq_right':    ((-2.6, 1.3, -2.6), (-0.8, 0.7, -1.7), 40),   # right rear quarter close-up
    'corner_rr':   ((-1.9, 0.9, -3.9), (-0.6, 0.6, -2.3), 40),   # crushed corner close-up
    'mirror_r':    ((-2.2, 1.2, 1.3), (-0.86, 0.85, 0.4), 30),
}

def load(path):
    b = open(path, 'rb').read(); r = R(b)
    assert r.raw(6) == b'sc6969'
    ver = r.u('i')
    if ver > 5: r.u('i')
    tex = {}
    for _ in range(r.u('i')):
        r.u('i'); name = r.st(); sz = r.u('i'); tex[name] = r.raw(sz)
    mats = []
    for _ in range(r.u('i')):
        name = r.st(); shader = r.st(); r.raw(2)
        if ver > 4: r.u('i')
        props = {}
        for _ in range(r.u('i')):
            pn = r.st(); props[pn] = r.u('f'); r.raw(36)
        res = []
        for _ in range(r.u('i')):
            r.st(); r.u('i'); res.append(r.st())
        mats.append((name, shader, props, res))
    nodes = parse_nodes(r, ver)
    return b, tex, mats, nodes

def world_M(nodes, i):
    M = np.eye(4); j = i
    while j != -1:
        n = nodes[j]
        if n['type'] == 1: M = M @ np.array(n['matrix']).reshape(4, 4)
        j = n['parent']
    return M

def tex_image(data):
    try:
        im = Image.open(io.BytesIO(data)).convert('RGB')
        if max(im.size) > 1024: im = im.resize((1024, 1024 * im.size[1] // im.size[0]))
        return np.asarray(im).astype(np.float32) / 255.0
    except Exception:
        return None

def scene(path, skip=()):
    b, tex, mats, nodes = load(path)
    cache = {}
    P = []; C = []; G = []
    for i, n in enumerate(nodes):
        if n['type'] != 2 or n['ic'] == 0 or any(s in n['name'] for s in skip): continue
        V = np.frombuffer(b, dtype='<f4', count=n['vc'] * 11, offset=n['vstart']).reshape(-1, 11).astype(np.float64)
        F = np.frombuffer(b, dtype='<u2', count=n['ic'], offset=n['istart']).astype(np.int64).reshape(-1, 3)
        w = (np.c_[V[:, :3], np.ones(len(V))] @ world_M(nodes, i))[:, :3]
        m = mats[n['mat']]; tn = m[3][0] if m[3] else None
        if tn not in cache: cache[tn] = tex_image(tex[tn]) if tn in tex else None
        im = cache[tn]
        if im is None: col = np.full((len(F), 3), 0.5)
        else:
            uv = V[F, 6:8].mean(1); h, wd = im.shape[:2]
            col = im[(np.mod(uv[:, 1], 1) * (h - 1)).astype(int), (np.mod(uv[:, 0], 1) * (wd - 1)).astype(int)]
        nm = m[0].lower(); glass = 0
        if nm == 'chrome': col = np.full((len(F), 3), 0.72)
        if nm == 'glass': col = np.full((len(F), 3), (0.10, 0.13, 0.15)); glass = 1
        if 'clearglass' in nm: col = np.full((len(F), 3), 0.8)
        if nm == 'redglass': col = np.full((len(F), 3), (0.7, 0.1, 0.1))
        if 'chasis' in nm: col = np.full((len(F), 3), 0.03)
        P.append(w[F]); C.append(col); G.append(np.full(len(F), glass))
    return np.concatenate(P), np.concatenate(C), np.concatenate(G)

def render(P, C, G, eye, target, fov, W=1600, H=900, ss=2):
    eye = np.array(eye, float); target = np.array(target, float)
    f = target - eye; f /= np.linalg.norm(f)
    rt = np.cross(f, [0, 1.0, 0])
    if np.linalg.norm(rt) < 1e-6: rt = np.array([1.0, 0, 0])
    rt /= np.linalg.norm(rt); up = np.cross(rt, f)
    d = P - eye
    x = d @ rt; y = d @ up; z = d @ f                       # (n,3) each
    ok = (z > 0.05).all(1)
    fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); ln = np.linalg.norm(fn, axis=1); ok &= ln > 1e-12
    fn = fn / np.maximum(ln, 1e-12)[:, None]
    facing = (fn * (P[:, 0] - eye)).sum(1)
    fn = np.where((facing > 0)[:, None], -fn, fn)            # double-sided: shade toward the viewer
    Ws, Hs = W * ss, H * ss; k = (Hs / 2) / np.tan(np.radians(fov) / 2)
    zc = np.maximum(z, 0.05)
    sx = Ws / 2 + k * x / zc; sy = Hs / 2 - k * y / zc
    ok &= (sx.max(1) > 0) & (sx.min(1) < Ws) & (sy.max(1) > 0) & (sy.min(1) < Hs)
    L = np.array([-0.35, 0.8, -0.45]); L /= np.linalg.norm(L)
    sh = 0.35 + 0.5 * np.clip(fn @ L, 0, 1) + 0.25 * np.clip(fn[:, 1], 0, 1) + 0.15 * np.clip(-(fn @ f), 0, 1)
    col = np.clip(C * sh[:, None], 0, 1)
    idx = np.where(ok)[0]; idx = idx[np.argsort(-z[idx].mean(1))]
    img = np.full((Hs, Ws, 3), (0.42, 0.45, 0.48), np.float32)
    pts = np.stack([sx, sy], -1)
    pts = np.round(pts * 16).astype(np.int32)                # 4 bits of sub-pixel precision
    bgr = (col[:, ::-1] * 255).astype(np.uint8)
    img8 = (img * 255).astype(np.uint8)
    for i in idx:
        if G[i]: continue
        cv2.fillConvexPoly(img8, pts[i], bgr[i].tolist(), lineType=cv2.LINE_8, shift=4)
    ov = img8.copy()
    for i in idx:
        if G[i]: cv2.fillConvexPoly(ov, pts[i], bgr[i].tolist(), lineType=cv2.LINE_8, shift=4)
    img8 = cv2.addWeighted(ov, 0.55, img8, 0.45, 0)
    return cv2.resize(img8, (W, H), interpolation=cv2.INTER_AREA)

if __name__ == '__main__':
    path, out = sys.argv[1], sys.argv[2]; names = sys.argv[3:] or list(VIEWS)
    os.makedirs(out, exist_ok=True)
    P, C, G = scene(path)
    for nm in names:
        eye, tgt, fov = VIEWS[nm]
        cv2.imwrite(os.path.join(out, nm + '.jpg'), render(P, C, G, eye, tgt, fov), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('rendered', nm)
