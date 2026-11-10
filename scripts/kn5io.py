"""Full KN5 reader/writer (static meshes). Loads a kn5 into plain Python objects, lets you edit them, and writes it
back; an unmodified load->save round trip is byte-identical.

    k = Kn5.load(path)
    k.textures            list of [active, name, bytes]
    k.materials           list of Material
    k.root                Node tree (Node.children); meshes carry V (n,11 float32: pos3 nrm3 uv2 tan3) and F (m,3)
    k.save(path)

Coordinates (world): +x = car's left, +y = up, +z = front.
"""
import struct, io
import numpy as np


class _R:
    def __init__(s, b): s.b = b; s.p = 0
    def u(s, fmt):
        n = struct.calcsize('<' + fmt); v = struct.unpack_from('<' + fmt, s.b, s.p); s.p += n
        return v if len(v) > 1 else v[0]
    def raw(s, n): v = s.b[s.p:s.p + n]; s.p += n; return v
    def st(s): return s.raw(s.u('i')).decode('utf8')


def _s(x): e = x.encode('utf8'); return struct.pack('<i', len(e)) + e


class Material:
    def __init__(s, name, shader, blend=0, alpha_tested=0, depth=0, props=None, tex=None):
        s.name = name; s.shader = shader; s.blend = blend; s.alpha_tested = alpha_tested; s.depth = depth
        s.props = props or []      # [name, value, 36 raw bytes]
        s.tex = tex or []          # [sampler, slot, texture name]

    def set(s, name, value):
        for p in s.props:
            if p[0] == name: p[1] = value; return
        s.props.append([name, value, b'\x00' * 36])

    def copy(s, name):
        return Material(name, s.shader, s.blend, s.alpha_tested, s.depth, [list(p) for p in s.props], [list(t) for t in s.tex])


class Node:
    def __init__(s, type_, name, active=1):
        s.type = type_; s.name = name; s.active = active; s.children = []
        s.matrix = np.eye(4, dtype=np.float32)
        s.flags = (1, 1, 0)        # cast shadows, visible, transparent
        s.V = None; s.F = None; s.mat = 0; s.layer = 0; s.lod = (0.0, 0.0); s.bs = None; s.renderable = 1
        s.raw = None               # skinned meshes are carried through untouched

    def walk(s, parent=None, M=None):
        """yield (node, parent, world matrix of the node's parent chain incl. itself), row-vector convention"""
        M = np.eye(4) if M is None else M
        if s.type == 1: M = s.matrix.astype(np.float64) @ M
        yield s, parent, M
        for c in s.children: yield from c.walk(s, M)


class Kn5:
    @staticmethod
    def load(path):
        k = Kn5(); b = open(path, 'rb').read(); r = _R(b)
        assert r.raw(6) == b'sc6969'
        k.version = r.u('i'); k.extra = r.u('i') if k.version > 5 else None
        k.textures = []
        for _ in range(r.u('i')):
            act = r.u('i'); name = r.st(); sz = r.u('i'); k.textures.append([act, name, r.raw(sz)])
        k.materials = []
        for _ in range(r.u('i')):
            m = Material(r.st(), r.st()); m.blend = r.u('B'); m.alpha_tested = r.u('B')
            if k.version > 4: m.depth = r.u('i')
            for _ in range(r.u('i')): m.props.append([r.st(), r.u('f'), r.raw(36)])
            for _ in range(r.u('i')): m.tex.append([r.st(), r.u('i'), r.st()])
            k.materials.append(m)

        def node():
            t = r.u('i'); n = Node(t, r.st()); cc = r.u('i'); n.active = r.u('B')
            if t == 1:
                n.matrix = np.array(r.u('16f'), dtype=np.float32).reshape(4, 4)
            elif t == 2:
                n.flags = r.u('3B'); vc = r.u('i')
                n.V = np.frombuffer(r.raw(vc * 44), dtype='<f4').reshape(vc, 11).copy()
                ic = r.u('i'); n.F = np.frombuffer(r.raw(ic * 2), dtype='<u2').astype(np.int64).reshape(-1, 3)
                n.mat = r.u('i'); n.layer = r.u('i'); n.lod = r.u('2f'); n.bs = r.u('4f'); n.renderable = r.u('B')
            else:
                raise NotImplementedError('skinned mesh nodes are not supported')
            for _ in range(cc): n.children.append(node())
            return n
        k.root = node()
        assert r.p == len(b), 'trailing bytes'
        return k

    def save(s, path):
        o = io.BytesIO(); w = o.write
        w(b'sc6969'); w(struct.pack('<i', s.version))
        if s.version > 5: w(struct.pack('<i', s.extra))
        w(struct.pack('<i', len(s.textures)))
        for act, name, data in s.textures: w(struct.pack('<i', act)); w(_s(name)); w(struct.pack('<i', len(data))); w(data)
        w(struct.pack('<i', len(s.materials)))
        for m in s.materials:
            w(_s(m.name)); w(_s(m.shader)); w(struct.pack('<BB', m.blend, m.alpha_tested))
            if s.version > 4: w(struct.pack('<i', m.depth))
            w(struct.pack('<i', len(m.props)))
            for pn, pv, rest in m.props: w(_s(pn)); w(struct.pack('<f', pv)); w(rest)
            w(struct.pack('<i', len(m.tex)))
            for sn, slot, tn in m.tex: w(_s(sn)); w(struct.pack('<i', slot)); w(_s(tn))

        def node(n):
            w(struct.pack('<i', n.type)); w(_s(n.name)); w(struct.pack('<iB', len(n.children), n.active))
            if n.type == 1:
                w(np.asarray(n.matrix, dtype='<f4').tobytes())
            else:
                assert len(n.V) < 65536, n.name + ': too many vertices for one mesh'
                w(struct.pack('<3B', *n.flags)); w(struct.pack('<i', len(n.V))); w(np.asarray(n.V, dtype='<f4').tobytes())
                F = np.asarray(n.F).reshape(-1)
                w(struct.pack('<i', len(F))); w(F.astype('<u2').tobytes())
                w(struct.pack('<ii', n.mat, n.layer)); w(struct.pack('<2f', *n.lod)); w(struct.pack('<4f', *n.bs)); w(struct.pack('<B', n.renderable))
            for c in n.children: node(c)
        node(s.root)
        open(path, 'wb').write(o.getvalue())

    # ---------------------------------------------------------------- helpers
    def nodes(s):
        return list(s.root.walk())

    def find(s, name, type_=2):
        """-> (node, parent, world matrix)"""
        hits = [t for t in s.root.walk() if t[0].name == name and t[0].type == type_]
        assert len(hits) == 1, '%s: %d matches' % (name, len(hits))
        return hits[0]

    def mat_id(s, name):
        return [m.name for m in s.materials].index(name)

    def add_material(s, m):
        s.materials.append(m); return len(s.materials) - 1

    def set_texture(s, name, data):
        for t in s.textures:
            if t[1] == name: t[2] = data; return
        s.textures.append([1, name, data])

    def add_mesh(s, name, V, F, mat, parent=None, transparent=0, shadows=1):
        """append a mesh (world coordinates) wrapped in its own dummy, like the meshes ksEditor exports"""
        d = Node(1, name); m = Node(2, name); d.children.append(m)
        m.V = np.asarray(V, dtype=np.float32); m.F = np.asarray(F, dtype=np.int64).reshape(-1, 3); m.mat = mat
        m.flags = (shadows, 1, transparent); update_bounds(m)
        (parent or s.root).children.append(d)
        return m


def update_bounds(n):
    p = n.V[:, :3].astype(np.float64); c = (p.min(0) + p.max(0)) / 2
    n.bs = (*c, float(np.linalg.norm(p - c, axis=1).max()))


def world_pos(n, M):
    return (np.c_[n.V[:, :3].astype(np.float64), np.ones(len(n.V))] @ M)[:, :3]


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def weld_normals(p, F):
    """area-weighted vertex normals, shared between vertices at the same position (keeps UV seams smooth)"""
    key = np.round(p * 1e4).astype(np.int64)
    _, wid = np.unique(key, axis=0, return_inverse=True); wid = wid.ravel()
    fn = np.cross(p[F[:, 1]] - p[F[:, 0]], p[F[:, 2]] - p[F[:, 0]])
    acc = np.zeros((wid.max() + 1, 3))
    for j in range(3): np.add.at(acc, wid[F[:, j]], fn)
    return acc[wid]


def deform(n, M, fn, thr=1e-5):
    """move a mesh's vertices with fn(world_xyz)->world_xyz; normals follow the change in the welded surface normal,
    so hand-authored hard edges / smoothing are preserved. Returns the number of vertices moved."""
    w = world_pos(n, M); w2 = fn(w)
    mv = np.linalg.norm(w2 - w, axis=1) > thr
    if not mv.any(): return 0
    Mi = np.linalg.inv(M)
    V = n.V.astype(np.float64); lp0 = V[:, :3].copy(); lp1 = (np.c_[w2, np.ones(len(w2))] @ Mi)[:, :3]
    n0 = unit(weld_normals(lp0, n.F)); n1 = unit(weld_normals(lp1, n.F))
    nn = unit(V[:, 3:6] + (n1 - n0)); m = mv[:, None]
    V[:, :3] = np.where(m, lp1, lp0); nf = np.where(m, nn, V[:, 3:6]); V[:, 3:6] = nf
    t = V[:, 8:11]; t = t - nf * (t * nf).sum(1, keepdims=True); V[:, 8:11] = np.where(m, unit(t), V[:, 8:11])
    n.V = V.astype(np.float32); update_bounds(n)
    return int(mv.sum())


def split_mesh(n, tri_mask):
    """take the triangles selected by tri_mask out of mesh n; returns (V, F) of the removed part (local coords)"""
    F = n.F; take = F[tri_mask]; keep = F[~tri_mask]
    def compact(Fs):
        used = np.unique(Fs); remap = -np.ones(len(n.V), np.int64); remap[used] = np.arange(len(used))
        return n.V[used].copy(), remap[Fs]
    Vt, Ft = compact(take); Vk, Fk = compact(keep)
    n.V, n.F = Vk, Fk; update_bounds(n)
    return Vt, Ft


if __name__ == '__main__':
    import sys, hashlib
    k = Kn5.load(sys.argv[1]); out = sys.argv[2]; k.save(out)
    a = hashlib.md5(open(sys.argv[1], 'rb').read()).hexdigest(); b = hashlib.md5(open(out, 'rb').read()).hexdigest()
    print('round trip', 'IDENTICAL' if a == b else 'DIFFERENT', a, b)
