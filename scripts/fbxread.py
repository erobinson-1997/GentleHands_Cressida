import struct, zlib, numpy as np
class Node:
    def __init__(s,name,props): s.name=name; s.props=props; s.children=[]
    def find(s,name): return [c for c in s.children if c.name==name]
def read_fbx(path):
    b=open(path,'rb').read()
    ver=struct.unpack_from('<I',b,23)[0]
    p=27
    big=ver>=7500
    def u32(o): return struct.unpack_from('<I',b,o)[0]
    def read_prop(o):
        t=chr(b[o]); o+=1
        if t=='Y': return struct.unpack_from('<h',b,o)[0],o+2
        if t=='C': return b[o],o+1
        if t=='I': return struct.unpack_from('<i',b,o)[0],o+4
        if t=='F': return struct.unpack_from('<f',b,o)[0],o+4
        if t=='D': return struct.unpack_from('<d',b,o)[0],o+8
        if t=='L': return struct.unpack_from('<q',b,o)[0],o+8
        if t in 'fdlib':
            n,enc,clen=struct.unpack_from('<III',b,o); o+=12
            dt={'f':'<f4','d':'<f8','l':'<i8','i':'<i4','b':'u1'}[t]
            raw=b[o:o+clen]; o+=clen
            if enc==1: raw=zlib.decompress(raw)
            return np.frombuffer(raw,dtype=dt,count=n).copy(),o
        if t in 'SR':
            n=u32(o); o+=4; raw=b[o:o+n]; o+=n
            return (raw.decode('utf8','replace') if t=='S' else raw),o
        raise ValueError(t)
    def read_node(o):
        if big: end,nprop,plen,nlen=struct.unpack_from('<QQQB',b,o); o+=25
        else:   end,nprop,plen,nlen=struct.unpack_from('<IIIB',b,o); o+=13
        if end==0: return None,o
        name=b[o:o+nlen].decode(); o+=nlen
        props=[]
        for _ in range(nprop):
            v,o=read_prop(o); props.append(v)
        node=Node(name,props)
        while o<end:
            ch,o=read_node(o)
            if ch is None: break
            node.children.append(ch)
        return node,end
    root=Node('root',[])
    while p<len(b)-200:
        n,p2=read_node(p)
        if n is None: break
        root.children.append(n); p=p2
    return root,ver
