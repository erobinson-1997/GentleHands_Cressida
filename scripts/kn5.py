import struct, io
class R:
    def __init__(s,b): s.b=b; s.p=0
    def u(s,fmt):
        n=struct.calcsize(fmt); v=struct.unpack_from('<'+fmt,s.b,s.p); s.p+=n; return v if len(v)>1 else v[0]
    def raw(s,n): v=s.b[s.p:s.p+n]; s.p+=n; return v
    def st(s): n=s.u('i'); return s.raw(n).decode('utf8','replace')
def parse(path):
    b=open(path,'rb').read(); r=R(b)
    assert r.raw(6)==b'sc6969'
    ver=r.u('i'); extra=r.u('i') if ver>5 else None
    tex=[]
    n=r.u('i')
    for _ in range(n):
        act=r.u('i'); name=r.st(); sz=r.u('i'); r.raw(sz); tex.append((name,sz))
    mats=[]
    n=r.u('i')
    for _ in range(n):
        name=r.st(); shader=r.st(); r.u('b'); r.u('b')
        if ver>4: r.u('i')
        pc=r.u('i'); props=[]
        for _ in range(pc):
            pn=r.st(); pv=r.u('f'); r.raw(36); props.append((pn,pv))
        tc=r.u('i'); res=[]
        for _ in range(tc):
            sn=r.st(); slot=r.u('i'); tn=r.st(); res.append((sn,slot,tn))
        mats.append((name,shader,props,res))
    return b,r,ver,tex,mats
if __name__=='__main__':
    import config
    b,r,ver,tex,mats=parse(str(config.ORIGINAL_KN5))
    print('ver',ver,'size',len(b),'tex',len(tex),'mats',len(mats),'nodes start',r.p)
    for i,m in enumerate(mats): print(i,m[0],m[1])

def parse_nodes(r, ver):
    out=[]
    def node(depth, parent):
        start=r.p
        t=r.u('i'); name=r.st(); cc=r.u('i'); act=r.u('b')
        d=dict(type=t,name=name,depth=depth,parent=parent,start=start)
        if t==1:
            d['matrix']=r.u('16f'); d['body_start']=None
        elif t in(2,3):
            d['flags']=r.u('3b'); vc=r.u('i'); d['vc']=vc; d['vstart']=r.p
            if t==2: r.raw(vc*44)
            else:
                r.raw(vc*44)  # skinned layout differs; handled below if encountered
            ic=r.u('i'); d['ic']=ic; d['istart']=r.p; r.raw(ic*2)
            d['mat']=r.u('i'); d['layer']=r.u('i'); r.u('2f'); d['bs']=r.u('4f'); r.u('b')
        d['end_header']=r.p
        out.append(d)
        idx=len(out)-1
        for _ in range(cc): node(depth+1,idx)
        return d
    node(0,-1)
    return out
