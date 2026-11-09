import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo import *
import numpy as np, struct

C=np.array([-0.80,0.60,-2.47]); R=np.array([0.55,0.52,0.78])
PUSH=0.19*np.array([0.42,0.0,0.91]); PUSH/=1.0
Q=np.array([0.35,0.45,0.82]); Q/=np.linalg.norm(Q)
EXCL=('TYRE','Brake','SSR','caliper','Exhaust','badge','SUSP','WHEEL')

def kernel(w):
    r2=(((w-C)/R)**2).sum(1)
    return np.where(r2<1,(1-r2)**2,0.0)

def displace(w):
    k=kernel(w)
    s=w@np.array([0.86,0.15,-0.48])
    wr=0.028*np.sin(26*s+2.0*np.sin(9*w[:,1]+3*w[:,0]))
    d=PUSH[None,:]*(k*(0.65+0.35*np.sin(17*w[:,1]+13*w[:,0])))[:,None]
    lift=np.where(w[:,1]>0.68,0.045,0.0)*k
    d[:,1]+=lift
    d+=Q[None,:]*(wr*k)[:,None]
    return d,k

def weld_normals(lp,idx):
    key=np.round(lp*1e4).astype(np.int64)
    _,wid=np.unique(key,axis=0,return_inverse=True)
    wid=wid.ravel()
    f=idx.reshape(-1,3)
    e1=lp[f[:,1]]-lp[f[:,0]]; e2=lp[f[:,2]]-lp[f[:,0]]
    fn=np.cross(e1,e2)
    acc=np.zeros((wid.max()+1,3))
    for j in range(3): np.add.at(acc,wid[f[:,j]],fn)
    return acc[wid]

def deform(buf):
    changed=[]
    for i,n in enumerate(nodes):
        if n['type']!=2 or any(x in n['name'] for x in EXCL): continue
        vc=n['vc']
        V=np.frombuffer(b,dtype='<f4',count=vc*11,offset=n['vstart']).reshape(vc,11).astype(np.float64).copy()
        M=world_M(i); Mi=np.linalg.inv(M); A=M[:3,:3]
        w=(np.c_[V[:,:3],np.ones(vc)]@M)[:,:3]
        d,k=displace(w)
        if k.max()<=0: continue
        idx=np.frombuffer(b,dtype='<u2',count=n['ic'],offset=n['istart']).astype(np.int64)
        lp0=V[:,:3].copy()
        w2=w+d
        lp1=(np.c_[w2,np.ones(vc)]@Mi)[:,:3]
        n0=weld_normals(lp0,idx); n1=weld_normals(lp1,idx)
        nn=V[:,3:6]
        def unit(a): return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        nnew=unit(nn+ (unit(n1)-unit(n0)))
        m=(k>0.002)[:,None]
        V[:,:3]=np.where(m,lp1,lp0)
        nfin=np.where(m,nnew,nn)
        V[:,3:6]=nfin
        t=V[:,8:11]; t=t-nfin*(t*nfin).sum(1,keepdims=True); V[:,8:11]=np.where(m,unit(t),V[:,8:11])
        buf[n['vstart']:n['vstart']+vc*44]=V.astype('<f4').tobytes()
        changed.append((n['name'],int((k>0.002).sum()),float(np.abs(d).max())))
    return changed

def tube(pts,rad,seg=14,caps=True,uvscale=4.0):
    pts=np.asarray(pts,float)
    # Catmull-Rom resample
    def cr(p0,p1,p2,p3,t):
        return 0.5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t**3)
    P=np.vstack([pts[0],pts,pts[-1]]); path=[]
    for i in range(1,len(P)-2):
        for t in np.linspace(0,1,12,endpoint=False): path.append(cr(P[i-1],P[i],P[i+1],P[i+2],t))
    path.append(pts[-1]); path=np.array(path)
    T=np.gradient(path,axis=0); T/=np.linalg.norm(T,axis=1,keepdims=True)
    up=np.array([0,1.0,0]); 
    if abs(T[0]@up)>0.9: up=np.array([1.0,0,0])
    nrm=np.cross(T[0],up); nrm/=np.linalg.norm(nrm)
    frames=[]
    for i in range(len(path)):
        if i>0:
            nrm=nrm-T[i]*(nrm@T[i]); nrm/=np.linalg.norm(nrm)
        frames.append((nrm.copy(),np.cross(T[i],nrm)))
    L=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    V=[];F=[]
    for i,p in enumerate(path):
        n_,b_=frames[i]
        for j in range(seg+1):
            a=2*np.pi*j/seg; dirn=np.cos(a)*n_+np.sin(a)*b_
            tang=T[i]
            V.append([*(p+rad*dirn),*dirn,j/seg,L[i]*uvscale,*tang])
    for i in range(len(path)-1):
        for j in range(seg):
            a=i*(seg+1)+j; bq=a+seg+1
            F+= [a,bq,a+1, a+1,bq,bq+1]
    if caps:
        for end,sign in ((0,-1),(len(path)-1,1)):
            c=len(V); p=path[end]; tang=T[end]*sign
            V.append([*p,*tang,0.5,0.5,*frames[end][0]])
            base=end*(seg+1)
            for j in range(seg+1):
                v=V[base+j]; V.append([v[0],v[1],v[2],*tang,0.5+0.5*np.cos(2*np.pi*j/seg),0.5+0.5*np.sin(2*np.pi*j/seg),*frames[end][0]])
            for j in range(seg):
                F+= [c,c+1+j,c+2+j] if sign>0 else [c,c+2+j,c+1+j]
    return np.array(V,float),np.array(F,np.int64)

def bar_mesh():
    main=[(0.80,0.37,-2.27),(0.80,0.35,-2.40),(0.74,0.335,-2.52),(0.60,0.33,-2.565),(0.30,0.33,-2.572),(0.0,0.33,-2.574),
          (-0.30,0.33,-2.570),(-0.52,0.335,-2.545),(-0.66,0.345,-2.47),(-0.76,0.36,-2.37),(-0.80,0.385,-2.27),(-0.72,0.40,-2.20)]
    parts=[tube(main,0.019)]
    for x in (0.46,-0.36):
        parts.append(tube([(x,0.331,-2.566),(x,0.331,-2.45),(x,0.34,-2.33)],0.011,seg=10))
    Vs=[];Fs=[];off=0
    for V,F in parts: Vs.append(V);Fs.append(F+off);off+=len(V)
    return np.vstack(Vs),np.concatenate(Fs)

def s(x): e=x.encode(); return struct.pack('<i',len(e))+e
def mesh_bytes(name,V,F,mat):
    c=V[:,:3].mean(0); rad=np.linalg.norm(V[:,:3]-c,axis=1).max()
    o=struct.pack('<i',2)+s(name)+struct.pack('<i',0)+b'\x01'+bytes([1,1,0])
    o+=struct.pack('<i',len(V))+V.astype('<f4').tobytes()
    o+=struct.pack('<i',len(F))+F.astype('<u2').tobytes()
    o+=struct.pack('<ii',mat,0)+struct.pack('<2f',0,0)+struct.pack('<4f',*c,rad)+b'\x01'
    return o
def dummy_bytes(name,children):
    return struct.pack('<i',1)+s(name)+struct.pack('<i',children)+b'\x01'+struct.pack('<16f',1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)

