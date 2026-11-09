import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geo import *
import numpy as np, struct
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from kn5build import tube, s, mesh_bytes, dummy_bytes, weld_normals

# ---------- crash field (right-rear corner) ----------
C1=np.array([-0.80,0.60,-2.47]); R1=np.array([0.55,0.52,0.78])
PUSH1=0.20*np.array([0.42,0.0,0.91])
C2=np.array([-0.76,0.34,-2.33]); R2=np.array([0.40,0.30,0.50])   # crushed lower corner
PUSH2=0.13*np.array([0.30,0.35,0.89])
Q=np.array([0.35,0.45,0.82]); Q/=np.linalg.norm(Q)
def kern(w,C,R):
    r2=(((w-C)/R)**2).sum(1); return np.where(r2<1,(1-r2)**2,0.0)
def crash(w):
    k1=kern(w,C1,R1); k2=kern(w,C2,R2)
    s_=w@np.array([0.86,0.15,-0.48])
    wr1=0.03*np.sin(26*s_+2.0*np.sin(9*w[:,1]+3*w[:,0]))
    wr2=0.045*np.sin(34*s_+3.0*np.sin(13*w[:,0]+7*w[:,1]))
    d=PUSH1[None,:]*(k1*(0.65+0.35*np.sin(17*w[:,1]+13*w[:,0])))[:,None]
    d+=PUSH2[None,:]*(k2*(0.6+0.4*np.sin(23*w[:,0]+19*w[:,1])))[:,None]
    d+=Q[None,:]*((wr1*k1)+(wr2*k2))[:,None]
    return w+d
# ---------- bumper -> flat rear lower panel ----------
def panelwarp(w):
    w=w.copy(); m=w[:,2]<-2.21
    t=np.clip((w[:,1]-0.44)/0.06,0,1); t=t*t*(3-2*t)
    sc=0.45+0.55*t                       # 0.45 below the ledge, 1.0 (original surface) at the top edge
    w[m,2]=-2.21+(w[m,2]+2.21)*sc[m]
    return w
# ---------- right taillight: rigid tilt ----------
PV=np.array([-0.30,0.60,-2.45]); PHI=np.radians(6.0); PSI=np.radians(6.0); TR=np.array([0.0,-0.004,0.0])
def tilt(p):
    d=p-PV; c,s_=np.cos(PHI),np.sin(PHI)
    x=d[:,0]*c-d[:,1]*s_; y=d[:,0]*s_+d[:,1]*c; z=d[:,2]
    c2,s2=np.cos(PSI),np.sin(PSI)
    x2=x*c2+z*s2; z2=-x*s2+z*c2
    return PV+np.c_[x2,y,z2]+TR

# ---- unified displacement: body around the right lamp follows the lamp rigidly ----
LO=np.array([-0.82,0.46,-2.50]); HI=np.array([-0.28,0.71,-2.20])
def ss(a,b,x):
    t=np.clip((x-a)/(b-a),0,1); return t*t*(3-2*t)
def wlamp(w):
    d=np.maximum(np.maximum(LO-w,0),np.maximum(w-HI,0)); dist=np.linalg.norm(d,axis=1)
    return 1-ss(0.02,0.22,dist)
def total(w):
    wl=wlamp(w)[:,None]
    return w+wl*(tilt(w)-w)+(1-wl)*(crash(w)-w)
TC=np.array([-0.66,0.82,-2.40]); TR_=np.array([0.40,0.28,0.60])
def trunk_extra(w):
    r2=(((w-TC)/TR_)**2).sum(1); k=np.where(r2<1,(1-r2)**2,0.0)
    up=ss(0.60,0.80,w[:,1])                      # only the lid skin, not the lower edge
    wr=0.02*np.sin(31*(w[:,0]*0.8+w[:,2]*0.6)+2*np.sin(11*w[:,0]))
    d=np.c_[0.045*np.ones(len(w)),0.075*up,0.030*np.ones(len(w))]*(k*(0.7+0.3*np.sin(15*w[:,0]+9*w[:,2])))[:,None]
    d[:,1]+=wr*k*up
    return w+d

LAMPS=['headlights_part003','brakelights_part004','brakelights_part005','brakelights_part003','headlights_part010','Object032','Object036','signal_r_glass','signal_rr_SUB0','signal_rr_SUB1','brakelights_part008']
def comp_labels(w,idx):
    idx=idx.reshape(-1,3)
    key=np.round(w*1e3).astype(np.int64); _,wid=np.unique(key,axis=0,return_inverse=True); wid=wid.ravel()
    e=np.r_[np.c_[wid[idx[:,0]],wid[idx[:,1]]],np.c_[wid[idx[:,1]],wid[idx[:,2]]]]
    g=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(wid.max()+1,)*2)
    nc,lab=connected_components(g,directed=False); return lab[wid],nc
def lamp_fn(w,idx,name):
    if name=='brakelights_part008':
        t=np.clip((-0.20-w[:,0])/0.15,0,1); t=t*t*(3-2*t)
        return w+(tilt(w)-w)*t[:,None]
    lab,nc=comp_labels(w,idx); out=w.copy()
    for c in range(nc):
        m=lab==c
        if m.sum()<4: continue
        if w[m,0].max()<-0.27: out[m]=tilt(w[m])
    return out
# ---------- exhaust: extend tip and bend down ----------
def exhaust_fn(w,bend=True):
    w=w.copy(); z=w[:,2]
    ext=np.clip((-2.25-z)/0.05,0,1); w[:,2]=z-0.08*ext          # push tip 10 cm further out
    if bend:
        z0=-2.05; dz=w[:,2]-z0; t=np.clip(-dz/0.45,0,1); th=np.radians(13)*t*t*(3-2*t)
        y0=0.16
        w[:,1]=np.where(dz<0,w[:,1]+dz*np.sin(th),w[:,1])
        w[:,2]=np.where(dz<0,z0+dz*np.cos(th),w[:,2])
    return w

def unit(a): return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
def apply(buf,i,fn,thr=1e-5):
    n=nodes[i]; vc=n['vc']
    V=np.frombuffer(b,dtype='<f4',count=vc*11,offset=n['vstart']).reshape(vc,11).astype(np.float64).copy()
    M=world_M(i); Mi=np.linalg.inv(M)
    w=(np.c_[V[:,:3],np.ones(vc)]@M)[:,:3]
    idx=np.frombuffer(b,dtype='<u2',count=n['ic'],offset=n['istart']).astype(np.int64)
    w2=fn(w,idx)
    mv=np.linalg.norm(w2-w,axis=1)>thr
    if not mv.any(): return 0
    lp0=V[:,:3].copy(); lp1=(np.c_[w2,np.ones(vc)]@Mi)[:,:3]
    n0=weld_normals(lp0,idx); n1=weld_normals(lp1,idx)
    nn=V[:,3:6]; nnew=unit(nn+(unit(n1)-unit(n0)))
    m=mv[:,None]
    V[:,:3]=np.where(m,lp1,lp0); nf=np.where(m,nnew,nn); V[:,3:6]=nf
    t=V[:,8:11]; t=t-nf*(t*nf).sum(1,keepdims=True); V[:,8:11]=np.where(m,unit(t),V[:,8:11])
    buf[n['vstart']:n['vstart']+vc*44]=V.astype('<f4').tobytes()
    return int(mv.sum())

EXCL=('TYRE','Brake','SSR','caliper','badge','SUSP','WHEEL','chasis','Tsurikawa','license_plate','Rear_Plate')
IDX={n['name']:i for i,n in enumerate(nodes) if n['type']==2}


def strap(p0,p1,w=0.04,t=0.008):
    p0=np.asarray(p0,float); p1=np.asarray(p1,float)
    d=p1-p0; d/=np.linalg.norm(d)
    u=np.array([1.0,0,0]); v=np.cross(d,u); v/=np.linalg.norm(v)
    def c(a,b_,e): return p0+e*(p1-p0)+a*(w/2)*u+b_*(t/2)*v
    Vs=[];Fs=[];off=0
    def add(V,F):
        nonlocal off
        Vs.append(V);Fs.append(F+off);off+=len(V)
    for a_ in (1,-1):
        add(*face(c(a_,-1,0),c(a_,1,0),c(a_,1,1),c(a_,-1,1),a_*u))
    for b2 in (1,-1):
        add(*face(c(-1,b2,0),c(1,b2,0),c(1,b2,1),c(-1,b2,1),b2*v))
    for e,sg in ((1,1),(0,-1)):
        add(*face(c(-1,-1,e),c(1,-1,e),c(1,1,e),c(-1,1,e),sg*d))
    return np.vstack(Vs),np.concatenate(Fs)

def bar_mesh(holes):
    y=0.395; zb=-2.435; r=0.02; ar=0.10
    pts=[]
    for sgn in (1,-1):
        pass
    # path left(+x) end -> right(-x) end, with short 90-degree returns at both ends
    def arc(xc,zc,a0,a1,sx):
        out=[]
        for a in np.linspace(a0,a1,9):
            out.append((xc+sx*ar*np.sin(a), y, zc-ar*np.cos(a)))
        return out
    xa=0.66
    left=[(0.765,y,-2.275),(0.765,y,-2.30)]
    left+=arc(xa,-2.335,np.pi/2,0.0,1)[1:]          # from forward-heading to straight
    left=[(xa+ar*np.sin(a), y, -2.335-ar*np.cos(a)) for a in np.linspace(np.pi/2,0,8)]
    left=[(0.765,y,-2.275)]+left
    right=[(-xa-ar*np.sin(a), y, -2.335-ar*np.cos(a)) for a in np.linspace(0,np.pi/2,8)]+[(-0.765,y,-2.275)]
    path=left+[(0.3,y,zb),(0.0,y,zb),(-0.3,y,zb)]+right
    parts=[tube(path,r,seg=14)]
    for (hx,hy,hz) in holes:
        parts.append(strap((hx,y-0.012,zb+0.012),(hx,hy+0.012,hz+0.022),0.040,0.009))
    Vs=[];Fs=[];off=0
    for V,F in parts: Vs.append(V);Fs.append(F+off);off+=len(V)
    V=np.vstack(Vs); F=np.concatenate(Fs)
    return V,fix_winding(V,F)

def quad(cx,cy,z,w_,h_):
    x0,x1,y0,y1=cx-w_/2,cx+w_/2,cy-h_/2,cy+h_/2
    V=[[x0,y0,z,0,0,-1,0,0,1,0,0],[x1,y0,z,0,0,-1,1,0,1,0,0],[x1,y1,z,0,0,-1,1,1,1,0,0],[x0,y1,z,0,0,-1,0,1,1,0,0]]
    return np.array(V,float),np.array([0,2,1,0,3,2])


def fix_winding(V,F):
    F=F.reshape(-1,3).copy()
    p=V[:,:3]; n=V[:,3:6]
    fn=np.cross(p[F[:,1]]-p[F[:,0]],p[F[:,2]]-p[F[:,0]])
    d=(fn*n[F].mean(1)).sum(1)
    bad=d<0
    F[bad]=F[bad][:,[0,2,1]]
    return F.reshape(-1)

def face(p0,p1,p2,p3,nrm):
    nrm=np.asarray(nrm,float); nrm/=np.linalg.norm(nrm)
    e=np.asarray(p1,float)-np.asarray(p0,float); e/=np.linalg.norm(e)
    V=np.array([[*p,*nrm,u,v,*e] for p,(u,v) in zip((p0,p1,p2,p3),((0,0),(1,0),(1,1),(0,1)))],float)
    return V,np.array([0,1,2,0,2,3])

def liner_mesh(damage,buf):
    from scipy.interpolate import NearestNDInterpolator
    xl=0.70; xr=-0.70; zr=-2.36; zf=-1.72; yf=0.60
    # lid underside from the final (deformed) trunk mesh
    n=nodes[IDX['trunk']]; vc=n['vc']
    Vt=np.frombuffer(buf,dtype='<f4',count=vc*11,offset=n['vstart']).reshape(vc,11).astype(float)
    wt=(np.c_[Vt[:,:3],np.ones(vc)]@world_M(IDX['trunk']))[:,:3]
    wt=wt[(wt[:,2]>-2.40)&(wt[:,2]<-1.70)]
    dx=0.05; dz=0.07
    xe=np.arange(xr,xl+1e-6,dx); ze=np.arange(zr,zf+1e-6,dz)
    nx,nz=len(xe)-1,len(ze)-1
    H=np.full((nx,nz),np.nan)
    for i in range(nx):
        for j in range(nz):
            m=(wt[:,0]>=xe[i]-dx*0.5)&(wt[:,0]<xe[i+1]+dx*0.5)&(wt[:,2]>=ze[j]-dz*0.5)&(wt[:,2]<ze[j+1]+dz*0.5)
            if m.any(): H[i,j]=wt[m,1].min()
    nanm=np.isnan(H)
    if nanm.any():
        ii,jj=np.where(~nanm); itp=NearestNDInterpolator(np.c_[ii,jj],H[~nanm])
        a,b2=np.where(nanm); H[nanm]=itp(np.c_[a,b2])
    xc=(xe[:-1]+xe[1:])/2
    T=H-0.012
    T=np.where(np.abs(xc)[:,None]>0.66,np.minimum(T,0.795),T)     # never above the deck beside the lid
    T=np.maximum(T,yf+0.05)
    Vs=[];Fs=[];off=0
    def add(V,F):
        nonlocal off
        Vs.append(V);Fs.append(F+off);off+=len(V)
    add(*face((xr,yf,zf),(xl,yf,zf),(xl,yf,zr),(xr,yf,zr),(0,1,0)))                      # floor
    for i in range(nx):
        for j in range(nz):
            x0,x1,z0,z1,h=xe[i],xe[i+1],ze[j],ze[j+1],T[i,j]
            add(*face((x0,h,z0),(x1,h,z0),(x1,h,z1),(x0,h,z1),(0,-1,0)))                 # top, faces down
    for j in range(nz):                                                                   # side walls
        z0,z1=ze[j],ze[j+1]
        add(*face((xl,yf,z0),(xl,yf,z1),(xl,T[-1,j],z1),(xl,T[-1,j],z0),(-1,0,0)))
        add(*face((xr,yf,z1),(xr,yf,z0),(xr,T[0,j],z0),(xr,T[0,j],z1),(1,0,0)))
    for i in range(nx):                                                                   # front wall + short rear wall
        x0,x1=xe[i],xe[i+1]
        add(*face((x1,yf,zf),(x0,yf,zf),(x0,T[i,-1],zf),(x1,T[i,-1],zf),(0,0,-1)))
        if T[i,0]>0.71: add(*face((x0,0.71,zr),(x1,0.71,zr),(x1,T[i,0],zr),(x0,T[i,0],zr),(0,0,1)))
    # step faces between neighbouring columns so the hull is closed
    for i in range(nx-1):
        for j in range(nz):
            h0,h1=T[i,j],T[i+1,j]
            if abs(h0-h1)>1e-4:
                xb=xe[i+1]; z0,z1=ze[j],ze[j+1]
                if h1>h0: add(*face((xb,h0,z1),(xb,h0,z0),(xb,h1,z0),(xb,h1,z1),(1,0,0)))
                else:     add(*face((xb,h1,z0),(xb,h1,z1),(xb,h0,z1),(xb,h0,z0),(-1,0,0)))
    for i in range(nx):
        for j in range(nz-1):
            h0,h1=T[i,j],T[i,j+1]
            if abs(h0-h1)>1e-4:
                zb=ze[j+1]; x0,x1=xe[i],xe[i+1]
                if h1>h0: add(*face((x0,h0,zb),(x1,h0,zb),(x1,h1,zb),(x0,h1,zb),(0,0,1)))
                else:     add(*face((x1,h1,zb),(x0,h1,zb),(x0,h0,zb),(x1,h0,zb),(0,0,-1)))
    V=np.vstack(Vs); F=np.concatenate(Fs)
    return V,fix_winding(V,F)

def plate_fill():
    # closes the slot beside the licence plate holder and backs the lamp cavity so nothing is see-through
    Vs=[];Fs=[];off=0
    def add(V,F):
        nonlocal off
        Vs.append(V);Fs.append(F+off);off+=len(V)
    y0,y1=0.47,0.71
    for nrm in ((-1,0,0),(1,0,0)):                                                         # holder side wall, both faces
        add(*face((-0.29,y0,-2.43),(-0.29,y0,-2.18),(-0.29,y1,-2.18),(-0.29,y1,-2.43),nrm))
        add(*face((0.29,y0,-2.43),(0.29,y0,-2.18),(0.29,y1,-2.18),(0.29,y1,-2.43),nrm))
    for nrm in ((0,1,0),(0,-1,0)):                                                         # holder top/bottom extension
        add(*face((-0.29,y1,-2.43),(0.29,y1,-2.43),(0.29,y1,-2.18),(-0.29,y1,-2.18),nrm))
        add(*face((-0.29,y0,-2.43),(0.29,y0,-2.43),(0.29,y0,-2.18),(-0.29,y0,-2.18),nrm))
    add(*face((0.76,y0,-2.18),(-0.76,y0,-2.18),(-0.76,y1,-2.18),(0.76,y1,-2.18),(0,0,-1)))  # baffle behind lamps, faces rear
    V=np.vstack(Vs); F=np.concatenate(Fs)
    return V,fix_winding(V,F)


EXHAUST_TEX=None
def make_exhaust_texture(pts,fr,ss_):
    import casehard
    from PIL import Image
    base=np.array(casehard.tex()).astype(float)            # 1024x1024 tileable case-hardened pattern
    H,W=2048,1024
    L=ss_[-1]-ss_[0]
    yy=(np.arange(H)+0.5)/H
    pi=yy*(len(pts)-1); i0=np.clip(np.floor(pi).astype(int),0,len(pts)-2); t=(pi-i0)[:,None]
    P=pts[i0]*(1-t)+pts[i0+1]*t
    Nn=np.array([f[0] for f in fr]); Bn=np.array([f[1] for f in fr])
    N_=Nn[i0]*(1-t)+Nn[i0+1]*t; B_=Bn[i0]*(1-t)+Bn[i0+1]*t
    # shade along the length: under the car body (z > -2.34) it is in shadow, easing out toward the tip
    z=P[:,2]
    k=np.clip((z+2.58)/(0.30),0,1); k=k*k*(3-2*k)          # 0 at z<=-2.52 (open air) .. 1 at z>=-2.32 (under the panel)
    ao_z=0.95*(1-k)+0.32*k
    ao_z=np.where(np.arange(H)>H-30,np.maximum(ao_z,0.9),ao_z) # keep the very tip clean
    a=2*np.pi*np.arange(W)/W
    rad=np.cos(a)[None,:,None]*N_[:,None,:]+np.sin(a)[None,:,None]*B_[:,None,:]   # (H,W,3)
    up=rad[...,1]
    fu=0.62+0.38*(1-up)/2                                  # top faces get less sky/ground bounce
    ao=ao_z[:,None]*fu
    rows=(((np.arange(H)/H)*(L*3.2))*1024).astype(int)%1024
    col=base[rows,:,:]                                     # (H,W,3)
    img=np.clip(col*ao[...,None]*1.15,0,255).astype(np.uint8)
    return Image.fromarray(img,'RGB')

# ---------- exhaust: muffler at the end of the tail pipe, bent sideways ----------
def exhaust_tail(damage):
    S=np.array([0.443,0.154,-2.04])
    yaw=np.radians(18.0 if damage else 0.0); pit=np.radians(10.0 if damage else 3.0)
    d0=np.array([0,0,-1.0]); d1=np.array([-np.sin(yaw)*np.cos(pit),-np.sin(pit),-np.cos(yaw)*np.cos(pit)]); d1/=np.linalg.norm(d1)
    A,B=0.10,0.14
    pipe,cone,body,taper,tip=0.03,0.025,0.17,0.025,0.04
    s_m0=A+B+pipe; Ltot=s_m0+cone+body+taper+tip
    ds=0.005
    # front section follows the original pipe's centreline (x,y,z) so there is no gap under the car
    FC=np.array([(0.384,0.178,-1.20),(0.387,0.175,-1.42),(0.390,0.172,-1.62),(0.395,0.158,-1.76),(0.418,0.150,-1.92),(0.443,0.154,-2.04)])
    seglen=np.linalg.norm(np.diff(FC,axis=0),axis=1); cs=np.r_[0,np.cumsum(seglen)]
    sf=np.arange(0,cs[-1],ds)
    front=np.c_[np.interp(sf,cs,FC[:,0]),np.interp(sf,cs,FC[:,1]),np.interp(sf,cs,FC[:,2])]
    Lf=cs[-1]
    def dirn(sv):
        if sv<A: return d0
        if sv<A+B:
            t=(sv-A)/B; t=t*t*(3-2*t); d=(1-t)*d0+t*d1; return d/np.linalg.norm(d)
        return d1
    srest=np.arange(0,Ltot+1e-9,ds)
    rest=[S.copy()]
    for sv in srest[1:]: rest.append(rest[-1]+dirn(sv)*ds)
    pts=np.vstack([front,np.array(rest)])
    ss_=np.r_[sf-Lf, srest]            # <0 while on the front section
    def rad(sv):
        r0,r1,r2=0.025,0.052,0.034
        if sv<s_m0: return r0
        if sv<s_m0+cone: t=(sv-s_m0)/cone; return r0+(r1-r0)*(t*t*(3-2*t))
        if sv<s_m0+cone+body: return r1
        if sv<s_m0+cone+body+taper: t=(sv-(s_m0+cone+body))/taper; return r1+(r2-r1)*(t*t*(3-2*t))
        return r2
    rr=np.array([rad(v) for v in ss_])
    T=np.gradient(pts,axis=0); T/=np.linalg.norm(T,axis=1,keepdims=True)
    nrm=np.cross(T[0],[0,1.0,0]); nrm/=np.linalg.norm(nrm)
    fr=[]
    for i_ in range(len(pts)):
        if i_>0:
            nrm=nrm-T[i_]*(nrm@T[i_]); nrm/=np.linalg.norm(nrm)
        fr.append((nrm.copy(),np.cross(T[i_],nrm)))
    seg=20; V=[];F=[]
    for i_,p in enumerate(pts):
        n_,b2=fr[i_]
        dr=(rr[min(i_+1,len(pts)-1)]-rr[max(i_-1,0)])/(2*ds)
        for j in range(seg+1):
            a=2*np.pi*j/seg; rad_dir=np.cos(a)*n_+np.sin(a)*b2
            nn=rad_dir-dr*T[i_]; nn/=np.linalg.norm(nn)
            V.append([*(p+rr[i_]*rad_dir),*nn,j/seg,(ss_[i_]-ss_[0])/(ss_[-1]-ss_[0]),*T[i_]])
    rows=seg+1
    i_split=int(np.argmax(pts[:,2]<-2.27))            # first ring that is out from under the body
    for i_ in range(len(pts)-1):
        for j in range(seg):
            a=i_*rows+j; bq=a+rows; F+=[a,bq,a+1,a+1,bq,bq+1]
    V=np.array(V,float); F=np.array(F,np.int64)
    global EXHAUST_TEX
    EXHAUST_TEX=make_exhaust_texture(pts,fr,ss_)
    Fall=F.reshape(-1,3)
    segidx=np.repeat(np.arange(len(pts)-1),seg*2)      # which ring-segment each triangle belongs to
    FA=Fall[segidx<i_split].reshape(-1); FB=Fall[segidx>=i_split].reshape(-1)
    VA=V[:(i_split+1)*rows]; FAo=fix_winding(VA,FA)
    off=i_split*rows; VB=V[off:]; FBo=fix_winding(VB,FB-off)
    # inside surface (whole tube): flipped normals, dark sooty material
    Vin=V.copy(); Vin[:,3:6]*=-1
    Fin=fix_winding(Vin,F)
    # rolled lip at the tip so the wall has thickness (belongs to the exposed part)
    pe=pts[-1]; de=T[-1]; n_,b2=fr[-1]; r_out=rr[-1]; r_in=r_out-0.0035
    Vl=[]
    for j in range(seg+1):
        a=2*np.pi*j/seg; rd=np.cos(a)*n_+np.sin(a)*b2
        Vl.append([*(pe+r_out*rd),*de,j/seg,1.0,*n_]); Vl.append([*(pe+r_in*rd),*de,j/seg,1.0,*n_])
    Vl=np.array(Vl,float); Fl=[]
    for j in range(seg):
        a=2*j; Fl+=[a,a+1,a+2,a+2,a+1,a+3]
    Fl=fix_winding(Vl,np.array(Fl,np.int64))
    VBo=np.vstack([VB,Vl]); FBo=np.concatenate([FBo,Fl+len(VB)])
    # black cap closing the far (front) end of the pipe, facing the opening
    c0=pts[0]+T[0]*0.02; n0,b0=fr[0]; Vc=[[*c0,*T[0],0.5,0.5,*n0]]
    for j in range(seg+1):
        a=2*np.pi*j/seg; q=c0+0.021*(np.cos(a)*n0+np.sin(a)*b0)
        Vc.append([*q,*T[0],0.5+0.5*np.cos(a),0.5+0.5*np.sin(a),*n0])
    Fc=[]
    for j in range(seg): Fc+=[0,1+j,2+j]
    Vc=np.array(Vc,float); Fc=fix_winding(Vc,np.array(Fc,np.int64))
    return VA,FAo,VBo,FBo,Vin,Fin,Vc,Fc

STEEL=len(mats)      # shiny case-hardened steel (appended after the existing materials)
MATTE=len(mats)+1    # matte version for the part under the body
def steel_material_bytes():
    o=s('exhaust_steel')+s('ksPerPixel')+b'\x00'*6
    props=[('ksAmbient',0.30),('ksDiffuse',0.70),('ksSpecular',1.1),('ksSpecularEXP',80.0),('ksEmissive',0.0),('ksAlphaRef',0.0)]
    o+=struct.pack('<i',len(props))
    for k,v in props: o+=s(k)+struct.pack('<f',v)+b'\x00'*36
    o+=struct.pack('<i',1)+s('txDiffuse')+struct.pack('<i',0)+s('exhaust_casehardened.png')
    return o
def matte_material_bytes():
    o=s('exhaust_steel_matte')+s('ksPerPixel')+b'\x00'*6
    props=[('ksAmbient',0.16),('ksDiffuse',0.60),('ksSpecular',0.22),('ksSpecularEXP',18.0),('ksEmissive',0.0),('ksAlphaRef',0.0)]
    o+=struct.pack('<i',len(props))
    for k,v in props: o+=s(k)+struct.pack('<f',v)+b'\x00'*36
    o+=struct.pack('<i',1)+s('txDiffuse')+struct.pack('<i',0)+s('exhaust_casehardened.png')
    return o
def texture_entry_bytes():
    import io
    buf_=io.BytesIO(); EXHAUST_TEX.save(buf_,format='PNG',optimize=True); png=buf_.getvalue()
    return struct.pack('<i',1)+s('exhaust_casehardened.png')+struct.pack('<i',len(png))+png
def material_count_pos():
    from kn5 import R
    r_=R(b); r_.raw(6); v_=r_.u('i')
    if v_>5: r_.u('i')
    n_=r_.u('i')
    for _ in range(n_):
        r_.u('i'); r_.st(); sz=r_.u('i'); r_.raw(sz)
    return r_.p

def build(out,damage=True):
    buf=bytearray(b); info={}
    # 1) bumper -> panel (+crash)
    i=IDX['rest_of_body']
    info['panel']=apply(buf,i,lambda w,idx:(total(panelwarp(w)) if damage else panelwarp(w)))
    for nm in ('rest_of_body_SUB1','rest_of_body_SUB2'):
        j=IDX[nm]; info[nm]=apply(buf,j,lambda w,idx:(total(panelwarp(w)) if damage else panelwarp(w)))
        nd=nodes[j]; p=nd['istart']+nd['ic']*2; struct.pack_into('<i',buf,p,15)
        # copy paint UVs from the nearest vertex of the original panel (matched on rear-view x,y)
        from scipy.spatial import cKDTree as _KD
        pn=nodes[i]; pvc=pn['vc']
        PV_=np.frombuffer(b,dtype='<f4',count=pvc*11,offset=pn['vstart']).reshape(pvc,11).astype(float)
        pw=(np.c_[PV_[:,:3],np.ones(pvc)]@world_M(i))[:,:3]
        sv=nd['vc']; SV=np.frombuffer(b,dtype='<f4',count=sv*11,offset=nd['vstart']).reshape(sv,11).astype(float)
        sw=(np.c_[SV[:,:3],np.ones(sv)]@world_M(j))[:,:3]
        _,nn_=_KD(pw[:,[0,1]]).query(sw[:,[0,1]])
        cur=np.frombuffer(buf,dtype='<f4',count=sv*11,offset=nd['vstart']).reshape(sv,11).astype(np.float64).copy()
        cur[:,6:8]=PV_[nn_,6:8]
        buf[nd['vstart']:nd['vstart']+sv*44]=cur.astype('<f4').tobytes()
    # panel face lookup for holes (final world positions)
    n=nodes[i]; vc=n['vc']; V=np.frombuffer(buf,dtype='<f4',count=vc*11,offset=n['vstart']).reshape(vc,11)
    wp=(np.c_[V[:,:3].astype(float),np.ones(vc)]@world_M(i))[:,:3]
    from scipy.spatial import cKDTree
    tr=cKDTree(wp[:,[0,1]])
    def facez(x,y):
        d,ids=tr.query([x,y],k=12); return wp[ids,2].min()
    # 2) crash on the rest of the body, lamps tilt
    if damage:
        for j,nd in enumerate(nodes):
            if nd['type']!=2 or nd['name'] in ('rest_of_body','rest_of_body_SUB1','rest_of_body_SUB2') or any(x in nd['name'] for x in EXCL) or nd['name']=='Exhaust': continue
            if nd['name']=='trunk': fn=lambda w,idx:trunk_extra(total(w))
            else: fn=lambda w,idx:total(w)
            c=apply(buf,j,fn,thr=1e-4)
            if c: info[nd['name']]=c
    # 3) exhaust
    # original can + tail pipe (everything behind z=-2.04): collapse those triangles so only the front pipe remains
    ex=nodes[IDX['Exhaust']]; evc=ex['vc']
    EV=np.frombuffer(b,dtype='<f4',count=evc*11,offset=ex['vstart']).reshape(evc,11).astype(float)
    ew=(np.c_[EV[:,:3],np.ones(evc)]@world_M(IDX['Exhaust']))[:,:3]
    eidx=np.frombuffer(buf,dtype='<u2',count=ex['ic'],offset=ex['istart']).copy().reshape(-1,3)
    cut=np.ones(len(eidx),bool)
    eidx[cut]=eidx[cut][:,[0,0,0]]
    buf[ex['istart']:ex['istart']+ex['ic']*2]=eidx.reshape(-1).astype('<u2').tobytes()
    info['exhaust_tris_removed']=int(cut.sum())
    # remove the rear tow hook (the 'Tsurikawa' dummy and its mesh)
    di=[k for k,nd_ in enumerate(nodes) if nd_['name']=='Tsurikawa' and nd_['type']==1][0]
    nxt=[k for k in range(di+1,len(nodes)) if nodes[k]['depth']<=nodes[di]['depth']]
    s0=nodes[di]['start']; e0=nodes[nxt[0]]['start'] if nxt else len(buf)
    buf_pre=buf                                   # untouched offsets: later steps that read the trunk must use this
    buf=bytearray(buf[:s0])+bytearray(buf[e0:])
    info['tow_hook_removed_bytes']=e0-s0
    # extra nodes to append
    V1,F1=bar_mesh([(0.436,0.325,facez(0.436,0.325)),(-0.39,0.32,facez(-0.39,0.32))])
    hole_specs=[(0.436,0.325,0.07,0.075),(-0.39,0.32,0.065,0.08),(0.65,0.30,0.03,0.07),(0.595,0.30,0.03,0.07),(0.54,0.30,0.03,0.07)]
    Vh=[];Fh=[];off=0
    for cx,cy,w_,h_ in hole_specs:
        zf=facez(cx,cy)-0.003
        Vq,Fq=quad(cx,cy,zf,w_,h_); Vh.append(Vq);Fh.append(Fq+off);off+=4
    Vh=np.vstack(Vh);Fh=np.concatenate(Fh)
    new=bytearray(buf)
    r0=nodes[0]['start']; nlen=struct.unpack_from('<i',new,r0+4)[0]; cp=r0+8+nlen
    cnt=struct.unpack_from('<i',new,cp)[0]
    struct.pack_into('<i',new,cp,cnt+8-1)
    new+=dummy_bytes('BASH_BAR',1)+mesh_bytes('BASH_BAR',V1,F1,1)
    new+=dummy_bytes('PANEL_HOLES',1)+mesh_bytes('PANEL_HOLES',Vh,Fh,4)
    VL,FL=liner_mesh(damage,buf_pre)
    new+=dummy_bytes('TRUNK_LINER',1)+mesh_bytes('TRUNK_LINER',VL,FL,12)
    VXa,FXa,VX,FX,VXi,FXi,VXc,FXc=exhaust_tail(damage)
    new+=dummy_bytes('EXHAUST_UNDER',1)+mesh_bytes('EXHAUST_UNDER',VXa,FXa,STEEL)
    new+=dummy_bytes('EXHAUST_TAIL',1)+mesh_bytes('EXHAUST_TAIL',VX,FX,STEEL)
    new+=dummy_bytes('EXHAUST_INNER',1)+mesh_bytes('EXHAUST_INNER',VXi,FXi,12)
    new+=dummy_bytes('EXHAUST_CAP',1)+mesh_bytes('EXHAUST_CAP',VXc,FXc,4)
    VP,FP=plate_fill()
    new+=dummy_bytes('PLATE_FILL',1)+mesh_bytes('PLATE_FILL',VP,FP,12)
    # root child count: removed 1, added 2  -> but dummy_bytes only counts one child each; root counts 2 new
    mcp=material_count_pos(); ins_at=nodes[0]['start']
    ins=steel_material_bytes()
    cntm=struct.unpack_from('<i',new,mcp)[0]; assert cntm==len(mats)
    new[ins_at:ins_at]=ins
    struct.pack_into('<i',new,mcp,cntm+1)
    te=texture_entry_bytes()
    new[mcp:mcp]=te                                   # textures end exactly where the material count starts
    struct.pack_into('<i',new,14,struct.unpack_from('<i',new,14)[0]+1)
    open(out,'wb').write(new)
    return info
# ---------- post-process: drop the (now unused) tow-hook texture and its material ----------
def strip_tow_hook_assets(path):
    import kn5 as K
    from kn5 import R
    data=bytearray(open(path,'rb').read())
    r=R(bytes(data)); r.raw(6); ver=r.u('i')
    if ver>5: r.u('i')
    ntex=r.u('i'); tex_spans=[]
    for _ in range(ntex):
        st=r.p; r.u('i'); name=r.st(); sz=r.u('i'); r.raw(sz); tex_spans.append((st,r.p,name))
    mat_count_pos=r.p; nm=r.u('i'); mat_spans=[]
    for _ in range(nm):
        st=r.p; name=r.st(); r.st(); r.raw(6); pc=r.u('i')
        for _ in range(pc): r.st(); r.u('f'); r.raw(36)
        tc=r.u('i'); texnames=[]
        for _ in range(tc): r.st(); r.u('i'); texnames.append(r.st())
        mat_spans.append((st,r.p,name,texnames))
    nodes_start=r.p
    tex_i=[k for k,t in enumerate(tex_spans) if t[2]=='Tsurikawa.png'][0]
    mat_i=[k for k,m in enumerate(mat_spans) if 'Tsurikawa.png' in m[3]]
    assert len(mat_i)==1, mat_i
    mat_i=mat_i[0]
    # parse nodes to find every mesh's material id
    r2=R(bytes(data)); r2.p=nodes_start
    nd=K.parse_nodes(r2,ver)
    used=[n['mat'] for n in nd if n['type']==2]
    assert mat_i not in used, 'material still used by a mesh'
    for n in nd:
        if n['type']==2 and n['mat']>mat_i:
            struct.pack_into('<i',data,n['istart']+n['ic']*2,n['mat']-1)
    struct.pack_into('<i',data,mat_count_pos,nm-1)
    struct.pack_into('<i',data,14,ntex-1)
    ms,me=mat_spans[mat_i][0],mat_spans[mat_i][1]
    del data[ms:me]
    ts,te=tex_spans[tex_i][0],tex_spans[tex_i][1]
    del data[ts:te]
    open(path,'wb').write(data)
    return te-ts, me-ms


def embed_paint(path):
    """Bake the silver body texture into the kn5 itself (same byte length as the white original)."""
    import config, paint
    from kn5 import R
    # take the pristine white texture from the ORIGINAL kn5 so re-running never tints twice
    ro=R(open(config.ORIGINAL_KN5,'rb').read()); ro.raw(6); v=ro.u('i')
    if v>5: ro.u('i')
    white=None
    for _ in range(ro.u('i')):
        ro.u('i'); name=ro.st(); sz=ro.u('i'); d=ro.raw(sz)
        if name=='carpaint_diffuse.dds': white=d
    new=paint.make_paint_dds(white)
    data=bytearray(open(path,'rb').read()); r=R(bytes(data)); r.raw(6); ver=r.u('i')
    if ver>5: r.u('i')
    for _ in range(r.u('i')):
        r.u('i'); name=r.st(); sz=r.u('i'); st=r.p; r.raw(sz)
        if name=='carpaint_diffuse.dds':
            assert len(new)==sz; data[st:st+sz]=new
    open(path,'wb').write(data)

if __name__=='__main__':
    import config
    out=str(config.OUTPUT_KN5); config.OUTPUT_KN5.parent.mkdir(parents=True,exist_ok=True)
    print('original :',config.ORIGINAL_KN5); print('output   :',out)
    info=build(out,True)
    print(info)
    print('stripped tow-hook texture/material bytes:',strip_tow_hook_assets(out))
    import wheels
    print('wheels replaced:',wheels.replace_wheels(out))
    import rotor
    print('rotors added:',rotor.replace_discs(out))
    embed_paint(out)
    print('done')
