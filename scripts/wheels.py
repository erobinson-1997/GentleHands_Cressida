import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, struct
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import kn5 as K

def fix_winding(V,F):
    F=np.asarray(F).reshape(-1,3).copy()
    p=V[:,:3]; n=V[:,3:6]
    fn=np.cross(p[F[:,1]]-p[F[:,0]],p[F[:,2]]-p[F[:,0]])
    bad=(fn*n[F].mean(1)).sum(1)<0
    F[bad]=F[bad][:,[0,2,1]]
    return F.reshape(-1)
def sstr(x): e=x.encode(); return struct.pack('<i',len(e))+e
def mesh_bytes(name,V,F,mat):
    c=V[:,:3].mean(0); rad=np.linalg.norm(V[:,:3]-c,axis=1).max()
    o=struct.pack('<i',2)+sstr(name)+struct.pack('<i',0)+b'\x01'+bytes([1,1,0])
    o+=struct.pack('<i',len(V))+V.astype('<f4').tobytes()
    o+=struct.pack('<i',len(F))+np.asarray(F).astype('<u2').tobytes()
    o+=struct.pack('<ii',mat,0)+struct.pack('<2f',0,0)+struct.pack('<4f',*c,rad)+b'\x01'
    return o

# ---------------------------------------------------------------- geometry
def revolve(profile,nseg=56):
    P=np.array(profile,float)
    area=0.5*np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1])
    sgn=1.0 if area>0 else -1.0
    V=[];F=[]
    for i in range(len(P)):
        a=P[i]; b=P[(i+1)%len(P)]
        d=b-a; L=np.hypot(*d)
        if L<1e-9 or (a[1]<1e-9 and b[1]<1e-9): continue
        n=sgn*np.array([d[1],-d[0]])/L              # outward normal in the (x,r) plane
        base=len(V)
        for j in range(nseg+1):
            ph=2*np.pi*j/nseg; c,s=np.cos(ph),np.sin(ph)
            for q in (a,b):
                V.append([q[0],q[1]*c,q[1]*s, n[0],n[1]*c,n[1]*s, j/nseg,0.0, 0.0,-s,c])
        for j in range(nseg):
            k=base+2*j
            F+=[k,k+1,k+2,k+2,k+1,k+3]
    V=np.array(V,float); return V,np.array(F,np.int64)
def quad(p0,p1,p2,p3,nrm):
    nrm=np.asarray(nrm,float); nrm=nrm/np.linalg.norm(nrm)
    e=np.asarray(p1,float)-np.asarray(p0,float); e/=np.linalg.norm(e)
    V=np.array([[*p,*nrm,0,0,*e] for p in (p0,p1,p2,p3)],float)
    return V,np.array([0,1,2,0,2,3])
def hex_nut(cy,cz,x_base,x_top,rad,rot=0.0):
    ang=rot+np.arange(6)*np.pi/3
    ring=[(cy+rad*np.cos(a),cz+rad*np.sin(a)) for a in ang]
    Vs=[];Fs=[];off=0
    def add(V,F):
        nonlocal off
        Vs.append(V);Fs.append(F+off);off+=len(V)
    for i in range(6):
        a=ring[i]; b=ring[(i+1)%6]
        mid=((a[0]+b[0])/2-cy,(a[1]+b[1])/2-cz)
        add(*quad((x_top,*a),(x_top,*b),(x_base,*b),(x_base,*a),(0,mid[0],mid[1])))
    # top cap (faces outward = -x)
    V=[[x_top,cy,cz,-1,0,0,0,0,0,1,0]]
    for a in ring: V.append([x_top,a[0],a[1],-1,0,0,0,0,0,1,0])
    F=[]
    for i in range(6): F+=[0,1+i,1+(i+1)%6]
    add(np.array(V,float),np.array(F))
    return np.vstack(Vs),np.concatenate(Fs)

# ------------------------------------------------------------------
# Proportions measured from the reference wheel (units there -> metres here)
KS=0.2136/2.24                 # reference lip radius 2.24 units -> our 0.2136 m lip
X_PEAK=-0.121                 # outermost point of the domed face (local x, outward = -x)
ZREF_R=[0.25,0.40,0.50,0.60,0.75,0.97,1.12,1.28,1.43,1.58,1.73,1.88,2.00,2.20]
ZREF_Z=[-0.03,0.0,0.145,0.18,0.212,0.217,0.21,0.197,0.169,0.138,0.108,0.079,0.06,0.03]
SEP_R=[0.8,0.95,1.18,1.32,1.48,1.63,1.78,1.93,2.2]
SEP_L=[0.32,0.46,0.467,0.428,0.422,0.39,0.388,0.374,0.36]
def x_front(rho):  return X_PEAK+(0.217-np.interp(np.asarray(rho)/KS,ZREF_R,ZREF_Z))*KS
def sep_half(rho): return 0.5*np.interp(np.asarray(rho)/KS,SEP_R,SEP_L)*KS
W_TOP,W_BASE=0.0165,0.0220
def thick(rho):    return np.interp(rho,[0.09,0.17,0.2085],[0.0181,0.0143,0.0035])
RHO0,RHO1=0.078,0.2085

def ribbon(c0,tdir,rho,off,xf,xb,wt,wb,chamfer=0.35):
    wt=np.broadcast_to(np.asarray(wt,float),np.shape(rho)).copy(); wb=np.broadcast_to(np.asarray(wb,float),np.shape(rho)).copy()
    """sweep a trapezoid-ish cross-section along a straight line; returns V,F,bright-mask"""
    n=len(rho)
    ctr=np.stack([c0[0]*rho+tdir[0]*off, c0[1]*rho+tdir[1]*off],1)       # (y,z) of the centre line
    xm=xf+chamfer*(xb-xf)
    def P(x,d): return np.c_[x,ctr+d[:,None]*np.array(tdir)[None,:]]
    one=np.ones(n)
    TL,TR=P(xf,-0.5*wt),P(xf,0.5*wt)
    MR,ML=P(xm,0.5*wb),P(xm,-0.5*wb)
    BR,BL=P(xb,0.5*wb),P(xb,-0.5*wb)
    cen=np.c_[(xf+xb)/2,ctr]
    strips=[(TL,TR,False),(TR,MR,True),(MR,BR,False),(BR,BL,False),(BL,ML,False),(ML,TL,True)]
    Vs=[];Fs=[];Ms=[];off_=0
    for A,B,br in strips:
        dA=np.gradient(A,axis=0); nr=np.cross(dA,B-A); nr/=np.maximum(np.linalg.norm(nr,axis=1,keepdims=True),1e-12)
        mid=(A+B)/2; flip=((nr*(mid-cen)).sum(1)<0); nr[flip]*=-1
        V=[]
        for i in range(n):
            tg=dA[i]/max(np.linalg.norm(dA[i]),1e-12)
            V.append([*A[i],*nr[i],0,0,*tg]); V.append([*B[i],*nr[i],1,0,*tg])
        F=[]
        for i in range(n-1):
            kk=2*i; F+=[kk,kk+1,kk+2,kk+1,kk+3,kk+2]
        Vs.append(np.array(V,float));Fs.append(np.array(F)+off_);Ms.append(np.full(len(V),br));off_+=len(V)
    return np.vstack(Vs),np.concatenate(Fs),np.concatenate(Ms)

def wheel_barrel():
    outer=[(-0.1170,0.2060),(-0.1145,0.2125),(-0.1090,0.2136),(-0.1040,0.2105),(-0.1015,0.2000),(-0.0980,0.1935),
           (0.1000,0.1935),(0.1035,0.2000),(0.1060,0.2105),(0.1115,0.2136),(0.1165,0.2125),(0.1190,0.2060)]
    inner=[(0.1190,0.2015),(0.1120,0.2050),(0.1050,0.2000),(0.0980,0.1883),
           (-0.0980,0.1883),(-0.1050,0.2000),(-0.1120,0.2050),(-0.1170,0.2015)]
    return revolve(outer+inner,nseg=96)

def wheel_face(phase=np.pi/2):
    Vs=[];Fs=[];Ms=[];off=0
    def add(V,F,bright=False,mask=None):
        nonlocal off
        Vs.append(V);Fs.append(np.asarray(F)+off);off+=len(V)
        Ms.append(mask if mask is not None else np.full(len(V),bright))
    # barrel: lip flanges polished, well painted
    Vb,Fb=wheel_barrel(); rb=np.hypot(Vb[:,1],Vb[:,2]); add(Vb,Fb,mask=(rb>0.2040))
    rho=np.linspace(RHO0,RHO1,36)
    for k in range(5):
        a=phase+k*2*np.pi/5; c0=(np.cos(a),np.sin(a)); td=(-np.sin(a),np.cos(a))
        xf=x_front(rho); xb=xf+thick(rho)
        flare=1.0+0.75*np.clip((0.150-rho)/0.075,0,1)**1.3     # arms flare into the hub web
        for sg in (1,-1):
            V,F,M=ribbon(c0,td,rho,sg*sep_half(rho),xf,xb,W_TOP*flare,W_BASE*flare); add(V,F,mask=M)
    # hub web (full disc, recessed centre) + domed centre cap + nuts
    rr=np.linspace(0.044,0.092,9); front=[(x_front(r_),r_) for r_ in rr]
    add(*revolve(front+[(-0.0780,0.092),(-0.0780,0.044)],nseg=72),bright=False)
    add(*revolve([(-0.1065,0.0),(-0.1055,0.020),(-0.1022,0.032),(-0.0990,0.0380),(-0.0800,0.0380),(-0.0800,0.0)],nseg=64),bright=True)
    for k in range(5):
        a=phase+k*2*np.pi/5
        xb_=float(x_front(0.0572))
        add(*hex_nut(0.0572*np.cos(a),0.0572*np.sin(a),xb_,xb_-0.0080,0.0095,rot=a),bright=True)
    V=np.vstack(Vs); F=np.concatenate(Fs); M=np.concatenate(Ms)
    return V,F,M


# ------------------------------------------------------------------
# Wheel adapted from "Advan AVS 5" by gunis.david (https://skfb.ly/oE7Rn), CC BY 4.0
# (scaled, re-oriented, re-shaded and re-textured for Assetto Corsa)
import config
REF_FBX=str(config.REF_FBX)
def ref_wheel(path=REF_FBX,protrude=0.5,crease_deg=38.0):
    from fbxmesh import load_meshes
    Vr,Tr=load_meshes(path)['Circle.001']
    KR=0.2136/2.24; S_AX=0.236/2.227
    phi=np.radians(-18.0)                       # put a spoke pair at the top
    x=Vr[:,0]*np.cos(phi)-Vr[:,1]*np.sin(phi); y=Vr[:,0]*np.sin(phi)+Vr[:,1]*np.cos(phi); z=Vr[:,2]
    zc=np.where(z>0.018,0.018+(z-0.018)*protrude,z)          # keep the dome from sticking out past the tyre
    P=np.c_[-0.117+(0.018-zc)*S_AX, y*KR, x*KR]
    T=Tr.copy()
    def tri_normals(P,T):
        n=np.cross(P[T[:,1]]-P[T[:,0]],P[T[:,2]]-P[T[:,0]]); a=np.linalg.norm(n,axis=1); return n/np.maximum(a[:,None],1e-15),a
    fn,area=tri_normals(P,T)
    cen=P[T].mean(1); rc=np.hypot(cen[:,1],cen[:,2])
    front=(cen[:,0]<-0.115)&(rc>0.05)&(rc<0.17)&(np.abs(fn[:,0])>0.9)
    if (fn[front,0]<0).mean()<0.5: T=T[:,[0,2,1]]; fn=-fn          # make winding agree with outward (-x) faces
    # smooth normals with a crease angle, welded per (vertex, normal)
    cosc=np.cos(np.radians(crease_deg))
    vt=[[] for _ in range(len(P))]
    for ti,t in enumerate(T):
        for v in t: vt[v].append(ti)
    corner_n=np.zeros((len(T),3,3))
    for ti,t in enumerate(T):
        for c,v in enumerate(t):
            adj=np.array(vt[v]); d=fn[adj]@fn[ti]; sel=adj[d>cosc]
            nn=(fn[sel]*area[sel,None]).sum(0); l=np.linalg.norm(nn)
            corner_n[ti,c]=nn/l if l>1e-15 else fn[ti]
    # polished (bright) vs painted: slanted/curved surfaces in the front 2 cm are polished, flat tops are paint
    n_out=-fn[:,0]
    bright_t=(cen[:,0]<-0.100)&(n_out<0.93)&(rc>0.05)
    key={}; V=[];F=[];Mk=[]
    for ti,t in enumerate(T):
        for c,v in enumerate(t):
            nq=tuple(np.round(corner_n[ti,c]*40).astype(int)); k=(v,nq,bool(bright_t[ti]))
            if k not in key:
                key[k]=len(V); n_=corner_n[ti,c]
                tg=np.cross(n_,[1.0,0,0]); l=np.linalg.norm(tg); tg=tg/l if l>1e-6 else np.array([0,1.0,0])
                V.append([*P[v],*n_,0,0,*tg]); Mk.append(bright_t[ti])
            F.append(key[k])
    return np.array(V,float),np.array(F,np.int64),np.array(Mk,bool)

def wheel_from_reference():
    Vs=[];Fs=[];Ms=[];off=0
    def add(V,F,mask):
        nonlocal off
        Vs.append(V);Fs.append(np.asarray(F)+off);off+=len(V);Ms.append(mask)
    V,F,M=ref_wheel(); add(V,F,M.astype(int))
    # polished centre cap filling the bore, sitting just proud of the hub
    V,F=revolve([(-0.1180,0.0),(-0.1170,0.010),(-0.1148,0.019),(-0.1120,0.0242),(-0.0950,0.0242),(-0.0950,0.0)],nseg=64)
    add(V,F,np.ones(len(V),int))
    # five hex nuts in the recessed seats (circle radius 0.0458 m, directly under each spoke pair)
    for k in range(5):
        a=np.pi/2+k*2*np.pi/5          # local frame: +z is up, spoke pair 0 points up
        yz=(0.0458*np.sin(a),0.0458*np.cos(a))
        V,F=hex_nut(yz[0],yz[1],-0.1032,-0.1152,0.0085,rot=a)
        add(V,F,np.ones(len(V),int))
    # two stacked bolt-on spacers bridging the gap between the car's hub face (x=-0.019) and the wheel's
    # mounting face (x=-0.0975); dark anodised look
    def spacer(x_back,x_front,rad,ch=0.0025):
        return revolve([(x_back,0.0),(x_back,rad-ch),(x_back-ch,rad),(x_front+ch,rad),(x_front,rad-ch),(x_front,0.0)],nseg=72)
    add(*spacer(-0.0190,-0.0580,0.0600),np.full(len(spacer(-0.0190,-0.0580,0.06)[0]),2))
    V,F=spacer(-0.0580,-0.0975,0.0620); add(V,F,np.full(len(V),2))
    V,F=revolve([(-0.0975,0.0),(-0.0975,0.0225),(-0.1075,0.0225),(-0.1075,0.0)],nseg=40)   # centring nub into the wheel bore
    add(V,F,np.full(len(V),2))
    return np.vstack(Vs),np.concatenate(Fs),np.concatenate(Ms)

# ---------------------------------------------------------------- file surgery
KEEP_SIZES={2800,847,693}      # barrel + the two small trim pieces; everything else is the old face
GUN={'front':(0.6816,-0.3770),'rear':(0.1836,-0.2246)}        # flat ~116 grey, low reflection (painted gunmetal)
POL={'front':(0.7959,-0.5117),'rear':(0.7959,-0.4482)}
DARK={'front':(0.5811,-0.9502),'rear':(0.5811,-0.9502)}      # flat ~38 grey (dark anodised)        # flat ~146 grey, full reflection (polished)
def replace_wheels(path):
    data=bytearray(open(path,'rb').read())
    r=K.R(bytes(data)); r.raw(6); ver=r.u('i')
    if ver>5: r.u('i')
    nt=r.u('i')
    for _ in range(nt):
        r.u('i'); r.st(); sz=r.u('i'); r.raw(sz)
    nm=r.u('i')
    for _ in range(nm):
        r.st(); r.st(); r.raw(6); pc=r.u('i')
        for _ in range(pc): r.st(); r.u('f'); r.raw(36)
        tc=r.u('i')
        for _ in range(tc): r.st(); r.u('i'); r.st()
    nodes=K.parse_nodes(r,ver)
    jobs=[]
    for name,ax in (('SSR_MKIILF','front'),('SSR_MKII','front'),('SSR_MKIII','rear'),('SSR_MKIIILR','rear')):
        i=[k for k,n in enumerate(nodes) if n['name']==name and n['type']==2][0]; n=nodes[i]; vc=n['vc']
        V=np.frombuffer(bytes(data),dtype='<f4',count=vc*11,offset=n['vstart']).reshape(vc,11).astype(float)
        idx=np.frombuffer(bytes(data),dtype='<u2',count=n['ic'],offset=n['istart']).astype(int).reshape(-1,3)
        key=np.round(V[:,:3]*1e4).astype(np.int64); _,wid=np.unique(key,axis=0,return_inverse=True); wid=wid.ravel()
        e=np.r_[np.c_[wid[idx[:,0]],wid[idx[:,1]]],np.c_[wid[idx[:,1]],wid[idx[:,2]]]]
        g=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(wid.max()+1,)*2)
        nc,lab=connected_components(g,directed=False); vl=lab[wid]; sizes=np.bincount(vl)
        # the entire old wheel (barrel, star, bolts, nuts, trim) is replaced
        Vf,Ff,Mf=wheel_from_reference(); Vf=Vf.copy()
        Vf[:,6],Vf[:,7]=GUN[ax]
        Vf[Mf==1,6],Vf[Mf==1,7]=POL[ax]
        Vf[Mf==2,6],Vf[Mf==2,7]=DARK[ax]
        Vo=Vf; Fo=fix_winding(Vo,Ff)
        end=nodes[i+1]['start'] if i+1<len(nodes) else len(data)
        jobs.append((n['start'],end,mesh_bytes(name,Vo,Fo,n['mat']),name,len(Vo),len(Fo)//3))
    for st,en,new,name,nv,nt_ in sorted(jobs,key=lambda t:-t[0]):
        data[st:en]=new
    open(path,'wb').write(data)
    return [(j[3],j[4],j[5]) for j in jobs]
