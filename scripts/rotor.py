import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, struct
import kn5 as K
from wheels import revolve, fix_winding, mesh_bytes

# wheel-local (RIM frame) rotor: hat cup from the hub face to the caliper slot + friction ring
SPEC={'front':dict(x_hub=-0.0190,x0=0.0780,x1=0.1060,r_out=0.168,r_in=0.085,r_hat=0.066),
      'rear': dict(x_hub=-0.0190,x0=0.0660,x1=0.0940,r_out=0.160,r_in=0.080,r_hat=0.066)}
UV_PER_M=2.92          # the texture's disc fills the image: rotor outer radius -> 0.49 of the width

def rotor_geometry(kind):
    s=SPEC[kind]
    prof=[(s['x_hub'],0.0),(s['x_hub'],s['r_hat']-0.002),(s['x_hub']+0.002,s['r_hat']),
          (s['x0'],s['r_hat']),(s['x0'],s['r_out']-0.002),(s['x0']+0.002,s['r_out']),
          (s['x1']-0.002,s['r_out']),(s['x1'],s['r_out']-0.002),(s['x1'],s['r_in']),(s['x1'],0.0)]
    V,F=revolve(prof,nseg=112)
    # planar UVs through the axis so the disc image lands on the friction faces
    V[:,6]=0.5+V[:,1]*UV_PER_M
    V[:,7]=-0.5+V[:,2]*UV_PER_M
    # tangent along +y for the flat faces
    flat=np.abs(V[:,3])>0.7
    V[flat,8:11]=[0,1,0]
    return V,F

def replace_discs(path):
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
    for name,kind in (('Brake_DiscLF','front'),('Brake_Disc','front'),('Brake_Disc1','rear'),('Brake_DiscLR','rear')):
        i=[k for k,n in enumerate(nodes) if n['name']==name and n['type']==2][0]; n=nodes[i]
        par=nodes[n['parent']]; assert par['type']==1
        Md=np.array(par['matrix']).reshape(4,4); Mi=np.linalg.inv(Md); A=Md[:3,:3]
        V,F=rotor_geometry(kind)
        Vn=V.copy()
        Vn[:,:3]=(np.c_[V[:,:3],np.ones(len(V))]@Mi)[:,:3]            # RIM-frame -> this node's frame
        for sl in (slice(3,6),slice(8,11)):
            t=V[:,sl]@A.T; t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12); Vn[:,sl]=t
        Fo=fix_winding(Vn,F)
        end=nodes[i+1]['start'] if i+1<len(nodes) else len(data)
        jobs.append((n['start'],end,mesh_bytes(name,Vn,Fo,n['mat']),name,len(Vn),len(Fo)//3))
    for st,en,new,name,nv,nt_ in sorted(jobs,key=lambda t:-t[0]): data[st:en]=new
    open(path,'wb').write(data)
    return [(j[3],j[4],j[5]) for j in jobs]
