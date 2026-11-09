import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fbxread import read_fbx
def load_meshes(path):
    root,ver=read_fbx(path); obj=root.find('Objects')[0]; out={}
    for c in [c for c in obj.children if c.name=='Geometry']:
        v=np.array(c.find('Vertices')[0].props[0]).reshape(-1,3)
        pi=np.array(c.find('PolygonVertexIndex')[0].props[0])
        tris=[];cur=[]
        for i in pi:
            if i<0: cur.append(~i); 
            else: cur.append(i); continue
            for k in range(1,len(cur)-1): tris.append((cur[0],cur[k],cur[k+1]))
            cur=[]
        out[c.props[1].split('\x00')[0]]=(v,np.array(tris))
    return out
