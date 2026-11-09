import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, struct
from kn5 import *
import config
if not config.ORIGINAL_KN5.exists():
    raise SystemExit('Original model not found: %s\nPut a pristine copy of the original mod\'s ggh_toyota_cressida_mx83_stage_1.kn5 there (or set CRESSIDA_ORIGINAL_KN5).'%config.ORIGINAL_KN5)
P=str(config.ORIGINAL_KN5)
b,r,ver,tex,mats=parse(P); nodes=parse_nodes(r,ver)
def world(i):
    M=np.eye(4)
    chain=[]; j=i
    while j!=-1: chain.append(j); j=nodes[j]['parent']
    for j in reversed(chain):
        n=nodes[j]
        if n['type']==1:
            m=np.array(n['matrix']).reshape(4,4)
            M=M@m   # row-vector convention: child local * parent
    return M
def world_M(i):
    # row vector: v_world = v @ M_child @ M_parent @ ... @ M_root
    chain=[]; j=i
    while j!=-1: chain.append(j); j=nodes[j]['parent']
    M=np.eye(4)
    for j in chain:   # child first
        n=nodes[j]
        if n['type']==1: M=M@np.array(n['matrix']).reshape(4,4)
    return M
def verts(i):
    n=nodes[i]; vc=n['vc']
    a=np.frombuffer(b,dtype='<f4',count=vc*11,offset=n['vstart']).reshape(vc,11)
    return a
