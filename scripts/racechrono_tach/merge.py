import sys, os, glob, csv, json, math, numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib'))
from dialfit import fit, proj
def load_gps(path):
    rows=list(csv.reader(open(path,encoding='utf-8'))); hi=[i for i,r in enumerate(rows) if r and r[0]=='timestamp'][0]
    data=rows[hi+3:]; g=lambda j: np.array([float(r[j]) if j<len(r) and r[j]!='' else np.nan for r in data])
    t=g(0); k=np.isclose(t%1,0)
    return dict(t=t[k],el=g(3)[k],v=g(13)[k],alt=g(6)[k],acc=g(5)[k],dist=g(4)[k])
def load_tach(dirn):
    out=[]
    for f in sorted(glob.glob(dirn+'/RC_*.csv')):
        for r in csv.DictReader(open(f)):
            if r['rpm']=='': continue
            out.append([float(r['unix_time']),float(r['theta']),float(r['ecc']),float(r['snr']),float(r['peak'])])
    return np.array(out)
def mapper(calib, pivot_true, r_mid, delta=None):
    d=json.load(open(calib)); O=np.array(d['pivot'],float); P=np.asarray(pivot_true,float)
    H,dl,res,pf=fit(d['numerals'],delta=delta)
    def f(theta):
        th=np.radians(theta); X=O[None]+r_mid*np.c_[np.cos(th),np.sin(th)]   # point the scan ray locks onto
        phi=np.arctan2(X[:,1]-P[1],X[:,0]-P[0])                               # direction from true pivot
        Q=P[None]+150*np.c_[np.cos(phi),np.sin(phi)]+(pf-P)                   # parallax shift to dial-plane centre
        q=proj(np.linalg.inv(H),Q); psi=np.degrees(np.arctan2(q[:,1],q[:,0]))
        z=proj(np.linalg.inv(H),np.array(d['numerals']['0'],float)[None]+ (pf-P)*0)[0]
        psi0=np.degrees(np.arctan2(z[1],z[0]))
        dpsi=(psi-psi0+90)%360-90
        return dpsi/dl*1000
    def g(theta):  # piecewise-linear between numeral angles measured from the true pivot
        th=np.radians(theta); X=O[None]+r_mid*np.c_[np.cos(th),np.sin(th)]
        phi=np.degrees(np.arctan2(X[:,1]-P[1],X[:,0]-P[0]))%360
        ks=sorted(int(k) for k in d['numerals']); a=np.array([math.degrees(math.atan2(d['numerals'][str(k)][1]-P[1],d['numerals'][str(k)][0]-P[0]))%360 for k in ks])
        a=np.where(a<a[0]-1,a+360,a); phi=np.where(phi<a[0]-40,phi+360,phi)
        return np.interp(phi,a,np.array(ks)*1000.0,left=np.nan,right=np.nan)
    return f,g,dl,res
