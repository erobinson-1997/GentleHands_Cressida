import numpy as np, math
from scipy.optimize import least_squares
def proj(H,P):
    P=np.c_[P,np.ones(len(P))]@H.T; return P[:,:2]/P[:,2:3]
def fit(numerals, delta=None, pivot=None, PW=0.5):
    ks=np.array(sorted(int(k) for k in numerals)); img=np.array([numerals[str(k)] for k in ks],float)
    def unpack(x):
        H=np.append(x[:8],1).reshape(3,3); d=x[8] if delta is None else delta; return H,d
    def res(x):
        H,d=unpack(x); ph=np.radians(ks*d); dial=np.c_[np.cos(ph),np.sin(ph)]
        r=(proj(H,dial)-img).ravel()
        if pivot is not None: r=np.r_[r,PW*(proj(H,np.zeros((1,2)))[0]-pivot)]
        return r
    c=img.mean(0); s=np.ptp(img,0).max()/2
    best=None
    for d0 in ([delta] if delta else [26,28,30,32,34]):
        for rot in range(0,360,30):
            for flip in (1,-1):
                a=math.radians(rot); R=np.array([[math.cos(a),-flip*math.sin(a)],[math.sin(a),flip*math.cos(a)]])*s
                x0=np.r_[R[0],c[0],R[1],c[1],0,0]
                if delta is None: x0=np.r_[x0,d0]
                try: r=least_squares(res,x0,method='lm')
                except Exception: continue
                if best is None or r.cost<best.cost: best=r
    H,d=unpack(best.x); 
    ph=np.radians(ks*d); resid=proj(H,np.c_[np.cos(ph),np.sin(ph)])-img
    return H,d,np.hypot(*resid.T),proj(H,np.zeros((1,2)))[0]
def rpm_from_image_angle(H,d,theta_deg,pivot_img,r=100):
    th=np.radians(theta_deg); P=np.c_[pivot_img[0]+r*np.cos(th),pivot_img[1]+r*np.sin(th)]
    q=proj(np.linalg.inv(H),P); psi=np.degrees(np.arctan2(q[:,1],q[:,0]))
    psi=np.where(psi<-90, psi+360, psi)  # keep continuity; numeral 0 at psi=0, increasing
    return psi/d*1000
def make_lut(numerals, pivot_true, thetas, r=150, delta=None):
    H,d,res,pf=fit(numerals,delta=delta)
    th=np.radians(thetas); P=np.c_[pivot_true[0]+r*np.cos(th),pivot_true[1]+r*np.sin(th)] + (pf-np.asarray(pivot_true))
    q=proj(np.linalg.inv(H),P); psi=np.degrees(np.arctan2(q[:,1],q[:,0]))
    psi=np.unwrap(np.radians(psi)); psi=np.degrees(psi)
    k0=np.argmin(np.abs(thetas-np.degrees(np.arctan2(*(np.array(numerals['0'])-pivot_true)[::-1]))%360))
    psi=psi-360*np.round((psi[k0]-0)/360)
    return psi/d*1000, d, res, pf
