import cv2, numpy as np, json, sys, os, math
class Tach:
    def __init__(s, calib_path):
        d=json.load(open(calib_path)); base=os.path.dirname(os.path.abspath(calib_path))
        s.d=d; s.sc=d['scale']
        ref=cv2.imread(os.path.join(base,d['ref_image']))
        x0,y0,x1,y1=d['tmpl_box']; s.c0=np.array([x0,y0],float)
        g=cv2.cvtColor(ref[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
        s.tmpl=cv2.resize(g,None,fx=1/s.sc,fy=1/s.sc,interpolation=cv2.INTER_AREA).astype(np.float32)
        s.tmpl=cv2.GaussianBlur(s.tmpl,(5,5),0)
        m=np.full(s.tmpl.shape,255,np.uint8)
        px,py=(np.array(d['pivot'])-s.c0)/s.sc
        cv2.circle(m,(int(px),int(py)),int(d['mask_r']/s.sc),0,-1); s.mask=m
        s.piv=np.array(d['pivot'],float)
        # calibration: image angle of each numeral -> rpm
        ang=[]; 
        for k,(x,y) in d['numerals'].items():
            ang.append((math.degrees(math.atan2(y-s.piv[1],x-s.piv[0]))%360,int(k)*1000))
        ang.sort(key=lambda a:a[1]); s.cal_ang=np.array([a[0] for a in ang]); s.cal_rpm=np.array([a[1] for a in ang],float)
        s.cal_ang=np.where(s.cal_ang<s.cal_ang[0]-1,s.cal_ang+360,s.cal_ang)
        s.H=None
        s.thetas=np.arange(s.cal_ang[0]-15,s.cal_ang[-1]+10,0.5)
        r=np.arange(d['needle_r'][0],d['needle_r'][1],2.0)
        s.R,s.TH=np.meshgrid(r,np.radians(s.thetas))
        s.ray_pts=np.stack([s.piv[0]+s.R*np.cos(s.TH), s.piv[1]+s.R*np.sin(s.TH)],-1)  # ref full-res
    def register(s, frame):
        g=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        g=cv2.resize(g,None,fx=1/s.sc,fy=1/s.sc,interpolation=cv2.INTER_AREA).astype(np.float32)
        g=cv2.GaussianBlur(g,(5,5),0)
        crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-5)
        inits=[]
        if s.H is not None: inits.append(s.H.copy())
        # translation search via template matching (whole template)
        res=cv2.matchTemplate(g,s.tmpl,cv2.TM_CCOEFF_NORMED); _,_,_,loc=cv2.minMaxLoc(res)
        inits.append(np.array([[1,0,loc[0]],[0,1,loc[1]],[0,0,1]],np.float32))
        best=(-1,None)
        for H0 in inits:
            try:
                # coarse affine then homography
                A=H0[:2].copy()
                m=cv2.warpPerspective(s.mask,H0.astype(np.float64),(g.shape[1],g.shape[0]),flags=cv2.INTER_NEAREST)
                cc,A=cv2.findTransformECC(s.tmpl,g,A,cv2.MOTION_AFFINE,crit,m,5)
                H=np.vstack([A,[0,0,1]]).astype(np.float32)
                m=cv2.warpPerspective(s.mask,H.astype(np.float64),(g.shape[1],g.shape[0]),flags=cv2.INTER_NEAREST)
                cc,H=cv2.findTransformECC(s.tmpl,g,H,cv2.MOTION_HOMOGRAPHY,crit,m,5)
            except cv2.error:
                continue
            if cc>best[0]: best=(cc,H)
            if cc>0.9: break
        if best[1] is not None: s.H=best[1]
        return best[0]
    def to_frame(s, P):
        q=(P-s.c0)/s.sc
        sh=q.shape; q=q.reshape(-1,1,2).astype(np.float32)
        p=cv2.perspectiveTransform(q,s.H).reshape(sh)*s.sc
        return p
    def read(s, frame):
        cc=s.register(frame)
        if s.H is None: return None
        P=s.to_frame(s.ray_pts).astype(np.float32)
        smp=cv2.remap(frame,P[...,0],P[...,1],cv2.INTER_LINEAR).astype(np.float32)
        B,G,R=smp[...,0],smp[...,1],smp[...,2]
        score=(R-B)  # salmon needle: red-dominant; gray face & white numerals ~0
        prof=np.median(score,axis=1)  # median along ray -> robust to small blobs
        i=int(np.argmax(prof)); pk=prof[i]
        # contrast: peak vs profile excluding +-8 deg around peak
        far=np.abs(s.thetas-s.thetas[i])>8
        bg=np.median(prof[far]); noise=np.std(prof[far])+1e-6
        # sub-sample peak refine
        if 0<i<len(prof)-1:
            a,b,c=prof[i-1],prof[i],prof[i+1]; den=a-2*b+c; off=0.5*(a-c)/den if den!=0 else 0
        else: off=0
        th=s.thetas[i]+off*0.5
        rpm=float(np.interp(th,s.cal_ang,s.cal_rpm,left=np.nan,right=np.nan))
        if th<s.cal_ang[0]: rpm=(th-s.cal_ang[0])/(s.cal_ang[1]-s.cal_ang[0])*1000  # below 0 mark
        return dict(rpm=rpm,theta=th,ecc=cc,peak=float(pk-bg),snr=float((pk-bg)/noise))
    def overlay(s, frame, r):
        f=frame.copy()
        piv=s.to_frame(s.piv[None])[0]
        th=math.radians(r["theta"]); L=s.d["needle_r"][1]; tip=s.to_frame(np.array([[s.piv[0]+L*math.cos(th),s.piv[1]+L*math.sin(th)]]))[0]
        cv2.line(f,tuple(piv.astype(int)),tuple(tip.astype(int)),(0,255,0),2)
        for k,(x,y) in s.d['numerals'].items():
            p=s.to_frame(np.array([[x,y]],float))[0]; cv2.circle(f,tuple(p.astype(int)),12,(255,255,0),2)
        cv2.putText(f,f"{r['rpm']:.0f} rpm ecc {r['ecc']:.2f} snr {r['snr']:.1f}",(20,60),0,1.5,(0,255,255),3)
        return f
