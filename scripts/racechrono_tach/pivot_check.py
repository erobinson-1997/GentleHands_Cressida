import cv2, sys, numpy as np, os
sys.path.insert(0,sys.argv[1]+'/lib'); from tach import Tach
T=Tach(sys.argv[2]); vid=sys.argv[3]; secs=[float(x) for x in sys.argv[4].split(',')]
x0,y0,x1,y1=T.d['tmpl_box']
X,Y=np.meshgrid(np.arange(x0,x1,1.0),np.arange(y0,y1,1.0)); grid=np.stack([X,Y],-1)
cap=cv2.VideoCapture(vid); lines=[]
for s in secs:
    cap.set(cv2.CAP_PROP_POS_MSEC,s*1000); ok,f=cap.read(); T.H=None; r=T.read(f)
    P=T.to_frame(grid).astype(np.float32); A=cv2.remap(f,P[...,0],P[...,1],cv2.INTER_LINEAR).astype(np.float32)
    sc=A[...,2]-A[...,0]
    d=np.hypot(X-T.piv[0],Y-T.piv[1]); ang=np.degrees(np.arctan2(Y-T.piv[1],X-T.piv[0]))%360
    da=(ang-r['theta']%360+180)%360-180
    r0,r1=T.d["needle_r"]; m=(sc>25)&(d>r0*0.8)&(d<r1*1.1)&(np.abs(da)<10)
    pts=np.c_[X[m],Y[m]]
    if len(pts)<50: print(s,'few pts'); continue
    L=cv2.fitLine(pts.astype(np.float32),cv2.DIST_HUBER,0,0.01,0.01).ravel(); vt=np.array([[L[0],L[1]],[-L[1],L[0]]]); c=L[2:]
    sv=[0,np.std((pts-c)@vt[1])*np.sqrt(len(pts))]
    lines.append((c,vt[0])); print(s,'npts',len(pts),'angle',round(np.degrees(np.arctan2(*vt[0][::-1]))%180,1),'width px',round(sv[1]/np.sqrt(len(pts))*2,1))
# least squares intersection
A=np.zeros((2,2)); b=np.zeros(2)
for c,dv in lines:
    n=np.array([-dv[1],dv[0]]); A+=np.outer(n,n); b+=np.outer(n,n)@c
p=np.linalg.solve(A,b); print('needle-line intersection (true pivot):',p.round(1))
for c,dv in lines:
    n=np.array([-dv[1],dv[0]]); print('  dist of line from pivot',round(abs((p-c)@n),1),' from clicked',round(abs((T.piv-c)@n),1))
