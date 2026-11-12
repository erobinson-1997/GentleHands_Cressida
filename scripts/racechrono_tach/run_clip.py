import cv2, sys, os, csv, json, time
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib')); from tach import Tach
calib, vid, start_unix, outdir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
step=int(sys.argv[5]) if len(sys.argv)>5 else 15
T=Tach(calib); cap=cv2.VideoCapture(vid); base=os.path.basename(vid)[:-4]
os.makedirs(outdir+'/qa',exist_ok=True)
w=csv.writer(open(f"{outdir}/{base}.csv","w",newline="")); w.writerow(["unix_time","video_s","frame","rpm","theta","ecc","peak","snr"])
i=0; t0=time.time(); lastqa=-999
while True:
    if i%step:
        if not cap.grab(): break
        i+=1; continue
    ok,f=cap.read()
    if not ok: break
    vs=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    r=T.read(f)
    if r is None: w.writerow([f"{start_unix+vs:.3f}",f"{vs:.3f}",i,"","","","",""])
    else:
        w.writerow([f"{start_unix+vs:.3f}",f"{vs:.3f}",i,f"{r['rpm']:.0f}",f"{r['theta']:.2f}",f"{r['ecc']:.3f}",f"{r['peak']:.1f}",f"{r['snr']:.1f}"])
        if vs-lastqa>=30 and T.H is not None:
            o=T.overlay(f,r); cv2.imwrite(f"{outdir}/qa/{base}_{vs:06.1f}.jpg",cv2.resize(o,(o.shape[1]//3,o.shape[0]//3))); lastqa=vs
        if r['ecc']<0.35: T.H=None
    i+=1
print(base,"done",i,"frames",round(time.time()-t0),"s")
