import sys, os, numpy as np, csv, json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from merge import *
sess, gps_csv, tach_dir, calib, pivot, r_mid, ecc_min, snr_min, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), sys.argv[9]
MAP=sys.argv[10] if len(sys.argv)>10 else 'interp'
G=load_gps(gps_csv); A=load_tach(tach_dir)
fM,fI,dl,res=mapper(calib,pivot,r_mid,delta=28.0 if MAP=='model28' else None)
t,th,ecc,snr,pk=A.T
rpmM=fM(th); rpmI=fI(th)
if MAP=='interp': rpmM,rpmI=rpmI,rpmM
good=(ecc>=ecc_min)&(snr>=snr_min)&(rpmM>-300)&(rpmM<8200)
# spike filter: compare each good sample with median of good samples within +-1.0 s
idx=np.where(good)[0]; spk=np.zeros(len(t),bool)
for j,i in enumerate(idx):
    w=idx[max(0,j-4):j+5]; w=w[np.abs(t[w]-t[i])<=1.0]
    if len(w)>=3 and abs(rpmM[i]-np.median(rpmM[w]))>350: spk[i]=True
good&=~spk
print(f"[{sess}] spike-rejected samples: {spk.sum()}")
print(f"[{sess}] tach samples {len(t)}, pass quality {good.sum()} ({good.mean()*100:.0f}%), dial delta {dl:.2f} deg/1000rpm")
# GPS accel at 1 Hz (central diff), interpolate to tach times
v=G['v']; acc=np.gradient(v,G['t'])
def at(tq,lag): return np.interp(tq-lag,G['t'],v,left=np.nan,right=np.nan)
TIRE_R=0.316; K=60/(2*np.pi*TIRE_R)   # rpm per (m/s) per unit overall ratio
W58=np.array([3.285,1.894,1.275,1.0,0.783]); FD=3.90
pred=K*W58*FD
# lag search: maximise fraction of samples within 3% of a predicted gear line (shape only; robust to scale via best-scale too)
def score(rpm,lag,scale=1.0):
    vs=at(t,lag); m=good&(vs>4)&~np.isnan(rpm)
    nv=rpm[m]/vs[m]; rel=np.min(np.abs(nv[:,None]/(pred[None]*scale)-1),axis=1)
    return np.mean(rel<0.03), m.sum()
for name,rpm in (("model",rpmM),("interp",rpmI)):
    if np.all(np.isnan(rpm)): continue
    best=max(((score(rpm,l,s)[0],l,s) for l in np.arange(-1.5,2.51,0.25) for s in np.arange(0.85,1.151,0.01)))
    print(f"  mapping={name}: best lag {best[1]:+.2f}s scale {best[2]:.2f} -> {best[0]*100:.0f}% of samples within 3% of a W58 gear line")
# choose model mapping; find lag with scale fixed at 1
lags=np.arange(-1.5,2.51,0.25); sc=[max(score(rpmM,l,x)[0] for x in np.arange(0.85,1.151,0.01)) for l in lags]; lag=lags[int(np.argmax(sc))]
print(f"  primary mapping={MAP}; lag scan (best scale per lag):",' '.join(f"{l:+.2f}:{s*100:.0f}%" for l,s in zip(lags,sc)))
vs=at(t,lag); m=good&(vs>4)
nv=rpmM/vs
# histogram peaks of N/V
h,e=np.histogram(nv[m],bins=np.arange(60,450,2)); c=(e[1:]+e[:-1])/2
from scipy.signal import find_peaks
hs=np.convolve(h,np.ones(3)/3,'same'); pk_i,_=find_peaks(hs,height=max(15,0.01*m.sum()),distance=8)
print("  N/V histogram peaks (rpm per m/s):",[(round(c[i]),int(hs[i])) for i in pk_i])
print("  predicted W58 x 3.90, r=0.316:",pred.round(1))
# assign gear: nearest predicted within 4% using measured peaks if close
gear=np.zeros(len(t),int)
for gi,p in enumerate(pred):
    near=[c[i] for i in pk_i if abs(c[i]/p-1)<0.08]
    ref=near[0] if near else p
    sel=m&(np.abs(nv/ref-1)<0.04)&(rpmM>900)
    gear[sel]=gi+1
# refine per gear: median N/V among steady samples
print("  per-gear N/V (median, IQR, n) and implied overall ratio with r=0.316 m, then gear ratio /3.90:")
stats={}
for gi in range(1,6):
    s=gear==gi
    if s.sum()<20: print(f"   gear {gi}: n={s.sum()} (insufficient)"); continue
    q=np.percentile(nv[s],[25,50,75]); ov=q[1]/K
    stats[gi]=(q[1],ov,s.sum())
    print(f"   gear {gi}: {q[1]:.1f} [{q[0]:.1f}-{q[2]:.1f}] n={s.sum()}  overall {ov:.3f}  gear {ov/FD:.3f} (W58 {W58[gi-1]})  speed range {vs[s].min()*2.237:.0f}-{vs[s].max()*2.237:.0f} mph, rpm {rpmM[s].min():.0f}-{rpmM[s].max():.0f}")
if 4 in stats:
    print("  ratios relative to 4th (independent of tyre size & final drive):", {g:round(stats[g][0]/stats[4][0],3) for g in stats}, " W58:",{g:round(W58[g-1],3) for g in stats})
print("  band-matched N/V (median) by engine-rpm band:")
for lo in range(1000,4500,500):
    cells=[]
    for gi in range(2,6):
        mm=(gear==gi)&(rpmM>=lo)&(rpmM<lo+500)
        cells.append(f"g{gi} {np.median(nv[mm]):6.1f} n{mm.sum():<4d}" if mm.sum()>=8 else f"g{gi}    -        ")
    print(f"    {lo}-{lo+500}: "+" | ".join(cells))
# linearity check within 5th/4th: N/V vs rpm slope (offset errors show as trend)
for gi in (4,5,3):
    s=gear==gi
    if s.sum()>50:
        b=np.polyfit(vs[s],rpmM[s],1); print(f"  gear {gi} fit rpm = {b[0]:.1f}*v + {b[1]:.0f}  (intercept != 0 => tach offset/nonlinearity)")
# idle
idle=good&(at(t,lag)<0.3)
if idle.sum(): print(f"  stationary samples n={idle.sum()}: rpm median {np.median(rpmM[idle]):.0f}, p10-p90 {np.percentile(rpmM[idle],10):.0f}-{np.percentile(rpmM[idle],90):.0f}")
print(f"  max rpm seen (quality pass): {np.nanmax(rpmM[good]):.0f}; p99 {np.nanpercentile(rpmM[good],99):.0f}")
# coverage gaps
tt=np.sort(t[good]); gaps=np.diff(tt); big=np.where(gaps>2)[0]
print(f"  gaps >2s in good RPM: {len(big)}, total {gaps[big].sum():.0f}s; longest {gaps.max():.1f}s; GPS span {G['t'][-1]-G['t'][0]:.0f}s, good-RPM coverage ~{(len(tt)*0.25):.0f}s")
# write combined
el0=G['t'][0]-G['el'][0]
with open(out,'w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['unix_time','elapsed_s','rpm_tach_video','rpm_quality_ok','rpm_alt_mapping','gps_speed_mps_lagcorr','gps_speed_mph_lagcorr','gps_accel_g_1hz','nv_rpm_per_mps','est_gear','reg_ecc','needle_snr','gps_accuracy_m','altitude_m'])
    ac=np.interp(t-lag,G['t'],acc)/9.81; accm=np.interp(t-lag,G['t'],G['acc']); alt=np.interp(t-lag,G['t'],G['alt'])
    for i in range(len(t)):
        w.writerow([f"{t[i]:.3f}",f"{t[i]-el0:.3f}",f"{rpmM[i]:.0f}",int(good[i]),'' if np.isnan(rpmI[i]) else f"{rpmI[i]:.0f}",f"{vs[i]:.3f}",f"{vs[i]*2.23694:.2f}",f"{ac[i]:.3f}",'' if not (vs[i]>1) else f"{nv[i]:.1f}",gear[i] if gear[i] else '',f"{ecc[i]:.3f}",f"{snr[i]:.1f}",f"{accm[i]:.2f}",f"{alt[i]:.0f}"])
print("  wrote",out)
