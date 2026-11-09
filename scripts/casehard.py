import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image
def tex(N=1024,seed=11):
    rng=np.random.default_rng(seed)
    def noise(sig,aniso=1.0):
        a=rng.standard_normal((N,N))
        a=gaussian_filter(a,(sig*aniso,sig),mode='wrap')
        a-=a.mean(); a/=a.std()+1e-9; return a
    yy,xx=np.mgrid[0:N,0:N].astype(float)
    # domain warp (tileable): swirly marbling
    wx=noise(34)*36+noise(12)*10; wy=noise(34)*36+noise(12)*10
    def sample(f): return map_coordinates(f,[(yy+wy)%N,(xx+wx)%N],order=1,mode='wrap')
    f=noise(48)*1.0+noise(21)*0.75+noise(8)*0.4+noise(3)*0.12
    f=sample(f)
    f=(f-f.min())/(f.max()-f.min())
    band=0.5+0.5*np.sin(f*np.pi*6.0+noise(14)*0.8)
    t=0.6*f+0.4*band
    t=(t-t.min())/(t.max()-t.min())
    # muted, steel-dominated ramp: gunmetal base with blue / plum / straw / bronze case colours
    stops=[(0.00,(58,60,68)),(0.14,(78,80,90)),(0.26,(56,78,122)),(0.36,(92,76,118)),(0.46,(150,128,88)),
           (0.56,(118,88,66)),(0.68,(98,100,110)),(0.84,(132,134,142)),(1.00,(168,170,178))]
    xs=[s_[0] for s_ in stops]
    img=np.stack([np.interp(t,xs,[s_[1][c] for s_ in stops]) for c in range(3)],-1)
    grain=noise(0.9)[...,None]*2.5+noise(2.2)[...,None]*1.8
    # faint machining streaks along the pipe (v axis = rows)
    streak=gaussian_filter(rng.standard_normal((N,N)),(0.4,28),mode='wrap'); streak/=streak.std()
    img=img*1.12+grain+streak[...,None]*2.0
    return Image.fromarray(np.clip(img,0,255).astype(np.uint8),'RGB')
if __name__=='__main__':
    im=tex(); im.save('casehard_preview.png'); print(im.size)
    a=np.array(im).reshape(-1,3).mean(0); print('mean rgb',a.round(0))
