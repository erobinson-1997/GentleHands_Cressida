"""Silver body paint, baked into the car's own carpaint_diffuse.dds (DXT5, 2048x2048, 12 mips).
Mip levels are built from the colour channels only: this texture's alpha is all zero, and resampling RGBA would
premultiply the colour by that alpha and turn every smaller mip black (the 'fades to black with distance' bug).
The original 128-byte DDS header is reused byte for byte."""
import io, numpy as np
from PIL import Image
TINT=np.array([0xB5,0xBA,0xBD])/255.0*0.45*np.array([0.90,0.97,1.10])  # = ~#49515E cool dark silver, picked in-game 2026-11-08 (test skin f_cool_darker)
# Lighter values (e.g. #B5B8BC) read as WHITE in-game under AC's auto exposure. Keep skins/silver_176/metal_detail.dds
# neutral white: the game multiplies the detail colour onto this, so a coloured detail tints the paint twice.

def make_paint_dds(orig):
    src=Image.open(io.BytesIO(orig)).convert('RGBA')
    a=np.array(src).astype(float); alpha=a[...,3].astype(np.uint8)
    rgb=Image.fromarray(np.clip(a[...,:3]*TINT[None,None,:],0,255).astype(np.uint8),'RGB')
    def block(rgba):
        b=io.BytesIO(); rgba.save(b,format='DDS',pixel_format='DXT5'); return b.getvalue()[128:]
    levels=[]; w=2048
    while w>=1:
        if w>=4:
            lv=rgb if w==2048 else rgb.resize((w,w),Image.BOX if w>=512 else Image.LANCZOS)
            al=Image.fromarray(alpha) if w==2048 else Image.fromarray(alpha).resize((w,w),Image.NEAREST)
            levels.append(block(Image.merge('RGBA',(*lv.split(),al))))
        else:                                    # 2x2 and 1x1 are stored as one 4x4 block
            px=rgb.resize((w,w),Image.BOX).resize((4,4),Image.NEAREST)
            levels.append(block(Image.merge('RGBA',(*px.split(),Image.new('L',(4,4),int(alpha.max()))))))
        w//=2
    out=bytes(orig[:128])+b''.join(levels)
    assert len(out)==len(orig),'size differs from the original texture'
    return out
