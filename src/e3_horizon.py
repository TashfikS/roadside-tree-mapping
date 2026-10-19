"""Horizon row from the dominant vanishing point of oblique line segments (street direction)."""
import cv2, numpy as np
def horizon_row(path, work_w=1536, seed=0, viz=None):
    im=cv2.imread(path); H0,W0=im.shape[:2]; s=work_w/W0
    g=cv2.cvtColor(cv2.resize(im,(work_w,int(H0*s))),cv2.COLOR_BGR2GRAY)
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD); L=lsd.detect(g)[0]
    L=L.reshape(-1,4); d=L[:,2:]-L[:,:2]; ln=np.hypot(*d.T); ang=np.degrees(np.arctan2(np.abs(d[:,1]),np.abs(d[:,0])))
    k=(ln>work_w*0.03)&(ang>8)&(ang<75); L=L[k]; ln=ln[k]
    if len(L)<6: return None
    P1=np.c_[L[:,:2],np.ones(len(L))]; P2=np.c_[L[:,2:],np.ones(len(L))]; Ls=np.cross(P1,P2); Ls/=np.linalg.norm(Ls[:,:2],axis=1,keepdims=True)
    rng=np.random.default_rng(seed); best=(0,None)
    for _ in range(3000):
        i,j=rng.choice(len(L),2,replace=False); v=np.cross(Ls[i],Ls[j])
        if abs(v[2])<1e-9: continue
        v=v/v[2]
        if not (-0.5*work_w<v[0]<1.5*work_w): continue
        # angular consistency: segment direction vs direction to VP
        mid=(L[:,:2]+L[:,2:])/2; dv=v[:2]-mid; dl=L[:,2:]-L[:,:2]
        cosang=np.abs((dv*dl).sum(1))/(np.hypot(*dv.T)*np.hypot(*dl.T)+1e-9)
        inl=cosang>np.cos(np.radians(2.0)); sc=(ln*inl).sum()
        if sc>best[0]: best=(sc,v,inl)
    if best[1] is None: return None
    _,v,inl=best
    A=Ls[inl][:,:2]; b=-Ls[inl][:,2]; w=ln[inl]
    vp=np.linalg.lstsq(A*w[:,None],b*w,rcond=None)[0]
    if viz:
        out=cv2.resize(im,(work_w,int(H0*s)))
        for x0,y0,x1,y1 in L[inl].astype(int): cv2.line(out,(x0,y0),(x1,y1),(0,255,255),2)
        cv2.line(out,(0,int(vp[1])),(work_w,int(vp[1])),(0,0,255),2); cv2.circle(out,(int(vp[0]),int(vp[1])),10,(0,0,255),3)
        cv2.imwrite(viz,out)
    return dict(vp_u=vp[0]/s, vp_v=vp[1]/s, n_inliers=int(inl.sum()), W=W0, H=H0)
