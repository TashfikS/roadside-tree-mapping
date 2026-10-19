import os, math, json, requests, numpy as np, mapbox_vector_tile, collections
tok=os.environ['MAPILLARY']
r=requests.get("https://router.project-osrm.org/route/v1/driving/90.4074,23.7104;91.1809,23.4607;91.7832,22.3569",params=dict(overview='full',geometries='geojson'),timeout=120).json()
line=np.array(r['routes'][0]['geometry']['coordinates'])
R=6371008.8
def hv(a,b):
    la1,lo1,la2,lo2=map(np.radians,(a[...,1],a[...,0],b[...,1],b[...,0]))
    return 2*R*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
seg=hv(line[:-1],line[1:]); s=np.concatenate([[0],np.cumsum(seg)]); L=s[-1]
ss=np.arange(0,L,20.0); pts=np.stack([np.interp(ss,s,line[:,0]),np.interp(ss,s,line[:,1])],1)
z=14; n=2**z
def tile(lon,lat): return int((lon+180)/360*n), int((1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n)
tiles=sorted({tile(*p) for p in pts}); feats=[]
for (x,y) in tiles:
    q=requests.get(f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}",params=dict(access_token=tok),timeout=120)
    if not q.ok: continue
    t=mapbox_vector_tile.decode(q.content); ext=t.get('image',{}).get('extent',4096)
    for f in t.get('image',{}).get('features',[]):
        gx,gy=f['geometry']['coordinates']
        lon=(x+gx/ext)/n*360-180; yy=y+1-gy/ext
        lat=math.degrees(math.atan(math.sinh(math.pi*(1-2*yy/n))))
        feats.append((lon,lat,f['properties']['captured_at']))
F=np.array([(a,b) for a,b,_ in feats]); yr=np.array([c for _,_,c in feats])/1000/3.15576e7+1970
cell=0.0005; idx=collections.defaultdict(list)
for i,(a,b) in enumerate(F): idx[(int(a/cell),int(b/cell))].append(i)
cov=np.zeros(len(pts),bool); recent=np.zeros(len(pts),bool)
for k,(a,b) in enumerate(pts):
    cand=[i for dx in (-1,0,1) for dy in (-1,0,1) for i in idx.get((int(a/cell)+dx,int(b/cell)+dy),[])]
    if not cand: continue
    d=hv(np.array([a,b])[None,:],F[cand]); near=d<=25
    cov[k]=near.any(); recent[k]=(near & (yr[cand]>=2022)).any()
def merged(c,gap_km=0.5):
    rr=[];st=None
    for i,v in enumerate(list(c)+[False]):
        if v and st is None: st=i
        if not v and st is not None: rr.append((st*0.02,i*0.02)); st=None
    m=[]
    for a,b in rr:
        if m and a-m[-1][1]<=gap_km: m[-1]=(m[-1][0],b)
        else: m.append((a,b))
    return m
mr=merged(recent); ma=merged(cov)
res=dict(route_km=round(L/1000,1), images_in_tiles=len(F), covered_km=round(cov.sum()*0.02,1), covered_km_2022plus=round(recent.sum()*0.02,1),
  longest_stretch_km=round(max((b-a for a,b in ma),default=0),1), longest_stretch_2022plus_km=round(max((b-a for a,b in mr),default=0),1),
  top_stretches_2022plus=[(round(a,1),round(b,1)) for a,b in sorted(mr,key=lambda t:t[0]-t[1])[:5]])
json.dump(dict(res, line=line.tolist()),open('n1_coverage.json','w')); print(json.dumps(res))
