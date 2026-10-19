import os
import math, random, sys, numpy as np
sys.path.insert(0,'.'); import treegeo as tg
rng=random.Random(0); errs=[]
for _ in range(2000):
    cam=dict(lat=23.75+rng.uniform(-.01,.01), lon=90.37+rng.uniform(-.01,.01), compass=rng.uniform(0,360), h_cam=rng.uniform(1.4,3.0), pitch=rng.uniform(-5,5))
    W,H=4000,3000; cam.update(f_px=tg.focal_px_from_mapillary(0.8,W,H), cx=W/2, cy=H/2)
    D=rng.uniform(4,40); rel=rng.uniform(-30,30)            # true tree: range D, bearing offset rel
    lat,lon=tg.destination(cam['lat'],cam['lon'],cam['compass']+rel,D)
    # forward pinhole projection of trunk base (ground point) -> pixel
    e,n=tg.to_enu(lat,lon,cam['lat'],cam['lon']); b=math.radians(cam['compass'])
    fwd= e*math.sin(b)+n*math.cos(b); right= e*math.cos(b)-n*math.sin(b)
    p=math.radians(cam['pitch']); h=cam['h_cam']
    zc=fwd*math.cos(p)-h*math.sin(p); yc=fwd*math.sin(p)+h*math.cos(p)
    u=cam['cx']+cam['f_px']*right/zc; v=cam['cy']+cam['f_px']*yc/zc
    r=tg.locate_tree(u,v,cam)
    errs.append(tg.haversine_m(lat,lon,r['lat'],r['lon']))
print('perspective round-trip max err m', round(max(errs),4))
sys.path.insert(0,os.path.join(os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection'), 'PythonAPI'))
import json, image_extract as ie
P=os.path.join(os.environ.get('PASADENA_DIR', 'data/external/PasadenaUrbanTrees_Detection'), 'PasadenaUrbanTrees')
panos=json.load(open(P+'/panos.json')); ex=ie.StreetViewExtractor(P,panos); ex.earth_radius=ie.EARTH_RADIUS; keys=list(panos)[:300]; e2=[]; e3=[]
for k in keys:
    pn=panos[k]; la0=float(pn['Location']['original_lat']); lo0=float(pn['Location']['original_lng'])
    D=rng.uniform(4,30); brg=rng.uniform(0,360); la,lo=tg.destination(la0,lo0,brg,D)
    x,y=ex.geo_coords_to_streetview_pixel(pn,la,lo)
    W=int(pn['Data']['image_width'])/2**(int(pn['Location']['zoomLevels'])-2); H=int(pn['Data']['image_height'])/2**(int(pn['Location']['zoomLevels'])-2)
    r=tg.pano_pixel_to_geo(x%W,y,W,H,la0,lo0,float(pn['Projection']['pano_yaw_deg']))
    e2.append(tg.haversine_m(la,lo,r['lat'],r['lon']))
    la2,lo2=ex.streetview_pixel_to_geo_coords(pn,x,y); e3.append(tg.haversine_m(la2,lo2,r['lat'],r['lon']))
print('pano: mine vs truth max err m', round(max(e2),3), '| mine vs dataset code max diff m', round(max(e3),3))
