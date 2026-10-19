import os, glob, json, random, numpy as np
from PIL import Image
R=glob.glob('/kaggle/input/**/roadside-trees-of-bangladesh-street-view', recursive=True)[0]
src={'BD_740':f'{R}/Train/BD_740','BD_755':f'{R}/Train/BD_755','BD_Test_Set_270':f'{R}/Test/BD_Test_Set_270'}
recs=[]
def dhash(im,n=16):
    g=np.asarray(im.convert('L').resize((n+1,n),Image.BILINEAR),dtype=np.int16); return (g[:,1:]>g[:,:-1]).flatten()
def thumb(im): return np.asarray(im.convert('RGB').resize((32,16),Image.BILINEAR),dtype=np.float32).flatten()
for k,d in src.items():
    for p in sorted(glob.glob(d+'/*.jpg'), key=lambda x:int(os.path.basename(x).split('.')[0])):
        im=Image.open(p); nb=sum(1 for l in open(p[:-4]+'.txt') if l.strip())
        recs.append(dict(folder=k,id=int(os.path.basename(p).split('.')[0]),w=im.size[0],h=im.size[1],n_boxes=nb,dh=dhash(im),th=thumb(im)))
N=len(recs); H=np.stack([r['dh'] for r in recs]).astype(np.uint8); T=np.stack([r['th'] for r in recs])
T=(T-T.mean(1,keepdims=True))/(T.std(1,keepdims=True)+1e-6)
ham=(H[:,None,:]!=H[None,:,:]).sum(2); corr=(T@T.T)/T.shape[1]
np.fill_diagonal(ham,999); np.fill_diagonal(corr,-1)
# does ID order reflect route continuity? consecutive vs random pairs
order=[(i,i+1) for i in range(N-1) if recs[i]['folder']==recs[i+1]['folder'] and recs[i+1]['id']-recs[i]['id']==1]
rng=np.random.default_rng(0); rp=rng.integers(0,N,(5000,2)); rp=rp[rp[:,0]!=rp[:,1]]
stats=dict(N=N, consecutive_pairs=len(order),
  ham_consec_median=float(np.median([ham[i,j] for i,j in order])), ham_random_median=float(np.median(ham[rp[:,0],rp[:,1]])),
  corr_consec_median=float(np.median([corr[i,j] for i,j in order])), corr_random_median=float(np.median(corr[rp[:,0],rp[:,1]])))
# near-duplicate links (strict): dHash<=20/256 or thumbnail corr>=0.97
dup=(ham<=20)|(corr>=0.97); iu=np.argwhere(np.triu(dup,1))
isTest=np.array([r['folder']=='BD_Test_Set_270' for r in recs])
stats.update(near_dup_pairs=int(len(iu)), imgs_with_near_dup=int(dup.any(1).sum()),
  test_imgs_with_train_near_dup=int((dup[isTest][:,~isTest]).any(1).sum()),
  test_imgs_nearest_train_corr_median=float(np.median(corr[isTest][:,~isTest].max(1))),
  train_imgs_nearest_other_train_corr_median=float(np.median(np.where(np.eye(N,dtype=bool),-1,corr)[~isTest][:,~isTest].max(1))))
# grouping: contiguous 25-ID blocks per folder, unioned with near-dup links
par=list(range(N))
def f(x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def u(a,b): par[f(a)]=f(b)
blk={}
for i,r in enumerate(recs):
    key=(r['folder'],r['id']//25)
    if key in blk: u(i,blk[key])
    else: blk[key]=i
for a,b in iu: u(int(a),int(b))
groups={}
for i in range(N): groups.setdefault(f(i),[]).append(i)
G=list(groups.values()); random.Random(42).shuffle(G)
tot=sum(sum(recs[i]['n_boxes'] for i in g) for g in G); acc=0; split={}
for g in G:
    s='train' if acc<0.70*tot else ('val' if acc<0.85*tot else 'test')
    for i in g: split[i]=s
    acc+=sum(recs[i]['n_boxes'] for i in g)
stats.update(n_groups=len(G), largest_group=max(len(g) for g in G))
for s in ['train','val','test']:
    ii=[i for i in range(N) if split[i]==s]; stats[f'{s}_images']=len(ii); stats[f'{s}_boxes']=int(sum(recs[i]['n_boxes'] for i in ii))
# leakage check on new split
lab=np.array([split[i] for i in range(N)])
stats['new_split_test_imgs_with_train_near_dup']=int(dup[lab=='test'][:,lab=='train'].any(1).sum())
import csv
with open('/kaggle/working/split_route_proxy.csv','w',newline='') as fh:
    w=csv.writer(fh); w.writerow(['folder','id','width','height','n_boxes','group','split'])
    for i,r in enumerate(recs): w.writerow([r['folder'],r['id'],r['w'],r['h'],r['n_boxes'],f(i),split[i]])
json.dump(stats,open('/kaggle/working/split_stats.json','w'),indent=1); print(json.dumps(stats,indent=1))
