"""Run detectors on the 150 highway TEST photos (hw_score_test.py reads the output). Refuses to run before the final
model choice is recorded in hw_final_model.json. Usage: python hw_test_predict.py RUN WEIGHTS [RUN WEIGHTS ...]"""
import os, sys, json
import pandas as pd, torch
assert os.path.exists('hw_final_model.json'), 'final model not chosen yet'
from ultralytics import YOLO
torch.set_num_threads(4)
ids = pd.read_csv('hw_test_answers_pass1.csv', dtype={'id': str}).id.unique()
paths = [f'hw/img/{i}.jpg' for i in ids]; assert all(os.path.exists(p) for p in paths)
for run, w in zip(sys.argv[1::2], sys.argv[2::2]):
    m = YOLO(w); rows = []
    for b in range(0, len(paths), 8):
        for r in m.predict(paths[b:b + 8], imgsz=1024, conf=0.05, device='cpu', verbose=False):
            H, W = r.orig_shape; k = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for j, ((a, bb, c, d), s) in enumerate(zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())):
                rows.append(dict(id=os.path.basename(r.path)[:-4], x0=a, y0=bb, x1=c, y1=d, score=s,
                                 ku=k[j, 0, 0] / W if k is not None else 0, kv=k[j, 0, 1] / H if k is not None else 0, kconf=k[j, 0, 2] if k is not None else 0))
    pd.DataFrame(rows, columns=['id', 'x0', 'y0', 'x1', 'y1', 'score', 'ku', 'kv', 'kconf']).to_csv(f'hw_test_preds_{run}.csv', index=False)
    print(run, len(rows), 'boxes on', len(paths), 'photos', flush=True)
