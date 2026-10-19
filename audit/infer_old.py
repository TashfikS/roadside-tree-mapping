"""Run the ORIGINAL thesis YOLOv4 darknet weights (OpenCV DNN 4.10, CPU) on a list of images.
Darknet-equivalent preprocessing: plain resize to the cfg input size (letter_box=0), RGB, /255.
Score floor 0.005 as in darknet `detector map` (OpenCV's [yolo] default floor is 0.2, so the cfg copy sets thresh=0.005);
class-wise NMS IoU 0.45. Writes <out.csv> (detections) and <out.csv>.done (every processed image, incl. zero detections).
Usage: python infer_old.py <cfg> <weights> <list.txt> <out.csv>   (resumable)"""
import sys, os, time, cv2, numpy as np
CFG, Wt, LST, OUT = sys.argv[1:5]
cv2.setNumThreads(int(os.environ.get('NT', '3')))
net = cv2.dnn.readNetFromDarknet(CFG, Wt); net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV); net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
names = net.getUnconnectedOutLayersNames()
W_IN = int([l.split('=')[1] for l in open(CFG) if l.strip().startswith('width')][0]); H_IN = int([l.split('=')[1] for l in open(CFG) if l.strip().startswith('height')][0])
paths = [l.strip() for l in open(LST) if l.strip()]
DONE = OUT + '.done'; done = set(open(DONE).read().splitlines()) if os.path.exists(DONE) else set()
if not os.path.exists(OUT): open(OUT, 'w').write('path,x0,y0,x1,y1,score\n')
t0 = time.time(); n = 0
with open(OUT, 'a') as fo, open(DONE, 'a') as fd:
    for p in paths:
        if p in done: continue
        im = cv2.imread(p)
        if im is None: print('UNREADABLE', p, flush=True); continue
        net.setInput(cv2.dnn.blobFromImage(im, 1 / 255.0, (W_IN, H_IN), swapRB=True, crop=False))
        outs = np.vstack(net.forward(names)); sc = outs[:, 5]; k = sc >= 0.005; o = outs[k]; sc = sc[k]
        bx = np.c_[o[:, 0] - o[:, 2] / 2, o[:, 1] - o[:, 3] / 2, o[:, 2], o[:, 3]]
        keep = cv2.dnn.NMSBoxes(bx.tolist(), sc.tolist(), 0.005, 0.45) if len(sc) else []
        for i in np.array(keep).flatten():
            x0, y0, w, h = bx[i]; fo.write(f"{p},{x0:.5f},{y0:.5f},{x0 + w:.5f},{y0 + h:.5f},{sc[i]:.5f}\n")
        fd.write(p + '\n'); n += 1
        if n % 25 == 0: fo.flush(); fd.flush(); print(n, len(paths), round((time.time() - t0) / n, 2), 's/img', flush=True)
print('done', n, round(time.time() - t0, 1), flush=True)
