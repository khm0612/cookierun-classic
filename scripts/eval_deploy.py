"""Deployment-sim sweep over the held-out val tail: confidence threshold x N-frame
persistence x jump-cooldown, reporting event hit-rate and false fires/min — pick the
operating point for the live test without retraining."""
import os, json, sys
from _runtime import DATA
import numpy as np, cv2, torch
from cookierun_bot.policies.learned import build_net_from_meta

REC = str(DATA / "demo")
META = json.load(open(os.path.join(REC, "model_meta.json")))
CLASSES = META["classes"]
fm = json.load(open(os.path.join(REC, "frames.json")))
frames = sorted(fm["frames"], key=lambda f: f["idx"])
keys = json.load(open(os.path.join(REC, "keys.json")))
ts = np.array([f["t"] for f in frames])
y = np.zeros(len(frames), np.int64)
for k in keys:
    cls = CLASSES.index(k["action"])
    lo = np.searchsorted(ts, k["t"] - META["win_pre"])
    hi = np.searchsorted(ts, k["t"] + META["win_post"])
    y[lo:hi] = cls
n = len(frames); cut = int(n * 0.85)

x0f, y0f, x1f, y1f = META["crop"]
H, W, K = META["H"], META["W"], META["K"]
imgs = np.zeros((n - cut + K, H, W), np.uint8)          # val tail + K warmup frames
for j, i in enumerate(range(cut - K, n)):
    im = cv2.imread(os.path.join(REC, "frames", f"{frames[i]['idx']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    h, w = im.shape
    imgs[j] = cv2.resize(im[int(h*y0f):int(h*y1f), int(w*x0f):int(w*x1f)], (W, H),
                         interpolation=cv2.INTER_AREA)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = build_net_from_meta(torch, META).to(dev)
net.load_state_dict(torch.load(os.path.join(REC, "model.pt"), map_location=dev)); net.eval()
X = np.stack([imgs[j:j+K] for j in range(n - cut)]).astype(np.float32) / 255.0
with torch.no_grad():
    P = []
    for b in range(0, len(X), 512):
        P.append(torch.softmax(net(torch.from_numpy(X[b:b+512]).to(dev)), 1).cpu().numpy())
P = np.concatenate(P)
yv = y[cut:]
pred = P.argmax(1); pmax = P.max(1)
dur_min = (ts[-1] - ts[cut]) / 60.0

events, i = [], 0
while i < len(yv):
    if yv[i] != 0:
        j = i
        while j + 1 < len(yv) and yv[j + 1] == yv[i]: j += 1
        events.append((i, j, yv[i])); i = j + 1
    else: i += 1

print(f"val tail: {len(yv)} frames / {dur_min:.1f} min / {len(events)} human action events")
print(f"{'conf':>5} {'persist':>7} | {'events hit':>10} | {'false/min':>9} | {'w/ 0.8s jump cooldown':>21}")
for conf in (0.6, 0.7, 0.8, 0.9, 0.95):
    for persist in (1, 2, 3):
        fire_raw = (pred != 0) & (pmax > conf)
        # persistence: require `persist` consecutive fire frames of the same class
        fire = fire_raw.copy()
        for p_ in range(1, persist):
            fire[p_:] &= fire_raw[:-p_] & (pred[p_:] == pred[:len(pred)-p_])
            fire[:p_] = False
        hits = sum(1 for a, b, c in events
                   if np.any(fire[max(0,a-2):b+3] & (pred[max(0,a-2):b+3] == c)))
        fam = (fire & (yv == 0)).sum() / dur_min
        # cooldown sim: after a fired jump, suppress jumps 0.8s (~28 frames @35fps)
        fired, cd = [], 0
        for t in range(len(fire)):
            if cd > 0: cd -= 1
            if fire[t] and pred[t] == 1:
                if cd == 0: fired.append(t); cd = 28
            elif fire[t]: fired.append(t)
        fmask = np.zeros(len(fire), bool); fmask[fired] = True
        hits_cd = sum(1 for a, b, c in events
                      if np.any(fmask[max(0,a-2):b+3] & (pred[max(0,a-2):b+3] == c)))
        fam_cd = (fmask & (yv == 0)).sum() / dur_min
        print(f"{conf:>5} {persist:>7} | {hits:>4}/{len(events):<5} | {fam:>9.0f} | "
              f"hits {hits_cd}/{len(events)} false/min {fam_cd:.0f}")
