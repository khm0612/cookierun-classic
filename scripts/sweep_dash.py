"""Live TRAINING dashboard — real-time learning curves for a running sweep.py.

Reads `sweep_progress.jsonl` (written per-epoch by sweep.py: {t:'epoch',loss} and
{t:'eval',score}) and serves a localhost page that plots, live and auto-refreshing:
  * held-out SCORE vs epoch, one line per config (the learning graph)
  * training LOSS vs epoch, one line per config
  * a status header (phase, configs done, current config/epoch, GPU util/mem)
  * a leaderboard (best score per config)

  python scripts/sweep_dash.py            # http://127.0.0.1:8848
Open the URL; no coupling to the sweep beyond the log file. Ctrl+C to stop.
"""
from __future__ import annotations
import http.server
import socketserver
import json
import sys
import subprocess

from _runtime import ROOT

PROG = ROOT / "sweep_progress.jsonl"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8848


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout.strip()
        u, m, t = [x.strip() for x in out.split(",")]
        return {"util": int(u), "mem": int(m), "total": int(t)}
    except Exception:
        return None


def parse():
    start, configs, order = None, {}, []
    if PROG.exists():
        for line in PROG.read_text(errors="ignore").splitlines():
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("t") == "start":
                start = o
            elif o.get("t") == "epoch":
                c = o["config"]
                if c not in configs:
                    configs[c] = {"K": o.get("K"), "epochs": [], "evals": []}
                    order.append(c)
                configs[c]["epochs"].append([o["epoch"], o["loss"]])
            elif o.get("t") == "eval":
                c = o["config"]
                if c not in configs:
                    configs[c] = {"K": o.get("K"), "epochs": [], "evals": []}
                    order.append(c)
                configs[c]["evals"].append([o["epoch"], o["score"], o.get("hits"), o.get("events"), o.get("fam")])
    total = start["total"] if start else 0
    done = total > 0 and len(order) >= total and all(
        configs[c]["epochs"] and configs[c]["epochs"][-1][0] >= (start.get("epochs", 30)) for c in order)
    cur = order[-1] if order else None
    cur_ep = configs[cur]["epochs"][-1][0] if cur and configs[cur]["epochs"] else 0
    return {"start": start, "order": order, "configs": configs, "total": total,
            "done": done, "cur": cur, "cur_ep": cur_ep, "gpu": gpu()}


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Training — live</title><style>
 body{background:#0f1216;color:#dfe6ee;font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;padding:16px}
 h1{font-size:17px;margin:0 0 2px} .sub{color:#8b98a5;font-size:12px;margin-bottom:12px}
 .pill{display:inline-block;padding:2px 10px;border-radius:20px;background:#1d2634;color:#9fb2c6;margin-left:6px;font-size:12px}
 .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
 canvas{background:#151a21;border-radius:10px}
 .card{background:#151a21;border-radius:10px;padding:10px 12px}
 table{border-collapse:collapse;font-size:12px} th,td{padding:4px 8px;text-align:right} th{color:#8b98a5;border-bottom:1px solid #263041}
 td:first-child,th:first-child{text-align:left}
 .leg{display:flex;flex-wrap:wrap;gap:10px;font-size:11px;color:#9fb2c6;margin:6px 0}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;vertical-align:middle}
 .bar{height:6px;border-radius:3px;background:#263041;overflow:hidden;margin-top:5px} .bar>i{display:block;height:100%;background:#57d977}
</style></head><body>
<h1>CookieRun training — live <span class=pill id=prog>…</span> <span class=pill id=stat></span> <span class=pill id=gpu></span></h1>
<div class=sub>held-out score &amp; training loss per epoch · updates every 1.5s</div>
<div class=bar><i id=pbar style=width:0%></i></div>
<div class=leg id=leg></div>
<div class=row>
 <div class=card><div style="font-size:12px;color:#8b98a5;margin-bottom:4px">held-out score vs epoch (higher = better)</div><canvas id=score width=560 height=300></canvas></div>
 <div class=card><div style="font-size:12px;color:#8b98a5;margin-bottom:4px">training loss vs epoch (lower = better)</div><canvas id=loss width=560 height=300></canvas></div>
</div>
<div class=card style=margin-top:14px;display:inline-block><table id=tbl><thead><tr><th>#</th><th>config</th><th>K</th><th>best score</th><th>hit%</th><th>false/min</th><th>@ep</th></tr></thead><tbody></tbody></table></div>
<script>
const CO=['#57d977','#5a8fd6','#f2b45a','#e06c9a','#57c7d9','#b98ce0','#e0704a','#9fd957'];
function line(cid,series,ymin,ymax,fmtY){const cv=document.getElementById(cid),g=cv.getContext('2d'),W=cv.width,H=cv.height,PL=44,PB=26,PT=12,PR=10;
 g.clearRect(0,0,W,H);let xmax=1;series.forEach(s=>s.pts.forEach(p=>xmax=Math.max(xmax,p[0])));xmax=Math.max(xmax,5);
 g.strokeStyle='#263041';g.fillStyle='#6b7a8c';g.font='10px system-ui';g.lineWidth=1;
 for(let k=0;k<=4;k++){const y=PT+k/4*(H-PT-PB),v=ymax-(k/4)*(ymax-ymin);g.beginPath();g.moveTo(PL,y);g.lineTo(W-PR,y);g.stroke();g.fillText(fmtY(v),4,y+3);}
 for(let e=0;e<=xmax;e+=Math.ceil(xmax/6)){const x=PL+(e/xmax)*(W-PL-PR);g.fillText(e,x-4,H-PB+14);}
 const X=e=>PL+(e/xmax)*(W-PL-PR),Y=v=>PT+(1-(v-ymin)/(ymax-ymin))*(H-PT-PB);
 series.forEach((s,i)=>{if(!s.pts.length)return;g.strokeStyle=CO[i%CO.length];g.lineWidth=2;g.beginPath();
  s.pts.forEach((p,j)=>{const x=X(p[0]),y=Y(p[1]);j?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
  const last=s.pts[s.pts.length-1];g.fillStyle=CO[i%CO.length];g.beginPath();g.arc(X(last[0]),Y(last[1]),3,0,7);g.fill();});}
async function tick(){let d;try{d=await(await fetch('/data',{cache:'no-store'})).json();}catch(e){return;}
 const order=d.order||[];
 document.getElementById('prog').textContent=(order.length)+' / '+(d.total||'?')+' configs';
 const st=document.getElementById('stat');st.textContent=d.done?'DONE ✓':(d.cur?('training '+d.cur+' · ep'+d.cur_ep):'starting…');st.style.background=d.done?'#193a24':'#1d2634';
 const gp=document.getElementById('gpu');gp.textContent=d.gpu?('GPU '+d.gpu.util+'% · '+(d.gpu.mem/1024).toFixed(1)+'/'+(d.gpu.total/1024).toFixed(0)+'GB'):'';
 const epochs=(d.start&&d.start.epochs)||30;const totEp=(d.total||1)*epochs;let doneEp=0;order.forEach(c=>doneEp+=(d.configs[c].epochs.slice(-1)[0]||[0])[0]);
 document.getElementById('pbar').style.width=Math.min(100,100*doneEp/totEp)+'%';
 document.getElementById('leg').innerHTML=order.map((c,i)=>`<span><span class=dot style=background:${CO[i%CO.length]}></span>${c}</span>`).join('');
 const sSeries=order.map(c=>({pts:d.configs[c].evals.map(e=>[e[0],e[1]])}));
 const lSeries=order.map(c=>({pts:d.configs[c].epochs}));
 let smin=0.5,smax=0.5;sSeries.forEach(s=>s.pts.forEach(p=>{smin=Math.min(smin,p[1]);smax=Math.max(smax,p[1]);}));smin=Math.min(smin,-0.1);smax=Math.max(smax,0.5);
 let lmax=0.1;lSeries.forEach(s=>s.pts.forEach(p=>lmax=Math.max(lmax,p[1])));
 line('score',sSeries,smin,smax,v=>v.toFixed(2));line('loss',lSeries,0,lmax,v=>v.toFixed(2));
 const rows=order.map(c=>{const ev=d.configs[c].evals;let best=null;ev.forEach(e=>{if(!best||e[1]>best[1])best=e;});return{c,K:d.configs[c].K,best};}).filter(r=>r.best).sort((a,b)=>b.best[1]-a.best[1]);
 const tb=document.querySelector('#tbl tbody');tb.innerHTML=rows.map((r,i)=>`<tr><td>${i+1}</td><td>${r.c}</td><td>${r.K}</td><td>${r.best[1].toFixed(3)}</td><td>${r.best[3]?Math.round(100*r.best[2]/r.best[3]):'-'}%</td><td>${r.best[4]?r.best[4].toFixed(0):'-'}</td><td>${r.best[0]}</td></tr>`).join('');
}
tick();setInterval(tick,1500);
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/data"):
            body = json.dumps(parse()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"training dashboard: http://127.0.0.1:{PORT}  (Ctrl+C to stop)", flush=True)
        httpd.serve_forever()
