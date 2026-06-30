#!/usr/bin/env python3
"""Lightweight FastAPI status dashboard for robot_sft (TensorBoard-like, zero JS deps).

It NEVER runs training — it only READS the session/run files that watchdog.py and
eval_watcher.py write, and serves one auto-refreshing HTML page plus JSON endpoints. This
keeps the CLI clean and lets the user watch a long run remotely from a browser.

Beyond the stage/run status table it plots, with plain <canvas> (no external/CDN libraries,
so it works offline):
  - the TRAINING LOSS curve (log-y), parsed from the run's train.log, and
  - the periodic OPEN-LOOP EVAL curve (mean MSE per checkpoint) from eval_results.jsonl,
    written by eval_watcher.py.

Run (background):
    python monitor_server.py --session <session_dir> [--host 0.0.0.0] [--port 8770]

Falls back to a stdlib http.server if FastAPI/uvicorn aren't installed, so it always works.
Endpoints: GET / (HTML), /api/session, /api/runs, /api/run/<id>, /api/metrics
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

SESSION_DIR = ""  # set in main

STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")
LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")
# one combined scan so each loss is tagged with the most recent tqdm step
TOKEN_RE = re.compile(r"(\d+)\s*/\s*\d+\s*\[|'loss':\s*([0-9.eE+-]+)")

_loss_cache: dict = {}


def _read(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def session_state():
    return _read(os.path.join(SESSION_DIR, "session.json")) or {}


def runs_state():
    out = []
    for rj in sorted(glob.glob(os.path.join(SESSION_DIR, "runs", "*", "run.json"))):
        d = _read(rj)
        if d:
            out.append(d)
    return out


def run_state(run_id: str):
    return _read(os.path.join(SESSION_DIR, "runs", run_id, "run.json"))


def _latest_train_log() -> str | None:
    logs = glob.glob(os.path.join(SESSION_DIR, "runs", "*", "train.log"))
    return max(logs, key=os.path.getmtime) if logs else None


def _parse_loss(path: str, max_points: int = 800):
    """Return [[step, loss], ...] from a train.log; cached by (size, mtime)."""
    try:
        st = os.stat(path)
    except OSError:
        return []
    key = (path, st.st_size, int(st.st_mtime))
    if _loss_cache.get("key") == key:
        return _loss_cache["val"]
    try:
        with open(path, errors="ignore") as f:
            text = f.read()
    except OSError:
        return []
    series = []
    last_step = 0
    for m in TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            last_step = int(m.group(1))
        elif m.group(2) is not None:
            try:
                series.append([last_step, float(m.group(2))])
            except ValueError:
                pass
    if len(series) > max_points:
        k = len(series) // max_points + 1
        series = series[::k]
    _loss_cache["key"] = key
    _loss_cache["val"] = series
    return series


def _eval_series():
    path = os.path.join(SESSION_DIR, "eval", "eval_results.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    rows.append(d)
                except Exception:  # noqa: BLE001
                    pass
    rows.sort(key=lambda d: d.get("step", 0))
    return rows


def compute_assessment(loss, evals):
    """A live training conclusion derived from the loss + eval curves (read-only fallback so
    the dashboard always shows a verdict, even before the watchdog writes its own). Mirrors
    watchdog.assess: train-loss plateau alone is NOT 'done' — the eval MSE curve decides."""
    if not loss:
        return {"verdict": "waiting for first training steps…", "stop_recommended": False}
    steps = [s for s, _ in loss]
    max_step = max(steps)
    # best (lowest) loss with a 3% improvement threshold → plateau length
    best, best_step = float("inf"), steps[0]
    for s, v in loss:
        if v < best * 0.97:
            best, best_step = v, s
    cur = loss[-1][0]
    plateau = cur - best_step
    flat = plateau >= max(1500, int(0.1 * max_step))
    ev = [(e.get("step"), e.get("mean_mse")) for e in evals if e.get("mean_mse") is not None]
    if not ev:
        etxt, best_e, stop = "no eval points yet", None, False
    else:
        best_e, best_mse = min(ev, key=lambda x: x[1])
        last_mse = ev[-1][1]
        if len(ev) < 2:
            etxt, stop = f"only 1 eval point (ckpt-{ev[-1][0]} mse={last_mse:.2f}); need ≥2 to judge", False
        else:
            improving = last_mse <= best_mse * 1.001
            etxt = f"eval MSE {'still improving' if improving else 'plateaued'}; best ckpt-{best_e} mse={best_mse:.2f}"
            stop = flat and not improving
    if stop:
        v = f"CONVERGED — safe to stop. loss flat {plateau} steps and {etxt}. Pick ckpt-{best_e}."
    elif flat and len(ev) < 2:
        v = f"loss plateaued ({plateau} steps since last >3% drop), but {etxt}. Let 1–2 more checkpoints score before stopping."
    elif flat:
        v = f"loss plateaued ({plateau} steps) but {etxt} — keep going."
    else:
        v = f"training healthy — loss {'plateaued' if flat else 'improving'}; {etxt}."
    return {"verdict": v, "stop_recommended": stop, "loss_plateau_steps": plateau,
            "best_eval_step": best_e, "n_eval_points": len(ev)}


def metrics_state():
    log = _latest_train_log()
    loss = _parse_loss(log) if log else []
    evals = _eval_series()
    # prefer the watchdog's own assessment (run.json) if present; else compute one here
    wd = None
    for r in runs_state():
        if isinstance(r.get("assessment"), dict):
            wd = r["assessment"]
    return {
        "loss": loss,
        "eval": [[e.get("step"), e.get("mean_mse"), e.get("mean_mae")] for e in evals],
        "eval_detail": evals[-1] if evals else None,
        "assessment": wd or compute_assessment(loss, evals),
    }


IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def plots_state():
    """Per-checkpoint eval artifacts (images), discovered GENERICALLY by scanning the standard
    layout <session>/eval/artifacts/ckpt-<step>/<group>/<files> (plus legacy .../plots/...).
    Tool-agnostic: any image an eval adapter drops there is surfaced, no schema needed.
    Returns [{step, datasets:{group:[relpath-from-eval/, ...]}}] sorted by step."""
    eval_root = os.path.join(SESSION_DIR, "eval")
    by_step: dict = {}
    for top in ("artifacts", "plots"):  # 'plots' kept for backward-compat
        for ck in glob.glob(os.path.join(eval_root, top, "ckpt-*")):
            m = re.search(r"ckpt-(\d+)$", ck)
            if not m:
                continue
            step = int(m.group(1))
            groups = by_step.setdefault(step, {})
            for grp in sorted(glob.glob(os.path.join(ck, "*"))):
                if not os.path.isdir(grp):
                    continue
                name = os.path.basename(grp)
                imgs = [
                    os.path.relpath(p, eval_root)
                    for p in sorted(glob.glob(os.path.join(grp, "*")))
                    if p.lower().endswith(IMG_EXT)
                ]
                if imgs:
                    groups.setdefault(name, []).extend(imgs)
    return [{"step": s, "datasets": by_step[s]} for s in sorted(by_step)]


def serve_plot(relpath: str):
    """Return (bytes, content_type) for an image under <session>/eval, or (None, None).
    Guards against path traversal."""
    # relpaths in eval_results.jsonl look like "plots/ckpt-2200/<ds>/traj_0.jpeg" (relative to eval/)
    base = os.path.realpath(os.path.join(SESSION_DIR, "eval"))
    full = os.path.realpath(os.path.join(base, relpath))
    if not full.startswith(base + os.sep) or not os.path.isfile(full):
        return None, None
    ext = full.rsplit(".", 1)[-1].lower()
    ctype = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
    with open(full, "rb") as f:
        return f.read(), ctype


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>robot_sft monitor</title>
<style>
 body{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0e1116;color:#d7dde5;margin:0;padding:24px}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#7d8794;font-size:12px;margin-bottom:18px}
 .card{background:#161b22;border:1px solid #2a313c;border-radius:8px;padding:14px 16px;margin:12px 0}
 .row{display:flex;flex-wrap:wrap;gap:18px;align-items:center} .k{color:#7d8794} .v{color:#e6edf3}
 .b{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px}
 .running{background:#16361f;color:#4ade80}.done{background:#16304a;color:#60a5fa}
 .failed,.early_stopping{background:#3a1620;color:#f87171}.restarting{background:#3a3216;color:#fbbf24}
 .pending{background:#23282f;color:#7d8794}.blocked{background:#3a1620;color:#f87171}
 .bar{height:8px;background:#23282f;border-radius:4px;overflow:hidden;margin-top:6px}
 .fill{height:100%;background:#4ade80}
 table{border-collapse:collapse;width:100%;font-size:13px} td,th{text-align:left;padding:4px 10px;border-bottom:1px solid #23282f}
 .charts{display:flex;flex-wrap:wrap;gap:18px} .chart{flex:1;min-width:340px}
 .ct{font-size:13px;color:#e6edf3;margin-bottom:6px} .cv{color:#7d8794;font-size:12px}
 canvas{width:100%;height:220px;background:#0b0e13;border:1px solid #23282f;border-radius:6px}
 .seltab{display:inline-block;padding:3px 10px;margin:0 6px 6px 0;border:1px solid #2a313c;border-radius:6px;cursor:pointer;font-size:12px;color:#9aa4b2}
 .seltab.on{background:#16304a;color:#cfe3ff;border-color:#2d4f74}
 .gal{display:flex;flex-wrap:wrap;gap:14px} .gcol{min-width:300px} .gname{font-size:12px;color:#7d8794;margin:4px 0}
 .gal img{width:340px;max-width:46vw;border:1px solid #23282f;border-radius:6px;background:#fff}
</style></head><body>
<h1>robot_sft &middot; <span id="sid"></span></h1>
<div class="sub">auto-refresh 5s &middot; reads watchdog + eval_watcher state files (no training runs here)</div>
<div id="assess" class="card" style="border-left:4px solid #4ade80"><span class="cv">assessment</span><div id="assessv" style="margin-top:4px"></div></div>
<div id="stages" class="card"></div>
<div id="runs"></div>
<div class="card"><div class="charts">
 <div class="chart"><div class="ct">training loss <span id="lossv" class="cv"></span></div><canvas id="lossc"></canvas></div>
 <div class="chart"><div class="ct">open-loop eval &middot; mean MSE (held-out) <span id="evalv" class="cv"></span></div><canvas id="evalc"></canvas>
   <div id="evaltable" style="margin-top:8px"></div></div>
</div></div>
<div class="card"><div class="ct">open-loop trajectory plots <span class="cv">(gt vs pred per action dim &middot; pick a checkpoint)</span></div>
 <div id="ckpttabs"></div><div id="gallery" class="gal"></div></div>
<script>
let SELCK=null;  // selected checkpoint step for the gallery (null => latest)
const cls=s=>({running:'running',done:'done',failed:'failed',early_stopping:'early_stopping',
 restarting:'restarting',blocked:'blocked'}[s]||'pending');
function draw(id,pts,o){
 o=o||{};const c=document.getElementById(id);const dpr=window.devicePixelRatio||1;
 const W=c.clientWidth,H=c.clientHeight;c.width=W*dpr;c.height=H*dpr;
 const g=c.getContext('2d');g.scale(dpr,dpr);g.clearRect(0,0,W,H);
 const pad={l:52,r:10,t:10,b:22};const pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
 if(!pts||pts.length===0){g.fillStyle='#7d8794';g.font='12px monospace';g.fillText('waiting for data…',pad.l,pad.t+16);return;}
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]).filter(v=>v!=null&&isFinite(v));
 let xmin=Math.min(...xs),xmax=Math.max(...xs);if(xmax===xmin)xmax=xmin+1;
 const ylog=!!o.ylog;let vmin=Math.min(...ys),vmax=Math.max(...ys);
 if(ylog){vmin=Math.max(vmin,1e-6);}if(vmax===vmin)vmax=vmin+1e-6;
 const tv=v=>ylog?Math.log10(Math.max(v,1e-6)):v;const lv=tv(vmin),hv=tv(vmax);
 const X=x=>pad.l+pw*(x-xmin)/(xmax-xmin);
 const Y=v=>pad.t+ph*(1-(tv(v)-lv)/(hv-lv||1));
 g.strokeStyle='#23282f';g.fillStyle='#7d8794';g.font='11px monospace';g.lineWidth=1;
 for(let i=0;i<=4;i++){const yy=pad.t+ph*i/4;g.beginPath();g.moveTo(pad.l,yy);g.lineTo(W-pad.r,yy);g.stroke();
   const val=ylog?Math.pow(10,hv-(hv-lv)*i/4):vmax-(vmax-vmin)*i/4;
   g.fillText(val>=0.01?val.toFixed(3):val.toExponential(1),4,yy+4);}
 g.fillText(xmin|0,pad.l,H-6);g.fillText(xmax|0,W-pad.r-34,H-6);
 g.strokeStyle=o.color||'#4ade80';g.lineWidth=1.6;g.beginPath();let started=false;
 for(const p of pts){if(p[1]==null||!isFinite(p[1]))continue;const x=X(p[0]),y=Y(p[1]);
   if(!started){g.moveTo(x,y);started=true;}else g.lineTo(x,y);}g.stroke();
 if(o.dots){g.fillStyle=o.color||'#60a5fa';for(const p of pts){if(p[1]==null)continue;g.beginPath();g.arc(X(p[0]),Y(p[1]),2.5,0,7);g.fill();}}
}
async function tick(){
 try{
  const s=await (await fetch('/api/session')).json();
  document.getElementById('sid').textContent=(s.session_id||'')+' ['+(s.status||'')+']';
  let h='<div class="row"><div><span class="k">goal</span> <span class="v">'+(s.goal||'—')+'</span></div>'
    +'<div><span class="k">stage</span> <span class="v">'+(s.current_stage||'')+'</span></div></div><table>';
  const order=['overview','dataset_explore','preprocess','training_plan','train'];
  for(const st of order){const o=(s.stages||{})[st]||{};
    h+='<tr><td>'+st+'</td><td><span class="b '+cls(o.status)+'">'+(o.status||'pending')+'</span></td><td>'+(o.summary||'')+'</td></tr>';}
  h+='</table>';document.getElementById('stages').innerHTML=h;
  const runs=await (await fetch('/api/runs')).json();
  let r='';
  for(const run of runs){
   const cur=run.last_step||0,max=run.max_step||0,pct=max?Math.min(100,100*cur/max):0;
   r+='<div class="card"><div class="row"><b>'+run.run_id+'</b>'
     +'<span class="b '+cls(run.status)+'">'+(run.status||'')+'</span>'
     +'<div><span class="k">step</span> <span class="v">'+cur+(max?'/'+max:'')+'</span></div>'
     +'<div><span class="k">loss</span> <span class="v">'+(run.last_loss??'—')+'</span></div>'
     +'<div><span class="k">restarts</span> <span class="v">'+(run.restarts||0)+'</span></div>'
     +'<div><span class="k">ckpt</span> <span class="v">'+(run.checkpoint||'—')+'</span></div></div>'
     +'<div class="bar"><div class="fill" style="width:'+pct+'%"></div></div>';
   if(run.reason) r+='<div class="sub" style="margin-top:8px">reason: '+run.reason+'</div>';
   if(run.resume) r+='<div class="sub">'+run.resume+(run.backoff_s?(' (backoff '+run.backoff_s+'s)'):'')+'</div>';
   r+='</div>';
  }
  document.getElementById('runs').innerHTML=r||'<div class="card sub">no runs yet</div>';
  const m=await (await fetch('/api/metrics')).json();
  const as=m.assessment||{};
  document.getElementById('assessv').textContent=as.verdict||'—';
  document.getElementById('assess').style.borderLeftColor=as.stop_recommended?'#fbbf24':'#4ade80';
  const ed=m.eval_detail;
  if(ed&&ed.per_dataset){let t='<table><tr><th>dataset (ckpt-'+ed.step+')</th><th>MSE</th><th>MAE</th></tr>';
    for(const[n,d]of Object.entries(ed.per_dataset)){const mm=d.metrics||d;
      t+='<tr><td>'+n+'</td><td>'+(mm.mse!=null?mm.mse.toFixed(3):'—')+'</td><td>'+(mm.mae!=null?mm.mae.toFixed(3):'—')+'</td></tr>';}
    t+='<tr><td><b>mean</b></td><td><b>'+(ed.mean_mse!=null?ed.mean_mse.toFixed(3):'—')+'</b></td><td><b>'+(ed.mean_mae!=null?ed.mean_mae.toFixed(3):'—')+'</b></td></tr></table>';
    document.getElementById('evaltable').innerHTML=t;}
  const loss=m.loss||[];draw('lossc',loss,{ylog:true,color:'#4ade80'});
  document.getElementById('lossv').textContent=loss.length?('last '+loss[loss.length-1][1]):'';
  const ev=(m.eval||[]).map(e=>[e[0],e[1]]);draw('evalc',ev,{color:'#60a5fa',dots:true});
  if(ev.length){const best=ev.filter(e=>e[1]!=null).reduce((a,b)=>b[1]<a[1]?b:a);
    document.getElementById('evalv').textContent='best ckpt-'+best[0]+' mse='+(best[1]!=null?best[1].toFixed(4):'—');}
  const plots=await (await fetch('/api/plots')).json();
  const steps=plots.map(p=>p.step);
  let tabs='';for(const st of steps){tabs+='<span class="seltab'+((SELCK===st||(SELCK===null&&st===steps[steps.length-1]))?' on':'')+'" onclick="SELCK='+st+';tick()">ckpt-'+st+'</span>';}
  document.getElementById('ckpttabs').innerHTML=tabs;
  const sel=SELCK!==null?plots.find(p=>p.step===SELCK):plots[plots.length-1];
  let g='';if(sel){for(const[name,imgs]of Object.entries(sel.datasets)){
    g+='<div class="gcol"><div class="gname">'+name+'</div>';
    for(const rel of imgs){g+='<img loading="lazy" src="/plot?p='+encodeURIComponent(rel)+'">';}
    g+='</div>';}}
  document.getElementById('gallery').innerHTML=g||'<div class="cv">no plots yet</div>';
 }catch(e){}
}
tick();setInterval(tick,5000);
</script></body></html>"""


def build_fastapi():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="robot_sft monitor")

    @app.get("/", response_class=HTMLResponse)
    def index():  # noqa: ANN202
        return PAGE

    @app.get("/api/session")
    def api_session():  # noqa: ANN202
        return JSONResponse(session_state())

    @app.get("/api/runs")
    def api_runs():  # noqa: ANN202
        return JSONResponse(runs_state())

    @app.get("/api/run/{run_id}")
    def api_run(run_id: str):  # noqa: ANN202
        return JSONResponse(run_state(run_id) or {})

    @app.get("/api/metrics")
    def api_metrics():  # noqa: ANN202
        return JSONResponse(metrics_state())

    @app.get("/api/plots")
    def api_plots():  # noqa: ANN202
        return JSONResponse(plots_state())

    @app.get("/plot")
    def plot(p: str):  # noqa: ANN202
        from fastapi.responses import Response
        body, ctype = serve_plot(p)
        if body is None:
            return Response(status_code=404)
        return Response(content=body, media_type=ctype)

    return app


def serve_stdlib(host: str, port: int) -> None:
    """Fallback with zero third-party deps."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                self._send(PAGE.encode(), "text/html")
            elif self.path == "/api/session":
                self._send(json.dumps(session_state()).encode(), "application/json")
            elif self.path == "/api/runs":
                self._send(json.dumps(runs_state()).encode(), "application/json")
            elif self.path == "/api/metrics":
                self._send(json.dumps(metrics_state()).encode(), "application/json")
            elif self.path == "/api/plots":
                self._send(json.dumps(plots_state()).encode(), "application/json")
            elif self.path.startswith("/plot?"):
                from urllib.parse import parse_qs, urlparse
                p = parse_qs(urlparse(self.path).query).get("p", [""])[0]
                body, ctype = serve_plot(p)
                if body is None:
                    self.send_response(404); self.end_headers()
                else:
                    self._send(body, ctype)
            elif self.path.startswith("/api/run/"):
                rid = self.path.rsplit("/", 1)[-1]
                self._send(json.dumps(run_state(rid) or {}).encode(), "application/json")
            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *a):  # silence
            pass

    print(f"[monitor] stdlib server on http://{host}:{port}  (session={SESSION_DIR})")
    ThreadingHTTPServer((host, port), H).serve_forever()


def main() -> None:
    global SESSION_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    SESSION_DIR = args.session

    try:
        import uvicorn  # noqa: F401
        app = build_fastapi()
        print(f"[monitor] FastAPI on http://{args.host}:{args.port}  (session={SESSION_DIR})")
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except Exception as e:  # noqa: BLE001
        print(f"[monitor] FastAPI/uvicorn unavailable ({e}); using stdlib server")
        serve_stdlib(args.host, args.port)


if __name__ == "__main__":
    main()
