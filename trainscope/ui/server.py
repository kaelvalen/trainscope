import json
import math
import re
from pathlib import Path

import pyarrow.ipc as ipc
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def _read_arrow(path: Path) -> list[dict]:
    if not path.exists():
        return []
    reader = ipc.open_file(str(path))
    table = reader.read_all()
    raw = table.to_pydict()
    n = len(next(iter(raw.values()))) if raw else 0
    rows = []
    for i in range(n):
        row = {}
        for k, v in raw.items():
            val = v[i]
            if hasattr(val, "as_py"):
                val = val.as_py()
            row[k] = val
        rows.append(row)
    return rows


def _safe_kl(p: list[float], q: list[float]) -> float:
    eps = 1e-10
    if not p or not q or len(p) != len(q):
        return 0.0
    sum_p = sum(p) + eps * len(p)
    sum_q = sum(q) + eps * len(q)
    kl = 0.0
    for pi, qi in zip(p, q):
        pi_n = (pi + eps) / sum_p
        qi_n = (qi + eps) / sum_q
        kl += pi_n * math.log(pi_n / qi_n)
    return max(0.0, kl)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def create_app(run_path: str) -> FastAPI:
    rp = Path(run_path)
    app = FastAPI(title="TrainScope UI")
    _cache: dict = {}  # keyed by "diff" and "ranked"

    static_dir = Path(__file__).parent / "static"

    @app.get("/api/meta")
    def get_meta():
        meta_file = rp / "meta.json"
        if not meta_file.exists():
            raise HTTPException(status_code=404, detail="meta.json not found")
        with open(meta_file) as f:
            return json.load(f)

    @app.get("/api/global")
    def get_global():
        return _read_arrow(rp / "global.arrow")

    @app.get("/api/layers")
    def get_layers():
        layers_dir = rp / "layers"
        if not layers_dir.exists():
            return []
        names = []
        for arrow_file in sorted(layers_dir.glob("*.arrow")):
            safe_name = arrow_file.stem
            original_name = safe_name.replace("__", "/")
            names.append(original_name)
        return names

    @app.get("/api/layers/{layer_name:path}")
    def get_layer(layer_name: str):
        safe_name = layer_name.replace("/", "__")
        path = rp / "layers" / f"{safe_name}.arrow"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Layer '{layer_name}' not found")
        return _read_arrow(path)

    @app.get("/api/spikes")
    def get_spikes():
        spikes_dir = rp / "spikes"
        if not spikes_dir.exists():
            return []
        result = []
        for f in sorted(spikes_dir.glob("spike_step_*.arrow")):
            match = re.search(r"spike_step_(\d+)\.arrow", f.name)
            if match:
                result.append({"step": int(match.group(1)), "file": f.name})
        return result

    @app.get("/api/spikes/{step}/layers")
    def get_spike_layers(step: int):
        layers_dir = rp / "spikes" / f"spike_step_{step}_layers"
        if not layers_dir.exists():
            return []
        return [f.stem.replace("__", "/") for f in sorted(layers_dir.glob("*.arrow"))]

    @app.get("/api/spikes/{step}/layers/{layer_name:path}")
    def get_spike_layer(step: int, layer_name: str):
        safe_name = layer_name.replace("/", "__")
        path = rp / "spikes" / f"spike_step_{step}_layers" / f"{safe_name}.arrow"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Layer '{layer_name}' not in spike {step}")
        return _read_arrow(path)

    @app.get("/api/spikes/{step}")
    def get_spike(step: int):
        path = rp / "spikes" / f"spike_step_{step}.arrow"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Spike at step {step} not found")
        return _read_arrow(path)

    def _get_diff_cache() -> dict[str, dict[int, list]]:
        if "diff" not in _cache:
            layers_dir = rp / "layers"
            index: dict[str, dict[int, list]] = {}
            if layers_dir.exists():
                for arrow_file in sorted(layers_dir.glob("*.arrow")):
                    rows = _read_arrow(arrow_file)
                    layer_name = arrow_file.stem.replace("__", "/")
                    index[layer_name] = {
                        r["step"]: r.get("hist_counts") or []
                        for r in rows
                    }
            _cache["diff"] = index
        return _cache["diff"]

    @app.get("/api/diff")
    def get_diff(step_a: int, step_b: int):
        index = _get_diff_cache()
        result = []
        for layer_name, step_map in index.items():
            counts_a = step_map.get(step_a)
            counts_b = step_map.get(step_b)
            if not counts_a or not counts_b:
                continue
            result.append({"layer": layer_name, "kl_divergence": _safe_kl(counts_a, counts_b)})
        result.sort(key=lambda x: x["kl_divergence"], reverse=True)
        return result

    @app.get("/api/layers/ranked")
    def get_layers_ranked(top_n: int = 8):
        if "ranked" not in _cache:
            layers_dir = rp / "layers"
            scored: list[tuple[str, float]] = []
            if layers_dir.exists():
                for arrow_file in sorted(layers_dir.glob("*.arrow")):
                    rows = _read_arrow(arrow_file)
                    layer_name = arrow_file.stem.replace("__", "/")
                    grad_norms = [r["grad_l2_norm"] for r in rows if "grad_l2_norm" in r]
                    scored.append((layer_name, _variance(grad_norms)))
            scored.sort(key=lambda x: x[1], reverse=True)
            _cache["ranked"] = [name for name, _ in scored]
        return _cache["ranked"][:top_n]

    if (static_dir / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:
        @app.get("/")
        def index():
            return HTMLResponse(content=_FALLBACK_HTML)

    return app


_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TrainScope</title>
<script src="https://cdn.plot.ly/plotly-2.29.0.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{padding:14px 24px;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;gap:16px;flex-shrink:0}
h1{font-size:17px;font-weight:600;letter-spacing:.02em}
#meta-info{font-size:12px;color:#666}
nav{display:flex;border-bottom:1px solid #2a2a2a;flex-shrink:0}
nav button{padding:11px 22px;background:none;border:none;border-bottom:2px solid transparent;color:#777;cursor:pointer;font-size:13px;transition:color .15s}
nav button:hover{color:#ccc}
nav button.active{color:#e0e0e0;border-bottom-color:#4f9de0}
.view{padding:20px 24px;display:none;overflow-y:auto;flex:1}
.view.active{display:block}
.chart-box{margin-bottom:20px}
.row{display:flex;gap:12px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
label{font-size:13px;color:#888}
select,input[type=number]{background:#1c1c1c;border:1px solid #333;color:#e0e0e0;padding:6px 10px;border-radius:4px;font-size:13px}
button.action{background:#4f9de0;color:#fff;border:none;padding:7px 16px;border-radius:4px;cursor:pointer;font-size:13px}
button.action:hover{background:#5facee}
.spike-count{background:#c0392b;color:#fff;padding:2px 7px;border-radius:3px;font-size:11px;font-weight:600}
</style>
</head>
<body>
<header>
  <h1>TrainScope</h1>
  <span id="meta-info">loading…</span>
</header>
<nav>
  <button class="active" data-view="timeline">Timeline</button>
  <button data-view="layers">Layer Drill-down</button>
  <button data-view="diff">Diff View</button>
  <button data-view="spike-inspector">Spike Inspector</button>
</nav>
<div id="timeline" class="view active">
  <div id="chart-loss" class="chart-box"></div>
  <div id="chart-gradnorm" class="chart-box"></div>
</div>
<div id="layers" class="view">
  <div class="row">
    <label>Layer</label>
    <select id="layer-sel"><option>—</option></select>
  </div>
  <div id="chart-kurtosis" class="chart-box"></div>
  <div id="chart-layer-grad" class="chart-box"></div>
  <div id="chart-layer-weight" class="chart-box"></div>
</div>
<div id="diff" class="view">
  <div class="row">
    <label>Step A</label><input type="number" id="step-a" value="0" style="width:80px">
    <label>Step B</label><input type="number" id="step-b" value="100" style="width:80px">
    <button class="action" id="btn-compare">Compare</button>
  </div>
  <div id="chart-diff" class="chart-box"></div>
</div>
<div id="spike-inspector" class="view">
  <div class="row">
    <label>Spike</label>
    <select id="spike-sel" style="width:140px"><option value="">— select —</option></select>
    <label>Layer</label>
    <select id="spike-layer-sel" style="width:300px"><option value="">— select spike first —</option></select>
  </div>
  <div id="spike-status" style="font-size:12px;color:#666;margin-bottom:12px"></div>
  <div id="chart-spike-loss" class="chart-box"></div>
  <div id="chart-spike-kurtosis" class="chart-box"></div>
  <div id="chart-spike-grad" class="chart-box"></div>
</div>
<script>
const DARK={paper_bgcolor:'#0f0f0f',plot_bgcolor:'#181818',font:{color:'#d0d0d0'},
  xaxis:{gridcolor:'#2a2a2a',linecolor:'#333',zerolinecolor:'#2a2a2a'},
  yaxis:{gridcolor:'#2a2a2a',linecolor:'#333',zerolinecolor:'#2a2a2a'},
  margin:{t:36,r:16,b:40,l:60}};

const api=p=>fetch('/api'+p).then(r=>r.ok?r.json():[]);

document.querySelectorAll('nav button').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('nav button').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById(b.dataset.view).classList.add('active');
  });
});

let globalData=[],spikes=[];

function spikeShapes(){
  return spikes.map(s=>({type:'line',x0:s.step,x1:s.step,y0:0,y1:1,yref:'paper',
    line:{color:'#c0392b',width:1,dash:'dot'}}));
}

async function init(){
  const [meta,global_,spikes_,layers,ranked]=await Promise.all([
    api('/meta').catch(()=>null),
    api('/global'),
    api('/spikes'),
    api('/layers'),
    api('/layers/ranked?top_n=8').catch(()=>[]),
  ]);
  globalData=global_; spikes=spikes_;

  if(meta){
    const rn=meta.trainscope_config?.run_name??'unknown';
    const sc=spikes_.length;
    document.getElementById('meta-info').innerHTML=
      `<b>${rn}</b> &nbsp; ${global_.length} steps &nbsp; `+
      (sc?`<span class="spike-count">${sc} spike${sc>1?'s':''}</span>`:'no spikes');
  }

  const steps=globalData.map(r=>r.step);
  Plotly.newPlot('chart-loss',[{x:steps,y:globalData.map(r=>r.loss),mode:'lines',
    line:{color:'#4f9de0',width:1.5},name:'Loss'}],
    {...DARK,title:'Training Loss',shapes:spikeShapes()},{responsive:true});

  Plotly.newPlot('chart-gradnorm',[{x:steps,y:globalData.map(r=>r.grad_norm_before_clip),mode:'lines',
    line:{color:'#e0a34f',width:1.5},name:'Grad Norm'}],
    {...DARK,title:'Global Grad Norm (pre-clip)',shapes:spikeShapes()},{responsive:true});

  const sel=document.getElementById('layer-sel');
  sel.innerHTML=layers.map(l=>`<option value="${l}">${l}</option>`).join('');
  const firstLayer=ranked.length?ranked[0]:(layers.length?layers[0]:null);
  if(firstLayer){
    sel.value=firstLayer;
    loadLayer(firstLayer);
  }
  sel.addEventListener('change',()=>loadLayer(sel.value));

  // Spike Inspector — populate spike list
  const spikeSel=document.getElementById('spike-sel');
  spikes_.forEach(s=>{
    const opt=document.createElement('option');
    opt.value=s.step;
    opt.textContent=`Step ${s.step}`;
    spikeSel.appendChild(opt);
  });
  spikeSel.addEventListener('change',()=>loadSpikeWindow(Number(spikeSel.value)));
  document.getElementById('spike-layer-sel').addEventListener('change',function(){
    const step=Number(spikeSel.value);
    if(step&&this.value) loadSpikeLayer(step,this.value);
  });
}

async function loadSpikeWindow(step){
  const status=document.getElementById('spike-status');
  status.textContent='Loading spike window…';
  const [globalRows,layerNames]=await Promise.all([
    api(`/spikes/${step}`),
    api(`/spikes/${step}/layers`),
  ]);
  if(!globalRows.length){status.textContent='No data for this spike.';return;}
  status.textContent=`${globalRows.length} steps in window · ${layerNames.length} layers`;

  const steps=globalRows.map(r=>r.step);
  const spikeShape=[{type:'line',x0:step,x1:step,y0:0,y1:1,yref:'paper',
    line:{color:'#c0392b',width:1.5,dash:'dot'}}];
  Plotly.newPlot('chart-spike-loss',[
    {x:steps,y:globalRows.map(r=>r.loss),mode:'lines',line:{color:'#4f9de0',width:1.5},name:'Loss'},
    {x:steps,y:globalRows.map(r=>r.grad_norm_before_clip),mode:'lines',
      line:{color:'#e0a34f',width:1.5},name:'Grad Norm',yaxis:'y2'},
  ],{...DARK,title:`Loss + Grad Norm around spike step ${step}`,shapes:spikeShape,
    yaxis2:{overlaying:'y',side:'right',gridcolor:'#2a2a2a',color:'#e0a34f'},
  },{responsive:true});

  const lsel=document.getElementById('spike-layer-sel');
  lsel.innerHTML=layerNames.map(l=>`<option value="${l}">${l}</option>`).join('');
  if(layerNames.length) loadSpikeLayer(step,layerNames[0]);
}

async function loadSpikeLayer(step,layerName){
  const rows=await api(`/spikes/${step}/layers/${encodeURIComponent(layerName)}`);
  if(!rows.length) return;
  const steps=rows.map(r=>r.step);
  const spikeShape=[{type:'line',x0:step,x1:step,y0:0,y1:1,yref:'paper',
    line:{color:'#c0392b',width:1.5,dash:'dot'}}];
  Plotly.newPlot('chart-spike-kurtosis',[
    {x:steps,y:rows.map(r=>r.act_kurtosis),mode:'lines',line:{color:'#9b59b6',width:1.5},name:'Kurtosis'},
  ],{...DARK,title:`Activation Kurtosis — ${layerName}`,shapes:spikeShape},{responsive:true});
  Plotly.newPlot('chart-spike-grad',[
    {x:steps,y:rows.map(r=>r.grad_l2_norm),mode:'lines',line:{color:'#2ecc71',width:1.5},name:'Grad L2'},
  ],{...DARK,title:`Gradient L2 — ${layerName}`,shapes:spikeShape},{responsive:true});
}

async function loadLayer(name){
  name=name??document.getElementById('layer-sel').value;
  const data=await api('/layers/'+encodeURIComponent(name));
  const steps=data.map(r=>r.step);
  const shapes=spikeShapes();
  Plotly.newPlot('chart-kurtosis',[{x:steps,y:data.map(r=>r.act_kurtosis),mode:'lines',
    line:{color:'#9b59b6',width:1.5},name:'Kurtosis'}],
    {...DARK,title:`Activation Kurtosis — ${name}`,shapes},{responsive:true});
  Plotly.newPlot('chart-layer-grad',[{x:steps,y:data.map(r=>r.grad_l2_norm),mode:'lines',
    line:{color:'#2ecc71',width:1.5},name:'Grad L2'}],
    {...DARK,title:`Gradient L2 Norm — ${name}`,shapes},{responsive:true});
  Plotly.newPlot('chart-layer-weight',[{x:steps,y:data.map(r=>r.weight_l2_norm),mode:'lines',
    line:{color:'#f1c40f',width:1.5},name:'Weight L2'}],
    {...DARK,title:`Weight L2 Norm — ${name}`},{responsive:true});
}

document.getElementById('btn-compare').addEventListener('click',async()=>{
  const a=document.getElementById('step-a').value;
  const b=document.getElementById('step-b').value;
  const data=await api(`/diff?step_a=${a}&step_b=${b}`);
  if(!data.length){
    document.getElementById('chart-diff').textContent='No layer data found for these steps.';
    return;
  }
  const colors=data.map((_,i)=>i<3?'#c0392b':'#4f9de0');
  Plotly.newPlot('chart-diff',[{
    x:data.map(d=>d.kl_divergence),y:data.map(d=>d.layer),
    type:'bar',orientation:'h',marker:{color:colors},name:'KL div',
  }],{...DARK,
    title:`KL Divergence step ${a} → ${b} (top 3 in red)`,
    height:Math.max(400,data.length*22+100),
    yaxis:{...DARK.yaxis,autorange:'reversed'},
  },{responsive:true});
});

init();
</script>
</body>
</html>"""


def start_server(run_path: str, host: str = "127.0.0.1", port: int = 7007):
    import uvicorn
    uvicorn.run(create_app(run_path), host=host, port=port)
