# Chart Fusion App – Frontend & Backend

This delivers a minimal **infinite-canvas chart builder** where users can:

- Upload a CSV
- Create chart nodes by selecting *any* dimension(s) and measure(s)
- **Fuse** two charts by drawing a connection ("noodle/thread") between nodes
- Fusion is only allowed when either:
  - **Same dimension + different measures** (→ grouped/stacked bar, or dual-axis line), or
  - **Same measure + different dimensions** (→ multi-series or 2D matrix/heatmap-ready)

---

## Backend (Python • FastAPI)

**Requirements**
```bash
pip install fastapi uvicorn pandas python-multipart
```

**app.py**
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import uuid

app = FastAPI(title="Chart Fusion Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# In-memory stores
# -----------------------
DATASETS: Dict[str, pd.DataFrame] = {}
CHARTS: Dict[str, Dict[str, Any]] = {}

# -----------------------
# Models
# -----------------------
class ChartCreate(BaseModel):
    dataset_id: str
    dimensions: List[str] = []
    measures: List[str] = []
    agg: str = "sum"  # future: support more
    title: Optional[str] = None

class FuseRequest(BaseModel):
    chart1_id: str
    chart2_id: str

# -----------------------
# Helpers
# -----------------------

def _agg(df: pd.DataFrame, dimensions: List[str], measures: List[str], agg: str = "sum") -> pd.DataFrame:
    if not measures:
        raise HTTPException(status_code=400, detail="At least one measure is required")
    for col in dimensions + measures:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column not found: {col}")
    if dimensions:
        grouped = df.groupby(dimensions)[measures].agg(agg).reset_index()
    else:
        grouped = df[measures].agg(agg).to_frame().T
    return grouped


def _same_dim_diff_measures(spec1, spec2):
    return spec1["dimensions"] == spec2["dimensions"] and set(spec1["measures"]) != set(spec2["measures"]) and len(spec1["dimensions"]) > 0


def _same_measure_diff_dims(spec1, spec2):
    common_measures = set(spec1["measures"]).intersection(set(spec2["measures"]))
    return (len(common_measures) == 1) and (spec1["dimensions"] != spec2["dimensions"]) and (len(spec1["dimensions"]) > 0 or len(spec2["dimensions"]) > 0)


# -----------------------
# Routes
# -----------------------

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    return {
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "rows": len(df)
    }


@app.post("/charts")
async def create_chart(spec: ChartCreate):
    if spec.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = DATASETS[spec.dataset_id]
    table = _agg(df, spec.dimensions, spec.measures, spec.agg)
    chart_id = str(uuid.uuid4())
    CHARTS[chart_id] = {
        "chart_id": chart_id,
        "dataset_id": spec.dataset_id,
        "dimensions": spec.dimensions,
        "measures": spec.measures,
        "agg": spec.agg,
        "title": spec.title or f"Chart {chart_id[:6]}",
        "table": table.to_dict(orient="records")
    }
    return CHARTS[chart_id]


@app.get("/charts/{chart_id}")
async def get_chart(chart_id: str):
    if chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    return CHARTS[chart_id]


@app.post("/fuse")
async def fuse(req: FuseRequest):
    if req.chart1_id not in CHARTS or req.chart2_id not in CHARTS:
        raise HTTPException(status_code=404, detail="One or both charts not found")

    c1, c2 = CHARTS[req.chart1_id], CHARTS[req.chart2_id]
    ds_id = c1["dataset_id"]
    if ds_id != c2["dataset_id"]:
        raise HTTPException(status_code=400, detail="Charts must come from the same dataset for fusion in this demo")
    df = DATASETS[ds_id]

    # Case A: Same Dimension + Different Measures
    if _same_dim_diff_measures(c1, c2):
        dims = c1["dimensions"]
        measures = sorted(list(set(c1["measures"]) | set(c2["measures"])) )
        fused_table = _agg(df, dims, measures, c1["agg"]).copy()
        strategy = {
            "type": "same-dimension-different-measures",
            "suggestion": "grouped-bar | stacked-bar | dual-axis-line"
        }
        title = f"Fusion: {', '.join(measures)} by {', '.join(dims)}"

    # Case B: Same Measure + Different Dimensions
    elif _same_measure_diff_dims(c1, c2):
        common_measure = list(set(c1["measures"]).intersection(set(c2["measures"])))[0]
        # Aggregate each chart separately on its own dims
        t1 = _agg(df, c1["dimensions"], [common_measure], c1["agg"]).copy()
        t2 = _agg(df, c2["dimensions"], [common_measure], c2["agg"]).copy()
        t1["__DimensionType__"] = ",".join(c1["dimensions"]) or "(none)"
        t2["__DimensionType__"] = ",".join(c2["dimensions"]) or "(none)"
        # Normalize to long format: DimensionValue = concatenation of keys
        def flatten_keys(row, dims):
            if not dims: return "(total)"
            return " | ".join(str(row[d]) for d in dims)
        t1["DimensionValue"] = t1.apply(lambda r: flatten_keys(r, c1["dimensions"]), axis=1)
        t2["DimensionValue"] = t2.apply(lambda r: flatten_keys(r, c2["dimensions"]), axis=1)
        fused_table = pd.concat([
            t1[["__DimensionType__", "DimensionValue", common_measure]],
            t2[["__DimensionType__", "DimensionValue", common_measure]],
        ], ignore_index=True)
        fused_table = fused_table.rename(columns={"__DimensionType__": "DimensionType", common_measure: "Value"})
        strategy = {
            "type": "same-measure-different-dimensions",
            "suggestion": "multi-series line/bar | heatmap-ready (if granular joint exists)"
        }
        title = f"Fusion: {common_measure} across dimension sets"

    else:
        raise HTTPException(status_code=400, detail="Fusion not allowed: charts must share either a dimension (and differ in measures) or share a single common measure (and differ in dimensions)")

    chart_id = str(uuid.uuid4())
    fused_payload = {
        "chart_id": chart_id,
        "dataset_id": ds_id,
        "dimensions": list({*c1["dimensions"], *c2["dimensions"]}),
        "measures": list({*c1["measures"], *c2["measures"]}),
        "title": title,
        "strategy": strategy,
        "table": fused_table.to_dict(orient="records"),
    }
    CHARTS[chart_id] = fused_payload
    return fused_payload
```

---

## Frontend (React • Infinite Canvas with React Flow)

**Requirements**
```bash
npm install react react-dom react-flow-renderer react-plotly.js plotly.js tailwindcss
```

> Tailwind setup not shown; any styling is optional. React Flow provides pan/zoom (infinite canvas). Connecting two nodes triggers fusion.

**App.jsx** (single-file demo)
```jsx
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import ReactFlow, { addEdge, Background, Controls, MiniMap } from 'react-flow-renderer';
import Plot from 'react-plotly.js';

const API = 'http://localhost:8000';

function ChartNode({ data }) {
  const { title, figure } = data;
  return (
    <div className="bg-white rounded-2xl shadow p-3 w-[380px]">
      <div className="font-semibold mb-2">{title}</div>
      {figure ? (
        <Plot {...figure} style={{ width: '100%', height: '260px' }} useResizeHandler />
      ) : (
        <div className="text-sm text-gray-500">No figure</div>
      )}
    </div>
  );
}

const nodeTypes = { chart: ChartNode };

export default function App() {
  const [datasetId, setDatasetId] = useState(null);
  const [columns, setColumns] = useState([]);
  const [dims, setDims] = useState([]);
  const [measures, setMeasures] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [idCounter, setIdCounter] = useState(1);

  const onConnect = useCallback(async (params) => {
    // Attempt fusion when user connects two chart nodes
    const c1 = params.source; // chart id
    const c2 = params.target; // chart id
    try {
      const res = await fetch(`${API}/fuse`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ chart1_id: c1, chart2_id: c2 }) });
      if (!res.ok) throw new Error(await res.text());
      const fused = await res.json();
      const newId = fused.chart_id;
      const position = { x: params.position?.x || Math.random()*200 + 400, y: params.position?.y || Math.random()*200 };
      const figure = figureFromPayload(fused);
      setNodes(nds => nds.concat({ id: newId, type: 'chart', position, data: { title: fused.title + `\n(${fused.strategy.type})`, figure } }));
      setEdges(eds => addEdge(params, eds));
    } catch (e) {
      alert('Fusion failed: ' + e.message);
    }
  }, []);

  function figureFromPayload(payload) {
    const rows = payload.table || [];
    // Strategy A: same-dimension-different-measures => grouped bar
    if (payload.strategy?.type === 'same-dimension-different-measures') {
      const dims = payload.dimensions; // union, but original dims are equal across inputs
      const xKey = dims[0];
      const measureKeys = payload.measures.filter(m => m !== xKey);
      const xValues = [...new Set(rows.map(r => r[xKey]))];
      const data = measureKeys.map(m => ({
        type: 'bar',
        name: m,
        x: xValues,
        y: xValues.map(v => (rows.find(r => r[xKey] === v)?.[m]) ?? 0)
      }));
      return { data, layout: { title: payload.title, barmode: 'group', margin: { t: 30 } } };
    }
    // Strategy B: same-measure-different-dimensions => multi-series line
    if (payload.strategy?.type === 'same-measure-different-dimensions') {
      const groups = {};
      rows.forEach(r => {
        const g = r['DimensionType'];
        if (!groups[g]) groups[g] = [];
        groups[g].push(r);
      });
      const data = Object.entries(groups).map(([g, arr]) => ({
        type: 'scatter', mode: 'lines+markers', name: g,
        x: arr.map(a => a['DimensionValue']),
        y: arr.map(a => a['Value'])
      }));
      return { data, layout: { title: payload.title, margin: { t: 30 } } };
    }
    // Fallback: table-like bar of first numeric column
    const keys = rows.length ? Object.keys(rows[0]) : [];
    const numKey = keys.find(k => rows.some(r => typeof r[k] === 'number'));
    const xKey = keys.find(k => k !== numKey) || 'x';
    return { data: [{ type: 'bar', x: rows.map(r => r[xKey]), y: rows.map(r => r[numKey] || 0) }], layout: { title: payload.title } };
  }

  const uploadCSV = async (file) => {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${API}/upload`, { method: 'POST', body: fd });
    const meta = await res.json();
    setDatasetId(meta.dataset_id);
    setColumns(meta.columns);
  };

  const createChart = async () => {
    if (!datasetId) return alert('Upload a CSV first.');
    if (dims.length === 0 && measures.length === 0) return alert('Pick at least one dimension or measure');
    const body = { dataset_id: datasetId, dimensions: dims, measures, title: `Chart ${idCounter}` };
    const res = await fetch(`${API}/charts`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) return alert('Create chart failed');
    const chart = await res.json();
    const id = chart.chart_id;
    const figure = figureFromPayload(chart);
    setNodes(nds => nds.concat({ id, type: 'chart', position: { x: 80 + (nds.length*30), y: 80 + (nds.length*30) }, data: { title: chart.title, figure } }));
    setIdCounter(c => c + 1);
  };

  return (
    <div className="w-screen h-screen flex">
      <div className="w-72 border-r p-3 space-y-3 bg-gray-50">
        <div className="font-semibold">Chart Fusion</div>
        <input type="file" accept=".csv" onChange={e => e.target.files[0] && uploadCSV(e.target.files[0])} />
        <div className="text-xs text-gray-500">{datasetId ? `Dataset: ${datasetId}` : 'Upload CSV to begin'}</div>

        <div>
          <div className="text-sm font-medium mb-1">Dimensions</div>
          <select multiple className="w-full border p-1 h-24" onChange={(e)=> setDims(Array.from(e.target.selectedOptions).map(o=>o.value))}>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <div className="text-sm font-medium mb-1">Measures</div>
          <select multiple className="w-full border p-1 h-24" onChange={(e)=> setMeasures(Array.from(e.target.selectedOptions).map(o=>o.value))}>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <button className="w-full bg-black text-white rounded-xl py-2" onClick={createChart}>Create Chart Node</button>
        <div className="text-xs text-gray-500">Tip: Drag nodes. Connect two nodes to fuse. Pan/zoom the canvas.</div>
      </div>

      <div className="flex-1">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={setNodes}
          onEdgesChange={setEdges}
          onConnect={onConnect}
          fitView
        >
          <MiniMap />
          <Controls />
          <Background gap={16} />
        </ReactFlow>
      </div>
    </div>
  );
}
```

---

### How the Fusion Rules Are Enforced
- **Allowed** when:
  1. **Same dimension, different measures**: backend aggregates on that dimension and unions measures → frontend renders **grouped bar**.
  2. **Same measure, different dimensions**: backend stacks results with `DimensionType` + `DimensionValue` → frontend renders **multi-series line**.
- **Blocked** when neither rule holds → backend returns HTTP 400 with a clear message.

You can extend the backend strategy to suggest **stacked** vs **grouped** vs **dual-axis** based on scale or units, and add a **heatmap** path when granular joint dimension data is available.

