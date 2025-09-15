from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import uuid
import re
import ast
import operator
import json
from gemini_llm import GeminiDataFormulator

app = FastAPI(title="Chart Fusion Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
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

class ChartTableRequest(BaseModel):
    chart_id: str

class HistogramRequest(BaseModel):
    dataset_id: str
    measure: str

class DimensionCountRequest(BaseModel):
    dataset_id: str
    dimension: str

class ExpressionRequest(BaseModel):
    dataset_id: str
    expression: str
    filters: Optional[Dict[str, Any]] = {}

class ExpressionValidateRequest(BaseModel):
    dataset_id: str
    expression: str

class AIExploreRequest(BaseModel):
    chart_id: str
    user_query: str

class MetricCalculationRequest(BaseModel):
    user_query: str
    dataset_id: str

# -----------------------
# Helpers
# -----------------------

def _parse_expression(expression: str, dataset_id: str) -> Dict[str, Any]:
    """Parse expression and extract field references like @Revenue.Sum"""
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = DATASETS[dataset_id]
    available_measures = []
    
    # Get available measures from the dataset
    for col in df.columns:
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
            available_measures.append(col)
    
    # Find all @Field.Aggregation patterns (case-insensitive for both field and aggregation)
    pattern = r'@([a-zA-Z_][a-zA-Z0-9_]*)\.(Sum|Avg|Min|Max|Count)'
    matches = re.findall(pattern, expression, re.IGNORECASE)
    
    field_refs = []
    errors = []
    
    for field, agg in matches:
        # Find the actual field name with correct casing
        actual_field = None
        for measure in available_measures:
            if measure.lower() == field.lower():
                actual_field = measure
                break
        
        if actual_field is None:
            errors.append(f"Field '{field}' not found in dataset")
        else:
            field_refs.append({
                "field": actual_field,
                "aggregation": agg.lower(),
                "token": f"@{field}.{agg}"  # Keep original casing in token for replacement
            })
    
    # Validate mathematical expression structure
    # Remove field references and check if remaining is valid math
    temp_expr = expression
    for field, agg in matches:
        temp_expr = temp_expr.replace(f"@{field}.{agg}", "1")
    
    # More lenient validation - allow empty expressions and basic math
    if temp_expr.strip() and not re.match(r'^[0-9+\-*/().\s]+$', temp_expr.strip()):
        errors.append("Expression contains invalid characters")
    
    return {
        "field_refs": field_refs,
        "errors": errors,
        "available_measures": available_measures,
        "valid": len(errors) == 0
    }

def _evaluate_expression(expression: str, dataset_id: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """Evaluate expression with actual data"""
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = DATASETS[dataset_id].copy()
    
    # Apply filters if provided
    if filters:
        for field, values in filters.items():
            if field in df.columns and values:
                if isinstance(values, list):
                    df = df[df[field].isin(values)]
                else:
                    df = df[df[field] == values]
    
    # Parse expression to get field references
    parsed = _parse_expression(expression, dataset_id)
    if not parsed["valid"]:
        raise HTTPException(status_code=400, detail=f"Invalid expression: {', '.join(parsed['errors'])}")
    
    # Calculate aggregated values for each field reference
    field_values = {}
    for ref in parsed["field_refs"]:
        field = ref["field"]
        agg = ref["aggregation"]
        token = ref["token"]
        
        if agg == "sum":
            value = df[field].sum()
        elif agg == "avg":
            value = df[field].mean()
        elif agg == "min":
            value = df[field].min()
        elif agg == "max":
            value = df[field].max()
        elif agg == "count":
            value = df[field].count()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported aggregation: {agg}")
        
        field_values[token] = float(value) if pd.notna(value) else 0.0
    
    # Replace field references with actual values in expression
    eval_expr = expression
    for token, value in field_values.items():
        eval_expr = eval_expr.replace(token, str(value))
    
    # Safely evaluate the mathematical expression
    try:
        # Use ast.literal_eval for safety, but it doesn't support math operations
        # So we'll use a simple evaluator with allowed operators
        result = _safe_eval(eval_expr)
        return {
            "result": result,
            "field_values": field_values,
            "expression": expression,
            "evaluated_expression": eval_expr,
            "filters_applied": filters or {}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to evaluate expression: {str(e)}")

def _safe_eval(expr: str) -> float:
    """Safely evaluate mathematical expressions"""
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def eval_node(node):
        if isinstance(node, ast.Num):  # Numbers
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = eval_node(node.left)
            right = eval_node(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = eval_node(node.operand)
            return operators[type(node.op)](operand)
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")
    
    try:
        tree = ast.parse(expr, mode='eval')
        return eval_node(tree.body)
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")

def _categorize_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Automatically categorize columns into dimensions and measures based on data types"""
    dimensions = []
    measures = []
    
    for col in df.columns:
        # Check if column contains numeric data
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
            # Additional check: if all values are integers and there are few unique values,
            # it might be a categorical dimension (like year, month, etc.)
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            # If less than 10% unique values and all integers, treat as dimension
            if (unique_count / total_count < 0.1 and 
                unique_count < 20 and 
                df[col].dtype in ['int64', 'int32', 'int']):
                dimensions.append(col)
            else:
                measures.append(col)
        else:
            # Non-numeric columns are dimensions
            dimensions.append(col)
    
    return {"dimensions": dimensions, "measures": measures}

def _agg(df: pd.DataFrame, dimensions: List[str], measures: List[str], agg: str = "sum") -> pd.DataFrame:
    # Map frontend aggregation names to pandas aggregation names
    agg_mapping = {
        "sum": "sum",
        "avg": "mean",  # Frontend sends "avg", pandas expects "mean"
        "min": "min", 
        "max": "max",
        "count": "count"
    }
    pandas_agg = agg_mapping.get(agg, agg)
    
    # Support count aggregation without explicit measures
    if agg == "count":
        for col in dimensions:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column not found: {col}")
        if dimensions:
            grouped = df.groupby(dimensions).size().reset_index(name="count")
        else:
            grouped = pd.DataFrame({"count": [len(df)]})
        return grouped

    if not measures:
        raise HTTPException(status_code=400, detail="At least one measure is required for non-count aggregations")
    for col in dimensions + measures:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column not found: {col}")
    if dimensions:
        grouped = df.groupby(dimensions)[measures].agg(pandas_agg).reset_index()
    else:
        grouped = df[measures].agg(pandas_agg).to_frame().T
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
    
    # Automatically categorize columns
    categorized = _categorize_columns(df)
    
    return {
        "dataset_id": dataset_id,
        "columns": list(df.columns),  # Keep for backward compatibility
        "dimensions": categorized["dimensions"],
        "measures": categorized["measures"],
        "rows": len(df)
    }


def _generate_chart_title(dimensions: List[str], measures: List[str], agg: str = "sum") -> str:
    """Generate a descriptive chart title based on dimensions and measures"""
    if not dimensions and not measures:
        return "Empty Chart"
    
    measure_text = ""
    if measures:
        if len(measures) == 1:
            measure_text = measures[0]
        else:
            measure_text = f"{', '.join(measures[:-1])} and {measures[-1]}"
    
    if not dimensions:
        return f"Total {measure_text}" if measures else "Chart"
    
    dimension_text = ""
    if len(dimensions) == 1:
        dimension_text = f"by {dimensions[0]}"
    else:
        dimension_text = f"by {', '.join(dimensions[:-1])} and {dimensions[-1]}"
    
    if measures:
        return f"{measure_text} {dimension_text}"
    else:
        return f"Distribution {dimension_text}"

@app.post("/charts")
async def create_chart(spec: ChartCreate):
    if spec.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = DATASETS[spec.dataset_id]
    table = _agg(df, spec.dimensions, spec.measures, spec.agg)
    chart_id = str(uuid.uuid4())
    
    # Generate descriptive title if none provided
    auto_title = _generate_chart_title(spec.dimensions, spec.measures, spec.agg)
    
    CHARTS[chart_id] = {
        "chart_id": chart_id,
        "dataset_id": spec.dataset_id,
        "dimensions": spec.dimensions,
        "measures": (spec.measures if spec.measures else (["count"] if spec.agg == "count" else [])),
        "agg": spec.agg,
        "title": spec.title or auto_title,
        "table": table.to_dict(orient="records")
    }
    return CHARTS[chart_id]


@app.get("/charts/{chart_id}")
async def get_chart(chart_id: str):
    if chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    return CHARTS[chart_id]


@app.post("/chart-table")
async def get_chart_table(req: ChartTableRequest):
    """Generate table data for a given chart"""
    if req.chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    chart = CHARTS[req.chart_id]
    
    try:
        # For AI-generated charts, use pre-computed table data
        if chart.get("is_ai_generated", False) and "table" in chart and chart["table"]:
            print(f"📊 Returning pre-computed table for AI-generated chart: {req.chart_id}")
            
            # Extract headers from first row or from dimensions + measures
            table_data = chart["table"]
            if table_data:
                headers = list(table_data[0].keys())
                
                # Format the rows for display
                rows = []
                for record in table_data:
                    formatted_row = []
                    for header in headers:
                        val = record.get(header)
                        # Format numbers nicely and handle JSON-unsafe values
                        if isinstance(val, (int, float)):
                            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                                formatted_row.append("N/A")
                            elif isinstance(val, float) and val.is_integer():
                                formatted_row.append(int(val))
                            elif isinstance(val, float):
                                formatted_row.append(round(val, 2))
                            else:
                                formatted_row.append(val)
                        else:
                            formatted_row.append(str(val) if val is not None else "N/A")
                    rows.append(formatted_row)
                
                return {
                    "chart_id": req.chart_id,
                    "title": chart.get("title", "AI Generated Chart Table"),
                    "headers": headers,
                    "rows": rows,
                    "total_rows": len(rows)
                }
            else:
                # Empty AI chart
                return {
                    "chart_id": req.chart_id,
                    "title": chart.get("title", "Empty AI Chart"),
                    "headers": [],
                    "rows": [],
                    "total_rows": 0
                }
        
        # For regular charts, use the original aggregation logic
        dataset_id = chart["dataset_id"]
        
        if dataset_id not in DATASETS:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = DATASETS[dataset_id]
        dimensions = chart.get("dimensions", [])
        measures = chart.get("measures", [])
        agg = chart.get("agg", "sum")
        
        # Filter out 'count' from measures if it's synthetic
        actual_measures = [m for m in measures if m != "count"]
        
        print(f"📊 Generating table for regular chart: {req.chart_id}")
        
        # Use the same _agg function that was used to create the chart
        table_df = _agg(df, dimensions, actual_measures if actual_measures else measures, agg)
        
        # Prepare headers and rows
        headers = list(table_df.columns)
        rows = []
        
        for _, row in table_df.iterrows():
            formatted_row = []
            for val in row:
                # Format numbers nicely
                if isinstance(val, (int, float)):
                    if isinstance(val, float) and val.is_integer():
                        formatted_row.append(int(val))
                    elif isinstance(val, float):
                        formatted_row.append(round(val, 2))
                    else:
                        formatted_row.append(val)
                else:
                    formatted_row.append(str(val))
            rows.append(formatted_row)
        
        return {
            "chart_id": req.chart_id,
            "title": chart.get("title", "Chart Table"),
            "headers": headers,
            "rows": rows,
            "total_rows": len(rows)
        }
        
    except Exception as e:
        print(f"❌ Failed to generate table for chart {req.chart_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate table: {str(e)}")


@app.post("/fuse")
async def fuse(req: FuseRequest):
    if req.chart1_id not in CHARTS or req.chart2_id not in CHARTS:
        raise HTTPException(status_code=404, detail="One or both charts not found")

    c1, c2 = CHARTS[req.chart1_id], CHARTS[req.chart2_id]
    ds_id = c1["dataset_id"]
    if ds_id != c2["dataset_id"]:
        raise HTTPException(status_code=400, detail="Charts must come from the same dataset for fusion in this demo")
    df = DATASETS[ds_id]

    # Helper detectors for 1-variable charts
    def _is_measure_histogram(chart: Dict[str, Any]) -> bool:
        # measure-only: one measure, no dimensions, and not a synthetic 'count'
        return len(chart.get("dimensions", [])) == 0 and len(chart.get("measures", [])) == 1 and chart["measures"][0] != "count"

    def _is_dimension_count(chart: Dict[str, Any]) -> bool:
        # dimension-only count: one dimension and either no measures or a single 'count' measure
        dims = chart.get("dimensions", [])
        meas = chart.get("measures", [])
        return len(dims) == 1 and (len(meas) == 0 or (len(meas) == 1 and meas[0] == "count"))

    def _pick_agg(*charts: Dict[str, Any]) -> str:
        for ch in charts:
            val = ch.get("agg") if isinstance(ch, dict) else None
            if val:
                return val
        return "sum"

    # Defaults for metadata
    dims_out = list({*c1["dimensions"], *c2["dimensions"]})
    measures_out = list({*c1["measures"], *c2["measures"]})

    # Case A: Same Dimension + Different Measures
    if _same_dim_diff_measures(c1, c2):
        dims = c1["dimensions"]
        measures = sorted(list(set(c1["measures"]) | set(c2["measures"])))
        agg = _pick_agg(c1, c2)
        fused_table = _agg(df, dims, measures, agg).copy()
        strategy = {
            "type": "same-dimension-different-measures",
            "suggestion": "grouped-bar | stacked-bar | dual-axis-line"
        }
        title = f"Combined: {', '.join(measures)} by {', '.join(dims)}"
        dims_out = dims
        measures_out = measures

    # Case B: Same Measure + Different Dimensions -> Generate HEATMAP data
    elif _same_measure_diff_dims(c1, c2):
        common_measure = list(set(c1["measures"]).intersection(set(c2["measures"])))[0]
        agg = _pick_agg(c1, c2)
        
        # Get the two dimensions 
        dim1 = c1["dimensions"][0] if c1["dimensions"] else None
        dim2 = c2["dimensions"][0] if c2["dimensions"] else None
        
        if dim1 and dim2:
            # Simple approach: Just aggregate by both dimensions 
            # This gives us regular row-based data that's easy to work with
            fused_table = df.groupby([dim1, dim2])[common_measure].agg(agg).reset_index().to_dict('records')
            
            strategy = {
                "type": "same-measure-different-dimensions-stacked",
                "suggestion": "stacked-bar | bubble-chart"  
            }
            title = f"Stacked Bar: {common_measure} by {dim1} vs {dim2}"
            dims_out = [dim1, dim2]
            measures_out = [common_measure]
        else:
            # Fallback to old behavior if no proper dimensions
            t1 = _agg(df, c1["dimensions"], [common_measure], agg).copy()
            t2 = _agg(df, c2["dimensions"], [common_measure], agg).copy()
            t1["__DimensionType__"] = ",".join(c1["dimensions"]) or "(none)"
            t2["__DimensionType__"] = ",".join(c2["dimensions"]) or "(none)"
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
            title = f"Comparison: {common_measure} across different dimensions"
            dims_out = list({*c1["dimensions"], *c2["dimensions"]})
            measures_out = [common_measure]

    # Case C: 1-variable charts
    elif (_is_measure_histogram(c1) and _is_dimension_count(c2)) or (_is_measure_histogram(c2) and _is_dimension_count(c1)):
        # Measure-only + Dimension-count -> build measure by dimension
        measure_chart = c1 if _is_measure_histogram(c1) else c2
        dimension_chart = c2 if _is_dimension_count(c2) else c1
        measure = measure_chart["measures"][0]
        dim = dimension_chart["dimensions"][0]
        agg = _pick_agg(measure_chart, dimension_chart)
        fused_table = _agg(df, [dim], [measure], agg).copy()
        strategy = {
            "type": "measure-by-dimension",
            "suggestion": "bar | line | heatmap"
        }
        title = f"Distribution of {measure} by {dim}"
        dims_out = [dim]
        measures_out = [measure]

    # Optional generic 1-variable cases
    elif _is_measure_histogram(c1) and _is_measure_histogram(c2):
        m1, m2 = c1["measures"][0], c2["measures"][0]
        fused_table = df[[m1, m2]].dropna().copy()
        strategy = {"type": "measure-vs-measure", "suggestion": "scatter | dual-histogram"}
        title = f"{m1} vs {m2}"
        dims_out = []
        measures_out = [m1, m2]
        agg = _pick_agg(c1, c2)

    elif _is_dimension_count(c1) and _is_dimension_count(c2):
        d1, d2 = c1["dimensions"][0], c2["dimensions"][0]
        fused_table = df.groupby([d1, d2]).size().reset_index(name="Count")
        strategy = {"type": "dimension-vs-dimension", "suggestion": "heatmap | mosaic | grouped-bar"}
        title = f"{d1} vs {d2} (frequency)"
        dims_out = [d1, d2]
        measures_out = ["Count"]
        agg = _pick_agg(c1, c2)

    # Case D: Permissive same-dimension union of measures (robust fallback)
    elif len(set(c1.get("dimensions", [])).intersection(set(c2.get("dimensions", [])))) >= 1 and \
         len(set(c1.get("measures", [])).union(set(c2.get("measures", [])))) >= 2:
        common_dims = list(set(c1["dimensions"]).intersection(set(c2["dimensions"])))
        # Preserve original order using c1's order
        common_dims = [d for d in c1["dimensions"] if d in common_dims]
        measures = sorted(list(set(c1.get("measures", [])).union(set(c2.get("measures", [])))))
        # Aggregate on first common dimension if multiple
        dims_use = [common_dims[0]] if common_dims else []
        if not dims_use:
            raise HTTPException(status_code=400, detail="Fusion failed: no common dimension found")
        agg = _pick_agg(c1, c2)
        fused_table = _agg(df, dims_use, measures, agg).copy()
        strategy = {"type": "same-dimension-different-measures", "suggestion": "grouped-bar | stacked-bar | dual-axis-line"}
        title = f"Combined: {', '.join(measures)} by {', '.join(dims_use)}"
        dims_out = dims_use
        measures_out = measures

    else:
        raise HTTPException(status_code=400, detail="Fusion not allowed: charts must share either a dimension (and differ in measures) or share a single common measure (and differ in dimensions)")

    chart_id = str(uuid.uuid4())
    
    # Handle different table formats (DataFrame vs dict for heatmap)
    if isinstance(fused_table, dict):
        # Already a dict (old heatmap case - shouldn't happen anymore)
        table_data = fused_table
    elif isinstance(fused_table, list):
        # Already a list of records (new stacked case)
        table_data = fused_table
    else:
        # DataFrame - convert to records
        table_data = fused_table.to_dict(orient="records")
    
    fused_payload = {
        "chart_id": chart_id,
        "dataset_id": ds_id,
        "dimensions": dims_out,
        "measures": measures_out,
        "agg": agg if 'agg' in locals() else _pick_agg(c1, c2),
        "title": title,
        "strategy": strategy,
        "table": table_data,
    }
    CHARTS[chart_id] = fused_payload
    return fused_payload


@app.post("/histogram")
async def histogram(req: HistogramRequest):
    if req.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = DATASETS[req.dataset_id]
    if req.measure not in df.columns:
        raise HTTPException(status_code=400, detail=f"Measure column not found: {req.measure}")

    # Convert to numeric and drop NaNs for clean histogram
    series = pd.to_numeric(df[req.measure], errors="coerce").dropna()
    stats = {
        "sum": float(series.sum()),
        "avg": float(series.mean()),
        "max": float(series.max()),
        "min": float(series.min()),
        "count": int(series.count()),
    }
    # Return raw values; frontend will build the histogram bins
    values = series.tolist()
    return {"measure": req.measure, "values": values, "stats": stats}


@app.post("/dimension_counts")
async def dimension_counts(req: DimensionCountRequest):
    if req.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = DATASETS[req.dataset_id]
    if req.dimension not in df.columns:
        raise HTTPException(status_code=400, detail=f"Dimension column not found: {req.dimension}")

    series = df[req.dimension].dropna()
    vc = series.value_counts(dropna=False)
    labels = [str(k) for k in vc.index.tolist()]
    counts = [int(v) for v in vc.tolist()]
    return {"dimension": req.dimension, "labels": labels, "counts": counts, "total": int(series.shape[0])}


@app.post("/expression/validate")
async def validate_expression(req: ExpressionValidateRequest):
    """Validate expression syntax and field references"""
    try:
        parsed = _parse_expression(req.expression, req.dataset_id)
        return {
            "valid": parsed["valid"],
            "errors": parsed["errors"],
            "field_refs": parsed["field_refs"],
            "available_measures": parsed["available_measures"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/expression/evaluate")
async def evaluate_expression(req: ExpressionRequest):
    """Evaluate expression and return result"""
    try:
        result = _evaluate_expression(req.expression, req.dataset_id, req.filters)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/dataset/{dataset_id}/measures")
async def get_dataset_measures(dataset_id: str):
    """Get available measures for autocomplete"""
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = DATASETS[dataset_id]
    measures = []
    dimensions = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
            measures.append(col)
        else:
            dimensions.append(col)
    
    return {
        "measures": measures,
        "dimensions": dimensions,
        "aggregations": ["Sum", "Avg", "Min", "Max", "Count"]
    }


# Initialize AI Data Formulator
ai_formulator = GeminiDataFormulator()

@app.post("/ai-explore")
async def ai_explore_data(request: AIExploreRequest):
    """
    AI-powered data exploration using Gemini + Data Formulator
    Takes a chart and natural language query, returns transformed data and new chart suggestion
    """
    if request.chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    # Get chart context
    chart = CHARTS[request.chart_id]
    dataset_id = chart["dataset_id"]
    
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Prepare chart context for AI
        chart_context = {
            "chart_id": request.chart_id,
            "dataset_id": dataset_id,
            "dimensions": chart.get("dimensions", []),
            "measures": chart.get("measures", []),
            "agg": chart.get("agg", "sum"),
            "table": chart.get("table", []),
            "title": chart.get("title", "")
        }
        
        # Debug logging
        print(f"🤖 AI Exploration started:")
        print(f"   Query: '{request.user_query}'")
        print(f"   Chart context: dimensions={chart_context['dimensions']}, measures={chart_context['measures']}")
        print(f"   Data rows: {len(chart_context['table'])}")
        
        # Call AI data exploration
        ai_result = ai_formulator.explore_data(request.user_query, chart_context)
        print(f"   AI result: dimensions={ai_result['dimensions']}, measures={ai_result['measures']}")
        print(f"   Transformations: {ai_result['transformations']}")
        
        # Generate new chart ID for the transformed result
        new_chart_id = str(uuid.uuid4())
        
        # Convert transformed DataFrame back to table format
        transformed_df = ai_result["data"]
        if not transformed_df.empty:
            # COMPREHENSIVE JSON serialization compatibility fix
            import numpy as np
            import json
            
            # Step 1: Handle NaN/inf in ALL columns (not just numeric)
            for col in transformed_df.columns:
                # Replace inf/-inf with None
                transformed_df[col] = transformed_df[col].replace([np.inf, -np.inf], None)
                # Replace NaN with None 
                transformed_df[col] = transformed_df[col].where(pd.notna(transformed_df[col]), None)
                
                # Additional safety: handle any remaining problematic numeric types
                if transformed_df[col].dtype.kind in 'fc':  # float or complex
                    # Convert any remaining non-finite values to None
                    mask = ~np.isfinite(transformed_df[col].astype(float, errors='ignore'))
                    transformed_df.loc[mask, col] = None
            
            # Step 2: Convert to records and validate JSON compatibility
            table_data = transformed_df.to_dict(orient="records")
            
            # Step 3: Final safety check - test JSON serialization
            try:
                json.dumps(table_data)  # Test serialization
            except (ValueError, TypeError) as json_error:
                print(f"JSON serialization error caught, applying emergency fix: {json_error}")
                # Emergency fix: convert any remaining problematic values
                for record in table_data:
                    for key, value in record.items():
                        if isinstance(value, float):
                            if not np.isfinite(value):
                                record[key] = None
                        elif hasattr(value, 'dtype') and value.dtype.kind in 'fc':
                            try:
                                float_val = float(value)
                                if not np.isfinite(float_val):
                                    record[key] = None
                            except (ValueError, TypeError, OverflowError):
                                record[key] = None
        else:
            table_data = []
        
        # Create new chart configuration
        new_chart = {
            "chart_id": new_chart_id,
            "dataset_id": dataset_id,
            "dimensions": ai_result["dimensions"],
            "measures": ai_result["measures"],
            "agg": chart.get("agg", "sum"),  # Keep original aggregation unless changed
            "title": f"AI Explored: {request.user_query[:50]}{'...' if len(request.user_query) > 50 else ''}",
            "table": table_data,
            "is_ai_generated": True,
            "source_chart_id": request.chart_id,
            "transformation_log": ai_result["transformations"],
            "ai_query": request.user_query
        }
        
        # Proactive JSON safety check for chart object before storing
        try:
            json.dumps(new_chart)
            print(f"   Chart object JSON check: ✅ PASSED")
        except (ValueError, TypeError) as chart_error:
            print(f"❌ Chart object JSON check failed: {chart_error}")
            # Apply JSON safety to chart object
            def make_chart_json_safe(obj):
                if isinstance(obj, dict):
                    return {k: make_chart_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_chart_json_safe(item) for item in obj]
                elif isinstance(obj, float):
                    if not np.isfinite(obj):
                        return None
                    return obj
                elif hasattr(obj, 'dtype') and obj.dtype.kind in 'fc':
                    try:
                        float_val = float(obj)
                        return float_val if np.isfinite(float_val) else None
                    except (ValueError, TypeError, OverflowError):
                        return None
                else:
                    return obj
            
            new_chart = make_chart_json_safe(new_chart)
            print(f"   Chart object JSON safety applied")
        
        # Store the new chart
        CHARTS[new_chart_id] = new_chart
        
        # Prepare response with JSON safety
        response_data = {
            "success": True,
            "new_chart": new_chart,
            "chart_suggestion": ai_result["chart_suggestion"],
            "transformations": ai_result["transformations"],
            "ai_response": ai_result.get("ai_response", ""),
            "original_query": request.user_query,
            "data_shape": {
                "rows": len(table_data),
                "columns": len(transformed_df.columns) if not transformed_df.empty else 0
            }
        }
        
        # Final JSON safety check for entire response with detailed debugging
        try:
            json.dumps(response_data)
            print(f"   JSON serialization check: ✅ PASSED")
        except (ValueError, TypeError) as final_error:
            print(f"❌ Final JSON safety check failed: {final_error}")
            print(f"   Analyzing response structure...")
            
            # Debug each part of the response
            for key, value in response_data.items():
                try:
                    json.dumps(value)
                    print(f"   ✅ {key}: OK")
                except Exception as part_error:
                    print(f"   ❌ {key}: FAILED - {part_error}")
                    if key == "new_chart" and isinstance(value, dict) and "table" in value:
                        print(f"      Checking table data...")
                        for i, row in enumerate(value["table"][:3]):  # Check first 3 rows
                            try:
                                json.dumps(row)
                                print(f"      ✅ Row {i}: OK")
                            except Exception as row_error:
                                print(f"      ❌ Row {i}: {row_error}")
                                for col, col_val in row.items():
                                    try:
                                        json.dumps(col_val)
                                    except Exception as col_error:
                                        print(f"         ❌ {col}: {col_val} ({type(col_val)}) - {col_error}")
            
            # Fallback: ensure all values are JSON-safe
            def make_json_safe(obj):
                if isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(item) for item in obj]
                elif isinstance(obj, float):
                    if not np.isfinite(obj):
                        return None
                    return obj
                elif hasattr(obj, 'dtype') and obj.dtype.kind in 'fc':
                    try:
                        float_val = float(obj)
                        return float_val if np.isfinite(float_val) else None
                    except (ValueError, TypeError, OverflowError):
                        return None
                else:
                    return obj
            
            print(f"   Applying comprehensive JSON safety fix...")
            response_data = make_json_safe(response_data)
            
            # Test again
            try:
                json.dumps(response_data)
                print(f"   ✅ JSON safety fix successful!")
            except Exception as final_final_error:
                print(f"   ❌ JSON safety fix still failed: {final_final_error}")
                # Last resort: return error response
                return {
                    "success": False,
                    "error": f"JSON serialization failed: {str(final_final_error)}",
                    "debug_info": "Comprehensive safety fix applied but still failing"
                }
        
        print(f"✅ AI Exploration completed successfully!")
        print(f"   New chart created: {new_chart_id}")
        print(f"   Data rows: {len(table_data)}")
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI exploration failed: {str(e)}")


@app.post("/ai-calculate-metric")
async def ai_calculate_metric(request: MetricCalculationRequest):
    """
    AI-powered metric calculation using natural language
    Takes a dataset and natural language query, returns calculated metric value
    """
    if request.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = DATASETS[request.dataset_id]
    
    try:
        print(f"🧮 AI Metric calculation request:")
        print(f"   Dataset: {request.dataset_id}")
        print(f"   Query: '{request.user_query}'")
        print(f"   Data shape: {df.shape}")
        
        # Use AI to calculate the metric
        result = ai_formulator.calculate_metric(request.user_query, request.dataset_id, df)
        
        print(f"🧮 AI Metric calculation result:")
        print(f"   Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"   Value: {result.get('value')}")
            print(f"   Formatted: {result.get('formatted_value')}")
        else:
            print(f"   Error: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ AI Metric calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate metric: {str(e)}")
