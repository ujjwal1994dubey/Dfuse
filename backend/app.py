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
    api_key: Optional[str] = None
    model: str = "gemini-2.0-flash"

class MetricCalculationRequest(BaseModel):
    user_query: str
    dataset_id: str
    api_key: Optional[str] = None
    model: str = "gemini-2.0-flash"

class ConfigTestRequest(BaseModel):
    api_key: str
    model: str = "gemini-2.0-flash"

# -----------------------
# Helpers
# -----------------------

def _parse_expression(expression: str, dataset_id: str) -> Dict[str, Any]:
    """
    Parse Expression Helper
    Parses mathematical expressions containing field references in @Field.Aggregation format.
    Validates field names against dataset columns and checks expression syntax.
    
    Args:
        expression: Mathematical expression string (e.g., "@Revenue.Sum - @Cost.Avg")
        dataset_id: ID of the dataset to validate fields against
    
    Returns:
        Dictionary containing field_refs, errors, available_measures, and valid flag
    
    Examples:
        "@Revenue.Sum * 2" -> extracts Revenue field with Sum aggregation
        "@Price.Avg + @Tax.Max" -> extracts Price (Avg) and Tax (Max)
    """
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
    """
    Evaluate Expression Helper
    Evaluates a parsed mathematical expression with actual aggregated values from the dataset.
    Applies optional filters before aggregation.
    
    Args:
        expression: Mathematical expression with @Field.Aggregation references
        dataset_id: ID of the dataset to evaluate against
        filters: Optional dictionary of dimension filters {dimension: [values]}
    
    Returns:
        Dictionary with result, field_values, expression, evaluated_expression, filters_applied
    
    Process:
        1. Apply filters to dataset
        2. Calculate aggregated values for each field reference
        3. Replace field references with actual values
        4. Safely evaluate the mathematical expression
    """
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
    """
    Safe Expression Evaluator
    Safely evaluates mathematical expressions using Python's AST module.
    Only allows basic arithmetic operations (no exec/eval vulnerabilities).
    
    Args:
        expr: Mathematical expression string (e.g., "100 + 50 * 2")
    
    Returns:
        float: Computed result
    
    Allowed Operations:
        - Addition, Subtraction, Multiplication, Division
        - Unary operations (negation, positive)
        - Numbers and constants
    
    Security: Uses AST parsing to prevent code injection attacks
    """
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
    """
    Column Categorization Helper
    Automatically categorizes DataFrame columns into dimensions (categorical) and measures (numeric).
    Uses heuristics to distinguish between numeric dimensions (e.g., Year) and true measures.
    
    Args:
        df: Pandas DataFrame to analyze
    
    Returns:
        Dictionary with 'dimensions' and 'measures' lists
    
    Logic:
        - Numeric columns with <10% unique values and <20 unique total -> dimension
        - Other numeric columns -> measures
        - Non-numeric columns -> dimensions
    """
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
    """
    Aggregation Helper
    Performs data aggregation across specified dimensions and measures.
    Supports sum, avg, min, max, and count aggregations.
    
    Args:
        df: Source DataFrame
        dimensions: List of columns to group by
        measures: List of numeric columns to aggregate
        agg: Aggregation method ('sum', 'avg', 'min', 'max', 'count')
    
    Returns:
        Aggregated DataFrame
    
    Special Cases:
        - count aggregation doesn't require measures
        - Maps 'avg' to pandas 'mean'
        - Handles both grouped and non-grouped aggregations
    """
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
    """
    Chart Fusion Pattern Detector: Same Dimension, Different Measures
    Checks if two charts share the same dimensions but have different measures.
    Used to determine if charts can be fused into a multi-measure visualization.
    
    Example: Both charts show data by "State" but one shows "Revenue", other shows "Population"
    """
    return spec1["dimensions"] == spec2["dimensions"] and set(spec1["measures"]) != set(spec2["measures"]) and len(spec1["dimensions"]) > 0


def _same_measure_diff_dims(spec1, spec2):
    """
    Chart Fusion Pattern Detector: Same Measure, Different Dimensions
    Checks if two charts share exactly one common measure but have different dimensions.
    Used to create comparison or stacked visualizations.
    
    Example: Both charts show "Revenue" but one groups by "Region", other by "Product"
    """
    common_measures = set(spec1["measures"]).intersection(set(spec2["measures"]))
    return (len(common_measures) == 1) and (spec1["dimensions"] != spec2["dimensions"]) and (len(spec1["dimensions"]) > 0 or len(spec2["dimensions"]) > 0)




# -----------------------
# Routes
# -----------------------

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    CSV Upload Endpoint
    Uploads and processes a CSV file, stores it in memory with a unique ID.
    Automatically categorizes columns into dimensions and measures.
    
    Returns:
        - dataset_id: Unique identifier for the uploaded dataset
        - columns: All column names (for backward compatibility)
        - dimensions: Categorical columns
        - measures: Numeric columns
        - rows: Total row count
    """
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
    """
    Chart Title Generator
    Automatically generates human-readable chart titles based on chart configuration.
    Creates natural language descriptions of what the chart displays.
    
    Args:
        dimensions: List of dimension columns
        measures: List of measure columns
        agg: Aggregation method
    
    Examples:
        dimensions=["State"], measures=["Revenue"] -> "Revenue by State"
        dimensions=["State", "Year"], measures=["Revenue", "Cost"] -> "Revenue and Cost by State and Year"
        dimensions=[], measures=["Revenue"] -> "Total Revenue"
    """
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
    """
    Chart Creation Endpoint
    Creates a new chart by aggregating data from a dataset.
    Stores chart metadata and aggregated table data in memory.
    
    Args:
        spec: ChartCreate model with dataset_id, dimensions, measures, agg, title
    
    Returns:
        Chart object with chart_id, metadata, and aggregated table data
    
    Process:
        1. Validates dataset exists
        2. Aggregates data using _agg helper
        3. Generates auto-title if not provided
        4. Stores chart in CHARTS registry
    """
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
    """
    Get Chart Endpoint
    Retrieves a previously created chart by its ID.
    Returns complete chart metadata including aggregated data.
    """
    if chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    return CHARTS[chart_id]


@app.post("/chart-table")
async def get_chart_table(req: ChartTableRequest):
    """
    Chart Table Endpoint
    Generates formatted table data for displaying chart data in tabular format.
    Handles both AI-generated and regular charts differently.
    
    Features:
        - Pre-computed table data for AI-generated charts
        - On-demand aggregation for regular charts
        - Number formatting (integers, decimals, N/A for nulls)
        - Returns headers and rows ready for UI display
    """
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
    """
    Chart Fusion Endpoint
    Intelligently merges two charts from the same dataset based on their structure.
    Detects fusion patterns and creates appropriate combined visualizations.
    
    Fusion Patterns Supported:
        1. Same Dimension + Different Measures -> Grouped/Stacked Bar
        2. Same Measure + Different Dimensions -> Stacked/Comparison Chart
        3. Measure Histogram + Dimension Count -> Distribution by Dimension
        4. Two Measure Histograms -> Scatter Plot
        5. Two Dimension Counts -> Heatmap/Frequency Chart
    
    Returns:
        Fused chart with merged data, strategy recommendation, and visualization hints
    """
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
    """
    Histogram Data Endpoint
    Returns raw numeric values for a measure to create histograms on the frontend.
    Includes statistical summary (sum, avg, min, max, count).
    """
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
    """
    Dimension Counts Endpoint
    Returns value counts for a categorical dimension.
    Used for bar charts and filter value lists.
    """
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
    """
    Expression Validation Endpoint
    Validates mathematical expression syntax and field references.
    Returns validation errors and available measures for autocomplete.
    """
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
    """
    Expression Evaluation Endpoint
    Evaluates a validated mathematical expression with actual dataset values.
    Supports filtering before aggregation.
    """
    try:
        result = _evaluate_expression(req.expression, req.dataset_id, req.filters)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/dataset/{dataset_id}/measures")
async def get_dataset_measures(dataset_id: str):
    """
    Dataset Measures Endpoint
    Returns available measures, dimensions, and aggregation options for a dataset.
    Used for autocomplete and validation in expression nodes.
    """
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


# Initialize AI Data Formulator per request now (removed global instance)

@app.post("/test-config")
async def test_config(request: ConfigTestRequest):
    """
    AI Configuration Test Endpoint
    Tests user's Gemini API key and model configuration.
    Verifies credentials work before using AI features.
    
    Returns:
        success: bool, error message if failed, token_usage for test query
    """
    try:
        print(f"🔧 Testing configuration:")
        print(f"   API Key: {'*' * (len(request.api_key)-8) + request.api_key[-8:] if len(request.api_key) > 8 else '***'}")
        print(f"   Model: {request.model}")
        
        # Create AI formulator with user's credentials
        ai_formulator = GeminiDataFormulator(api_key=request.api_key, model=request.model)
        
        # Test the configuration
        result = ai_formulator.test_configuration()
        
        print(f"✅ Configuration test result: {result.get('success', False)}")
        return result
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return {
            "success": False,
            "error": f"Configuration test failed: {str(e)}"
        }

@app.post("/ai-explore")
async def ai_explore_data(request: AIExploreRequest):
    """
    AI Data Exploration Endpoint
    AI-powered data exploration using Gemini LLM and pandas DataFrame analysis.
    Answers natural language questions about chart data.
    
    Features:
        - Natural language query processing
        - Generates and executes pandas code
        - Returns text answers with optional tabular data
        - Tracks token usage for cost estimation
    
    Args:
        request: Contains chart_id, user_query, api_key, model
    
    Returns:
        success, answer, code_steps, reasoning_steps, tabular_data, token_usage
    """
    if request.chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    # Get chart context
    chart = CHARTS[request.chart_id]
    dataset_id = chart["dataset_id"]
    
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Get the full original dataset
        full_dataset = DATASETS[dataset_id]
        
        print(f"🤖 AI Exploration started:")
        print(f"   Query: '{request.user_query}'")
        print(f"   Dataset shape: {full_dataset.shape}")
        print(f"   Model: {request.model}")
        print(f"   API Key: {'*' * (len(request.api_key or '')-8) + (request.api_key or '')[-8:] if (request.api_key or '') and len(request.api_key) > 8 else '***'}")
        
        # Create AI formulator with user's credentials
        ai_formulator = GeminiDataFormulator(api_key=request.api_key, model=request.model)
        
        # Use pandas DataFrame agent for text-based results
        ai_result = ai_formulator.get_text_analysis(request.user_query, full_dataset)
        
        print(f"✅ AI Analysis completed successfully!")
        print(f"   Result length: {len(ai_result.get('answer', ''))}")
        
        # Return complete AI analysis response including code_steps and token_usage
        return {
            "success": ai_result.get("success", True),
            "answer": ai_result.get("answer", "I couldn't process your query."),
            "query": request.user_query,
            "dataset_info": f"Dataset: {full_dataset.shape[0]} rows, {full_dataset.shape[1]} columns",
            "code_steps": ai_result.get("code_steps", []),
            "reasoning_steps": ai_result.get("reasoning_steps", []),
            "tabular_data": ai_result.get("tabular_data", []),
            "has_table": ai_result.get("has_table", False),
            "token_usage": ai_result.get("token_usage", {})
        }
        
    except Exception as e:
        print(f"❌ AI Exploration failed: {str(e)}")
        error_message = str(e)
        if "401" in error_message or "403" in error_message or "API key" in error_message:
            error_message += " Please check your API key in Settings."
        
        return {
            "success": False,
            "answer": f"I encountered an error while processing your query: {error_message}",
            "query": request.user_query,
            "dataset_info": "",
            "code_steps": [],
            "reasoning_steps": [],
            "tabular_data": [],
            "has_table": False,
            "token_usage": {}
        }


@app.post("/ai-calculate-metric")
async def ai_calculate_metric(request: MetricCalculationRequest):
    """
    AI Metric Calculation Endpoint
    Calculates metrics from natural language descriptions using AI.
    Used by expression nodes to compute values from text queries.
    
    Example Queries:
        - "What is the average revenue per state?"
        - "Calculate total population growth from 2018 to 2023"
    
    Returns:
        success, value, formatted_value, explanation, token_usage
    """
    if request.dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = DATASETS[request.dataset_id]
    
    try:
        print(f"🧮 AI Metric calculation started:")
        print(f"   Query: '{request.user_query}'")
        print(f"   Dataset: {request.dataset_id}")
        print(f"   Data shape: {df.shape}")
        print(f"   Model: {request.model}")
        print(f"   API Key: {'*' * (len(request.api_key or '')-8) + (request.api_key or '')[-8:] if (request.api_key or '') and len(request.api_key) > 8 else '***'}")
        
        # Create AI formulator with user's credentials
        ai_formulator = GeminiDataFormulator(api_key=request.api_key, model=request.model)
        
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
        error_message = str(e)
        if "401" in error_message or "403" in error_message or "API key" in error_message:
            error_message += " Please check your API key in Settings."
        
        raise HTTPException(status_code=500, detail=f"Failed to calculate metric: {error_message}")




@app.post("/list-models")
def list_models(api_key: str) -> List[Dict[str, str]]:
    """
    List Available AI Models Endpoint
    Returns list of available Gemini models for user selection.
    Currently returns a static list of Gemini models.
    """
    # Google Gemini doesn't expose a simple model list API yet.
    return [
        {"label": "Gemini 1.5 Pro", "value": "gemini-1.5-pro"},
        {"label": "Gemini 1.5 Flash", "value": "gemini-1.5-flash"},
        {"label": "Gemini 2.5 Flash", "value": "gemini-2.5-flash"}
    ]

class ChartInsightRequest(BaseModel):
    chart_id: str
    api_key: str
    model: str = "gemini-2.0-flash"

@app.post("/chart-insights")
async def generate_chart_insights(request: ChartInsightRequest):
    """
    Chart Insights Generation Endpoint
    Generates AI-powered statistical insights for a chart.
    Creates concise summaries highlighting key findings.
    
    Process:
        1. Calculates basic statistics (min, max, mean, total)
        2. Uses Gemini LLM to generate human-readable insight
        3. Returns 2-3 sentence summary with token usage
    
    Used by: Insight sticky notes feature
    """
    if request.chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    chart = CHARTS[request.chart_id]
    dataset = DATASETS.get(chart["dataset_id"])
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create statistical summary prompt
    dimensions = chart.get("dimensions", [])
    measures = chart.get("measures", [])
    table_data = chart.get("table", [])
    
    # Calculate basic statistics
    stats = {}
    for measure in measures:
        values = [row.get(measure, 0) for row in table_data if isinstance(row.get(measure), (int, float))]
        if values:
            stats[measure] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "total": sum(values)
            }
    
    # Generate insight using Gemini
    formulator = GeminiDataFormulator(api_key=request.api_key, model=request.model)
    
    prompt = f"""Generate a brief statistical summary (2-3 sentences) for this chart.

Chart Title: {chart.get('title', 'Untitled Chart')}
Dimensions: {dimensions}
Measures: {measures}
Statistics: {json.dumps(stats, indent=2)}

Top 5 data points:
{json.dumps(table_data[:5], indent=2)}

Provide only the insight text without any headers or formatting."""

    response, token_usage = formulator.run_gemini_with_usage(prompt)
    
    return {
        "success": True,
        "insight": response.strip(),
        "statistics": stats,
        "token_usage": token_usage
    }

class ReportSectionRequest(BaseModel):
    chart_id: str
    api_key: str
    model: str = "gemini-2.0-flash"
    ai_explore_result: Optional[str] = None

@app.post("/generate-report-section")
async def generate_report_section(request: ReportSectionRequest):
    """
    Report Section Generation Endpoint
    Creates LLM-enhanced report sections for charts with professional formatting.
    Combines statistical analysis with AI exploration results.
    
    Features:
        - Auto-generates clear, concise subheadings (3-6 words)
        - Produces short, crisp analysis (3-4 sentences)
        - Incorporates AI exploration results if available
        - Uses markdown formatting for easy editing
        - Optimized for readability and actionable insights
    
    Returns:
        success, report_section (markdown), chart_title, statistics, token_usage
    
    Used by: "Add to Report" feature in chart context menus
    """
    if request.chart_id not in CHARTS:
        raise HTTPException(status_code=404, detail="Chart not found")
    
    chart = CHARTS[request.chart_id]
    dataset = DATASETS.get(chart["dataset_id"])
    
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Get chart metadata
    title = chart.get('title', 'Untitled Chart')
    dimensions = chart.get("dimensions", [])
    measures = chart.get("measures", [])
    table_data = chart.get("table", [])
    
    # Calculate statistics
    stats = {}
    for measure in measures:
        values = [row.get(measure, 0) for row in table_data if isinstance(row.get(measure), (int, float))]
        if values:
            stats[measure] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "total": sum(values)
            }
    
    # Build comprehensive prompt for LLM
    formulator = GeminiDataFormulator(api_key=request.api_key, model=request.model)
    
    ai_section = f"""

**AI Exploration Results:**
{request.ai_explore_result}
""" if request.ai_explore_result else ""
    
    prompt = f"""You are a data analyst writing a concise, clear report section. Generate SHORT, CRISP, and EASY-TO-UNDERSTAND analysis.

**Chart Information:**
Title: {title}
Dimensions: {dimensions}
Measures: {measures}

**Statistical Summary:**
{json.dumps(stats, indent=2)}

**Top Data Points:**
{json.dumps(table_data[:10], indent=2)}
{ai_section}

**Instructions:**
1. Start with a SHORT, CLEAR subheading (3-6 words max, use ## markdown format) - NOT just the chart title
2. Write ONLY 3-4 SHORT sentences (max 2 paragraphs)
3. Be CRISP and DIRECT - no fluff or verbose language
4. Highlight ONLY the most important finding or trend
5. Use SIMPLE, EASY-TO-UNDERSTAND language (avoid jargon)
6. If AI Exploration Results are provided, incorporate those insights
7. Keep it CONCISE and ACTIONABLE

Example good subheading: "## Strong Growth in Q3"
Example bad subheading: "## Analysis of Population by State Chart"

Generate the SHORT, CRISP report section now:"""

    response, token_usage = formulator.run_gemini_with_usage(prompt)
    
    return {
        "success": True,
        "report_section": response.strip(),
        "chart_title": title,
        "chart_id": request.chart_id,
        "statistics": stats,
        "token_usage": token_usage
    }

           
