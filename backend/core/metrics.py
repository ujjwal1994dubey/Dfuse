import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import re
import google.generativeai as genai

# Pandas DataFrame Agent imports
try:
    from langchain.agents.agent_types import AgentType
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_google_genai import GoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False

class DfuseMetrics:
    """
    Data Formulator with Gemini 2.0 Flash for natural language data transformations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "AIzaSyCme3aE7H9TmRgKqHCqhqV8f-9FhIqDfOM"
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize pandas DataFrame agent if langchain is available
        self.pandas_agent = None
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = GoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                print("✅ Pandas DataFrame Agent initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize pandas DataFrame agent: {e}")
                self.llm = None
        else:
            print("⚠️  LangChain not available, using structured parsing only")


    def calculate_metric(self, user_query: str, dataset_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate a single metric value using natural language via pandas DataFrame agent"""
        try:
            print(f"🧮 Metric calculation started:")
            print(f"   Query: '{user_query}'")
            print(f"   Dataset: {dataset_id}")
            print(f"   Data shape: {data.shape}")
            
            # Get sample of data for AI context
            sample_data = data.head(3).to_dict('records') if not data.empty else []
            
            # Enhanced prompt for metric calculation using pandas DataFrame agent approach
            prompt = f"""
You are a data analyst calculating metrics from a dataset using pandas.

DATASET INFO:
- Columns: {list(data.columns)}
- Shape: {data.shape[0]} rows, {data.shape[1]} columns
- Sample data: {sample_data}

USER REQUEST: {user_query}

Generate a single line of pandas code to calculate the requested metric. The DataFrame is available as 'data'.

EXAMPLES:
Request: "What's our total revenue?"
Response: data['Revenue'].sum()

Request: "Calculate profit margin as percentage"  
Response: ((data['Revenue'].sum() - data['Cost'].sum()) / data['Revenue'].sum()) * 100

Request: "Average sales per customer"
Response: data['Sales'].sum() / data['Customers'].nunique()

Request: "What's the profit?"
Response: data['Revenue'].sum() - data['Cost'].sum()

Request: "How many customers do we have?"
Response: data['Customers'].nunique()

Request: "What's the maximum population?"
Response: data['Population2023'].max()

IMPORTANT RULES:
- Provide ONLY the pandas expression, no explanations
- Use exact column names from the dataset  
- Result should be a single numeric value
- Use appropriate aggregation (sum, mean, max, min, count, nunique, etc.)
- For percentages, multiply by 100
- Handle division carefully to avoid errors

PANDAS EXPRESSION:"""

            # Get AI response
            ai_response = self.run_gemini(prompt)
            pandas_expr = ai_response.strip()
            print(f"   AI pandas code: {pandas_expr}")
            
            # Execute the pandas expression safely
            result = self._execute_metric_calculation(pandas_expr, data)
            
            if result is not None:
                # Format the result nicely
                formatted_result = self._format_metric_result(result, user_query)
                
                return {
                    "success": True,
                    "value": result,
                    "formatted_value": formatted_result,
                    "expression": pandas_expr,
                    "interpretation": f"Calculated '{user_query}' = {formatted_result}",
                    "traditional_syntax": self._suggest_traditional_expression(pandas_expr, data)
                }
            else:
                return {
                    "success": False,
                    "error": "Could not calculate the metric - expression execution failed",
                    "expression": pandas_expr,
                    "suggestion": "Try rephrasing your request or check if column names exist in your data"
                }
                
        except Exception as e:
            print(f"❌ Metric calculation failed: {str(e)}")
            return {
                "success": False, 
                "error": f"Failed to calculate metric: {str(e)}",
                "suggestion": "Please try a simpler calculation or check your data"
            }
    
    def _execute_metric_calculation(self, pandas_expr: str, data: pd.DataFrame) -> Optional[float]:
        """Safely execute pandas expression and return scalar result"""
        try:
            # Create safe execution environment
            safe_globals = {
                'data': data,
                'pd': pd,
                'np': np,
                '__builtins__': {}
            }
            
            # Execute the pandas expression
            result = eval(pandas_expr, safe_globals)
            print(f"   Raw result: {result} (type: {type(result)})")
            
            # Convert to scalar if needed
            if hasattr(result, 'iloc') and len(result) > 0:
                result = result.iloc[0]
            elif hasattr(result, 'item'):
                result = result.item()
            elif pd.isna(result):
                return None
                
            # Ensure it's a number
            if isinstance(result, (int, float, np.integer, np.floating)):
                # Handle infinity and NaN
                if not np.isfinite(result):
                    print(f"   Result is not finite: {result}")
                    return None
                return float(result)
            else:
                print(f"   Result is not numeric: {result} (type: {type(result)})")
                return None
                
        except Exception as e:
            print(f"   Execution error: {str(e)}")
            return None


    def _format_metric_result(self, value: float, query: str) -> str:
        """Format the metric result for display"""
        try:
            # Detect if it's likely a percentage
            if 'percentage' in query.lower() or 'percent' in query.lower() or 'margin' in query.lower():
                return f"{value:.2f}%"
            
            # Detect if it's likely money
            if any(word in query.lower() for word in ['revenue', 'cost', 'profit', 'sales', 'price', 'amount']):
                return f"${value:,.2f}"
            
            # Detect if it's a count
            if any(word in query.lower() for word in ['count', 'number', 'how many']):
                return f"{int(value):,}"
            
            # Default formatting
            if value == int(value):
                return f"{int(value):,}"
            else:
                return f"{value:,.2f}"
                
        except:
            return str(value)


    def _suggest_traditional_expression(self, pandas_expr: str, data: pd.DataFrame) -> Optional[str]:
        """Convert pandas expression to traditional @MeasureName.Aggregation syntax if possible"""
        try:
            # Simple pattern matching for common cases
            pandas_expr_lower = pandas_expr.lower()
            
            # Extract column names from pandas expression
            import re
            column_pattern = r"data\['([^']+)'\]"
            columns = re.findall(column_pattern, pandas_expr)
            
            if not columns:
                return None
            
            # Simple aggregation mapping
            if '.sum()' in pandas_expr_lower:
                if len(columns) == 1:
                    return f"@{columns[0]}.Sum"
                elif len(columns) == 2 and '-' in pandas_expr:
                    return f"@{columns[0]}.Sum - @{columns[1]}.Sum"
            elif '.mean()' in pandas_expr_lower or '.avg()' in pandas_expr_lower:
                if len(columns) == 1:
                    return f"@{columns[0]}.Avg"
            elif '.max()' in pandas_expr_lower:
                if len(columns) == 1:
                    return f"@{columns[0]}.Max"
            elif '.min()' in pandas_expr_lower:
                if len(columns) == 1:
                    return f"@{columns[0]}.Min"
            elif '.count()' in pandas_expr_lower or '.nunique()' in pandas_expr_lower:
                if len(columns) == 1:
                    return f"@{columns[0]}.Count"
            
            return None
            
        except:
            return None
