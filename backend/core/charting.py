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

class DfuseCharting:
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


    def _suggest_chart_type(self, dimensions: List[str], measures: List[str], data: pd.DataFrame) -> str:
        """Suggest appropriate chart type based on data characteristics"""
        if len(dimensions) == 1 and len(measures) == 1:
            # Check if it's good for pie chart (positive values, not too many categories)
            if len(data) <= 10 and measures[0] in data.columns and (data[measures[0]] >= 0).all():
                return "pie"
            else:
                return "bar"
        elif len(dimensions) == 0 and len(measures) == 1:
            return "histogram"
        elif len(dimensions) == 2 and len(measures) == 1:
            return "heatmap"
        else:
            return "bar"


    def explore_data_enhanced(self, user_query: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ENHANCED AI data exploration with full dataset context
        Now has access to complete original dataset, not just chart's aggregated data
        """
        # Extract FULL dataset context
        full_dataset = chart_data.get("full_dataset")
        current_chart_data = pd.DataFrame(chart_data.get("table", []))
        
        # Current chart context for reference
        current_dimensions = chart_data.get("dimensions", [])
        current_measures = chart_data.get("measures", [])
        dataset_id = chart_data.get("dataset_id", "")
        
        # All available columns from full dataset
        available_dims = chart_data.get("available_columns", {}).get("dimensions", [])
        available_measures = chart_data.get("available_columns", {}).get("measures", [])
        
        print(f"🤖 ENHANCED Data exploration request: '{user_query}'")
        print(f"📊 Full dataset shape: {full_dataset.shape}")
        print(f"📈 Current chart context: dims={current_dimensions}, measures={current_measures}")
        print(f"🔍 Available dimensions: {available_dims}")
        print(f"🔍 Available measures: {available_measures}")
        
        # PRIORITY 1: Try enhanced pandas DataFrame agent with FULL dataset
        if LANGCHAIN_AVAILABLE and self.llm is not None:
            enhanced_result = self._use_pandas_agent_enhanced(user_query, full_dataset, current_dimensions, current_measures, available_dims, available_measures)
            if enhanced_result:
                enhanced_result.update({
                    "original_query": user_query,
                    "dataset_id": dataset_id
                })
                print(f"✅ Enhanced pandas agent succeeded")
                return enhanced_result
            else:
                print(f"❌ Enhanced pandas agent failed, trying fallback...")
        
        # FALLBACK: Use original method with chart data
        print(f"⚠️ Using fallback to original explore_data method...")
        return self.explore_data_original(user_query, chart_data)

    def explore_data_original(self, user_query: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Original AI data exploration (fallback method)
        Enhanced with pandas DataFrame agent for flexible natural language processing
        """
        # Extract current chart context
        current_data = pd.DataFrame(chart_data.get("table", []))
        dimensions = chart_data.get("dimensions", [])
        measures = chart_data.get("measures", [])
        dataset_id = chart_data.get("dataset_id", "")
        
        print(f"🤖 Data exploration request: '{user_query}'")
        print(f"📊 Data shape: {current_data.shape}")
        print(f"📈 Current chart: dimensions={dimensions}, measures={measures}")
        
        # PRIORITY 1: Try pandas DataFrame agent (most flexible)
        if LANGCHAIN_AVAILABLE and self.llm is not None:
            pandas_result = self._use_pandas_agent(user_query, current_data, dimensions, measures)
            if pandas_result:
                pandas_result.update({
                    "original_query": user_query,
                    "dataset_id": dataset_id
                })
                print(f"✅ Pandas agent succeeded")
                return pandas_result
            else:
                print(f"❌ Pandas agent failed, falling back to structured parsing")
        else:
            print(f"⚠️  Pandas agent not available, using structured parsing")
        
        # FALLBACK: Use structured parsing approach (existing logic)
        prompt = f"""You are a data transformation assistant. Analyze the user's natural language request and provide a structured data transformation instruction.

User Query: {user_query}

Current Chart Context:
- Dimensions (categorical columns): {dimensions}
- Measures (numerical columns): {measures}
- Data shape: {current_data.shape if not current_data.empty else 'No data'}
- Available columns: {list(current_data.columns) if not current_data.empty else 'None'}
- Sample data: {current_data.head(3).to_dict('records') if not current_data.empty else 'None'}

INSTRUCTIONS:
Please analyze the user's request and respond with ONE of these transformation commands:

1. For filtering: "FILTER: [column_name] == '[value]'" (use exact column names)
2. For calculated columns: "ADD_COLUMN: [new_name] = [pandas_expression]" (use exact column names in expressions)
3. For aggregation: "AGGREGATION: [measure] -> [agg_type]" where agg_type is sum, avg, min, max, count
4. For grouping: "GROUP_BY: [column_name]"
5. For top N records: "TOP_N: [number]" (highest values)
6. For bottom N records: "BOTTOM_N: [number]" (lowest values)

For ADD_COLUMN, use pandas expressions with exact column names in square brackets:
- Addition: [Col1] + [Col2] 
- Subtraction: [Col1] - [Col2]
- Multiplication: [Col1] * [Col2]
- Division: [Col1] / [Col2] (handles division by zero safely)
- Percentages: (([Col2] - [Col1]) / [Col1]) * 100
- Constants: [Col1] * 1.5, [Col1] + 100
- Ratios/Densities: [Population] / [Area], [Revenue] / [Units]

Examples:
- "Filter to Electronics" → "FILTER: Category == 'Electronics'"
- "Add profit margin" → "ADD_COLUMN: Profit = [Revenue] - [Cost]"
- "Calculate density" → "ADD_COLUMN: Density = [Population] / [Area]"
- "Calculate percentage change" → "ADD_COLUMN: Percent_Change = (([Population2023] - [Population2018]) / [Population2018]) * 100"
- "Show top 5" → "TOP_N: 5"

Please respond with the appropriate transformation command based on the user's request:"""
        
        # Get AI response
        ai_response = self.run_gemini(prompt)
        
        # Parse and apply transformation
        result = self.parse_transformation(ai_response, current_data, dimensions, measures)
        
        # Add metadata
        result.update({
            "original_query": user_query,
            "ai_response": ai_response,
            "dataset_id": dataset_id,
            "method": "structured_parsing"
        })
        
        print(f"📊 Structured parsing result: {len(result.get('transformations', []))} transformations")
        return result


    def explore_data_original(self, user_query: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Original AI data exploration (fallback method)
        Enhanced with pandas DataFrame agent for flexible natural language processing
        """
        # Extract current chart context
        current_data = pd.DataFrame(chart_data.get("table", []))
        dimensions = chart_data.get("dimensions", [])
        measures = chart_data.get("measures", [])
        dataset_id = chart_data.get("dataset_id", "")
        
        print(f"🤖 Data exploration request: '{user_query}'")
        print(f"📊 Data shape: {current_data.shape}")
        print(f"📈 Current chart: dimensions={dimensions}, measures={measures}")
        
        # PRIORITY 1: Try pandas DataFrame agent (most flexible)
        if LANGCHAIN_AVAILABLE and self.llm is not None:
            pandas_result = self._use_pandas_agent(user_query, current_data, dimensions, measures)
            if pandas_result:
                pandas_result.update({
                    "original_query": user_query,
                    "dataset_id": dataset_id
                })
                print(f"✅ Pandas agent succeeded")
                return pandas_result
            else:
                print(f"❌ Pandas agent failed, falling back to structured parsing")
        else:
            print(f"⚠️  Pandas agent not available, using structured parsing")
        
        # FALLBACK: Use structured parsing approach (existing logic)
        prompt = f"""You are a data transformation assistant. Analyze the user's natural language request and provide a structured data transformation instruction.

User Query: "{user_query}"

Current Chart Context:
- Dimensions (categorical columns): {dimensions}
- Measures (numerical columns): {measures}
- Data shape: {current_data.shape if not current_data.empty else 'No data'}
- Available columns: {list(current_data.columns) if not current_data.empty else 'None'}
- Sample data: {current_data.head(3).to_dict('records') if not current_data.empty else 'None'}

INSTRUCTIONS:
Please analyze the user's request and respond with ONE of these transformation commands:

1. For filtering: "FILTER: [column_name] == '[value]'" (use exact column names)
2. For calculated columns: "ADD_COLUMN: [new_name] = [pandas_expression]" (use exact column names in expressions)
3. For aggregation: "AGGREGATION: [measure] -> [agg_type]" where agg_type is sum, avg, min, max, count
4. For grouping: "GROUP_BY: [column_name]"
5. For top N records: "TOP_N: [number]" (highest values)
6. For bottom N records: "BOTTOM_N: [number]" (lowest values)

For ADD_COLUMN, use pandas expressions with exact column names in square brackets:
- Addition: [Col1] + [Col2] 
- Subtraction: [Col1] - [Col2]
- Multiplication: [Col1] * [Col2]
- Division: [Col1] / [Col2] (handles division by zero safely)
- Percentages: (([Col2] - [Col1]) / [Col1]) * 100
- Constants: [Col1] * 1.5, [Col1] + 100
- Ratios/Densities: [Population] / [Area], [Revenue] / [Units]

Examples:
- "Filter to Electronics" → "FILTER: Category == 'Electronics'"
- "Add profit margin" → "ADD_COLUMN: Profit = [Revenue] - [Cost]"
- "Calculate density" → "ADD_COLUMN: Density = [Population] / [Area]"
- "Calculate percentage change" → "ADD_COLUMN: Percent_Change = (([Population2023] - [Population2018]) / [Population2018]) * 100"
- "Show top 5" → "TOP_N: 5"

Please respond with the appropriate transformation command based on the user's request:"""
        
        # Get AI response
        ai_response = self.run_gemini(prompt)
        
        # Parse and apply transformation
        result = self.parse_transformation(ai_response, current_data, dimensions, measures)
        
        # Add metadata
        result.update({
            "original_query": user_query,
            "ai_response": ai_response,
            "dataset_id": dataset_id,
            "method": "structured_parsing"
        })
        
        print(f"📊 Structured parsing result: {len(result.get('transformations', []))} transformations")
        return result
    
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
