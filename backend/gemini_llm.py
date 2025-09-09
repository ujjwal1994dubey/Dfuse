"""
Gemini LLM Client for Data Formulator Integration
Provides natural language data transformation capabilities
"""
import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import re
import google.generativeai as genai

class GeminiDataFormulator:
    """
    Data Formulator with Gemini 2.0 Flash for natural language data transformations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "AIzaSyDTs3BYcLe_1XF8q3VW-blr_6wcG_mepgE"
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def run_gemini(self, prompt: str, model: str = "gemini-2.0-flash-exp") -> str:
        """
        Calls Gemini with a natural language prompt and returns the response text.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback to simulation if API fails
            return self._simulate_gemini_response(prompt)
    
    def _simulate_gemini_response(self, prompt: str) -> str:
        """Simulate Gemini responses for common data transformation patterns"""
        query = prompt.lower()
        
        # Pattern matching for common transformations
        if "filter" in query and "=" in query:
            # Extract filter condition
            filter_match = re.search(r'filter.*?(\w+)\s*=\s*[\'"]?(\w+)[\'"]?', query, re.IGNORECASE)
            if filter_match:
                column, value = filter_match.groups()
                return f"FILTER: {column} == '{value}'"
        
        if "profit" in query and ("revenue" in query or "cost" in query):
            return "ADD_COLUMN: Profit = Revenue - Cost"
        
        if "average" in query or "avg" in query:
            measure_match = re.search(r'average\s+(\w+)', query, re.IGNORECASE)
            if measure_match:
                measure = measure_match.group(1)
                return f"AGGREGATION: {measure} -> avg"
        
        if "group by" in query or "by" in query:
            group_match = re.search(r'by\s+(\w+)', query, re.IGNORECASE)
            if group_match:
                dimension = group_match.group(1)
                return f"GROUP_BY: {dimension}"
        
        if "top" in query or "highest" in query:
            number_match = re.search(r'top\s+(\d+)', query, re.IGNORECASE)
            if number_match:
                n = number_match.group(1)
                return f"TOP_N: {n}"
        
        # Default transformation
        return "TRANSFORM: Apply user request to data"
    
    def parse_transformation(self, ai_response: str, current_data: pd.DataFrame, 
                           dimensions: List[str], measures: List[str]) -> Dict[str, Any]:
        """
        Parse AI response and apply transformations to the DataFrame
        Enhanced to handle both structured responses and natural language responses from Gemini
        """
        response = ai_response.strip()
        transformed_data = current_data.copy()
        transformation_log = []
        new_dimensions = dimensions.copy()
        new_measures = measures.copy()
        
        # Try to parse structured response first
        result = self._parse_structured_response(response, transformed_data, transformation_log, new_dimensions, new_measures)
        if result:
            # Successfully parsed structured response
            return result
        else:
            # Fallback: try to parse natural language response
            self._parse_natural_language_response(response, transformed_data, transformation_log, new_dimensions, new_measures)
        
        return {
            "data": transformed_data,
            "dimensions": new_dimensions,
            "measures": new_measures,
            "transformations": transformation_log,
            "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, transformed_data)
        }
    
    def _parse_structured_response(self, response: str, transformed_data: pd.DataFrame,
                                 transformation_log: List[str], new_dimensions: List[str],
                                 new_measures: List[str]) -> Dict[str, Any]:
        """Parse structured AI response format"""
        try:
            if response.startswith("FILTER:"):
                # Parse filter: "FILTER: Category == 'Electronics'"
                filter_expr = response.replace("FILTER:", "").strip()
                if "==" in filter_expr:
                    column, value = filter_expr.split("==", 1)
                    column = column.strip()
                    value = value.strip().strip("'\"")
                    if column in transformed_data.columns:
                        mask = transformed_data[column].astype(str) == value
                        filtered_data = transformed_data[mask]
                        transformation_log.append(f"Filtered {column} = {value}")
                        return {
                            "data": filtered_data,
                            "dimensions": new_dimensions,
                            "measures": new_measures,
                            "transformations": transformation_log,
                            "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, filtered_data)
                        }
                    else:
                        # Try to find similar column name
                        similar_cols = [col for col in transformed_data.columns if col.lower() == column.lower()]
                        if similar_cols:
                            column = similar_cols[0]
                            mask = transformed_data[column].astype(str) == value
                            filtered_data = transformed_data[mask]
                            transformation_log.append(f"Filtered {column} = {value}")
                            return {
                                "data": filtered_data,
                                "dimensions": new_dimensions,
                                "measures": new_measures,
                                "transformations": transformation_log,
                                "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, filtered_data)
                            }
                return None
            
            elif response.startswith("ADD_COLUMN:"):
                # Parse new column: "ADD_COLUMN: Percent_Change = (([Population2023] - [Population2018]) / [Population2018]) * 100"
                column_expr = response.replace("ADD_COLUMN:", "").strip()
                if "=" in column_expr:
                    new_col, formula = column_expr.split("=", 1)
                    new_col = new_col.strip()
                    formula = formula.strip()
                    
                    # Use flexible pandas expression evaluator
                    result = self._evaluate_pandas_expression(new_col, formula, transformed_data, transformation_log)
                    if result is not None:
                        modified_data, log_msg = result
                        # Make the new calculated column the PRIMARY measure for the chart
                        # This ensures the chart shows the calculated values, not the original data
                        new_measures = [new_col]  # Replace measures with just the new calculated column
                        transformation_log.append(log_msg)
                        return {
                            "data": modified_data,
                            "dimensions": new_dimensions,
                            "measures": new_measures,
                            "transformations": transformation_log,
                            "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, modified_data)
                        }
                return None
            
            elif response.startswith("AGGREGATION:"):
                # Parse aggregation change: "AGGREGATION: Revenue -> avg"
                agg_expr = response.replace("AGGREGATION:", "").strip()
                if "->" in agg_expr:
                    measure, agg_func = agg_expr.split("->", 1)
                    measure, agg_func = measure.strip(), agg_func.strip()
                    transformation_log.append(f"Changed aggregation to {agg_func}")
                    return {
                        "data": transformed_data,
                        "dimensions": new_dimensions,
                        "measures": new_measures,
                        "transformations": transformation_log,
                        "aggregation_change": agg_func,
                        "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, transformed_data)
                    }
                return None
            
            elif response.startswith("GROUP_BY:"):
                # Parse grouping: "GROUP_BY: Region"
                group_col = response.replace("GROUP_BY:", "").strip()
                if group_col in transformed_data.columns and group_col not in new_dimensions:
                    new_dimensions.append(group_col)
                    transformation_log.append(f"Added grouping by {group_col}")
                    return {
                        "data": transformed_data,
                        "dimensions": new_dimensions,
                        "measures": new_measures,
                        "transformations": transformation_log,
                        "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, transformed_data)
                    }
                return None
            
            elif response.startswith("TOP_N:"):
                # Parse top N: "TOP_N: 5"
                n_str = response.replace("TOP_N:", "").strip()
                try:
                    n = int(n_str)
                    if new_measures and new_measures[0] in transformed_data.columns:
                        # Sort by first measure and take top N (highest values)
                        top_data = transformed_data.nlargest(n, new_measures[0])
                        transformation_log.append(f"Showing top {n} records")
                        return {
                            "data": top_data,
                            "dimensions": new_dimensions,
                            "measures": new_measures,
                            "transformations": transformation_log,
                            "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, top_data)
                        }
                except Exception as e:
                    transformation_log.append(f"Top N error: {str(e)}")
                return None
            
            elif response.startswith("BOTTOM_N:"):
                # Parse bottom N: "BOTTOM_N: 5"
                n_str = response.replace("BOTTOM_N:", "").strip()
                try:
                    n = int(n_str)
                    if new_measures and new_measures[0] in transformed_data.columns:
                        # Sort by first measure and take bottom N (lowest values)
                        bottom_data = transformed_data.nsmallest(n, new_measures[0])
                        transformation_log.append(f"Showing bottom {n} records")
                        return {
                            "data": bottom_data,
                            "dimensions": new_dimensions,
                            "measures": new_measures,
                            "transformations": transformation_log,
                            "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, bottom_data)
                        }
                except Exception as e:
                    transformation_log.append(f"Bottom N error: {str(e)}")
                return None
                
        except Exception as e:
            transformation_log.append(f"Structured parsing error: {str(e)}")
            
        return None
    
    def _evaluate_pandas_expression(self, new_col: str, formula: str, data: pd.DataFrame, log: List[str]) -> Optional[tuple]:
        """
        Safely evaluate pandas expressions for calculated columns
        Supports complex formulas like: (([Col2] - [Col1]) / [Col1]) * 100
        """
        try:
            # Replace column names in brackets with pandas column references
            # Example: [Population2023] -> data['Population2023']
            import re
            
            # Find all column references in brackets: [ColumnName]
            column_pattern = r'\[([^\]]+)\]'
            columns_in_formula = re.findall(column_pattern, formula)
            
            # Verify all referenced columns exist (with smart matching)
            missing_cols = []
            column_mapping = {}
            
            for col in columns_in_formula:
                if col in data.columns:
                    column_mapping[col] = col  # Exact match
                else:
                    # Smart column matching for common variations
                    possible_matches = []
                    col_lower = col.lower()
                    
                    for actual_col in data.columns:
                        actual_lower = actual_col.lower()
                        # Check for partial matches or common variations
                        if (col_lower in actual_lower or actual_lower in col_lower or
                            col_lower.replace('_', '') == actual_lower.replace('_', '') or
                            col_lower.replace(' ', '') == actual_lower.replace(' ', '')):
                            possible_matches.append(actual_col)
                    
                    if possible_matches:
                        # Use the best match (shortest name usually most accurate)
                        best_match = min(possible_matches, key=len)
                        column_mapping[col] = best_match
                        log.append(f"Mapped '{col}' to '{best_match}'")
                    else:
                        missing_cols.append(col)
            
            if missing_cols:
                log.append(f"Error: Missing columns {missing_cols} in formula: {formula}")
                available_cols = list(data.columns)
                log.append(f"Available columns: {available_cols}")
                return None
            
            # Create a safe evaluation context with only the data
            modified_data = data.copy()
            
            # Replace [ColumnName] with modified_data['ActualColumnName'] in the formula using mapping
            safe_formula = formula
            for col in columns_in_formula:
                actual_col = column_mapping[col]
                safe_formula = safe_formula.replace(f'[{col}]', f'modified_data["{actual_col}"]')
            
            # Evaluate the expression safely
            # Only allow basic math operations and pandas column access
            allowed_names = {
                'modified_data': modified_data,
                '__builtins__': {},  # Remove access to builtin functions for security
            }
            
            # Add common math operations
            import operator
            import math
            allowed_names.update({
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len
            })
            
            # Evaluate the expression
            result = eval(safe_formula, allowed_names)
            
            # Add the new column
            modified_data[new_col] = result
            
            # Handle NaN and infinity values for JSON compliance
            import numpy as np
            if modified_data[new_col].dtype.kind in 'fc':  # float or complex
                # Replace inf and -inf with None (will become null in JSON)
                modified_data[new_col] = modified_data[new_col].replace([np.inf, -np.inf], None)
                # Replace NaN with None for JSON serialization
                modified_data[new_col] = modified_data[new_col].where(pd.notna(modified_data[new_col]), None)
            
            # Create readable log message with actual column names used
            readable_formula = formula
            for col in columns_in_formula:
                actual_col = column_mapping[col]
                if col != actual_col:
                    readable_formula = readable_formula.replace(f'[{col}]', f'[{actual_col}]')
            readable_formula = readable_formula.replace('[', '').replace(']', '')
            log_msg = f"Created {new_col} = {readable_formula}"
            
            return modified_data, log_msg
            
        except ZeroDivisionError:
            log.append(f"Error: Division by zero in formula: {formula}")
            return None
        except Exception as e:
            log.append(f"Error evaluating formula '{formula}': {str(e)}")
            return None
    
    def _parse_natural_language_response(self, response: str, transformed_data: pd.DataFrame,
                                       transformation_log: List[str], new_dimensions: List[str],
                                       new_measures: List[str]) -> bool:
        """Fallback: Parse natural language response using keyword matching"""
        response_lower = response.lower()
        
        # Try to extract transformation intent from natural language
        if "filter" in response_lower:
            transformation_log.append("AI suggested filtering, but specific parameters could not be parsed")
            return True
        elif "add" in response_lower and ("column" in response_lower or "calculate" in response_lower):
            transformation_log.append("AI suggested adding a calculated column, but specific formula could not be parsed")
            return True
        elif "group" in response_lower:
            transformation_log.append("AI suggested grouping, but specific dimension could not be parsed")
            return True
        elif "top" in response_lower or "bottom" in response_lower or "limit" in response_lower:
            transformation_log.append("AI suggested limiting results, but specific number could not be parsed")
            return True
        
        transformation_log.append(f"AI response: {response[:100]}...")
        return True
    
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
    
    def explore_data(self, user_query: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for AI data exploration
        """
        # Extract current chart context
        current_data = pd.DataFrame(chart_data.get("table", []))
        dimensions = chart_data.get("dimensions", [])
        measures = chart_data.get("measures", [])
        dataset_id = chart_data.get("dataset_id", "")
        
        # Create enhanced context-aware prompt for Gemini
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
            "dataset_id": dataset_id
        })
        
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
