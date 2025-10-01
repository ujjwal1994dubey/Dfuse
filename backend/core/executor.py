"""
Gemini LLM Client for Data Formulator Integration
Provides natural language data transformation capabilities using both structured parsing and pandas DataFrame agent
"""
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

class GeminiDataFormulator:
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



    def _execute_enhanced_parsing(self, log_entry: str, data: pd.DataFrame, dimensions: List[str], measures: List[str]) -> Optional[Dict[str, Any]]:
        """Execute the filter operation detected by enhanced parsing"""
        import re
        
        # Extract filter details from log entry: "Enhanced parsing: Filtered Sales_Units > 600"
        filter_match = re.search(r'Enhanced parsing: Filtered (\w+) ([><=]+) (.+)', log_entry)
        if not filter_match:
            return None
        
        column, operator, value = filter_match.groups()
        
        try:
            if column not in data.columns:
                return None
            
            # Apply the filter
            if operator == '>':
                mask = data[column] > float(value)
            elif operator == '<':
                mask = data[column] < float(value)
            elif operator == '=' or operator == '==':
                # Handle both numeric and string values
                try:
                    mask = data[column] == float(value)
                except ValueError:
                    mask = data[column].astype(str) == str(value).strip("'\"")
            else:
                return None
            
            filtered_data = data[mask]
            
            return {
                "data": filtered_data,
                "dimensions": dimensions,
                "measures": measures,
                "transformations": [f"Applied filter: {column} {operator} {value}", f"Filtered {len(data)} rows to {len(filtered_data)} rows"],
                "chart_suggestion": self._suggest_chart_type(dimensions, measures, filtered_data)
            }
            
        except Exception as e:
            return None

    def _generate_pandas_fallback(self, user_query: str, data: pd.DataFrame, dimensions: List[str], 
                                measures: List[str], transformation_log: List[str]) -> Optional[Dict[str, Any]]:
        """Generate pandas code directly as final fallback"""
        try:
            # Create a focused prompt for pandas code generation
            prompt = f"""
Convert this user request into a single pandas operation on DataFrame 'data':
USER REQUEST: {user_query}
AVAILABLE COLUMNS: {list(data.columns)}
CURRENT CHART: dimensions={dimensions}, measures={measures}

Generate ONE line of pandas code that filters/transforms the data. Examples:
- "show products with sales > 600" → data[data['Sales_Units'] > 600]
- "filter to electronics" → data[data['Category'] == 'Electronics']
- "top 5 by revenue" → data.nlargest(5, 'Revenue')

PANDAS CODE:"""
            
            # Get AI response for pandas code
            ai_pandas_code = self.run_gemini(prompt).strip()
            
            # Clean up the response (remove any explanations)
            if '\n' in ai_pandas_code:
                ai_pandas_code = ai_pandas_code.split('\n')[0]
            
            # Execute the pandas code safely
            result_data = self._execute_pandas_safely(ai_pandas_code, data)
            if result_data is not None:
                transformation_log.append(f"Applied pandas operation: {ai_pandas_code}")
                transformation_log.append(f"Filtered {len(data)} rows to {len(result_data)} rows")
                
                return {
                    "data": result_data,
                    "dimensions": dimensions,
                    "measures": measures,
                    "transformations": transformation_log,
                    "chart_suggestion": self._suggest_chart_type(dimensions, measures, result_data)
                }
                
        except Exception as e:
            transformation_log.append(f"Pandas fallback failed: {str(e)}")
        
        return None

    def _execute_pandas_safely(self, pandas_code: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Safely execute pandas code and return result DataFrame"""
        try:
            # Create safe execution environment
            safe_globals = {
                'data': data,
                'pd': pd,
                'np': np,
                '__builtins__': {}
            }
            
            # Execute the pandas operation
            result = eval(pandas_code, safe_globals)
            
            # Ensure result is a DataFrame
            if isinstance(result, pd.DataFrame):
                return result
            else:
                return None
                
        except Exception as e:
            return None


    def _execute_pandas_filter_safely(self, code: str, data: pd.DataFrame, user_query: str) -> Optional[pd.DataFrame]:
        """Safely execute pandas filtering code"""
        try:
            # Extract filtering conditions from the code
            # Look for patterns like: df[(df['col'] > value) & (df['col2'] < value2)]
            import re
            
            # Simple pattern matching for common filtering operations
            if 'Revenue' in code and 'Sales_Units' in code:
                # Extract conditions for Revenue and Sales_Units
                revenue_match = re.search(r"df\['Revenue'\]\s*>\s*(\d+)", code)
                sales_match = re.search(r"df\['Sales_Units'\]\s*>\s*(\d+)", code)
                
                if revenue_match and sales_match:
                    revenue_threshold = int(revenue_match.group(1))
                    sales_threshold = int(sales_match.group(1))
                    
                    print(f"🤖 Applying filter: Revenue > {revenue_threshold} AND Sales_Units > {sales_threshold}")
                    
                    # Apply the filtering
                    filtered_data = data[(data['Revenue'] > revenue_threshold) & (data['Sales_Units'] > sales_threshold)]
                    return filtered_data
            
            # Generic safe execution as fallback
            safe_globals = {
                'df': data,
                'data': data,
                'pd': pd,
                'np': np,
                '__builtins__': {}
            }
            
            # Simple code cleanup - just get the filtering part
            if 'df[(' in code and ')]' in code:
                start = code.find('df[(')
                end = code.find(')]', start) + 2
                filter_code = code[start:end]
                
                result = eval(filter_code, safe_globals)
                if isinstance(result, pd.DataFrame):
                    return result
                    
        except Exception as e:
            print(f"🤖 Safe execution failed: {e}")
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


    def _extract_and_apply_filter(self, user_query: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract filtering logic from user query and apply to data"""
        try:
            query_lower = user_query.lower()
            
            # Pattern matching for common filtering scenarios
            if 'revenue' in query_lower and 'greater than' in query_lower and 'sales' in query_lower:
                # Extract numeric values
                import re
                numbers = re.findall(r'\d+', user_query)
                
                if len(numbers) >= 2:
                    # Assume first number is revenue threshold, second is sales threshold
                    revenue_threshold = int(numbers[0]) * 1000 if len(numbers[0]) <= 3 else int(numbers[0])  # Handle 100k format
                    sales_threshold = int(numbers[1])
                    
                    print(f"🤖 Direct filter application: Revenue > {revenue_threshold} AND Sales_Units > {sales_threshold}")
                    
                    # Apply compound filter
                    if 'Revenue' in data.columns and 'Sales_Units' in data.columns:
                        filtered_data = data[(data['Revenue'] > revenue_threshold) & (data['Sales_Units'] > sales_threshold)]
                        return filtered_data
            
            return None
            
        except Exception as e:
            print(f"🤖 Direct filter extraction failed: {e}")
            return None
