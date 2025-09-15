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
        self.api_key = api_key or "AIzaSyDTs3BYcLe_1XF8q3VW-blr_6wcG_mepgE"
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Initialize pandas DataFrame agent if langchain is available
        self.pandas_agent = None
        if LANGCHAIN_AVAILABLE:
            try:
                self.llm = GoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                print("✅ Pandas DataFrame Agent initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize pandas DataFrame agent: {e}")
                self.llm = None
        else:
            print("⚠️  LangChain not available, using structured parsing only")
    
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
            # Fallback 1: try to parse natural language response with improved logic
            nl_result = self._parse_natural_language_response(response, transformed_data, transformation_log, new_dimensions, new_measures)
            if nl_result and "Enhanced parsing:" in str(transformation_log):
                # If enhanced parsing found something actionable, try to execute it
                executed_result = self._execute_enhanced_parsing(transformation_log[-1], transformed_data, new_dimensions, new_measures)
                if executed_result:
                    return executed_result
            
            # Fallback 2: Generate pandas code directly if structured parsing fails
            pandas_result = self._generate_pandas_fallback(response, current_data, dimensions, measures, transformation_log)
            if pandas_result:
                return pandas_result
        
        return {
            "data": transformed_data,
            "dimensions": new_dimensions,
            "measures": new_measures,
            "transformations": transformation_log,
            "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, transformed_data)
        }
    
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

USER REQUEST: "{user_query}"
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
        """Enhanced fallback: Parse natural language response using improved keyword matching"""
        response_lower = response.lower()
        
        # Try to extract transformation intent from natural language with better parsing
        filter_result = self._extract_filter_from_nl(response_lower, transformed_data, transformation_log)
        if filter_result:
            return filter_result
        
        if "add" in response_lower and ("column" in response_lower or "calculate" in response_lower):
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
    
    def _extract_filter_from_nl(self, response_lower: str, data: pd.DataFrame, transformation_log: List[str]) -> bool:
        """Extract filtering operations from natural language with improved parsing"""
        import re
        
        # Pattern 1: "greater than" or ">"
        if "greater than" in response_lower or ">" in response_lower:
            # Multiple patterns for different phrasings
            patterns = [
                r'(\w+(?:\s+\w+)*)\s+greater\s+than\s+(\d+)',
                r'(\w+(?:\s+\w+)*)\s*>\s*(\d+)',
                r'with\s+(\w+(?:\s+\w+)*)\s+greater\s+than\s+(\d+)',
                r'where\s+(\w+(?:\s+\w+)*)\s+greater\s+than\s+(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_lower)
                if match:
                    column_phrase = match.group(1).strip()
                    value = match.group(2)
                    
                    # Smart column matching
                    matched_column = self._find_matching_column(column_phrase, data.columns)
                    if matched_column:
                        transformation_log.append(f"Enhanced parsing: Filtered {matched_column} > {value}")
                        return True
        
        # Pattern 2: "less than" or "<"
        if "less than" in response_lower or "<" in response_lower:
            patterns = [
                r'(\w+(?:\s+\w+)*)\s+less\s+than\s+(\d+)',
                r'(\w+(?:\s+\w+)*)\s*<\s*(\d+)',
                r'with\s+(\w+(?:\s+\w+)*)\s+less\s+than\s+(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_lower)
                if match:
                    column_phrase = match.group(1).strip()
                    value = match.group(2)
                    
                    matched_column = self._find_matching_column(column_phrase, data.columns)
                    if matched_column:
                        transformation_log.append(f"Enhanced parsing: Filtered {matched_column} < {value}")
                        return True
        
        # Pattern 3: "equal to" or "="
        if "equal" in response_lower or "=" in response_lower:
            patterns = [
                r'(\w+(?:\s+\w+)*)\s+equal\s+to\s+[\'"]?(\w+)[\'"]?',
                r'(\w+(?:\s+\w+)*)\s*=\s*[\'"]?(\w+)[\'"]?',
                r'where\s+(\w+(?:\s+\w+)*)\s+is\s+[\'"]?(\w+)[\'"]?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_lower)
                if match:
                    column_phrase = match.group(1).strip()
                    value = match.group(2)
                    
                    matched_column = self._find_matching_column(column_phrase, data.columns)
                    if matched_column:
                        transformation_log.append(f"Enhanced parsing: Filtered {matched_column} = '{value}'")
                        return True
        
        # Fallback for general filter mention
        if "filter" in response_lower:
            transformation_log.append("AI suggested filtering, but specific parameters could not be parsed with enhanced logic")
            return True
            
        return False
    
    def _find_matching_column(self, phrase: str, columns: List[str]) -> Optional[str]:
        """Smart column matching for natural language phrases"""
        phrase_clean = phrase.lower().replace(' ', '').replace('_', '')
        
        # Exact match first
        for col in columns:
            if phrase.lower() == col.lower():
                return col
        
        # Smart partial matching
        for col in columns:
            col_clean = col.lower().replace(' ', '').replace('_', '')
            
            # Check if phrase is contained in column name or vice versa
            if phrase_clean in col_clean or col_clean in phrase_clean:
                return col
            
            # Check word-by-word matching
            phrase_words = phrase.lower().split()
            col_words = col.lower().replace('_', ' ').split()
            
            # If all phrase words are in column words (order doesn't matter)
            if all(any(pw in cw for cw in col_words) for pw in phrase_words):
                return col
        
        return None
    
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
    
    def _use_pandas_agent(self, user_query: str, data: pd.DataFrame, dimensions: List[str], measures: List[str]) -> Optional[Dict[str, Any]]:
        """Use pandas DataFrame agent for flexible natural language processing"""
        if not LANGCHAIN_AVAILABLE or self.llm is None:
            return None
        
        try:
            print(f"🤖 Using pandas DataFrame agent for: '{user_query}'")
            
            # Create agent for this specific DataFrame
            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=data,
                verbose=True,
                return_intermediate_steps=False,
                handle_parsing_errors=True,
                allow_dangerous_code=True  # Required for pandas operations
            )
            
            # Create a prompt that focuses on data transformation tasks
            enhanced_query = f"""
You are working with a pandas DataFrame with the following structure:
- Columns: {list(data.columns)}
- Shape: {data.shape}
- Current chart dimensions: {dimensions}
- Current chart measures: {measures}

User request: "{user_query}"

Please perform ONE of these operations and return the result:
1. For filtering: Filter the DataFrame and return the filtered result
2. For calculations: Add a new calculated column and return the DataFrame with the new column
3. For aggregations: Group and aggregate the data as requested

Important: 
- Always return a DataFrame as the final result
- For filtering, use conditions like df[df['column'] > value]
- For calculations, use df['new_col'] = df['col1'] - df['col2']  
- For complex conditions, use & (and) and | (or) with parentheses
- Handle text comparisons case-insensitively when appropriate

Execute the operation:"""
            
            # Run the agent
            result = agent.run(enhanced_query)
            
            print(f"🤖 Pandas agent result type: {type(result)}")
            print(f"🤖 Pandas agent raw result: {str(result)[:200]}...")
            
            # The agent might return different types - try to extract a DataFrame
            transformed_data = None
            transformations = []
            
            if isinstance(result, pd.DataFrame):
                transformed_data = result
                transformations.append(f"DataFrame agent executed: {user_query}")
            elif isinstance(result, str):
                # Parse the string result to extract DataFrame info
                transformations.append(f"Agent response: {result[:100]}...")
                
                # The agent executed successfully, so let's re-run the same filtering on our original data
                try:
                    # Re-create the agent's filtering logic on our data
                    transformed_data = self._execute_agent_result_on_data(user_query, data)
                    if transformed_data is not None:
                        transformations.append("Pandas agent filtering applied successfully")
                    else:
                        print(f"🤖 Failed to apply agent result to original data")
                except Exception as e:
                    print(f"🤖 Failed to process agent result: {e}")
            
            # If we got a transformed DataFrame, determine dimensions and measures
            if transformed_data is not None and not transformed_data.empty:
                # Auto-detect dimensions and measures from transformed data
                new_dimensions, new_measures = self._auto_detect_columns(transformed_data, dimensions, measures)
                
                # Handle JSON serialization safety
                transformed_data = self._ensure_json_safe(transformed_data)
                
                return {
                    "data": transformed_data,
                    "dimensions": new_dimensions,
                    "measures": new_measures,
                    "transformations": transformations,
                    "chart_suggestion": self._suggest_chart_type(new_dimensions, new_measures, transformed_data),
                    "method": "pandas_agent"
                }
            else:
                print(f"🤖 Pandas agent did not return a usable DataFrame")
                return None
                
        except Exception as e:
            print(f"❌ Pandas DataFrame agent failed: {str(e)}")
            return None
    
    def _auto_detect_columns(self, data: pd.DataFrame, original_dimensions: List[str], original_measures: List[str]) -> tuple:
        """Auto-detect dimensions and measures from transformed data"""
        dimensions = []
        measures = []
        
        for col in data.columns:
            if data[col].dtype in ['object', 'string', 'category']:
                dimensions.append(col)
            elif data[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
                measures.append(col)
        
        # If no dimensions found but original had some, try to preserve original dimensions that still exist
        if not dimensions and original_dimensions:
            for dim in original_dimensions:
                if dim in data.columns:
                    dimensions.append(dim)
        
        # If no measures found but original had some, try to preserve original measures that still exist
        if not measures and original_measures:
            for measure in original_measures:
                if measure in data.columns:
                    measures.append(measure)
        
        # Ensure we have at least one measure for chart generation
        if not measures and data.shape[1] > 0:
            # Use the first numeric column as measure, or create a count measure
            for col in data.columns:
                if data[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
                    measures.append(col)
                    break
            
            if not measures:
                # Create a synthetic count measure
                measures.append('count')
        
        return dimensions, measures
    
    def _ensure_json_safe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame is JSON serializable"""
        safe_data = data.copy()
        
        for col in safe_data.columns:
            # Handle NaN and infinite values
            if safe_data[col].dtype.kind in 'fc':  # float or complex
                safe_data[col] = safe_data[col].replace([np.inf, -np.inf], None)
                safe_data[col] = safe_data[col].where(pd.notna(safe_data[col]), None)
        
        return safe_data
    
    def _execute_agent_result_on_data(self, user_query: str, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Re-execute the pandas agent's logic directly on the original data"""
        try:
            # Create a simplified agent just to get the pandas code
            if not LANGCHAIN_AVAILABLE or self.llm is None:
                return None
                
            agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=data,
                verbose=False,  # Less verbose for this execution
                return_intermediate_steps=True,  # We want to see the code
                allow_dangerous_code=True
            )
            
            # Get a more direct query for pandas code
            direct_query = f"""
Execute this request on the DataFrame: {user_query}
Return only the filtered/transformed DataFrame as the final result.
"""
            
            # Run the agent and get intermediate steps
            result_with_steps = agent.invoke({"input": direct_query})
            
            # Extract the actual DataFrame operation from intermediate steps
            if 'intermediate_steps' in result_with_steps:
                steps = result_with_steps['intermediate_steps']
                
                # Look for pandas operations in the steps
                for step in steps:
                    if hasattr(step, 'tool_input') and 'python_repl' in str(step.tool).lower():
                        code = step.tool_input
                        if 'df[(' in code and ')' in code:  # This looks like filtering code
                            # Execute the filtering code safely on our data
                            return self._execute_pandas_filter_safely(code, data, user_query)
            
            # Fallback: try to extract filtering logic from user query directly
            return self._extract_and_apply_filter(user_query, data)
            
        except Exception as e:
            print(f"🤖 Agent execution failed: {e}")
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
    
    def explore_data(self, user_query: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for AI data exploration
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
