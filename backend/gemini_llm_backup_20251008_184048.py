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

# Pandas DataFrame Agent imports - temporarily disabled due to compatibility issues
try:
    # from langchain.agents.agent_types import AgentType
    # from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    # from langchain_google_genai import GoogleGenerativeAI
    LANGCHAIN_AVAILABLE = False  # Temporarily disabled
except ImportError as e:
    print(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False


class GeminiDataFormulator:
    """
    Data Formulator with Gemini 2.0 Flash for natural language data transformations
    Generic implementation that adapts to any dataset type
    """
    
    # Generic optimization configuration
    OPTIMIZATION_CONFIG = {
        'max_relevant_columns': 15,        # Maximum columns to include
        'min_relevant_columns': 3,         # Minimum columns to ensure context
        'max_sample_rows': 5,              # Maximum sample rows
        'min_sample_rows': 2,              # Minimum sample rows
        'max_stats_columns': 4,            # Maximum columns for detailed stats
        'categorical_uniqueness_threshold': 0.1,  # Threshold for categorical vs text classification
        'complexity_score_weights': {
            'columns_per_10': 1.0,         # Weight for column count complexity
            'rows_per_10k': 0.3,           # Weight for row count complexity  
            'numeric_column_factor': 0.5,   # Weight for numeric columns
            'text_column_factor': 0.3       # Weight for text columns
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key or "AIzaSyCme3aE7H9TmRgKqHCqhqV8f-9FhIqDfOM"
        self.model_name = model
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Initialize pandas DataFrame agent if langchain is available
        self.pandas_agent = None
        if LANGCHAIN_AVAILABLE:
            try:
                # Map frontend model names to LangChain compatible names
                langchain_model = self._get_langchain_model_name(model)
                self.llm = GoogleGenerativeAI(
                    model=langchain_model,
                    google_api_key=self.api_key,
                    temperature=0.1
                )
                print(f"✅ Pandas DataFrame Agent initialized successfully with {langchain_model}")
            except Exception as e:
                print(f"❌ Failed to initialize pandas DataFrame agent: {e}")
                self.llm = None
        else:
            print("⚠️  LangChain not available, using structured parsing only")
    
    def _get_langchain_model_name(self, model: str) -> str:
        """Convert frontend model names to LangChain compatible names"""
        model_mapping = {
            "gemini-1.5-flash": "gemini-1.5-flash",
            "gemini-2.0-flash": "gemini-2.0-flash", 
            "gemini-2.0-flash-exp": "gemini-2.0-flash"  # fallback to stable version for LangChain
        }
        return model_mapping.get(model, "gemini-2.0-flash")
    
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
    
    def test_configuration(self) -> Dict[str, Any]:
        """
        Test the API key and model configuration
        """
        try:
            # Test direct Gemini API call
            test_prompt = "Hello! Please respond with 'Configuration test successful' to verify the API key and model are working correctly."
            
            response = self.model.generate_content(test_prompt)
            
            if response and response.text:
                # Extract token usage if available
                token_usage = {}
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    token_usage = {
                        'inputTokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                        'outputTokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                        'totalTokens': getattr(response.usage_metadata, 'total_token_count', 0)
                    }
                
                return {
                    'success': True,
                    'message': 'Configuration test successful! API key and model are working correctly.',
                    'model_used': self.model_name,
                    'response': response.text.strip(),
                    'token_usage': token_usage
                }
            else:
                return {
                    'success': False,
                    'error': 'Model returned empty response'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Configuration test failed: {str(e)}"
            }

    def _use_pandas_agent_enhanced(self, user_query: str, full_dataset: pd.DataFrame, current_dimensions: List[str], current_measures: List[str], available_dims: List[str], available_measures: List[str]) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: This method contained dataset-specific logic and examples.
        Now using generic _execute_real_pandas_analysis for all datasets.
        """
        print(f"🤖 Skipping legacy dataset-specific enhanced pandas agent - using generic pandas execution instead")
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

    def _auto_detect_columns_enhanced(self, data: pd.DataFrame, current_dimensions: List[str], current_measures: List[str], available_dims: List[str], available_measures: List[str]) -> tuple:
        """Enhanced auto-detection with full dataset context awareness"""
        dimensions = []
        measures = []
        
        # Detect based on data types
        for col in data.columns:
            if data[col].dtype in ['object', 'string', 'category']:
                dimensions.append(col)
            elif data[col].dtype in ['int64', 'int32', 'float64', 'float32', 'int', 'float']:
                measures.append(col)
        
        # Enhanced: Prioritize meaningful columns from full dataset
        # If we have common geographical/categorical columns, prioritize them
        priority_dims = ['State', 'Region', 'Category', 'Type', 'Zone', 'District']
        for dim in priority_dims:
            if dim in data.columns and dim not in dimensions:
                dimensions.insert(0, dim)  # Add at beginning for priority
        
        # Enhanced: Prioritize meaningful measures
        priority_measures = ['Population2023', 'Population2018', 'Area', 'Count', 'Total', 'Average']
        for measure in priority_measures:
            if measure in data.columns and measure not in measures:
                measures.insert(0, measure)  # Add at beginning for priority
        
        # If no dimensions found, try to preserve from current context or available
        if not dimensions:
            for dim in current_dimensions + available_dims:
                if dim in data.columns:
                    dimensions.append(dim)
                    break
        
        # If no measures found, try to preserve from current context or available  
        if not measures:
            for measure in current_measures + available_measures:
                if measure in data.columns:
                    measures.append(measure)
                    break
        
        # Final fallback: ensure we have at least something
        if not dimensions and len(data.columns) > 0:
            dimensions.append(data.columns[0])
        if not measures and len(data.columns) > 1:
            measures.append(data.columns[-1])
        elif not measures and len(data.columns) == 1:
            measures.append('count')
        
        return dimensions, measures

    def _parse_agent_string_result(self, agent_result: str, user_query: str, full_dataset: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        DEPRECATED: This method contained dataset-specific parsing logic.
        Now using generic _execute_real_pandas_analysis for all datasets.
        """
        print(f"🤖 Skipping legacy dataset-specific parsing - using generic pandas execution instead")
        return None

    def _execute_enhanced_agent_result_on_data(self, user_query: str, full_dataset: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        DEPRECATED: This method contained dataset-specific logic.
        Now using generic _execute_real_pandas_analysis for all datasets.
        """
        print(f"🤖 Skipping legacy dataset-specific enhanced execution - using generic pandas execution instead")
        return None

    def get_text_analysis(self, user_query: str, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        PANDAS DATAFRAME AGENT: Interactive pandas analysis with real code execution
        Returns text answers with actual computed results and code transparency
        """
        try:
            print(f"🐼 PANDAS DATAFRAME AGENT Analysis for: '{user_query}'")
            print(f"📊 Dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
            print(f"🔍 Columns: {list(dataset.columns)}")
            
            if not LANGCHAIN_AVAILABLE or self.llm is None:
                print("❌ LangChain not available, falling back to direct Gemini")
                return self._direct_gemini_analysis(user_query, dataset)
            
            # PRIORITY: Use real dataset with custom pandas execution
            return self._execute_real_pandas_analysis(user_query, dataset)
            
            # Try using a more compatible model for LangChain
            try:
                print("🔧 Creating pandas DataFrame agent with gemini-2.0-flash...")
                
                # Create agent with better error handling
                agent = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=dataset,
                    verbose=True,
                    return_intermediate_steps=True,
                    allow_dangerous_code=True,
                    # Add output parsing settings
                    agent_type="openai-functions" if hasattr(self.llm, 'bind') else "zero-shot-react-description",
                    max_iterations=3,
                    early_stopping_method="generate"
                )
                print("✅ Pandas DataFrame Agent created successfully")
                
            except Exception as agent_error:
                print(f"❌ Failed to create pandas DataFrame agent: {agent_error}")
                print("🔄 Falling back to direct Gemini analysis...")
                return self._direct_gemini_analysis(user_query, dataset)
            
            # Create enhanced prompt for better parsing
            enhanced_prompt = f"""
You are a pandas data analyst. Use the provided DataFrame 'df' to answer the user's question.

CRITICAL INSTRUCTIONS:
1. ALWAYS use the actual DataFrame 'df' provided - it contains real data
2. Use proper pandas operations like df.nlargest(), df.groupby(), etc.
3. Provide clear, specific answers with actual values from the data
4. Format your response clearly

User Question: "{user_query}"

DataFrame Info:
- Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns  
- Columns: {list(dataset.columns)}
- Sample data:
{dataset.head(3).to_string()}

Please analyze the data and provide your answer.
"""
            
            # Execute with better error handling
            try:
                print("🚀 Executing pandas DataFrame agent...")
                result = agent.invoke({"input": enhanced_prompt})
                
                print("=" * 60)
                print("🔍 PANDAS DATAFRAME AGENT RAW OUTPUT:")
                print("=" * 60)
                print(f"Result type: {type(result)}")
                print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                print("=" * 60)
                
                if isinstance(result, dict) and "output" in result:
                    final_answer = result["output"]
                    intermediate_steps = result.get("intermediate_steps", [])
                    
                    print("📝 FINAL ANSWER FROM AGENT:")
                    print("-" * 40)
                    print(final_answer)
                    print("-" * 40)
                    
                    print(f"🔧 INTERMEDIATE STEPS ({len(intermediate_steps)} steps):")
                    
                    # Extract reasoning and code from intermediate steps
                    reasoning_steps = []
                    code_steps = []
                    
                    for i, step in enumerate(intermediate_steps):
                        print(f"\n--- Step {i+1} ---")
                        if len(step) >= 2:
                            action, observation = step[0], step[1]
                            
                            print(f"Action: {type(action).__name__}")
                            if hasattr(action, 'log') and action.log:
                                print(f"Log: {action.log[:200]}...")
                            
                            print(f"Observation: {str(observation)[:200]}...")
                            
                            # Extract code from action
                            if hasattr(action, 'tool_input') and action.tool_input:
                                code = str(action.tool_input)
                                code_steps.append(code)
                                print(f"💻 Code executed: {code[:100]}...")
                            
                            # Extract reasoning
                            if hasattr(action, 'log') and action.log:
                                if "Thought:" in action.log:
                                    thought = action.log.split("Thought:")[1].split("Action:")[0].strip()
                                    if thought:
                                        reasoning_steps.append(thought)
                                        print(f"💭 Reasoning: {thought[:100]}...")
                    
                    print("=" * 60)
                    print(f"✅ Pandas agent analysis completed successfully!")
                    print(f"📊 Generated {len(reasoning_steps)} reasoning steps and {len(code_steps)} code steps")
                    print("=" * 60)
                    
                    return {
                        "answer": str(final_answer),
                        "success": True,
                        "reasoning_steps": reasoning_steps,
                        "code_steps": code_steps,
                        "tabular_data": [],
                        "has_table": False
                    }
                else:
                    raise ValueError(f"Unexpected result format: {type(result)}")
                
            except Exception as execution_error:
                print(f"❌ Pandas agent execution failed: {execution_error}")
                print("🔄 Trying direct agent.run() method...")
                
                try:
                    # Fallback to simple run method
                    simple_result = agent.run(enhanced_prompt)
                    
                    print("=" * 60)
                    print("🔄 PANDAS DATAFRAME AGENT SIMPLE RUN OUTPUT:")
                    print("=" * 60)
                    print(f"Result type: {type(simple_result)}")
                    print("📝 SIMPLE AGENT ANSWER:")
                    print("-" * 40)
                    print(simple_result)
                    print("-" * 40)
                    print("=" * 60)
                    
                    return {
                        "answer": str(simple_result),
                        "success": True,
                        "reasoning_steps": ["Used pandas DataFrame agent with simple execution"],
                        "code_steps": ["Executed pandas operations on your actual dataset"],
                        "tabular_data": [],
                        "has_table": False
                    }
                    
                except Exception as run_error:
                    print(f"❌ Agent.run() also failed: {run_error}")
                    print("🔄 Falling back to direct Gemini analysis...")
                    return self._direct_gemini_analysis(user_query, dataset)
            
        except Exception as e:
            print(f"❌ Pandas DataFrame agent failed: {str(e)}")
            print("🔄 Falling back to direct Gemini analysis...")
            return self._direct_gemini_analysis(user_query, dataset)

    def _execute_real_pandas_analysis(self, user_query: str, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute pandas operations on REAL dataset to prevent fabricated data
        """
        print("🔬 EXECUTING PANDAS ANALYSIS ON REAL DATA")
        print(f"📊 Real dataset shape: {dataset.shape}")
        print(f"📋 Real columns: {list(dataset.columns)}")
        
        try:
            # Show actual data sample to prove we're using real data
            real_data_sample = dataset.head(3).to_string(index=False)
            print(f"📄 REAL DATA SAMPLE (first 3 rows):")
            print(real_data_sample)
            
            # Automatically detect data types and structure for generic examples
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create generic example based on actual dataset structure
            if len(numeric_columns) > 0 and len(categorical_columns) > 0:
                example_numeric = numeric_columns[0]
                example_categorical = categorical_columns[0]
                generic_example = f"""# Answer the user's query using df
result = df.nlargest(5, '{example_numeric}')[['{example_categorical}', '{example_numeric}']]
print("Top 5 records by {example_numeric}:")
print(result.to_string(index=False))"""
            elif len(numeric_columns) > 0:
                example_numeric = numeric_columns[0]
                generic_example = f"""# Answer the user's query using df
result = df.nlargest(5, '{example_numeric}')
print("Top 5 records by {example_numeric}:")
print(result.to_string(index=False))"""
            else:
                # Use first available column
                first_col = dataset.columns[0] if len(dataset.columns) > 0 else 'column'
                generic_example = f"""# Answer the user's query using df
result = df['{first_col}'].value_counts().head(5)
print("Top 5 most frequent values in {first_col}:")
print(result)"""

            # Generate Python code using Gemini that works with real dataset
            code_generation_prompt = f"""You are a data analyst. Generate Python pandas code to answer the user's query using the provided DataFrame.

REAL DATASET CONTEXT:
- Variable name: 'df' 
- Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns
- Columns: {list(dataset.columns)}
- Data types: {dict(dataset.dtypes.astype(str))}
- Sample data (first 3 rows):
{real_data_sample}

USER QUERY: "{user_query}"

Generate ONLY Python pandas code that:
1. Uses ONLY the variable 'df' (which contains the real data above)
2. NEVER creates or recreates the DataFrame 
3. Uses appropriate pandas methods (nlargest, groupby, mean, sum, etc.)
4. Includes print statements to show results clearly
5. Provides the exact answer to the user's question
6. Works with the actual column names and data types shown above

Example format:
```python
{generic_example}
```

Generate ONLY the code, no explanations:"""

            print("🤖 Generating pandas code for real dataset...")
            code_response = self.model.generate_content(code_generation_prompt)
            
            # Extract token usage from the API response
            token_usage = {}
            if hasattr(code_response, 'usage_metadata') and code_response.usage_metadata:
                token_usage = {
                    'inputTokens': getattr(code_response.usage_metadata, 'prompt_token_count', 0),
                    'outputTokens': getattr(code_response.usage_metadata, 'candidates_token_count', 0),
                    'totalTokens': getattr(code_response.usage_metadata, 'total_token_count', 0)
                }
                print(f"🪙 Token usage - Input: {token_usage.get('inputTokens', 0)}, Output: {token_usage.get('outputTokens', 0)}, Total: {token_usage.get('totalTokens', 0)}")
            
            if code_response and code_response.text:
                # Extract Python code from response  
                generated_code = code_response.text.strip()
                
                # Clean up code (remove markdown formatting)
                if "```python" in generated_code:
                    code_lines = generated_code.split("```python")[1].split("```")[0].strip()
                elif "```" in generated_code:
                    code_lines = generated_code.split("```")[1].strip()
                else:
                    code_lines = generated_code
                
                print("💻 GENERATED PANDAS CODE FOR REAL DATA:")
                print("-" * 50)
                print(code_lines)
                print("-" * 50)
                
                # Execute code on real dataset
                print("⚡ EXECUTING CODE ON REAL DATASET...")
                
                # Create safe execution environment
                import io
                import sys
                from contextlib import redirect_stdout
                
                # Capture output
                captured_output = io.StringIO()
                
                # Create execution globals with REAL dataset
                execution_globals = {
                    'df': dataset.copy(),  # ✅ REAL dataset, not fabricated
                    'pd': pd,
                    'numpy': __import__('numpy'),
                    'print': lambda *args, **kwargs: print(*args, **kwargs, file=captured_output)
                }
                
                try:
                    # Execute generated code on REAL data
                    exec(code_lines, execution_globals)
                    
                    # Get the output
                    execution_output = captured_output.getvalue()
                    
                    print("✅ CODE EXECUTION SUCCESSFUL ON REAL DATA!")
                    print("📋 REAL DATA ANALYSIS RESULTS:")
                    print("-" * 50)
                    print(execution_output)
                    print("-" * 50)
                    
                    # Create analysis text
                    analysis_text = f"Based on your real dataset, here are the results for '{user_query}':\n\n{execution_output.strip()}"
                    
                    # Try to extract tabular data from output using enhanced parser
                    tabular_data = []
                    has_table = False
                    
                    if execution_output.strip():
                        try:
                            parsed_table = self._parse_pandas_output_to_table(execution_output)
                            if parsed_table:
                                tabular_data = [parsed_table]  # Wrap in array for frontend
                                has_table = True
                                print(f"✅ Successfully parsed tabular data: {len(parsed_table.get('rows', []))} rows")
                        except Exception as parse_error:
                            print(f"⚠️ Table parsing failed: {parse_error}")
                            has_table = False
                    
                    return {
                        "answer": analysis_text,
                        "success": True,
                        "reasoning_steps": ["✅ Executed pandas code on REAL uploaded dataset"],
                        "code_steps": [code_lines],  # Show the actual pandas code
                        "tabular_data": tabular_data,
                        "has_table": has_table,
                        "token_usage": token_usage
                    }
                    
                except Exception as exec_error:
                    print(f"❌ CODE EXECUTION FAILED: {exec_error}")
                    error_msg = f"Error executing pandas code on real dataset: {str(exec_error)}"
                    
                    return {
                        "answer": f"I generated pandas code for your real dataset but encountered an execution error: {error_msg}. The code was: {code_lines}",
                        "success": False,
                        "reasoning_steps": [f"❌ Code execution failed: {str(exec_error)}"],
                        "code_steps": [code_lines],
                        "tabular_data": [],
                        "has_table": False,
                        "token_usage": token_usage
                    }
                    
            else:
                print("❌ Failed to generate pandas code")
                raise Exception("No pandas code generated by Gemini")
                
        except Exception as e:
            print(f"❌ Real pandas analysis failed: {e}")
            print("🔄 Falling back to direct Gemini analysis...")
            return self._direct_gemini_analysis(user_query, dataset)
    
    def _parse_pandas_output_to_table(self, output_text: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced parser for pandas output formats (Series, DataFrame, etc.)
        """
        try:
            lines = output_text.strip().split('\n')
            
            # Remove empty lines and title lines
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.endswith(':') and line != 'Name: Revenue, dtype: float64' and 'dtype:' not in line:
                    content_lines.append(line)
            
            if len(content_lines) < 2:
                return None
                
            print(f"🔍 Parsing pandas output - {len(content_lines)} content lines")
            
            # Pattern 1: Pandas Series with index (most common for groupby)
            # Format: "Product\nBookshelf    20500\nDesk    42000"
            if len(content_lines) >= 2:
                # Check if first line looks like a column/index name
                first_line = content_lines[0].strip()
                
                # Check if subsequent lines have consistent format: "name    value"
                data_rows = []
                series_format = True
                
                for i in range(1, len(content_lines)):
                    line = content_lines[i].strip()
                    # Split on whitespace, but handle names with spaces
                    parts = line.split()
                    
                    if len(parts) >= 2:
                        # Last part should be numeric (value)
                        try:
                            float(parts[-1])
                            # Everything except the last part is the name/index
                            name = ' '.join(parts[:-1])
                            value = parts[-1]
                            data_rows.append([name, value])
                        except ValueError:
                            series_format = False
                            break
                    else:
                        series_format = False
                        break
                
                # If we detected Series format, create table
                if series_format and len(data_rows) > 0:
                    print(f"✅ Detected pandas Series format with {len(data_rows)} rows")
                    return {
                        "type": "table",
                        "columns": [first_line, "Value"],
                        "rows": data_rows
                    }
            
            # Pattern 2: DataFrame format with column headers
            # Format: "  Product  Revenue\n0  Laptop  30000\n1  Phone   25000"
            if len(content_lines) >= 3:
                # Check if first line has multiple column names
                potential_headers = content_lines[0].split()
                
                if len(potential_headers) >= 2:
                    data_rows = []
                    df_format = True
                    
                    for i in range(1, len(content_lines)):
                        line = content_lines[i].strip()
                        parts = line.split()
                        
                        # Skip index column if present (starts with number)
                        if len(parts) >= len(potential_headers):
                            if parts[0].isdigit():
                                row_data = parts[1:1+len(potential_headers)]
                            else:
                                row_data = parts[:len(potential_headers)]
                            
                            if len(row_data) == len(potential_headers):
                                data_rows.append(row_data)
                            else:
                                df_format = False
                                break
                        else:
                            df_format = False
                            break
                    
                    if df_format and len(data_rows) > 0:
                        print(f"✅ Detected pandas DataFrame format with {len(data_rows)} rows")
                        return {
                            "type": "table",
                            "columns": potential_headers,
                            "rows": data_rows
                        }
            
            # Pattern 3: Simple key-value pairs
            # Format: "Laptop: 40000\nPhone: 25000"
            if len(content_lines) >= 2:
                kv_rows = []
                kv_format = True
                
                for line in content_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        kv_rows.append([key.strip(), value.strip()])
                    else:
                        kv_format = False
                        break
                
                if kv_format and len(kv_rows) > 0:
                    print(f"✅ Detected key-value format with {len(kv_rows)} rows")
                    return {
                        "type": "table",
                        "columns": ["Item", "Value"],
                        "rows": kv_rows
                    }
            
            print("ℹ️ No recognizable table format detected")
            return None
            
        except Exception as e:
            print(f"❌ Error parsing pandas output: {e}")
            return None
    
    def _direct_gemini_analysis(self, user_query: str, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Direct analysis using main Gemini model with optimized token usage
        """
        try:
            # OPTIMIZATION 0: Analyze dataset characteristics for adaptive optimization
            dataset_characteristics = self._analyze_dataset_characteristics(dataset)
            
            # OPTIMIZATION 1: Smart column filtering based on query relevance
            relevant_columns = self._detect_relevant_columns(user_query, dataset.columns.tolist())
            filtered_dataset = dataset[relevant_columns] if relevant_columns else dataset
            
            # OPTIMIZATION 2: Adaptive sample size based on query complexity and dataset characteristics
            sample_size = self._determine_optimal_sample_size_adaptive(user_query, dataset_characteristics)
            sample_data = filtered_dataset.head(sample_size)
            
            # OPTIMIZATION 3: Selective statistical summary based on dataset type and relevance
            stats_summary = self._generate_adaptive_stats_summary(filtered_dataset, dataset_characteristics, relevant_columns)
            
            # OPTIMIZATION 4: Compact dataset info with essential context only
            dataset_info = f"""
Dataset Context:
- Total: {dataset.shape[0]} rows, {dataset.shape[1]} columns (showing {len(relevant_columns)} relevant)
- Types: {len(dataset_characteristics['numeric_columns'])} numeric, {len(dataset_characteristics['categorical_columns'])} categorical, {len(dataset_characteristics['datetime_columns'])} temporal
- Relevant columns: {relevant_columns}
- Sample data ({sample_size} rows):
{sample_data.to_string()}

Key Statistics:
{stats_summary}
"""
            
            # Automatically detect numeric columns for generic examples
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create generic example based on actual dataset structure
            if len(numeric_columns) > 0 and len(categorical_columns) > 0:
                example_numeric = numeric_columns[0]
                example_categorical = categorical_columns[0]
                generic_code_example = f"""# Get top 5 records by {example_numeric}
top_5 = df.nlargest(5, '{example_numeric}')[['{example_categorical}', '{example_numeric}']]
print(top_5)"""
                generic_analysis_example = f"Based on the dataset, here are the top 5 records by {example_numeric}..."
            elif len(numeric_columns) > 0:
                example_numeric = numeric_columns[0]
                generic_code_example = f"""# Get top 5 records by {example_numeric}
top_5 = df.nlargest(5, '{example_numeric}')
print(top_5)"""
                generic_analysis_example = f"Based on the dataset, here are the top 5 records by {example_numeric}..."
            else:
                first_col = dataset.columns[0] if len(dataset.columns) > 0 else 'column'
                generic_code_example = f"""# Get most frequent values in {first_col}
result = df['{first_col}'].value_counts().head(5)
print(result)"""
                generic_analysis_example = f"Based on the dataset, here are the most frequent values in {first_col}..."

            # OPTIMIZATION 5: Streamlined prompt with essential instructions only
            direct_prompt = f"""
Analyze this dataset and provide a comprehensive answer with Python code.

{dataset_info}

User Question: "{user_query}"

RESPONSE FORMAT:
## Analysis
[Your detailed analysis with specific numbers and insights from the actual data]

## Python Code
```python
# Exact pandas code using 'df' variable
[Show the pandas operations that generate these results]
```

REQUIREMENTS:
1. Use the ACTUAL data provided above - not mock data
2. Reference specific column names: {relevant_columns}
3. Provide concrete numerical results from the dataset
4. Use 'df' as the DataFrame variable in code
5. Include column names exactly as shown

Now analyze: "{user_query}"."""

            print(f"🔍 Using direct Gemini analysis for: {user_query}")
            print(f"📊 Generic token optimization applied:")
            print(f"   • Dataset: {dataset_characteristics['total_rows']} rows, {dataset_characteristics['total_columns']} columns")
            print(f"   • Types: {len(dataset_characteristics['numeric_columns'])} numeric, {len(dataset_characteristics['categorical_columns'])} categorical")
            print(f"   • Columns: {len(dataset.columns)} → {len(relevant_columns)} ({100*(1-len(relevant_columns)/len(dataset.columns)):.1f}% reduction)")
            print(f"   • Sample size: {sample_size} rows (complexity score: {dataset_characteristics['complexity_score']:.1f})")
            print(f"   • Stats: Adaptive summary based on dataset characteristics")
            
            response = self.model.generate_content(direct_prompt)
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    'inputTokens': getattr(response.usage_metadata, 'prompt_token_count', 0),
                    'outputTokens': getattr(response.usage_metadata, 'candidates_token_count', 0),
                    'totalTokens': getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            print("=" * 60)
            print("🤖 DIRECT GEMINI MODEL OUTPUT:")
            print("=" * 60)
            print(f"Response type: {type(response)}")
            print(f"Has text: {bool(response and response.text)}")
            print("=" * 60)
            
            if response and response.text:
                answer_text = response.text.strip()
                
                print("📝 RAW GEMINI RESPONSE:")
                print("-" * 40)
                print(answer_text)
                print("-" * 40)
                
                # Extract Python code from the response
                python_code = []
                analysis_text = answer_text
                
                print("🔍 EXTRACTING PYTHON CODE FROM RESPONSE:")
                print(f"📄 Response contains '## Python Code': {'## Python Code' in answer_text}")
                
                # Parse the structured response format
                if "## Python Code" in answer_text:
                    # Split analysis and code sections
                    parts = answer_text.split("## Python Code")
                    if len(parts) >= 2:
                        analysis_text = parts[0].replace("## Analysis", "").strip()
                        code_section = parts[1].strip()
                        
                        print(f"📝 Code section found: {code_section[:200]}...")
                        
                        # Extract code from markdown code blocks
                        if "```python" in code_section:
                            print("✅ Found '```python' code blocks")
                            code_blocks = code_section.split("```python")
                            for i, block in enumerate(code_blocks[1:]):  # Skip first split (before first code block)
                                if "```" in block:
                                    code = block.split("```")[0].strip()
                                    if code:
                                        python_code.append(code)
                                        print(f"📝 Extracted code block {i+1}: {code[:100]}...")
                        
                        elif "```" in code_section:
                            # Handle cases where it's just ``` without python specifier
                            code_blocks = code_section.split("```")
                            for i in range(1, len(code_blocks), 2):  # Get odd indices (code blocks)
                                code = code_blocks[i].strip()
                                if code and not code.startswith('#'):  # Avoid just comments
                                    python_code.append(code)
                
                # Try to extract tabular data using enhanced parser
                tabular_data = []
                has_table = False
                
                # Use the same enhanced parser as the main analysis method
                try:
                    parsed_table = self._parse_pandas_output_to_table(analysis_text)
                    if parsed_table:
                        tabular_data = [parsed_table]  # Wrap in array for frontend
                        has_table = True
                        print(f"✅ Direct Gemini analysis: parsed tabular data with {len(parsed_table.get('rows', []))} rows")
                except Exception as parse_error:
                    print(f"⚠️ Direct Gemini table parsing failed: {parse_error}")
                    has_table = False
                
                # Prepare reasoning and code steps for display
                reasoning_steps = ["Used direct Gemini 2.0 Flash analysis for maximum reliability"]
                code_steps = python_code if python_code else ["Generated analysis from dataset summary and statistical data"]
                
                print("=" * 60)
                print("🎯 FINAL RESPONSE SUMMARY:")
                print("=" * 60)
                print(f"📝 Analysis text length: {len(analysis_text)}")
                print(f"🐍 Python code blocks found: {len(python_code)}")
                if python_code:
                    for i, code in enumerate(python_code):
                        print(f"   Code Block {i+1}: {len(code)} characters")
                        print(f"   Preview: {code[:50]}...")
                print(f"📊 Tabular data: {tabular_data}")
                print(f"🔢 Has table: {has_table}")
                print("=" * 60)
                
                response_data = {
                    "answer": analysis_text,
                    "success": True,
                    "reasoning_steps": reasoning_steps,
                    "code_steps": code_steps,
                    "tabular_data": tabular_data,
                    "has_table": has_table,
                    "token_usage": token_usage
                }
                
                return response_data
            else:
                return {
                    "answer": "I was unable to generate an analysis for your query.",
                    "success": False,
                    "reasoning_steps": [],
                    "code_steps": [],
                    "tabular_data": [],
                    "has_table": False,
                    "token_usage": {}
                }
                
        except Exception as direct_error:
            print(f"❌ Direct Gemini analysis also failed: {direct_error}")
            return {
                "answer": f"I encountered an error while analyzing your data: {str(direct_error)[:200]}...",
                "success": False,
                "reasoning_steps": [],
                "code_steps": [],
                "tabular_data": [],
                "has_table": False,
                "token_usage": {}
            }
    
    def _validate_agent_used_real_data(self, user_query: str, agent_answer: str, dataset: pd.DataFrame) -> bool:
        """Validate if the pandas agent used the actual dataset or created mock data"""
        try:
            query_lower = user_query.lower()
            
            # For "highest core area" queries, check if answer matches mock data patterns
            if 'highest' in query_lower and 'core area' in query_lower:
                # Agent using mock data would likely say "Chhattisgarh with 2797"
                if "chhattisgarh" in agent_answer.lower() and "2797" in agent_answer:
                    print("🚨 Detected agent using mock data (Chhattisgarh 2797)")
                    return False
            
            # For "top 5" queries, check if results match expected real data patterns
            elif 'top' in query_lower and ('5' in query_lower or 'five' in query_lower):
                # Check if agent mentions realistic reserve names from our dataset
                real_reserve_names = set(dataset['TigerReserve'].str.lower() if 'TigerReserve' in dataset.columns else [])
                mentioned_reserves = set()
                for reserve in real_reserve_names:
                    if reserve in agent_answer.lower():
                        mentioned_reserves.add(reserve)
                
                # If less than 3 real reserve names mentioned, likely using mock data
                if len(mentioned_reserves) < 3:
                    print(f"🚨 Detected agent using mock data (only {len(mentioned_reserves)} real reserves mentioned)")
                    return False
            
            # Check for common mock data indicators
            mock_indicators = ["sample data", "mock data", "example data", "data = {"]
            for indicator in mock_indicators:
                if indicator in agent_answer.lower():
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return True  # Assume valid if we can't validate
    
    def _provide_correct_calculation(self, user_query: str, dataset: pd.DataFrame) -> str:
        """Provide correct calculation when agent used mock data"""
        try:
            query_lower = user_query.lower()
            
            # Handle "highest core area by state" queries
            if 'highest' in query_lower and 'core area' in query_lower and 'state' in query_lower:
                if 'State' in dataset.columns and 'CoreArea_km2' in dataset.columns:
                    state_areas = dataset.groupby('State')['CoreArea_km2'].sum()
                    max_area = state_areas.max()
                    state_with_max = state_areas.idxmax()
                    return f"The state with the highest total core area is {state_with_max} with {max_area:.2f} sq km."
            
            # Handle "highest core area" (single reserve) queries  
            elif 'highest' in query_lower and 'core area' in query_lower:
                if 'CoreArea_km2' in dataset.columns and 'TigerReserve' in dataset.columns:
                    max_idx = dataset['CoreArea_km2'].idxmax()
                    max_reserve = dataset.loc[max_idx, 'TigerReserve']
                    max_area = dataset.loc[max_idx, 'CoreArea_km2']
                    state = dataset.loc[max_idx, 'State'] if 'State' in dataset.columns else 'Unknown'
                    return f"The tiger reserve with the highest core area is {max_reserve} in {state} with {max_area:.2f} sq km."
            
            # Handle "top 5" queries
            elif 'top' in query_lower and ('5' in query_lower or 'five' in query_lower):
                if 'Population2023' in dataset.columns and 'TigerReserve' in dataset.columns:
                    top_5 = dataset.nlargest(5, 'Population2023')
                    results = []
                    for _, row in top_5.iterrows():
                        results.append(f"{row['TigerReserve']} ({int(row['Population2023'])})")
                    return f"The top 5 tiger reserves by population in 2023 are: {', '.join(results)}."
            
            # Handle Karnataka reserves queries
            elif 'karnataka' in query_lower and 'reserve' in query_lower:
                if 'State' in dataset.columns and 'TigerReserve' in dataset.columns:
                    karnataka_reserves = dataset[dataset['State'] == 'Karnataka']['TigerReserve'].tolist()
                    return f"Karnataka has {len(karnataka_reserves)} tiger reserves: {', '.join(karnataka_reserves)}."
            
            return None
            
        except Exception as e:
            print(f"Correction calculation error: {e}")
            return None
    
    def _extract_tabular_data_from_observation(self, observation: str) -> Dict[str, Any]:
        """Extract tabular data from pandas agent observations"""
        try:
            # Check if observation contains tabular output (DataFrame print)
            lines = observation.strip().split('\n')
            
            # Look for DataFrame patterns
            if len(lines) < 3:
                return None
                
            # Pattern 1: Check for standard DataFrame output with index and columns
            # Example:
            #                    Change
            # State                    
            # Andhra Pradesh      -16.0
            # Assam                62.0
            
            header_line = None
            data_rows = []
            index_column_name = None
            
            for i, line in enumerate(lines):
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Check for column headers (often indented with column names)
                if any(char.isalpha() for char in line) and line.strip() and not line.startswith(' ' * 10):
                    if header_line is None and ('Change' in line or 'Population' in line or 'Area' in line):
                        header_line = line.strip()
                        # Check if next line might be index column name
                        if i + 1 < len(lines) and lines[i + 1].strip() and not any(char.isdigit() for char in lines[i + 1][:20]):
                            index_column_name = lines[i + 1].strip()
                        continue
                
                # Look for data rows (contain numbers and state/reserve names)
                if line.strip() and any(char.isdigit() for char in line):
                    # Extract index and values
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # Try to parse the last part as a number
                            float(parts[-1])
                            index_name = ' '.join(parts[:-1])
                            value = parts[-1]
                            data_rows.append({'index': index_name, 'value': value})
                        except ValueError:
                            continue
            
            # If we found tabular data, format it
            if data_rows and len(data_rows) > 1:
                # Create table structure
                table_data = {
                    'type': 'table',
                    'columns': [index_column_name or 'Item', header_line or 'Value'],
                    'rows': [[row['index'], row['value']] for row in data_rows],
                    'title': f"Results ({len(data_rows)} rows)"
                }
                
                print(f"📊 Extracted table data: {len(data_rows)} rows")
                return table_data
            
            # Pattern 2: Simple single-value results
            # Example: "198" or "43.56603773584906"
            if len(lines) == 1 and lines[0].strip():
                try:
                    value = float(lines[0].strip())
                    return {
                        'type': 'value',
                        'value': value,
                        'formatted': lines[0].strip()
                    }
                except ValueError:
                    pass
                    
            return None
            
        except Exception as e:
            print(f"Table extraction error: {e}")
            return None
    
    def _ensure_json_safe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame is JSON serializable with comprehensive safety checks"""
        safe_data = data.copy()
        
        print(f"🔧 JSON safety check for DataFrame: {safe_data.shape}")
        
        for col in safe_data.columns:
            # Handle all numeric columns (float, int, complex)
            if safe_data[col].dtype.kind in 'fiuc':  # float, int, unsigned int, complex
                # Step 1: Replace infinite values with None
                safe_data[col] = safe_data[col].replace([np.inf, -np.inf], None)
                
                # Step 2: Replace NaN values with None
                safe_data[col] = safe_data[col].where(pd.notna(safe_data[col]), None)
                
                # Step 3: Handle extremely large values that might not be JSON safe
                if safe_data[col].dtype.kind == 'f':  # float columns
                    # Check for values that might be too large for JSON
                    mask = safe_data[col].notna()
                    if mask.any():
                        # Replace values outside reasonable JSON range
                        safe_data.loc[mask & (safe_data[col].abs() > 1e15), col] = None
                        
                # Step 4: Convert to standard Python types
                safe_data[col] = safe_data[col].astype(object, errors='ignore')
            
            # Handle string columns that might have issues
            elif safe_data[col].dtype == 'object':
                # Replace any remaining NaN in object columns
                safe_data[col] = safe_data[col].where(pd.notna(safe_data[col]), None)
        
        print(f"🔧 JSON safety check completed")
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
    
    def _detect_relevant_columns(self, user_query: str, all_columns: List[str]) -> List[str]:
        """
        Smart column detection based on query content to reduce token usage
        """
        query_lower = user_query.lower()
        relevant_columns = []
        
        # Always include essential identifier columns (generic patterns)
        essential_patterns = ['id', 'name', 'key', 'code', 'year', 'date', 'time', 'index']
        for col in all_columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in essential_patterns):
                relevant_columns.append(col)
        
        # Extract specific column mentions from query
        for col in all_columns:
            col_lower = col.lower().replace('_', ' ')
            # Direct mentions
            if col_lower in query_lower or col.lower() in query_lower:
                if col not in relevant_columns:
                    relevant_columns.append(col)
            # Partial word matches for compound terms
            col_words = col_lower.split('_')
            if len(col_words) > 1:
                for word in col_words:
                    if len(word) > 3 and word in query_lower:
                        if col not in relevant_columns:
                            relevant_columns.append(col)
                            break
        
        # Add semantically related columns based on query context
        query_keywords = self._extract_query_keywords(query_lower)
        for col in all_columns:
            col_lower = col.lower()
            for keyword in query_keywords:
                if keyword in col_lower and col not in relevant_columns:
                    relevant_columns.append(col)
        
        # Fallback: if too few columns found, add some numeric/categorical columns for context
        if len(relevant_columns) < 3:
            # Add numeric columns with common analytical patterns
            numeric_indicators = ['count', 'total', 'amount', 'rate', 'avg', 'average', 'sum', 'value', 'score', 'percent', 'ratio']
            numeric_cols = [col for col in all_columns if any(
                indicator in col.lower() for indicator in numeric_indicators
            )]
            
            # If no pattern-based numeric columns, add any numeric columns
            if not numeric_cols:
                try:
                    # This would need the actual dataset to determine numeric columns
                    # For now, we'll use heuristics based on column names
                    potential_numeric = [col for col in all_columns if 
                                       not any(text_indicator in col.lower() for text_indicator in 
                                              ['name', 'description', 'text', 'comment', 'note', 'address'])]
                    numeric_cols = potential_numeric[:5]
                except:
                    numeric_cols = []
            
            for col in numeric_cols[:5]:  # Add up to 5 numeric columns
                if col not in relevant_columns:
                    relevant_columns.append(col)
        
        # Limit maximum columns to prevent token explosion
        max_columns = self.OPTIMIZATION_CONFIG['max_relevant_columns']
        if len(relevant_columns) > max_columns:
            # Prioritize columns mentioned in query first
            direct_mentions = []
            indirect_mentions = []
            
            for col in relevant_columns:
                if col.lower() in query_lower or any(word in query_lower for word in col.lower().split('_')):
                    direct_mentions.append(col)
                else:
                    indirect_mentions.append(col)
            
            relevant_columns = direct_mentions + indirect_mentions[:max_columns-len(direct_mentions)]
        
        print(f"🎯 Column optimization: {len(all_columns)} → {len(relevant_columns)} columns")
        print(f"📋 Relevant columns: {relevant_columns}")
        
        return relevant_columns if relevant_columns else all_columns[:self.OPTIMIZATION_CONFIG['max_relevant_columns']//2]  # Fallback
    
    def _extract_query_keywords(self, query_lower: str) -> List[str]:
        """Extract meaningful keywords from user query for column matching (generic approach)"""
        analysis_keywords = []
        
        # Extract potential metric/aggregation names (universal patterns)
        metric_patterns = ['average', 'avg', 'mean', 'median', 'total', 'sum', 'count', 'max', 'maximum', 
                          'min', 'minimum', 'top', 'highest', 'lowest', 'first', 'last', 'most', 'least',
                          'percent', 'percentage', 'ratio', 'rate', 'frequency', 'distribution']
        for pattern in metric_patterns:
            if pattern in query_lower:
                analysis_keywords.append(pattern)
        
        # Extract comparison and analytical terms
        analytical_terms = ['compare', 'comparison', 'versus', 'vs', 'between', 'correlation', 'relationship',
                           'trend', 'pattern', 'growth', 'increase', 'decrease', 'change', 'difference',
                           'above', 'below', 'greater', 'less', 'higher', 'lower', 'more', 'fewer']
        for term in analytical_terms:
            if term in query_lower:
                analysis_keywords.append(term)
        
        # Extract temporal terms
        temporal_terms = ['year', 'month', 'day', 'date', 'time', 'period', 'duration', 'recent', 'latest',
                         'historical', 'past', 'current', 'future', 'annual', 'monthly', 'daily', 'weekly']
        for term in temporal_terms:
            if term in query_lower:
                analysis_keywords.append(term)
        
        # Extract domain-agnostic quantitative terms
        quantitative_terms = ['number', 'amount', 'quantity', 'volume', 'size', 'length', 'width', 'height',
                             'weight', 'distance', 'area', 'value', 'price', 'cost', 'revenue', 'profit',
                             'score', 'rating', 'grade', 'level', 'rank', 'position']
        for term in quantitative_terms:
            if term in query_lower:
                analysis_keywords.append(term)
        
        return list(set(analysis_keywords))  # Remove duplicates
    
    def _analyze_dataset_characteristics(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset characteristics to inform optimization decisions (generic approach)
        """
        characteristics = {
            'total_columns': len(dataset.columns),
            'total_rows': len(dataset),
            'numeric_columns': [],
            'categorical_columns': [],
            'datetime_columns': [],
            'identifier_columns': [],
            'text_columns': [],
            'has_nulls': False,
            'complexity_score': 0
        }
        
        try:
            # Analyze column types
            for col in dataset.columns:
                col_lower = col.lower()
                dtype = str(dataset[col].dtype)
                
                # Identify column types
                if 'int' in dtype or 'float' in dtype:
                    characteristics['numeric_columns'].append(col)
                elif 'datetime' in dtype or any(date_term in col_lower for date_term in ['date', 'time', 'year', 'month']):
                    characteristics['datetime_columns'].append(col)
                elif any(id_term in col_lower for id_term in ['id', 'key', 'code', 'index']):
                    characteristics['identifier_columns'].append(col)
                elif dataset[col].dtype == 'object':
                    # Check if it's likely categorical vs text
                    unique_ratio = len(dataset[col].unique()) / len(dataset) if len(dataset) > 0 else 0
                    threshold = self.OPTIMIZATION_CONFIG['categorical_uniqueness_threshold']
                    if unique_ratio < threshold:  # Less than threshold suggests categorical
                        characteristics['categorical_columns'].append(col)
                    else:
                        characteristics['text_columns'].append(col)
                else:
                    characteristics['categorical_columns'].append(col)
            
            # Check for null values
            characteristics['has_nulls'] = dataset.isnull().any().any()
            
            # Calculate complexity score using configurable weights
            weights = self.OPTIMIZATION_CONFIG['complexity_score_weights']
            complexity_score = 0
            complexity_score += min(len(dataset.columns) / 10, 5) * weights['columns_per_10']
            complexity_score += min(len(dataset) / 10000, 3) * weights['rows_per_10k']
            complexity_score += len(characteristics['numeric_columns']) * weights['numeric_column_factor']
            complexity_score += len(characteristics['text_columns']) * weights['text_column_factor']
            
            characteristics['complexity_score'] = complexity_score
            
        except Exception as e:
            print(f"⚠️ Dataset analysis failed: {e}")
            # Fallback to basic characteristics
            characteristics['complexity_score'] = 2.0  # Default medium complexity
        
        return characteristics
    
    def _determine_optimal_sample_size_adaptive(self, user_query: str, dataset_characteristics: Dict[str, Any]) -> int:
        """
        Determine optimal sample size based on query complexity and dataset characteristics (generic)
        """
        base_size = self.OPTIMIZATION_CONFIG['min_sample_rows']
        max_size = self.OPTIMIZATION_CONFIG['max_sample_rows']
        
        # Increase sample size for complex queries (generic complexity indicators)
        complexity_indicators = ['compare', 'comparison', 'trend', 'pattern', 'distribution', 'correlation', 
                               'relationship', 'versus', 'between', 'analyze', 'analysis', 'detailed']
        complexity_bonus = sum(1 for indicator in complexity_indicators if indicator in user_query.lower())
        
        # Adjust based on dataset complexity score
        complexity_factor = min(int(dataset_characteristics.get('complexity_score', 2) / 2), 2)
        
        # Adjust based on column count (more columns = slightly larger sample for context)
        column_factor = min(len(dataset_characteristics.get('numeric_columns', [])) // 5, 1)
        
        optimal_size = base_size + complexity_bonus + complexity_factor + column_factor
        return min(optimal_size, max_size)  # Cap at configured maximum
    
    def _generate_adaptive_stats_summary(self, filtered_dataset: pd.DataFrame, 
                                       dataset_characteristics: Dict[str, Any], 
                                       relevant_columns: List[str]) -> str:
        """
        Generate statistical summary based on dataset type and relevance (generic approach)
        """
        try:
            numeric_cols = [col for col in relevant_columns if col in dataset_characteristics.get('numeric_columns', [])]
            max_stats_cols = self.OPTIMIZATION_CONFIG['max_stats_columns']
            
            if len(relevant_columns) <= 8 and len(numeric_cols) <= max_stats_cols:
                # Small relevant set - provide detailed stats
                if numeric_cols:
                    stats_summary = filtered_dataset[numeric_cols].describe().to_string()
                else:
                    # No numeric columns in relevant set
                    categorical_cols = [col for col in relevant_columns if col in dataset_characteristics.get('categorical_columns', [])]
                    if categorical_cols and len(categorical_cols) <= 3:
                        value_counts = []
                        for col in categorical_cols[:2]:  # Limit to 2 categorical columns
                            try:
                                top_values = filtered_dataset[col].value_counts().head(3)
                                value_counts.append(f"{col}: {dict(top_values)}")
                            except:
                                continue
                        stats_summary = "Top values:\n" + "\n".join(value_counts) if value_counts else "Categorical data available"
                    else:
                        stats_summary = f"Dataset contains {len(relevant_columns)} relevant columns with mixed data types"
            elif len(numeric_cols) > 0:
                # Medium relevant set - provide summary stats
                stats_summary = f"Numeric columns ({len(numeric_cols)}): {numeric_cols[:5]}"
                if len(numeric_cols) <= 2:
                    # Show basic stats for 1-2 numeric columns
                    try:
                        basic_stats = filtered_dataset[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(3)
                        stats_summary += f"\nBasic statistics:\n{basic_stats.to_string()}"
                    except:
                        stats_summary += "\nBasic statistics available on request"
                else:
                    stats_summary += "\nDetailed statistics available on request"
            else:
                # Large relevant set or no numeric columns
                stats_summary = f"Dataset with {len(relevant_columns)} relevant columns. "
                if dataset_characteristics.get('categorical_columns'):
                    stats_summary += f"Contains {len(dataset_characteristics['categorical_columns'])} categorical columns. "
                if dataset_characteristics.get('datetime_columns'):
                    stats_summary += f"Contains {len(dataset_characteristics['datetime_columns'])} temporal columns. "
                stats_summary += "Detailed statistics available on request."
                
        except Exception as e:
            print(f"⚠️ Stats generation failed: {e}")
            stats_summary = f"Statistical summary available for {len(relevant_columns)} relevant columns"
        
        return stats_summary
    
    def _determine_optimal_sample_size(self, user_query: str, num_columns: int) -> int:
        """
        Determine optimal sample size based on query complexity and column count
        """
        base_size = 2  # Minimum sample size
        
        # Increase sample size for complex queries
        complexity_indicators = ['compare', 'trend', 'pattern', 'distribution', 'correlation', 'relationship']
        complexity_bonus = sum(1 for indicator in complexity_indicators if indicator in user_query.lower())
        
        # Adjust based on column count
        column_factor = min(num_columns // 10, 2)  # Add 1 row per 10 columns, max 2 extra
        
        optimal_size = base_size + complexity_bonus + column_factor
        return min(optimal_size, 4)  # Cap at 4 rows to control token usage
    
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
