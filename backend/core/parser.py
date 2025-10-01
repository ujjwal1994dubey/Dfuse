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

class DfuseParser:
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
                        "columns": ["Item", "Value"],
                        "rows": kv_rows
                    }
            
            print("ℹ️ No recognizable table format detected")
            return None
            
        except Exception as e:
            print(f"❌ Error parsing pandas output: {e}")
            return None
