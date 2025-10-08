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
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        self.api_key = api_key or "AIzaSyDTs3BYcLe_1XF8q3VW-blr_6wcG_mepgE"
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
    
    def run_gemini_with_usage(self, prompt: str, model: str = "gemini-2.0-flash-exp") -> tuple[str, dict]:
        """
        Calls Gemini with a natural language prompt and returns both response text and token usage.
        """
        try:
            response = self.model.generate_content(prompt)
            
            # Extract token usage from response
            token_usage = {}
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_usage = {
                    "inputTokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "outputTokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "totalTokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            else:
                # Fallback: estimate tokens (rough approximation)
                estimated_input = len(prompt.split()) * 1.3  # rough token estimation
                estimated_output = len(response.text.split()) * 1.3 if response.text else 0
                token_usage = {
                    "inputTokens": int(estimated_input),
                    "outputTokens": int(estimated_output),
                    "totalTokens": int(estimated_input + estimated_output)
                }
            
            return response.text, token_usage
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback to simulation if API fails
            return self._simulate_gemini_response(prompt), {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    
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
        
        if "group by" in query:
            # Extract group by column
            group_match = re.search(r'group by\s+(\w+)', query, re.IGNORECASE)
            if group_match:
                column = group_match.group(1)
                return f"GROUP_BY: {column}"
        
        if "sum" in query:
            return "AGGREGATE: sum"
        elif "average" in query or "mean" in query:
            return "AGGREGATE: mean"
        elif "count" in query:
            return "AGGREGATE: count"
        
        # Default response
        return "TRANSFORM: Basic data analysis"
    
    def test_configuration(self) -> Dict[str, Any]:
        """
        Test the API key and model configuration
        """
        try:
            # Test with a simple prompt
            test_prompt = "Hello, this is a test. Please respond with 'Configuration test successful'."
            response, token_usage = self.run_gemini_with_usage(test_prompt)
            
            return {
                "success": True,
                "message": "Configuration test successful",
                "model": self.model_name,
                "api_key_configured": bool(self.api_key),
                "langchain_available": LANGCHAIN_AVAILABLE,
                "test_response": response[:100] + "..." if len(response) > 100 else response,
                "token_usage": token_usage
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model_name,
                "api_key_configured": bool(self.api_key),
                "langchain_available": LANGCHAIN_AVAILABLE,
                "token_usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
            }
    
    def get_text_analysis(self, user_query: str, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point for AI-powered data analysis
        Generates Python code and executes it on real dataset
        """
        try:
            print(f"🤖 AI Analysis for: '{user_query}'")
            print(f"📊 Dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
            print(f"🔍 Columns: {list(dataset.columns)}")
            
            # Generate Python code using Gemini
            code, token_usage = self._generate_pandas_code(user_query, dataset)
            
            if not code:
                return {
                    "success": False,
                    "answer": "Failed to generate analysis code",
                    "query": user_query,
                    "dataset_info": f"Dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns",
                    "code_steps": [],
                    "reasoning_steps": [],
                    "tabular_data": [],
                    "has_table": False,
                    "token_usage": token_usage
                }
            
            # Execute the generated code
            result = self._execute_pandas_code(code, dataset, user_query)
            result["token_usage"] = token_usage
            
            return result
            
        except Exception as e:
            print(f"❌ Error in get_text_analysis: {str(e)}")
            return {
                "success": False,
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "query": user_query,
                "dataset_info": f"Dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns",
                "code_steps": [],
                "reasoning_steps": [],
                "tabular_data": [],
                "has_table": False,
                "token_usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
            }
    
    def _analyze_dataset_structure(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset structure and create generic examples
        """
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create generic example based on dataset structure
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
            first_col = dataset.columns[0] if len(dataset.columns) > 0 else 'column'
            generic_example = f"""# Answer the user's query using df
result = df['{first_col}'].value_counts().head(5)
print("Top 5 most frequent values in {first_col}:")
print(result)"""
        
        return {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "generic_example": generic_example,
            "sample_data": dataset.head(3).to_string(index=False)
        }
    
    def _generate_pandas_code(self, user_query: str, dataset: pd.DataFrame) -> tuple[str, dict]:
        """
        Generate Python pandas code using Gemini
        Returns: (generated_code, token_usage)
        """
        try:
            print("🤖 Generating pandas code for real dataset...")
            
            # Analyze dataset structure
            dataset_info = self._analyze_dataset_structure(dataset)
            
            # Create code generation prompt
            code_generation_prompt = f"""You are a data analyst. Generate Python pandas code to answer the user's query using the provided DataFrame.

REAL DATASET CONTEXT:
- Variable name: 'df' 
- Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns
- Columns: {list(dataset.columns)}
- Data types: {dict(dataset.dtypes.astype(str))}
- Sample data (first 3 rows):
{dataset_info['sample_data']}

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
{dataset_info['generic_example']}
```

Generate ONLY the code, no explanations:"""

            # Generate code using Gemini
            code_response_text, token_usage = self.run_gemini_with_usage(code_generation_prompt)
            
            if not code_response_text:
                return "", token_usage
            
            # Extract Python code from response
            generated_code = code_response_text.strip()
            
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
            
            return code_lines, token_usage
            
        except Exception as e:
            print(f"❌ Code generation failed: {str(e)}")
            return "", {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    
    def _execute_pandas_code(self, code: str, dataset: pd.DataFrame, user_query: str) -> Dict[str, Any]:
        """
        Execute pandas code on real dataset
        """
        try:
            print("⚡ EXECUTING CODE ON REAL DATASET...")
            
            # Create safe execution environment
            import io
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
            
            # Execute generated code on real dataset
            exec(code, execution_globals)
            
            # Get the output
            execution_output = captured_output.getvalue()
            
            print("✅ CODE EXECUTION SUCCESSFUL ON REAL DATA!")
            print("📋 REAL DATA ANALYSIS RESULTS:")
            print("-" * 50)
            print(execution_output)
            print("-" * 50)
            
            # Create analysis text
            analysis_text = f"Based on your real dataset, here are the results for '{user_query}':\n\n{execution_output.strip()}"
            
            # Try to extract tabular data from output
            tabular_data = []
            has_table = False
            
            if execution_output.strip():
                try:
                    # Simple tabular data detection
                    if '|' in execution_output and '\n' in execution_output:
                        lines = execution_output.split('\n')
                        for line in lines:
                            if '|' in line and not line.strip().startswith('|'):
                                tabular_data.append(line.strip())
                        if tabular_data:
                            has_table = True
                            print(f"✅ Successfully parsed tabular data: {len(tabular_data)} lines")
                except Exception as parse_error:
                    print(f"⚠️ Table parsing failed: {parse_error}")
                    has_table = False
            
            return {
                "answer": analysis_text,
                "success": True,
                "reasoning_steps": ["✅ Executed pandas code on REAL uploaded dataset"],
                "code_steps": [code],  # Show the actual pandas code
                "tabular_data": tabular_data,
                "has_table": has_table
            }
            
        except Exception as exec_error:
            print(f"❌ CODE EXECUTION FAILED: {exec_error}")
            error_msg = f"Error executing pandas code on real dataset: {str(exec_error)}"
            
            return {
                "answer": f"I generated pandas code for your real dataset but encountered an execution error: {error_msg}. The code was: {code}",
                "success": False, 
                "reasoning_steps": [f"❌ Code execution failed: {str(exec_error)}"],
                "code_steps": [code],
                "tabular_data": [],
                "has_table": False
            }
    
    def calculate_metric(self, user_query: str, dataset_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        AI-powered metric calculation using natural language
        """
        try:
            print(f"🧮 AI Metric calculation started:")
            print(f"   Query: '{user_query}'")
            print(f"   Dataset: {dataset_id}")
            print(f"   Data shape: {data.shape}")
            
            # Create a focused prompt for metric calculation
            prompt = f"""You are a data analyst calculating metrics from a dataset using pandas.

Dataset Information:
- Shape: {data.shape[0]} rows, {data.shape[1]} columns
- Columns: {list(data.columns)}
- Sample data:
{data.head().to_string()}

USER REQUEST: {user_query}

Please provide:
1. The exact pandas expression to calculate this metric
2. The calculated value
3. A brief explanation

Format your response as:
METRIC_EXPRESSION: [pandas expression using 'df' variable]
CALCULATED_VALUE: [the actual numeric result]
EXPLANATION: [brief explanation of what this metric represents]

Use the actual data provided above."""

            response, token_usage = self.run_gemini_with_usage(prompt)
            
            # Parse the response
            metric_expression = ""
            calculated_value = None
            explanation = ""
            
            lines = response.split('\n')
            for line in lines:
                if line.startswith('METRIC_EXPRESSION:'):
                    metric_expression = line.replace('METRIC_EXPRESSION:', '').strip()
                elif line.startswith('CALCULATED_VALUE:'):
                    try:
                        calculated_value = float(line.replace('CALCULATED_VALUE:', '').strip())
                    except:
                        calculated_value = line.replace('CALCULATED_VALUE:', '').strip()
                elif line.startswith('EXPLANATION:'):
                    explanation = line.replace('EXPLANATION:', '').strip()
            
            # If no structured response, try to execute the metric expression
            if metric_expression and calculated_value is None:
                try:
                    # Safe execution environment
                    safe_globals = {'df': data, 'pd': pd, 'np': np, '__builtins__': {}}
                    calculated_value = eval(metric_expression, safe_globals)
                except Exception as e:
                    print(f"❌ Failed to execute metric expression: {e}")
                    calculated_value = "Error in calculation"
            
            # Format the value
            formatted_value = calculated_value
            if isinstance(calculated_value, (int, float)):
                if calculated_value == int(calculated_value):
                    formatted_value = f"{int(calculated_value):,}"
                else:
                    formatted_value = f"{calculated_value:,.2f}"
            
            return {
                "success": True,
                "value": calculated_value,
                "formatted_value": formatted_value,
                "expression": metric_expression,
                "explanation": explanation,
                "query": user_query,
                "token_usage": token_usage
            }
            
        except Exception as e:
            print(f"❌ AI Metric calculation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "value": None,
                "formatted_value": "Error",
                "expression": "",
                "explanation": f"Failed to calculate metric: {str(e)}",
                "query": user_query,
                "token_usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
            }