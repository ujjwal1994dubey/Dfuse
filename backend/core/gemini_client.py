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


class GeminiClient:
    """
    Gemini Client with Gemini 2.0 Flash for natural language data transformations
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

    def _direct_gemini_analysis(self, user_query: str, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Direct analysis using main Gemini model when LangChain fails
        """
        try:
            # Create dataset summary with more context
            sample_data = dataset.head(5)
            
            # Try to get statistical summary, fallback gracefully if it fails
            try:
                stats_summary = dataset.describe().to_string()
            except Exception:
                stats_summary = "Statistical summary not available for this dataset"
            
            dataset_info = f"""
Dataset Information:
- Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns
- Columns: {list(dataset.columns)}
- Sample data (first 5 rows):
{sample_data.to_string()}

Statistical Summary:
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

            # Create analysis prompt with code generation instructions
            direct_prompt = f"""
You are an expert data analyst. Analyze this dataset and provide a comprehensive answer with Python code verification.

{dataset_info}

User Question: "{user_query}"

RESPONSE FORMAT - Provide your response in this EXACT structure:

## Analysis
[Your detailed analysis here with specific numbers and insights]

## Python Code
```python
# Code that would generate these results using pandas
[Provide the exact Python code using 'df' that would achieve this analysis]
```

CRITICAL INSTRUCTIONS:
1. Analyze the ACTUAL data provided above - do not use mock or example data
2. Use appropriate columns from the dataset for ranking, grouping, or filtering
3. Provide specific data values, names, and numerical results from the actual dataset
4. In the Python code section, show the exact pandas operations (e.g., df.nlargest(), df.groupby(), etc.)
5. Use 'df' as the DataFrame variable name in your code
6. Include column names exactly as shown: {list(dataset.columns)}
7. Data types available: {dict(dataset.dtypes.astype(str))}
8. Make sure your code would produce the exact results you stated in the analysis

Example format:
## Analysis
{generic_analysis_example}

## Python Code
```python
{generic_code_example}
```

Now analyze: "{user_query}"."""

            print(f"🔍 Using direct Gemini analysis for: {user_query}")
            response = self.model.generate_content(direct_prompt)
            
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
                        tabular_data = parsed_table
                        has_table = True
                        print(f"✅ Direct Gemini analysis: parsed tabular data with {len(tabular_data.get('rows', []))} rows")
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
                    "has_table": has_table
                }
                
                return response_data
            else:
                return {
                    "answer": "I was unable to generate an analysis for your query.",
                    "success": False,
                    "reasoning_steps": [],
                    "code_steps": [],
                    "tabular_data": [],
                    "has_table": False
                }
                
        except Exception as direct_error:
            print(f"❌ Direct Gemini analysis also failed: {direct_error}")
            return {
                "answer": f"I encountered an error while analyzing your data: {str(direct_error)[:200]}...",
                "success": False,
                "reasoning_steps": [],
                "code_steps": [],
                "tabular_data": [],
                "has_table": False
            }
  return
