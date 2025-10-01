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

class DfusePandasAgent:
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

User request: {user_query}

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


    def _use_pandas_agent_enhanced(self, user_query: str, full_dataset: pd.DataFrame, current_dimensions: List[str], current_measures: List[str], available_dims: List[str], available_measures: List[str]) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: This method contained dataset-specific logic and examples.
        Now using generic _execute_real_pandas_analysis for all datasets.
        """
        print(f"🤖 Skipping legacy dataset-specific enhanced pandas agent - using generic pandas execution instead")
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

User Question: {user_query}

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

USER QUERY: {user_query}

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
                                tabular_data = parsed_table
                                has_table = True
                                print(f"✅ Successfully parsed tabular data: {len(tabular_data.get('rows', []))} rows")
                        except Exception as parse_error:
                            print(f"⚠️ Table parsing failed: {parse_error}")
                            has_table = False
                    
                    return {
                        "answer": analysis_text,
                        "success": True,
                        "reasoning_steps": ["✅ Executed pandas code on REAL uploaded dataset"],
                        "code_steps": [code_lines],  # Show the actual pandas code
                        "tabular_data": tabular_data,
                        "has_table": has_table
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
                        "has_table": False
                    }
                    
            else:
                print("❌ Failed to generate pandas code")
                raise Exception("No pandas code generated by Gemini")
                
        except Exception as e:
            print(f"❌ Real pandas analysis failed: {e}")
            print("🔄 Falling back to direct Gemini analysis...")
            return self._direct_gemini_analysis(user_query, dataset)


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
