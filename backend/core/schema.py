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

class DfuseSchema:
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
