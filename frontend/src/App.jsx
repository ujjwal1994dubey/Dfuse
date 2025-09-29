import React, { useCallback, useMemo, useState, useEffect, useRef } from 'react';
import ReactFlow, { Background, Controls, MiniMap, applyNodeChanges, applyEdgeChanges, ReactFlowProvider, useStore } from 'react-flow-renderer';
import Plot from 'react-plotly.js';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import { Button, Badge, Card, CardHeader, CardContent, FileUpload, RadioGroup, DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuLabel, Input, Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui';
import { MousePointer2, MoveUpRight, Type, SquareSigma, Merge, X, ChartColumn, Funnel, SquaresExclude, Menu, BarChart, Table, Send, File, Wand, PieChart, Circle, TrendingUp, BarChart2, Settings, Check, AlertCircle, Eye, EyeOff, Edit } from 'lucide-react';
import './tiptap-styles.css';

const API = 'http://localhost:8000';

// Chart Types Registry - Defines all supported chart types and their capabilities
const CHART_TYPES = {
  BAR: {
    id: 'bar',
    label: 'Bar Chart',
    icon: BarChart,
    isSupported: (dims, measures) => dims === 1 && measures === 1,
    createFigure: (data, payload) => {
      const xKey = payload.dimensions[0];
      const yKey = payload.measures[0];
      return {
        data: [{
          type: 'bar',
          x: data.map(r => r[xKey]),
          y: data.map(r => r[yKey] || 0),
          marker: { color: '#3182ce' }
        }],
        layout: {
          xaxis: {
            title: { text: xKey, font: { size: 14, color: '#4a5568' } },
            tickangle: -45
          },
          yaxis: {
            title: { text: yKey || 'Value', font: { size: 14, color: '#4a5568' } }
          },
          margin: { t: 20, b: 80, l: 80, r: 30 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: false,
          legend: undefined
        }
      };
    }
  },
  PIE: {
    id: 'pie',
    label: 'Pie Chart',
    icon: PieChart,
    isSupported: (dims, measures) => dims === 1 && measures === 1,
    createFigure: (data, payload) => {
      const labelKey = payload.dimensions[0];
      const valueKey = payload.measures[0];
      return {
        data: [{
          type: 'pie',
          labels: data.map(r => r[labelKey]),
          values: data.map(r => r[valueKey] || 0),
          hole: 0.3,
          marker: {
            colors: ['#3182ce', '#38a169', '#d69e2e', '#e53e3e', '#805ad5', '#dd6b20', '#38b2ac', '#ed64a6']
          }
        }],
        layout: {
          margin: { t: 20, b: 20, l: 20, r: 20 },
          paper_bgcolor: 'white',
          showlegend: true,
          legend: { orientation: 'v', x: 1.05, y: 0.5 }
        }
      };
    }
  },
  SCATTER: {
    id: 'scatter',
    label: 'Scatter Plot',
    icon: Circle,
    isSupported: (dims, measures) => dims === 1 && measures === 2,
    createFigure: (data, payload) => {
      const labelKey = payload.dimensions[0];
      const xKey = payload.measures[0];
      const yKey = payload.measures[1];
      return {
        data: [{
          type: 'scatter',
          mode: 'markers',
          x: data.map(r => r[xKey] || 0),
          y: data.map(r => r[yKey] || 0),
          text: data.map(r => r[labelKey]),
          marker: { 
            size: 10, 
            color: '#3182ce',
            opacity: 0.7,
            line: { color: 'white', width: 1 }
          },
          hovertemplate: '<b>%{text}</b><br>' +
                         `${xKey}: %{x}<br>` +
                         `${yKey}: %{y}<br>` +
                         '<extra></extra>'
        }],
        layout: {
          xaxis: {
            title: { text: xKey, font: { size: 14, color: '#4a5568' } }
          },
          yaxis: {
            title: { text: yKey, font: { size: 14, color: '#4a5568' } }
          },
          margin: { t: 20, b: 60, l: 80, r: 30 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white'
        }
      };
    }
  },
  // 3-Variable Chart Types (2 Measures + 1 Dimension)
  GROUPED_BAR: {
    id: 'grouped_bar',
    label: 'Grouped Bar',
    icon: BarChart,
    isSupported: (dims, measures) => dims === 1 && measures === 2,
    createFigure: (data, payload) => {
      const xKey = payload.dimensions[0];
      const measureKeys = payload.measures;
      const xValues = [...new Set(data.map(r => r[xKey]))];
      
      return {
        data: measureKeys.map((measure, i) => ({
          type: 'bar',
          name: measure,
          x: xValues,
          y: xValues.map(v => (data.find(r => r[xKey] === v)?.[measure]) ?? 0),
          marker: { color: ['#3182ce', '#38a169', '#d69e2e'][i] }
        })),
        layout: {
          barmode: 'group',
          xaxis: {
            title: { text: xKey, font: { size: 14, color: '#4a5568' } },
            tickangle: -45
          },
          yaxis: {
            title: { text: 'Value', font: { size: 14, color: '#4a5568' } }
          },
          margin: { t: 20, b: 80, l: 80, r: 30 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: measureKeys.length > 1,
          legend: measureKeys.length > 1 ? {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.15,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E8F0',
            borderwidth: 1
          } : undefined
        }
      };
    }
  },
  DUAL_AXIS: {
    id: 'dual_axis',
    label: 'Dual Axis',
    icon: TrendingUp,
    isSupported: (dims, measures) => dims === 1 && measures === 2,
    createFigure: (data, payload) => {
      const xKey = payload.dimensions[0];
      const [m1, m2] = payload.measures;
      const xValues = [...new Set(data.map(r => r[xKey]))];
      const m1Values = xValues.map(v => (data.find(r => r[xKey] === v)?.[m1]) ?? 0);
      const m2Values = xValues.map(v => (data.find(r => r[xKey] === v)?.[m2]) ?? 0);
      
      return {
        data: [
          {
            type: 'scatter',
            mode: 'lines+markers',
            name: m1,
            x: xValues,
            y: m1Values,
            yaxis: 'y',
            line: { color: '#3182ce', width: 3 },
            marker: { color: '#3182ce', size: 8 }
          },
          {
            type: 'scatter',
            mode: 'lines+markers',
            name: m2,
            x: xValues,
            y: m2Values,
            yaxis: 'y2',
            line: { color: '#38a169', width: 3 },
            marker: { color: '#38a169', size: 8 }
          }
        ],
        layout: {
          xaxis: {
            title: { text: xKey, font: { size: 14, color: '#4a5568' } },
            tickangle: -45
          },
          yaxis: {
            title: { text: m1, font: { size: 14, color: '#3182ce' } },
            side: 'left'
          },
          yaxis2: {
            title: { text: m2, font: { size: 14, color: '#38a169' } },
            side: 'right',
            overlaying: 'y'
          },
          margin: { t: 20, b: 80, l: 80, r: 80 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: true,
          legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.15,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E8F0',
            borderwidth: 1
          }
        }
      };
    }
  },
  // 3-Variable Chart Types (2 Dimensions + 1 Measure) - HEATMAP REMOVED
  STACKED_BAR: {
    id: 'stacked_bar',
    label: 'Stacked Bar',
    icon: BarChart2,
    isSupported: (dims, measures) => dims === 2 && measures === 1,
    createFigure: (data, payload) => {
      const [dim1, dim2] = payload.dimensions;
      const measure = payload.measures[0];
      
      // Safety check for empty data
      if (!data || data.length === 0) {
        console.warn('STACKED_BAR: No data provided');
        return { 
          data: [], 
          layout: sanitizeLayout({
            xaxis: { title: { text: dim1 } },
            yaxis: { title: { text: measure } },
            showlegend: false,
            legend: undefined
          }) 
        };
      }
      
      // Simple row-based data handling only
      const groups = {};
      data.forEach(row => {
        const category = row[dim1];    // First dimension (e.g., Category)
        const product = row[dim2];     // Second dimension (e.g., Product)
        const value = row[measure] || 0; // Measure value
        
        if (category && product) { // Ensure valid values
          if (!groups[product]) groups[product] = {};
          groups[product][category] = value;
        }
      });
      
      const uniqueProducts = [...new Set(data.map(r => r[dim2]))];
      const uniqueCategories = [...new Set(data.map(r => r[dim1]))];
      
      const chartData = uniqueProducts.map((product, i) => ({
        type: 'bar',
        name: product,
        x: uniqueCategories,
        y: uniqueCategories.map(cat => groups[product]?.[cat] || 0),
        marker: { color: ['#3182ce', '#38a169', '#d69e2e', '#e53e3e', '#805ad5', '#dd6b20', '#38b2ac', '#ed64a6'][i % 8] }
      }));
      
      // Safety check for empty chart data
      if (chartData.length === 0) {
        console.warn('STACKED_BAR: No chart data generated');
        return { 
          data: [{ type: 'bar', x: [], y: [] }], 
          layout: sanitizeLayout({ 
            title: { text: 'No data available' },
            xaxis: { title: { text: dim1 } },
            yaxis: { title: { text: measure } },
            showlegend: false,
            legend: undefined
          })
        };
      }
      
      return {
        data: chartData,
        layout: sanitizeLayout({
          barmode: 'stack',
          xaxis: {
            title: { text: dim1, font: { size: 14, color: '#4a5568' } },
            tickangle: -45
          },
          yaxis: {
            title: { text: measure, font: { size: 14, color: '#4a5568' } }
          },
          margin: { t: 20, b: 80, l: 80, r: 30 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: chartData.length > 1,
          legend: chartData.length > 1 ? {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.15,
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E8F0',
            borderwidth: 1
          } : undefined
        })
      };
    }
  },
  BUBBLE: {
    id: 'bubble',
    label: 'Bubble Chart',
    icon: Circle,
    isSupported: (dims, measures) => dims === 2 && measures === 1,
    createFigure: (data, payload) => {
      const [dim1, dim2] = payload.dimensions;
      const measure = payload.measures[0];
      
      // Simple row-based data handling only
      const validData = data.filter(r => r[measure] && r[measure] > 0);
      const maxValue = Math.max(...validData.map(r => r[measure]));
      
      return {
        data: [{
          type: 'scatter',
          mode: 'markers',
          x: validData.map(r => r[dim2]), // Product names directly
          y: validData.map(r => r[dim1]), // Category names directly
          text: validData.map(r => `${dim1}: ${r[dim1]}<br>${dim2}: ${r[dim2]}<br>${measure}: ${r[measure]}`),
          marker: {
            size: validData.map(r => Math.max(8, Math.sqrt(r[measure] / maxValue * 2000) + 5)),
            color: validData.map(r => r[measure]),
            colorscale: [
              [0, '#e6f3ff'], [0.3, '#66c2ff'], [0.6, '#1a8cff'], [1, '#003d80']
            ],
            colorbar: { 
              title: { text: measure, side: 'right' },
              thickness: 15
            },
            opacity: 0.8,
            line: { color: 'white', width: 2 }
          },
          hovertemplate: '%{text}<extra></extra>'
        }],
        layout: sanitizeLayout({
          xaxis: {
            title: { text: dim2, font: { size: 14, color: '#4a5568' } },
            type: 'category',
            tickangle: -45
          },
          yaxis: {
            title: { text: dim1, font: { size: 14, color: '#4a5568' } },
            type: 'category'
          },
          margin: { t: 20, b: 80, l: 100, r: 120 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: false,
          legend: undefined
        })
      };
    }
  },
  LINE: {
    id: 'line',
    label: 'Line Chart',
    icon: TrendingUp,
    isSupported: (dims, measures) => dims === 1 && measures === 1,
    createFigure: (data, payload) => {
      const xKey = payload.dimensions[0];
      const yKey = payload.measures[0];
      return {
        data: [{
          type: 'scatter',
          mode: 'lines+markers',
          x: data.map(r => r[xKey]),
          y: data.map(r => r[yKey] || 0),
          line: { color: '#3182ce', width: 3 },
          marker: { color: '#3182ce', size: 6 }
        }],
        layout: {
          xaxis: {
            title: { text: xKey, font: { size: 14, color: '#4a5568' } },
            tickangle: -45
          },
          yaxis: {
            title: { text: yKey || 'Value', font: { size: 14, color: '#4a5568' } }
          },
          margin: { t: 20, b: 80, l: 80, r: 30 },
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: false,
          legend: undefined
        }
      };
    }
  }
};

// Helper function to get supported chart types for given dimensions and measures
const getSupportedChartTypes = (dims, measures) => {
  return Object.values(CHART_TYPES).filter(chartType => 
    chartType.isSupported(dims, measures)
  );
};

// Helper function to get default chart type for dimensions and measures
const getDefaultChartType = (dims, measures) => {
  const supported = getSupportedChartTypes(dims, measures);
  return supported.length > 0 ? supported[0] : CHART_TYPES.BAR;
};

// Universal layout sanitizer to ensure all layouts have proper legend configuration
const sanitizeLayout = (layout) => {
  return {
    ...layout,
    // Ensure legend is always properly defined
    showlegend: layout.showlegend !== undefined ? layout.showlegend : false,
    legend: layout.showlegend && layout.legend ? {
      ...layout.legend,
      bgcolor: layout.legend.bgcolor || 'rgba(255,255,255,0.8)',
      bordercolor: layout.legend.bordercolor || '#E2E8F0',
      borderwidth: layout.legend.borderwidth || 1
    } : undefined
  };
};

// For now, let's use a simple approach without custom extensions
// We'll implement autocomplete manually using a simple input approach

// Arrow Node Component - SVG in local bbox space
function ArrowNode({ data }) {
  const { id, start, end } = data;

  // Compute local coordinates inside a bbox anchored at (minX, minY)
  const minX = Math.min(start.x, end.x);
  const minY = Math.min(start.y, end.y);
  const sx = start.x - minX;
  const sy = start.y - minY;
  const ex = end.x - minX;
  const ey = end.y - minY;
  const width = Math.max(sx, ex) + 20;
  const height = Math.max(sy, ey) + 20;

  return (
    <svg width={width} height={height} style={{ pointerEvents: 'none', overflow: 'visible' }}>
      <defs>
        <marker id={`arrow-head-${id}`} markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
          <path d="M0,0 L12,6 L0,12 Z" fill="#2563eb" />
        </marker>
      </defs>
      <line x1={sx} y1={sy} x2={ex} y2={ey} stroke="#2563eb" strokeWidth="3" markerEnd={`url(#arrow-head-${id})`} />
    </svg>
  );
}

// Table Node Component
function TableNode({ data }) {
  const { title, headers, rows, totalRows } = data;
  
  return (
    <Card className="max-w-2xl">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="font-semibold text-gray-800">{title}</div>
          <Badge variant="secondary">
            {totalRows} rows
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        <div className="border rounded-lg" style={{ height: '384px', overflowY: 'scroll' }}>
          <table className="w-full text-sm">
            <thead className="bg-gray-50 sticky top-0">
              <tr>
                {headers?.map((header, i) => (
                  <th key={i} className="px-3 py-2 text-left font-medium text-gray-700 border-b">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows?.map((row, i) => (
                <tr key={i} className={`${i % 2 === 0 ? 'bg-white' : 'bg-gray-25'} hover:bg-blue-50`}>
                  {row.map((cell, j) => (
                    <td key={j} className="px-3 py-2 border-b text-gray-600">
                      {typeof cell === 'number' ? cell.toLocaleString() : cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {(!rows || rows.length === 0) && (
          <div className="text-center py-8 text-gray-500">
            No data available
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Text Box Node Component  
function TextBoxNode({ data, id }) {
  const [isEditing, setIsEditing] = useState(data.isNew || false);
  const [text, setText] = useState(data.text || 'Double-click to edit');
  const [tempText, setTempText] = useState(text);
  
  const handleDoubleClick = () => {
    setIsEditing(true);
    setTempText(text);
  };
  
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      setText(tempText);
      setIsEditing(false);
      // Update node data
      data.onTextChange?.(id, tempText);
    } else if (e.key === 'Escape') {
      setIsEditing(false);
      setTempText(text);
    }
  };
  
  const handleBlur = () => {
    setText(tempText);
    setIsEditing(false);
    data.onTextChange?.(id, tempText);
  };
  
  return (
    <div 
      className="bg-yellow-100 border-2 border-yellow-300 rounded-lg p-3 min-w-[200px] min-h-[60px] cursor-text shadow-sm"
      onDoubleClick={handleDoubleClick}
    >
      {isEditing ? (
        <textarea
          value={tempText}
          onChange={(e) => setTempText(e.target.value)}
          onKeyDown={handleKeyPress}
          onBlur={handleBlur}
          className="w-full h-full bg-transparent border-none outline-none resize-none font-medium text-gray-800"
          autoFocus
          style={{ minHeight: '40px' }}
        />
      ) : (
        <div className="whitespace-pre-wrap font-medium text-gray-800">
          {text}
        </div>
      )}
    </div>
  );
}

// TipTap-based Expression Node Component
function ExpressionNode({ data, id, apiKey, selectedModel, setShowSettings, updateTokenUsage }) {
  const [expression, setExpression] = useState(data.expression || '');
  const [result, setResult] = useState(data.result || null);
  const [isEditing, setIsEditing] = useState(data.isNew || false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState(data.filters || {});
  const [availableMeasures, setAvailableMeasures] = useState([]);
  const [availableDimensions, setAvailableDimensions] = useState([]);
  const [validationErrors, setValidationErrors] = useState([]);
  
  // Autocomplete state
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionType, setSuggestionType] = useState(''); // 'measures' or 'aggregations'
  const [suggestionPosition, setSuggestionPosition] = useState({ top: 0, left: 0 });
  // const [currentQuery, setCurrentQuery] = useState(''); // Removed unused variable
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const editorContainerRef = useRef(null);
  const editorRef = useRef(null);
  
  // AI Metric Calculation state
  const [aiMode, setAiMode] = useState(false);
  const [aiQuery, setAiQuery] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  
  // Menu dropdown state
  const [showDropdownMenu, setShowDropdownMenu] = useState(false);
  
  // Available aggregation methods
  const aggregationMethods = useMemo(() => [
    { name: 'SUM', description: 'Sum of all values' },
    { name: 'AVG', description: 'Average of all values' },
    { name: 'MIN', description: 'Minimum value' },
    { name: 'MAX', description: 'Maximum value' },
    { name: 'COUNT', description: 'Count of records' },
    { name: 'COUNT_DISTINCT', description: 'Count of unique values' },
    { name: 'MEDIAN', description: 'Median value' },
    { name: 'STDDEV', description: 'Standard deviation' }
  ], []);
  
  // Autocomplete helper functions
  const getCursorPosition = useCallback((editor) => {
    if (!editor || !editor.view || !editorContainerRef.current) return null;
    
    try {
      const { state } = editor;
      const { selection } = state;
      const { from } = selection;
      
      // Get the DOM position of the cursor
      const coords = editor.view.coordsAtPos(from);
      const containerRect = editorContainerRef.current.getBoundingClientRect();
      
      // Position dropdown below the cursor, accounting for container padding
      return {
        top: coords.top - containerRect.top + 25, // 25px below cursor
        left: coords.left - containerRect.left
      };
    } catch (error) {
      console.warn('Could not get cursor position:', error);
      // Fallback position if cursor position can't be determined
      return {
        top: 60, // Below the editor area
        left: 20
      };
    }
  }, []);
  
  const checkForAutocompleteTriggers = useCallback((editor) => {
    if (!editor || !editor.view || !isEditing) return;
    
    const { state } = editor;
    const { selection } = state;
    const { from } = selection;
    
    // Get text before cursor (last 50 characters to handle long measure names)
    const textBefore = state.doc.textBetween(Math.max(0, from - 50), from);
    
    // Check for @ trigger (measures)
    const atMatch = textBefore.match(/@([a-zA-Z0-9_]*)$/);

    if (atMatch) {
      const query = atMatch[1].toLowerCase();
      const filteredMeasures = availableMeasures.filter(measure => 
        measure.toLowerCase().includes(query)
      ).map(measure => ({ name: measure, type: 'measure' }));
      
      const position = getCursorPosition(editor);
      if (position) {
        setSuggestions(filteredMeasures);
        setSuggestionType('measures');
        setSuggestionPosition(position);
        // setCurrentQuery(query); // Not needed for functionality
        setSelectedSuggestionIndex(0);
        setShowSuggestions(true);
      }
      return;
    }
    
    // Check for . trigger (aggregations)
    const dotMatch = textBefore.match(/@[a-zA-Z0-9_]+\.([a-zA-Z]*)$/);

    if (dotMatch) {
      const query = dotMatch[1].toLowerCase();
      const filteredAggregations = aggregationMethods.filter(agg => 
        agg.name.toLowerCase().includes(query)
      ).map(agg => ({ name: agg.name, description: agg.description, type: 'aggregation' }));
      
      const position = getCursorPosition(editor);
      if (position) {
        setSuggestions(filteredAggregations);
        setSuggestionType('aggregations');
        setSuggestionPosition(position);
        // setCurrentQuery(query); // Not needed for functionality
        setSelectedSuggestionIndex(0);
        setShowSuggestions(true);
      }
      return;
    }
    
    // Hide suggestions if no triggers
    setShowSuggestions(false);
  }, [availableMeasures, aggregationMethods, getCursorPosition, isEditing]);
  
  const insertSuggestion = useCallback((suggestion) => {
    if (!editorRef.current || !editorRef.current.view || !editorRef.current.commands) return;
    

    try {
      const { state } = editorRef.current;
      const { selection } = state;
      const { from } = selection;
      
      // Get text before cursor to find what to replace
      const textBefore = state.doc.textBetween(Math.max(0, from - 50), from);
      
      let replaceFrom = from;
      let replaceText = suggestion.name;
      
      if (suggestionType === 'measures') {
        // Replace @query with @MeasureName
        const atMatch = textBefore.match(/@([a-zA-Z0-9_]*)$/);
        if (atMatch) {
          replaceFrom = from - atMatch[0].length;
          replaceText = `@${suggestion.name}`;
        } else {
          // Fallback: just replace the @ character if no query found
          const simpleAtMatch = textBefore.match(/@$/);
          if (simpleAtMatch) {
            replaceFrom = from - 1;
            replaceText = `@${suggestion.name}`;
          }
        }
      } else if (suggestionType === 'aggregations') {
        // Replace .query with .AGGREGATION
        const dotMatch = textBefore.match(/\.([a-zA-Z]*)$/);
        if (dotMatch) {
          replaceFrom = from - dotMatch[0].length;
          replaceText = `.${suggestion.name}`;
        } else {
          // Fallback: just replace the . character if no query found
          const simpleDotMatch = textBefore.match(/\.$/);
          if (simpleDotMatch) {
            replaceFrom = from - 1;
            replaceText = `.${suggestion.name}`;
          }
        }
      }
      
      // Replace the text
      editorRef.current.commands.deleteRange({ from: replaceFrom, to: from });
      editorRef.current.commands.insertContent(replaceText);
      
      // Hide suggestions
      setShowSuggestions(false);
    } catch (error) {
      console.warn('Could not insert suggestion:', error);
    }
  }, [suggestionType]);
  
  const handleKeyDown = useCallback((event) => {
    if (!showSuggestions || suggestions.length === 0) return;
    
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setSelectedSuggestionIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        event.preventDefault();
        setSelectedSuggestionIndex(prev => 
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
        break;
      case 'Enter':
      case 'Tab':
        event.preventDefault();

        if (suggestions[selectedSuggestionIndex]) {
          insertSuggestion(suggestions[selectedSuggestionIndex]);
        }
        break;
      case 'Escape':
        event.preventDefault();
        setShowSuggestions(false);
        break;
      default:
        // No action needed for other keys
        break;
    }
  }, [showSuggestions, suggestions, selectedSuggestionIndex, insertSuggestion]);

  // Add keyboard event listener for autocomplete
  useEffect(() => {
    const handleGlobalKeyDown = (event) => {
      if (isEditing && showSuggestions) {
        handleKeyDown(event);
      }
    };
    
    document.addEventListener('keydown', handleGlobalKeyDown);
    return () => document.removeEventListener('keydown', handleGlobalKeyDown);
  }, [isEditing, showSuggestions, handleKeyDown]);

  // Hide suggestions when not editing
  useEffect(() => {
    if (!isEditing) {
      setShowSuggestions(false);
    }
  }, [isEditing]);
  
  // Handle expression change from TipTap editor
  const handleExpressionChange = useCallback((newExpression) => {
    setExpression(newExpression);
    if (data.onExpressionChange) {
      data.onExpressionChange(newExpression);
    }
  }, [data]);

  // Simple TipTap editor without custom extensions
  const editor = useEditor({
    extensions: [StarterKit],
    content: expression ? `<p>${expression}</p>` : '<p></p>',
    editable: isEditing,
    onUpdate: ({ editor }) => {
      const content = editor.getText();
      handleExpressionChange(content);
      
      // Check for autocomplete triggers when editing
      if (isEditing) {
        checkForAutocompleteTriggers(editor);
      }
    },
    onCreate: ({ editor }) => {
      console.log('TipTap editor created:', editor);
      editorRef.current = editor;
      // Ensure editor is properly initialized
      if (isEditing) {
        setTimeout(() => {
          if (editor && editor.view && editor.commands && !editor.isDestroyed) {
            try {
              editor.setEditable(true);
              editor.commands.focus();
            } catch (error) {
              console.warn('Could not focus editor on create:', error);
            }
          }
        }, 100);
      }
    },
    onFocus: () => {
      console.log('TipTap editor focused');
    },
  }, [isEditing, checkForAutocompleteTriggers]);

  // Update editor content when expression changes externally
  useEffect(() => {
    if (editor && editor.view && editor.commands && !editor.isDestroyed) {
      try {
        const currentText = editor.getText();
        if (expression !== currentText) {
          editor.commands.setContent(expression ? `<p>${expression}</p>` : '<p></p>');
        }
      } catch (error) {
        console.warn('Could not update editor content:', error);
      }
    }
  }, [editor, expression]);

  // Cleanup editor on unmount
  useEffect(() => {
    return () => {
      if (editorRef.current && !editorRef.current.isDestroyed) {
        try {
          editorRef.current.destroy();
        } catch (error) {
          console.warn('Could not destroy editor:', error);
        }
      }
    };
  }, []);

  // Fetch available measures and dimensions
  useEffect(() => {
    if (data.datasetId) {
      console.log('Fetching measures for dataset:', data.datasetId);
      fetch(`${API}/dataset/${data.datasetId}/measures`)
        .then(res => res.json())
        .then(data => {
          console.log('Received measures data:', data);
          const measures = [...new Set(data.measures || [])];
          const dimensions = [...new Set(data.dimensions || [])];
          
          console.log('Unique measures:', measures);
          console.log('Unique dimensions:', dimensions);
          
          setAvailableMeasures(measures);
          setAvailableDimensions(dimensions);
        })
        .catch(err => console.error('Failed to fetch measures:', err));
    }
  }, [data.datasetId]);

  // Validate expression on change
  useEffect(() => {
    if (expression && data.datasetId) {
      fetch(`${API}/expression/validate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: data.datasetId,
          expression: expression
        })
      })
      .then(res => res.json())
      .then(validation => {
        setValidationErrors(validation.errors || []);
      })
      .catch(err => console.error('Validation failed:', err));
    }
  }, [expression, data.datasetId]);

  // Evaluate expression
  const evaluateExpression = useCallback(async () => {
    if (!expression || !data.datasetId) return;
    
    try {
      const response = await fetch(`${API}/expression/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: data.datasetId,
          expression: expression,
          filters: filters
        })
      });
      
      if (response.ok) {
        const evalResult = await response.json();
        setResult(evalResult.result);
        data.onExpressionChange?.(id, expression, evalResult.result, filters);
      } else {
        const error = await response.json();
        console.error('Evaluation failed:', error.detail);
        setResult(null);
      }
    } catch (err) {
      console.error('Failed to evaluate expression:', err);
      setResult(null);
    }
  }, [expression, data.datasetId, filters, id, data.onExpressionChange]);

  // Auto-evaluate when expression or filters change
  useEffect(() => {
    if (!isEditing) {
      evaluateExpression();
    }
  }, [expression, filters, isEditing, evaluateExpression]);

  const handleSave = useCallback(() => {
    setIsEditing(false);
    setShowSuggestions(false); // Hide autocomplete when saving
    if (editor && editor.view && !editor.isDestroyed) {
      try {
        editor.setEditable(false);
      } catch (error) {
        console.warn('Could not set editor editable state:', error);
      }
    }
  }, [editor]);

  const handleCancel = useCallback(() => {
    setExpression(data.expression || '');
    setIsEditing(false);
    setShowSuggestions(false); // Hide autocomplete when canceling
    if (editor && editor.view && editor.commands && !editor.isDestroyed) {
      try {
        editor.setEditable(false);
        editor.commands.setContent(`<p>${data.expression || ''}</p>`);
      } catch (error) {
        console.warn('Could not cancel editor changes:', error);
      }
    }
  }, [data.expression, editor]);

  const handleEdit = useCallback(() => {
    setIsEditing(true);
    // Use a longer delay to ensure the editor is fully ready
    setTimeout(() => {
      if (editor && editor.view && editor.commands && !editor.isDestroyed) {
        try {
          editor.setEditable(true);
          editor.commands.focus();
          // Force cursor to end of content
          editor.commands.setTextSelection(editor.state.doc.content.size);
        } catch (error) {
          console.warn('Could not focus editor:', error);
        }
      }
    }, 150);
  }, [editor]);

  const toggleFilter = (dimension, value) => {
    setFilters(prev => {
      const current = prev[dimension] || [];
      const updated = current.includes(value) 
        ? current.filter(v => v !== value)
        : [...current, value];
      
      return updated.length > 0 
        ? { ...prev, [dimension]: updated }
        : { ...prev, [dimension]: undefined };
    });
  };

  // AI Metric Calculation Handler
  const handleAIMetricCalculation = useCallback(async () => {
    if (!aiQuery.trim() || !data.datasetId) return;
    
    // Check if API key is configured
    const currentApiKey = apiKey || localStorage.getItem('gemini_api_key');
    const currentModel = selectedModel || localStorage.getItem('gemini_model') || 'gemini-2.0-flash';
    
    if (!currentApiKey.trim()) {
      setAiResult({
        success: false,
        error: '⚠️ Please configure your Gemini API key in Settings first.'
      });
      setShowSettings(true);
      return;
    }
    
    setAiLoading(true);
    try {
      console.log('🧮 Calculating AI metric:', aiQuery);
      console.log('🧮 Dataset ID:', data.datasetId);
      console.log('🧮 Full data object:', data);
      
      if (!data.datasetId) {
        throw new Error('No dataset ID available. Please ensure you have uploaded data first.');
      }
      
      const response = await fetch(`${API}/ai-calculate-metric`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_query: aiQuery,
          dataset_id: data.datasetId,
          api_key: currentApiKey,
          model: currentModel
        })
      });
      
      const result = await response.json();
      
      if (result.success) {
        console.log('✅ AI metric calculated:', result);
        
        // Track token usage
        if (result.token_usage) {
          updateTokenUsage(result.token_usage);
        }
        
        // 🐍 LOG PYTHON CODE TO BROWSER CONSOLE (if available)
        if (result.code_steps && result.code_steps.length > 0) {
          console.log('🐍 PYTHON CODE GENERATED BY AI:');
          console.log('='.repeat(50));
          result.code_steps.forEach((code, index) => {
            console.log(`📝 Code Step ${index + 1}:`);
            console.log(code);
            console.log('-'.repeat(30));
          });
          console.log('💡 This code was executed on your real uploaded dataset');
        }
        
        setAiResult(result);
        setResult(result.value);
        
        // If AI suggests a traditional expression, update the expression field
        if (result.traditional_syntax) {
          setExpression(result.traditional_syntax);
        }
        
        // Show success message
        console.log(`AI calculated: ${result.interpretation}`);
      } else {
        console.error('❌ AI metric calculation failed:', result.error);
        setAiResult(result);
      }
    } catch (error) {
      console.error('Failed to calculate AI metric:', error);
      setAiResult({
        success: false,
        error: `Network error occurred while calculating metric. ${error.message.includes('401') || error.message.includes('403') ? 'Please check your API key in Settings.' : ''}`
      });
    } finally {
      setAiLoading(false);
    }
  }, [aiQuery, data.datasetId, apiKey, selectedModel, updateTokenUsage]);
  
  // Handle AI mode toggle
  const handleAIModeToggle = useCallback(() => {
    setAiMode(!aiMode);
    setAiQuery('');
    setAiResult(null);
    if (!aiMode) {
      // When switching to AI mode, clear any autocomplete
      setShowSuggestions(false);
    }
  }, [aiMode]);

  // Click outside handler to close dropdown menu
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showDropdownMenu && editorContainerRef.current && !editorContainerRef.current.contains(event.target)) {
        setShowDropdownMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showDropdownMenu]);

  return (
    <Card className="min-w-[400px]">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <span className="text-lg font-semibold text-gray-800">Expression</span>
          <div className="flex items-center space-x-3">
            <Button
              onClick={handleAIModeToggle}
              variant={aiMode ? "default" : "ghost"}
              size="icon"
              title={aiMode ? "Switch to Manual Expression" : "Switch to AI Assistant"}
              className={aiMode ? "bg-blue-600 text-white hover:bg-blue-700" : ""}
            >
              <Wand size={16} />
            </Button>
            <div className="relative">
            <Button
              onClick={() => setShowDropdownMenu(!showDropdownMenu)}
              variant="ghost"
              size="icon"
              title="Options"
            >
              <Menu size={16} />
            </Button>
            
            {/* Custom Dropdown Menu */}
            {showDropdownMenu && (
              <div 
                className="absolute right-0 top-8 mt-1 w-48 bg-white rounded-md shadow-lg border border-gray-200 py-1 z-50"
                style={{ zIndex: 9999 }}
              >
                <button
                  onClick={() => {
                    isEditing ? handleSave() : handleEdit();
                    setShowDropdownMenu(false);
                  }}
                  className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 flex items-center"
                >
                  <Type className="mr-2 h-4 w-4" />
                  {isEditing ? "Save Expression" : "Edit Expression"}
                </button>
                <button
                  onClick={() => {
                    setShowFilters(!showFilters);
                    setShowDropdownMenu(false);
                  }}
                  className="w-full px-3 py-2 text-left text-sm hover:bg-gray-100 flex items-center"
                >
                  <Funnel className="mr-2 h-4 w-4" />
                  {showFilters ? "Hide Filters" : "Show Filters"}
                </button>
              </div>
            )}
            </div>
          </div>
        </div>
        
        {/* Calculated Metric Pill - Now below the Expression label with 2x font size */}
        {result !== null && (
          <div className="mt-3">
            <Badge 
              variant="outline" 
              className="font-bold px-4 py-2 text-xl"
              style={{ fontSize: '1.5rem' }}
            >
              {typeof result === 'number' ? result.toLocaleString() : result}
            </Badge>
          </div>
        )}
      </CardHeader>
      <CardContent className="relative" ref={editorContainerRef}>
        <div className="space-y-3">
          {aiMode ? (
            /* AI Natural Language Input */
            <div className="w-full max-w-md">
              <div className="border rounded-lg overflow-hidden bg-blue-50 border-blue-200">
                <div className="p-3" style={{ minHeight: '48px' }}>
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-sm font-medium text-blue-800">AI Calculator</span>
                  </div>
                  <input
                    type="text"
                    placeholder="e.g., 'Calculate total revenue' or 'Profit margin'"
                    value={aiQuery}
                    onChange={(e) => setAiQuery(e.target.value)}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !aiLoading) {
                        handleAIMetricCalculation();
                      }
                    }}
                    className="w-full px-3 py-2 border border-blue-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    disabled={aiLoading}
                  />
                  {aiResult && (
                    <div className="mt-2 text-sm font-medium">
                      {aiResult.success ? (
                        <span className="text-green-700">✅ {aiResult.formatted_value}</span>
                      ) : (
                        <span className="text-red-700">❌ Error</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
              
              {/* Calculate button - positioned below AI input, aligned right */}
              <div className="flex justify-end space-x-2 mt-2">
                <Button
                  onClick={handleAIMetricCalculation}
                  disabled={aiLoading || !aiQuery.trim()}
                  size="sm"
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {aiLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-2"></div>
                      Calculating...
                    </>
                  ) : (
                    <>
                      <Send size={14} className="mr-1" />
                      Calculate
                    </>
                  )}
                </Button>
              </div>
              
              {aiResult && aiResult.success && aiResult.traditional_syntax && (
                <div className="text-xs text-blue-700 bg-blue-100 p-2 rounded mt-2">
                  <strong>Equivalent expression:</strong> {aiResult.traditional_syntax}
                </div>
              )}
            </div>
          ) : (
            /* Traditional Expression Input */
            <div className="w-full max-w-md">
              <div className={`border rounded-lg overflow-hidden ${isEditing ? 'bg-white border-blue-300' : 'bg-gray-50'}`}>
                <div className="p-3 relative" style={{ minHeight: '48px' }}>
                  {editor ? (
                    <div 
                      className={`w-full ${isEditing ? 'cursor-text' : 'pointer-events-none cursor-default'}`}
                      onClick={() => {
                        if (isEditing && editor) {
                          editor.commands.focus();
                        }
                      }}
                    >
                      <EditorContent 
                        editor={editor}
                        className="prose prose-sm max-w-none focus:outline-none w-full"
                        style={{ 
                          minHeight: '24px', 
                          maxHeight: '24px', 
                          lineHeight: '24px',
                          overflow: 'hidden',
                          whiteSpace: 'nowrap'
                        }}
                      />
                    </div>
                  ) : (
                    <div className="text-gray-400 italic font-mono text-sm leading-6">
                      {expression || 'Click Edit to create an expression...'}
                    </div>
                  )}
                </div>
              </div>
              
              {/* Save/Cancel buttons - positioned below expression input */}
              {isEditing && (
                <div className="flex justify-end space-x-2 mt-2">
                  <Button
                    onClick={handleCancel}
                    variant="secondary"
                    size="sm"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSave}
                    size="sm"
                  >
                    Save
                  </Button>
                </div>
              )}
            </div>
          )}
        
        {/* Autocomplete Suggestions Dropdown - Positioned outside editor */}
        {showSuggestions && suggestions.length > 0 && (
          <div 
            className="absolute bg-white border border-gray-300 rounded-lg shadow-xl max-h-80 overflow-y-auto"
            style={{
              top: suggestionPosition.top,
              left: suggestionPosition.left,
              minWidth: '250px',
              maxWidth: '350px',
              zIndex: 9999,
              boxShadow: '0 10px 25px rgba(0, 0, 0, 0.15)'
            }}
          >
            <div className="p-2">
              <div className="text-xs text-gray-500 mb-2 font-medium">
                {suggestionType === 'measures' ? 'Available Measures' : 'Aggregation Methods'}
              </div>
              {suggestions.map((suggestion, index) => (
                <div
                  key={suggestion.name}
                  className={`px-3 py-2 cursor-pointer rounded-md text-sm transition-colors ${
                    index === selectedSuggestionIndex
                      ? 'bg-blue-100 text-blue-800'
                      : 'hover:bg-gray-100 text-gray-700'
                  }`}
                  onClick={() => insertSuggestion(suggestion)}
                >
                  <div className="font-medium">{suggestion.name}</div>
                  {suggestion.description && (
                    <div className="text-xs text-gray-500 mt-1">
                      {suggestion.description}
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div className="border-t border-gray-200 px-3 py-2 bg-gray-50 text-xs text-gray-500">
              ↑↓ Navigate • Enter/Tab Select • Esc Cancel
            </div>
          </div>
        )}
        
        {validationErrors.length > 0 && (
          <div className="text-red-600 text-sm">
            {validationErrors.map((error, i) => (
              <div key={i}>• {error}</div>
            ))}
          </div>
        )}
        <div className="text-xs text-gray-500 mt-2 px-3">
          {aiMode ? (
            <span>Describe what you want to calculate in natural language</span>
          ) : (
            <span>Use @MeasureName.Aggregation syntax (e.g., @Revenue.Sum, @Cost.Avg)</span>
          )}
        </div>
        </div>
      </CardContent>

      {/* Filters Panel */}
      {showFilters && (
        <div className="border-t border-gray-200 p-4">
          <div className="text-sm font-medium text-gray-700 mb-3">Filters</div>
          <div className="space-y-3">
            {availableDimensions.map(dimension => (
              <FilterDimension
                key={dimension}
                dimension={dimension}
                datasetId={data.datasetId}
                selectedValues={filters[dimension] || []}
                onToggle={(value) => toggleFilter(dimension, value)}
              />
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// Filter Dimension Component
function FilterDimension({ dimension, datasetId, selectedValues, onToggle }) {
  const [values, setValues] = useState([]);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    fetch(`${API}/dimension_counts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: datasetId,
        dimension: dimension
      })
    })
    .then(res => res.json())
    .then(data => {
      setValues(data.labels || []);
    })
    .catch(err => console.error('Failed to fetch dimension values:', err));
  }, [datasetId, dimension]);

  return (
    <div className="border rounded-lg">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 text-left hover:bg-gray-50 transition-colors"
      >
        <span className="font-medium text-gray-700">{dimension}</span>
        <div className="flex items-center space-x-2">
          {selectedValues.length > 0 && (
            <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
              {selectedValues.length}
            </span>
          )}
          <span className={`transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </div>
      </button>
      
      {isExpanded && (
        <div className="border-t border-gray-200 p-3 max-h-48 overflow-y-auto">
          <div className="space-y-2">
            {values.map(value => (
              <label key={value} className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedValues.includes(value)}
                  onChange={() => onToggle(value)}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">{value}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Toolbar Component
function Toolbar({ activeTool, onToolChange, selectedCharts = [], onMergeCharts, onClearSelection }) {
  const tools = [
    { id: 'select', name: 'Select', icon: MousePointer2, description: 'Select and move items' },
    { id: 'arrow', name: 'Arrow', icon: MoveUpRight, description: 'Create arrows between points' },
    { id: 'textbox', name: 'Text', icon: Type, description: 'Add text boxes' },
    { id: 'expression', name: 'Expression', icon: SquareSigma, description: 'Create calculated expressions' },
    // Future tools can be added here
    // { id: 'sticky', name: 'Sticky', icon: '🗒️', description: 'Add sticky notes' },
    // { id: 'highlighter', name: 'Highlight', icon: '🖍️', description: 'Highlight areas' },
    // { id: 'lasso', name: 'Lasso', icon: '⭕', description: 'Lasso selection' },
  ];

  const canMerge = selectedCharts.length === 2;
  const mergeDescription = selectedCharts.length === 0 ? 'Select 2 charts to merge' : 
                          selectedCharts.length === 1 ? 'Select 1 more chart to merge' : 
                          'Merge selected charts';
  
  return (
    <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-white border border-gray-300 rounded-2xl shadow-lg p-2 z-50">
      <div className="flex items-center space-x-1">
        {tools.map(tool => {
          const IconComponent = tool.icon;
          return (
            <button
              key={tool.id}
              onClick={() => onToolChange(tool.id)}
              className={`flex flex-col items-center justify-center p-3 rounded-xl transition-all duration-200 group ${
                activeTool === tool.id 
                  ? 'bg-blue-100 text-blue-700 shadow-inner'
                  : 'hover:bg-gray-100 text-gray-600'
              }`}
              title={tool.description}
            >
              <IconComponent size={18} className="mb-1" />
              <span className="text-xs font-medium">{tool.name}</span>
            </button>
          );
        })}
        
        {/* Separator */}
        <div className="h-12 w-px bg-gray-300 mx-2"></div>
        
        {/* Merge Tool */}
        <button
          onClick={canMerge ? onMergeCharts : undefined}
          className={`flex flex-col items-center justify-center p-3 rounded-xl transition-all duration-200 group relative ${
            canMerge 
              ? 'hover:bg-green-100 text-green-700 cursor-pointer'
              : 'text-gray-400 cursor-not-allowed opacity-60'
          }`}
          title={mergeDescription}
        >
          <Merge size={18} className="mb-1" />
          <span className="text-xs font-medium">Merge</span>
          {selectedCharts.length > 0 && (
            <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
              {selectedCharts.length}
            </span>
          )}
        </button>
        
        {/* Clear Selection Tool - only show if there are selections */}
        {selectedCharts.length > 0 && (
          <button
            onClick={onClearSelection}
            className="flex flex-col items-center justify-center p-3 rounded-xl transition-all duration-200 group hover:bg-red-100 text-red-600 cursor-pointer"
            title="Clear chart selection"
          >
            <X size={18} className="mb-1" />
            <span className="text-xs font-medium">Clear</span>
          </button>
        )}
      </div>
    </div>
  );
}

// Table Component for AI Results
function DataTable({ data }) {
  if (!data || data.type !== 'table') return null;
  
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs border-collapse border border-gray-300">
        <thead>
          <tr className="bg-gray-100">
            {data.columns.map((col, idx) => (
              <th key={idx} className="border border-gray-300 px-2 py-1 text-left font-medium text-gray-700">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row, rowIdx) => (
            <tr key={rowIdx} className="hover:bg-gray-50">
              {row.map((cell, cellIdx) => (
                <td key={cellIdx} className="border border-gray-300 px-2 py-1">
                  {cellIdx === row.length - 1 && !isNaN(cell) ? 
                    parseFloat(cell).toLocaleString() : cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {data.title && (
        <p className="text-xs text-gray-500 mt-1 text-center">{data.title}</p>
      )}
    </div>
  );
}

// Chart Type Selector Component
function ChartTypeSelector({ dimensions = [], measures = [], currentType, onTypeChange }) {
  const dims = dimensions.length;
  const meas = measures.length;
  
  const supportedTypes = getSupportedChartTypes(dims, meas);
  
  // Don't show selector if only one chart type is supported
  if (supportedTypes.length <= 1) return null;
  
  return (
    <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
      {supportedTypes.map(type => {
        const IconComponent = type.icon;
        const isActive = currentType === type.id;
        
        return (
          <button
            key={type.id}
            onClick={() => onTypeChange(type.id)}
            className={`p-1.5 rounded transition-all duration-150 ${
              isActive 
                ? 'bg-blue-500 text-white shadow-sm' 
                : 'text-gray-600 hover:bg-white hover:text-gray-800'
            }`}
            title={type.label}
          >
            <IconComponent size={14} />
          </button>
        );
      })}
    </div>
  );
}

function ChartNode({ data, id, selected, onSelect, apiKey, selectedModel, setShowSettings, updateTokenUsage }) {
  const { title, figure, isFused, strategy, stats, agg, dimensions = [], measures = [], onAggChange, onShowTable, table = [], onChartHover } = data;
  const [menuOpen, setMenuOpen] = useState(false);
  const [statsVisible, setStatsVisible] = useState(false);
  const [aiExploreOpen, setAiExploreOpen] = useState(false);
  const [aiQuery, setAiQuery] = useState('');
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResult, setAiResult] = useState(null);
  const [showTableView, setShowTableView] = useState(false);
  
  // Chart type switching state
  const defaultChartType = getDefaultChartType(dimensions.length, measures.length);
  const [chartType, setChartType] = useState(defaultChartType.id);
  const [currentFigure, setCurrentFigure] = useState(figure);
  const [hasUserChangedType, setHasUserChangedType] = useState(false);
  
  // Sync currentFigure with figure prop changes, but preserve user's chart type choice
  useEffect(() => {
    if (figure) {
      if (!hasUserChangedType) {
        // No manual chart type change, just use the new figure
        setCurrentFigure(figure);
      } else {
        // User has selected a specific chart type, regenerate that type with new data
        const chartTypeConfig = CHART_TYPES[chartType.toUpperCase()];
        
        // Check if we have data (either array format or heatmap format)
        const hasData = (Array.isArray(table) && table.length > 0) || 
                       (table && typeof table === 'object' && table.x && table.y && table.z);
                       
        if (chartTypeConfig && hasData) {
          const payload = {
            table: table,
            dimensions: dimensions,
            measures: measures,
            strategy: strategy ? { type: strategy } : undefined
          };
          const newFigure = chartTypeConfig.createFigure(Array.isArray(table) ? table : [], payload);
          setCurrentFigure(newFigure);
        } else {
          // Fallback to the provided figure if chart type regeneration fails
          setCurrentFigure(figure);
        }
      }
    }
  }, [figure, chartType, hasUserChangedType, table, dimensions, measures, strategy]);
  
  const handleSelect = (e) => {
    e.stopPropagation();
    onSelect(id);
  };
  
  // Handle chart type changes
  const handleChartTypeChange = useCallback((newChartType) => {
    setChartType(newChartType);
    setHasUserChangedType(true); // Mark that user has manually changed chart type
    
    // Regenerate figure with new chart type using the chart registry
    const chartTypeConfig = CHART_TYPES[newChartType.toUpperCase()];
    
    // Check if we have data (either array format or heatmap format)
    const hasData = (Array.isArray(table) && table.length > 0) || 
                   (table && typeof table === 'object' && table.x && table.y && table.z);
    
    if (chartTypeConfig && hasData) {
      const payload = {
        table: table,
        dimensions: dimensions,
        measures: measures,
        strategy: strategy ? { type: strategy } : undefined
      };
      
      const newFigure = chartTypeConfig.createFigure(Array.isArray(table) ? table : [], payload);
      setCurrentFigure(newFigure);
    } else {
      console.warn('Chart type switching failed:', {
        chartType: newChartType,
        hasChartTypeConfig: !!chartTypeConfig,
        tableLength: Array.isArray(table) ? table.length : 0,
        hasHeatmapData: table?.x && table?.y && table?.z,
        dimensions,
        measures
      });
    }
  }, [table, dimensions, measures, strategy]);
  
  const handleAIExplore = async () => {
    if (!aiQuery.trim() || aiLoading) return;
    
    // Check if API key is configured
    const currentApiKey = apiKey || localStorage.getItem('gemini_api_key');
    const currentModel = selectedModel || localStorage.getItem('gemini_model') || 'gemini-2.0-flash';
    
    if (!currentApiKey.trim()) {
      setAiResult({
        success: false,
        answer: '⚠️ Please configure your Gemini API key in Settings first.'
      });
      setShowSettings(true);
      return;
    }
    
    setAiLoading(true);
    setShowTableView(false);  // Reset to text view for new queries
    try {
      const response = await fetch(`${API}/ai-explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chart_id: id,
          user_query: aiQuery.trim(),
          api_key: currentApiKey,
          model: currentModel
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }
      
      const result = await response.json();
      
      // Track token usage
      if (result.token_usage) {
        updateTokenUsage(result.token_usage);
      }
      
      // 🐍 LOG PYTHON CODE TO BROWSER CONSOLE
      console.log('🤖 AI Analysis Result:', result);
      if (result.code_steps && result.code_steps.length > 0) {
        console.log('🐍 PYTHON CODE GENERATED BY AI:');
        console.log('='.repeat(50));
        result.code_steps.forEach((code, index) => {
          console.log(`📝 Code Step ${index + 1}:`);
          console.log(code);
          console.log('-'.repeat(30));
        });
        console.log('💡 This code was executed on your real uploaded dataset');
      } else {
        console.log('ℹ️ No Python code generated for this query');
      }
      
      // Store AI result for display
      setAiResult(result);
      
      // Keep the query visible and don't close the section for easier follow-up questions
      // setAiQuery('');
      // setAiExploreOpen(false);
      setMenuOpen(false);
      
    } catch (error) {
      console.error('AI exploration failed:', error);
      setAiResult({
        success: false,
        answer: `AI exploration failed: ${error.message}. ${error.message.includes('401') || error.message.includes('403') ? 'Please check your API key in Settings.' : ''}`
      });
    } finally {
      setAiLoading(false);
    }
  };
  
  // Determine chart size based on chart type
  const isDualAxis = strategy === 'same-dimension-different-measures' && title?.includes('(Dual Scale)');
  const isHeatmap = strategy === 'same-dimension-different-dimensions-heatmap';
  const isMultiVariable = dimensions.length >= 1 && measures.length >= 1;
  const isThreeVariable = (dimensions.length >= 2 && measures.length >= 1) || (dimensions.length >= 1 && measures.length >= 2);
  
  const chartWidth = (isDualAxis || isHeatmap) ? '1000px' : isMultiVariable ? '760px' : '380px';
  const chartHeight = (isDualAxis || isHeatmap || isThreeVariable) ? '400px' : '300px';
  
  const canChangeAgg = Array.isArray(dimensions) && dimensions.length >= 1 && Array.isArray(measures) && measures.length >= 1 && (agg || 'sum') !== 'count';
  
  return (
    <div 
      className={`bg-white rounded-2xl shadow p-3 border-2 transition-all cursor-pointer ${
        selected 
          ? 'border-blue-500 bg-blue-50 shadow-lg' 
          : 'border-transparent hover:border-gray-300'
      } ${isFused ? 'ring-2 ring-green-200' : ''}`}
      style={{ width: chartWidth }}
      onClick={handleSelect}
      onMouseEnter={() => onChartHover?.(true)}
      onMouseLeave={() => onChartHover?.(false)}
    >
      {/* Clean Header with Title and Menu */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex-1">
          <div className="font-semibold">{title}</div>
        </div>
        
        {/* Chart Type Selector - show for charts with supported dimension/measure combinations */}
        {(() => {
          const dims = dimensions?.length || 0;
          const meas = measures?.length || 0;
          const supportedTypes = getSupportedChartTypes(dims, meas);
          
          
          // Show selector if multiple chart types are supported
          // Now we support 3-variable charts too!
          const showSelector = supportedTypes.length > 1;
          
          return showSelector ? (
            <ChartTypeSelector
              dimensions={dimensions}
              measures={measures}
              currentType={chartType}
              onTypeChange={handleChartTypeChange}
            />
          ) : null;
        })()}
        
        <div className="flex items-center space-x-2">
          {/* AI Explore Button */}
          <Button
            onClick={(e) => {
              e.stopPropagation();
              setAiExploreOpen(!aiExploreOpen);
            }}
            variant={aiExploreOpen ? "default" : "ghost"}
            size="icon"
            title={aiExploreOpen ? "Close AI Explorer" : "Explore with AI"}
            className={`p-1 ${aiExploreOpen ? "bg-blue-600 text-white hover:bg-blue-700" : "hover:bg-gray-100"}`}
          >
            <Wand size={16} />
          </Button>
          
          {/* Chart Options Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger
              className="p-1 hover:bg-gray-100 rounded"
              onClick={(e) => {
                e.stopPropagation();
                setMenuOpen(!menuOpen);
              }}
            >
              <Menu size={16} className="text-gray-600" />
            </DropdownMenuTrigger>
            
            <DropdownMenuContent 
              isOpen={menuOpen} 
              onClose={() => setMenuOpen(false)}
            >
              <DropdownMenuLabel>Chart Options</DropdownMenuLabel>
              <DropdownMenuSeparator />
              
              {/* Show Table Option */}
              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation();
                  onShowTable?.(id);
                  setMenuOpen(false);
                }}
              >
                <Table size={14} className="mr-2" />
                Data Table
              </DropdownMenuItem>
              
              {/* Toggle Stats Option */}
              {stats && (
                <DropdownMenuItem
                  onClick={(e) => {
                    e.stopPropagation();
                    setStatsVisible(!statsVisible);
                    setMenuOpen(false);
                  }}
                >
                  <BarChart size={14} className="mr-2" />
                  {statsVisible ? 'Hide' : 'Show'} Statistics
                </DropdownMenuItem>
              )}
              
              {/* Aggregation Options */}
              {canChangeAgg && (
                <>
                  <DropdownMenuSeparator />
                  <DropdownMenuLabel>Aggregation</DropdownMenuLabel>
                  {['sum', 'avg', 'min', 'max'].map(aggType => (
                    <DropdownMenuItem
                      key={aggType}
                      onClick={(e) => {
                        e.stopPropagation();
                        onAggChange?.(id, aggType);
                        setMenuOpen(false);
                      }}
                    >
                      <div className="flex items-center justify-between w-full">
                        <span className="capitalize">{aggType === 'avg' ? 'Average' : aggType}</span>
                        {(agg || 'sum') === aggType && (
                          <span className="text-blue-600">✓</span>
                        )}
                      </div>
                    </DropdownMenuItem>
                  ))}
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
      
      {/* Chart Plot - Now with more space! */}
      {currentFigure && currentFigure.data && currentFigure.layout ? (
        <Plot 
          data={currentFigure.data || []} 
          layout={sanitizeLayout(currentFigure.layout)} 
          style={{ width: '100%', height: chartHeight }} 
          useResizeHandler 
          config={{ displayModeBar: false }}
        />
      ) : (
        <div className="text-sm text-gray-500">Loading chart...</div>
      )}

      {/* Collapsible Stats - only show when toggled */}
      {stats && statsVisible && (
        <div className="mt-2 grid grid-cols-4 gap-2 text-xs text-gray-700">
          <div className="bg-gray-50 rounded-md p-2">
            <div className="font-semibold">Sum</div>
            <div className="tabular-nums">{Number(stats.sum).toLocaleString()}</div>
          </div>
          <div className="bg-gray-50 rounded-md p-2">
            <div className="font-semibold">Avg</div>
            <div className="tabular-nums">{Number(stats.avg).toLocaleString()}</div>
          </div>
          <div className="bg-gray-50 rounded-md p-2">
            <div className="font-semibold">Max</div>
            <div className="tabular-nums">{Number(stats.max).toLocaleString()}</div>
          </div>
          <div className="bg-gray-50 rounded-md p-2">
            <div className="font-semibold">Min</div>
            <div className="tabular-nums">{Number(stats.min).toLocaleString()}</div>
          </div>
        </div>
      )}

      {/* AI Explore Input Box */}
      {aiExploreOpen && (
        <div className="mt-3 border-t border-gray-200 pt-3">
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
              <Wand size={16} className="text-blue-600" />
              Explore with AI
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={aiQuery}
                onChange={(e) => setAiQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAIExplore();
                  }
                }}
                placeholder="e.g., show top 5 tiger reserves, percentage change from 2018 to 2023"
                className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                disabled={aiLoading}
              />
              <Button
                onClick={handleAIExplore}
                disabled={!aiQuery.trim() || aiLoading}
                size="sm"
                className="px-3"
              >
                {aiLoading ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                ) : (
                  <Send size={14} />
                )}
              </Button>
            </div>
            
            {/* AI Result Display */}
            {aiResult && (
              <div className={`p-3 rounded-md text-sm ${
                aiResult.success 
                  ? 'bg-blue-50 border border-blue-200 text-blue-800' 
                  : 'bg-red-50 border border-red-200 text-red-800'
              }`}>
                <div className="flex items-start gap-2">
                  {aiResult.success ? (
                    <div className="w-4 h-4 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
          </div>
                  ) : (
                    <div className="w-4 h-4 rounded-full bg-red-600 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
                    </div>
                  )}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <div className="font-medium">
                        {aiResult.success ? 'AI Analysis:' : 'Error:'}
                      </div>
                      {/* Toggle Button - only show if we have table data */}
                      {aiResult.success && aiResult.has_table && (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => setShowTableView(false)}
                            className={`px-2 py-1 text-xs rounded transition-colors ${
                              !showTableView 
                                ? 'bg-blue-600 text-white' 
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }`}
                          >
                            Text
                          </button>
                          <button
                            onClick={() => setShowTableView(true)}
                            className={`px-2 py-1 text-xs rounded transition-colors ${
                              showTableView 
                                ? 'bg-blue-600 text-white' 
                                : 'bg-gray-200 text-gray-600 hover:bg-gray-300'
                            }`}
                          >
                            Table
                          </button>
                        </div>
                      )}
                    </div>
                    {/* Content Display - Text or Table View - Scrollable Container */}
                    <div className="max-h-96 overflow-y-auto">
                      {showTableView && aiResult.tabular_data && aiResult.tabular_data.length > 0 ? (
                      /* Table View */
                      <div className="space-y-3">
                        {aiResult.tabular_data.map((tableData, idx) => (
                          <DataTable key={idx} data={tableData} />
                        ))}
                        {/* Show summary text below tables */}
                        <div className="text-xs text-gray-600 mt-2">
                          {aiResult.answer.split('--- AI Analysis Details ---')[0].trim()}
                        </div>
                      </div>
                    ) : (
                      /* Text View */
                      <div className="whitespace-pre-wrap leading-relaxed space-y-2">
                        <div className="text-sm">
                          {aiResult.answer.split('--- AI Analysis Details ---')[0].trim()}
                        </div>
                        
                        {/* AI Analysis Details - show code_steps if available */}
                        {(aiResult.code_steps && aiResult.code_steps.length > 0) && (
                          <details className="mt-3">
                            <summary className="cursor-pointer text-xs font-medium text-blue-600 hover:text-blue-800 flex items-center gap-1">
                              <span>🐍 View Python Code</span>
                              <span className="text-xs">(verification & reproducibility)</span>
                            </summary>
                            <div className="mt-2 p-3 bg-gray-900 rounded text-xs space-y-3">
                              {aiResult.code_steps.map((code, idx) => (
                                <div key={idx}>
                                  {aiResult.code_steps.length > 1 && (
                                    <div className="text-gray-400 mb-1">Step {idx + 1}:</div>
                                  )}
                                  <pre className="text-green-400 font-mono text-xs whitespace-pre-wrap overflow-x-auto">
                                    <code>{code}</code>
                                  </pre>
                                </div>
                              ))}
                              <div className="text-gray-400 text-xs mt-2 pt-2 border-t border-gray-700">
                                💡 This code shows how the analysis was performed using your actual dataset
                              </div>
                            </div>
                          </details>
                        )}
                        
                        {/* Fallback for legacy analysis details format */}
                        {aiResult.answer.includes('--- AI Analysis Details ---') && !(aiResult.code_steps && aiResult.code_steps.length > 0) && (
                          <details className="mt-3">
                            <summary className="cursor-pointer text-xs font-medium text-gray-600 hover:text-gray-800 flex items-center gap-1">
                              <span>🔍 Show Analysis Details</span>
                              <span className="text-xs">(reasoning & code)</span>
                            </summary>
                            <div className="mt-2 p-3 bg-gray-50 rounded text-xs space-y-2">
                              <div className="whitespace-pre-wrap font-mono text-gray-700">
                                {aiResult.answer.split('--- AI Analysis Details ---')[1]}
                              </div>
                            </div>
                          </details>
                        )}
                      </div>
                    )}
                    </div>
                    {aiResult.dataset_info && (
                      <div className="text-xs opacity-75 mt-2">
                        {aiResult.dataset_info}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ReactFlowWrapper() {
  const [datasetId, setDatasetId] = useState(null);
  const [csvFileName, setCsvFileName] = useState('');
  const [availableDimensions, setAvailableDimensions] = useState([]);
  const [availableMeasures, setAvailableMeasures] = useState([]);
  const [selectedDimension, setSelectedDimension] = useState('');
  const [selectedMeasure, setSelectedMeasure] = useState('');
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedCharts, setSelectedCharts] = useState([]);
  
  // Toolbar and tools state
  const [activeTool, setActiveTool] = useState('select');
  const [arrowStart, setArrowStart] = useState(null);
  const [nodeIdCounter, setNodeIdCounter] = useState(1000);

  // State to control zoom behavior
  const [isHoveringChart, setIsHoveringChart] = useState(false);
  
  // Settings state
  const [showSettings, setShowSettings] = useState(false);
  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_api_key') || '');
  const [selectedModel, setSelectedModel] = useState(localStorage.getItem('gemini_model') || 'gemini-2.0-flash');
  const [configStatus, setConfigStatus] = useState('idle'); // idle, testing, success, error
  const [configMessage, setConfigMessage] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isConfigLocked, setIsConfigLocked] = useState(false);
  const [tokenUsage, setTokenUsage] = useState({
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
    estimatedCost: 0
  });
  
  // Handler to control chart hover state
  const handleChartHover = useCallback((isHovering) => {
    setIsHoveringChart(isHovering);
  }, []);

  // Helper function to calculate token costs (Gemini pricing)
  const updateTokenUsage = useCallback((newUsage) => {
    if (!newUsage) return;
    
    const inputCostPer1K = 0.00075; // $0.00075 per 1K input tokens for Gemini
    const outputCostPer1K = 0.003;  // $0.003 per 1K output tokens for Gemini
    
    const inputCost = (newUsage.inputTokens / 1000) * inputCostPer1K;
    const outputCost = (newUsage.outputTokens / 1000) * outputCostPer1K;
    
    setTokenUsage(prev => ({
      inputTokens: prev.inputTokens + (newUsage.inputTokens || 0),
      outputTokens: prev.outputTokens + (newUsage.outputTokens || 0),
      totalTokens: prev.totalTokens + (newUsage.inputTokens || 0) + (newUsage.outputTokens || 0),
      estimatedCost: prev.estimatedCost + inputCost + outputCost
    }));
  }, []);

  // Initialize locked state if configuration is already stored
  useEffect(() => {
    const storedApiKey = localStorage.getItem('gemini_api_key');
    
    if (storedApiKey && storedApiKey.trim()) {
      setIsConfigLocked(true);
      setConfigStatus('success');
      setConfigMessage('✅ Configuration loaded from previous session.');
    }
  }, []);

  // Node types with access to settings state
  const nodeTypes = useMemo(() => ({
    chart: (props) => (
      <ChartNode 
        {...props} 
        selected={props.data.selected}
        onSelect={props.data.onSelect}
        apiKey={apiKey}
        selectedModel={selectedModel}
        setShowSettings={setShowSettings}
        updateTokenUsage={updateTokenUsage}
      />
    ),
    arrow: ArrowNode,
    textbox: TextBoxNode,
    table: TableNode,
    expression: (props) => (
      <ExpressionNode
        {...props}
        apiKey={apiKey}
        selectedModel={selectedModel}
        setShowSettings={setShowSettings}
        updateTokenUsage={updateTokenUsage}
      />
    )
  }), [apiKey, selectedModel, setShowSettings, updateTokenUsage]);

  // Viewport transform: [translateX, translateY, zoom]
  const transform = useStore(s => s.transform);
  const tx = transform ? transform[0] : 0;
  const ty = transform ? transform[1] : 0;
  const zoom = transform ? transform[2] : 1;

  // Convert a pane click event to flow-space coordinates
  const toFlowPosition = useCallback((event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const paneX = event.clientX - rect.left;
    const paneY = event.clientY - rect.top;
    return {
      x: (paneX - tx) / zoom,
      y: (paneY - ty) / zoom
    };
  }, [tx, ty, zoom]);

  // Get the center of the current viewport in flow coordinates
  const getViewportCenter = useCallback(() => {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    
    // Calculate center in screen coordinates
    const centerScreenX = viewportWidth / 2;
    const centerScreenY = viewportHeight / 2;
    
    // Convert to flow coordinates using current transform
    return {
      x: (centerScreenX - tx) / zoom,
      y: (centerScreenY - ty) / zoom,
    };
  }, [tx, ty, zoom]);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );
  
  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );
  
  // Canvas click handler for tools
  const onPaneClick = useCallback((event) => {
    if (activeTool === 'select') return;
    
    // Convert click to flow coordinates using current transform
    const position = toFlowPosition(event);
    
    if (activeTool === 'arrow') {
      if (!arrowStart) {
        // First click - set arrow start point in flow space
        setArrowStart(position);
      } else {
        // Second click - create arrow with absolute start/end in flow space
        const start = arrowStart;
        const end = position;
        const minX = Math.min(start.x, end.x);
        const minY = Math.min(start.y, end.y);

        const arrowId = `arrow-${nodeIdCounter}`;
        const newArrow = {
          id: arrowId,
          type: 'arrow',
          position: { x: minX, y: minY },
          data: { id: arrowId, start, end },
          draggable: true,
          selectable: true
        };

        setNodes(nds => [...nds, newArrow]);
        setNodeIdCounter(c => c + 1);
        setArrowStart(null);
      }
    } else if (activeTool === 'textbox') {
      // Create text box
      const newTextBox = {
        id: `textbox-${nodeIdCounter}`,
        type: 'textbox',
        position,
        data: {
          text: '',
          isNew: true,
          onTextChange: (id, newText) => {
            setNodes(nds => nds.map(node => 
              node.id === id 
                ? { ...node, data: { ...node.data, text: newText, isNew: false } }
                : node
            ));
          }
        },
        draggable: true,
        selectable: true
      };
      
      setNodes(nds => [...nds, newTextBox]);
      setNodeIdCounter(c => c + 1);
    } else if (activeTool === 'expression') {
      // Create expression node
      if (!datasetId) {
        alert('Please upload a dataset first to create expressions');
        return;
      }
      
      const newExpression = {
        id: `expression-${nodeIdCounter}`,
        type: 'expression',
        position,
        data: {
          expression: '',
          result: null,
          isNew: true,
          datasetId: datasetId,
          filters: {},
          onExpressionChange: (id, expression, result, filters) => {
            setNodes(nds => nds.map(node => 
              node.id === id 
                ? { ...node, data: { ...node.data, expression, result, filters, isNew: false } }
                : node
            ));
          }
        },
        draggable: true,
        selectable: true
      };
      
      setNodes(nds => [...nds, newExpression]);
      setNodeIdCounter(c => c + 1);
    }
  }, [activeTool, arrowStart, nodeIdCounter, toFlowPosition, datasetId]);
  
  // Tool change handler
  const handleToolChange = useCallback((toolId) => {
    setActiveTool(toolId);
    setArrowStart(null); // Reset arrow state when changing tools
  }, []);

  const handleChartSelect = useCallback((chartId) => {
    setSelectedCharts(prev => {
      if (prev.includes(chartId)) {
        // Deselect if already selected
        return prev.filter(id => id !== chartId);
      } else {
        // Select chart (max 2 charts can be selected)
        if (prev.length >= 2) {
          // Replace oldest selection with new one
          return [prev[1], chartId];
        }
        return [...prev, chartId];
      }
    });
  }, []);

  const handleShowTable = useCallback(async (chartId) => {
    try {
      console.log('Showing table for chart:', chartId);
      
      // Call the backend to get table data
      const res = await fetch(`${API}/chart-table`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chart_id: chartId })
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(errorText);
      }
      
      const tableData = await res.json();
      console.log('Table data received:', tableData);
      
      // Use setNodes with functional update to get current nodes
      setNodes(currentNodes => {
        // Find the chart node to position the table next to it
        const chartNode = currentNodes.find(n => n.id === chartId);
        if (!chartNode) {
          console.error('Chart node not found in current nodes:', chartId);
          console.error('Available nodes:', currentNodes.map(n => ({ id: n.id, type: n.type })));
          alert('Chart node not found');
          return currentNodes;
        }
        
        // Calculate position for table node (to the right of chart with offset)
        const tablePosition = {
          x: chartNode.position.x + (chartNode.data.strategy === 'same-dimension-different-measures' || 
              chartNode.data.strategy === 'same-measure-different-dimensions-stacked' ? 520 : 400),
          y: chartNode.position.y
        };
        
        // Create table node
        const tableId = `table-${chartId}-${Date.now()}`;
        const newTableNode = {
          id: tableId,
          type: 'table',
          position: tablePosition,
          data: {
            title: `${tableData.title} - Data Table`,
            headers: tableData.headers,
            rows: tableData.rows,
            totalRows: tableData.total_rows,
            sourceChartId: chartId
          },
          draggable: true,
          selectable: true
        };
        
        // Add table node to canvas
        return [...currentNodes, newTableNode];
      });
      
      setNodeIdCounter(c => c + 1);
      
    } catch (error) {
      console.error('Failed to show table:', error);
      alert('Failed to show table: ' + error.message);
    }
  }, []);

  const mergeSelectedCharts = useCallback(async () => {
    if (selectedCharts.length !== 2) {
      alert('Please select exactly 2 charts to merge');
      return;
    }

    const [c1, c2] = selectedCharts;
    try {
      const res = await fetch(`${API}/fuse`, { 
        method: 'POST', 
        headers: { 'Content-Type': 'application/json' }, 
        body: JSON.stringify({ chart1_id: c1, chart2_id: c2 }) 
      });
      
      if (!res.ok) throw new Error(await res.text());
      
      const fused = await res.json();
      const newId = fused.chart_id;
      
      // Position the fused node in the center of the current viewport
      const position = getViewportCenter();
      
      const figure = figureFromPayload(fused);
      
      // Add the new merged chart
      
      // For 2D+1M charts, backend now sends clean row-based data
      let finalDimensions = fused.dimensions || [];
      let finalMeasures = fused.measures || [];
      
      setNodes(nds => nds.concat({ 
        id: newId, 
        type: 'chart', 
        position, 
        data: { 
          title: fused.title, 
          figure,
          selected: false,
          onSelect: handleChartSelect,
          onShowTable: handleShowTable,
          onAggChange: updateChartAgg, // Add aggregation handler for fused charts
          onAIExplore: handleAIExplore,
          isFused: true,
          strategy: fused.strategy.type,
          dimensions: finalDimensions,
          measures: finalMeasures,
          agg: fused.agg || 'sum',
          table: fused.table || [] // Add table data for chart type switching
        } 
      }));
      
      // Clear selections after successful merge
      setSelectedCharts([]);
      
    } catch (e) {
      alert('Merge failed: ' + e.message);
    }
  }, [selectedCharts, nodes, handleChartSelect, getViewportCenter]);

  // Update aggregation on an existing chart node
  const updateChartAgg = useCallback(async (nodeId, newAgg) => {
    
    setNodes(currentNodes => {
      const node = currentNodes.find(n => n.id === nodeId);
      if (!node) {
        console.log('Node not found in current nodes:', nodeId);
        return currentNodes;
      }
      
      const dims = node.data.dimensions || [];
      const meas = node.data.measures || [];
      
      if (!datasetId || dims.length === 0 || meas.length === 0) {
        console.warn('Missing required data for aggregation update:', { nodeId, dims, meas, datasetId });
        return currentNodes;
      }

      // Optimistically update UI so the dropdown reflects immediately
      const updatedNodes = currentNodes.map(n => 
        n.id === nodeId ? ({ ...n, data: { ...n.data, agg: (newAgg || 'sum') } }) : n
      );

      // Make the API call asynchronously
      (async () => {
        try {
          const body = { dataset_id: datasetId, dimensions: dims, measures: meas, agg: newAgg };
          const res = await fetch(`${API}/charts`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          if (!res.ok) throw new Error(await res.text());
          const chart = await res.json();
          const figure = figureFromPayload(chart);
          const title = chart.title || `${(newAgg || 'sum').toUpperCase()} ${meas.join(', ')} by ${dims.join(', ')}`;
          
          setNodes(nds => nds.map(n => n.id === nodeId ? ({
            ...n,
            data: { 
              ...n.data, 
              title, 
              figure, 
              agg: (newAgg || 'sum'), 
              dimensions: chart.dimensions, 
              measures: chart.measures,
              table: chart.table || [] // Update table data for chart type switching
            }
          }) : n));
        } catch (e) {
          console.error('Aggregation update failed:', e);
          // Revert optimistic change on error
          setNodes(nds => nds.map(n => n.id === nodeId ? ({ ...n, data: { ...n.data, agg: (node.data.agg || 'sum') } }) : n));
          alert('Aggregation update failed: ' + e.message);
        }
      })();

      return updatedNodes;
    });
  }, [datasetId]);

  // Update nodes with current selection status
  const nodesWithSelection = useMemo(() => {
    return nodes.map(node => ({
      ...node,
      data: {
        ...node.data,
        selected: selectedCharts.includes(node.id),
        // Add hover handler for chart nodes
        onChartHover: node.type === 'chart' ? handleChartHover : undefined
      }
    }));
  }, [nodes, selectedCharts, handleChartHover]);

  function figureFromPayload(payload, chartType = null) {
    // Internal helper to ensure all figures have sanitized layouts
    const createSafeFigure = (data, layout) => ({
      data,
      layout: sanitizeLayout(layout)
    });
    const rows = payload.table || [];
    const dims = payload.dimensions?.length || 0;
    const measures = payload.measures?.length || 0;
    
    // Strategy A: same-dimension-different-measures => grouped bar or dual-axis
    if (payload.strategy?.type === 'same-dimension-different-measures') {
      // Check if a specific chart type is requested via the chart registry
      if (chartType && CHART_TYPES[chartType.toUpperCase()]) {
        const chartTypeConfig = CHART_TYPES[chartType.toUpperCase()];
        if (chartTypeConfig.isSupported(dims, measures)) {
          return chartTypeConfig.createFigure(rows, payload);
        }
      }
      
      const dims = payload.dimensions;
      const xKey = dims[0];
      const measureKeys = payload.measures.filter(m => m !== xKey);
      const xValues = [...new Set(rows.map(r => r[xKey]))];
      
      // Check if we need dual Y-axis (2 measures with different scales)
      if (measureKeys.length === 2) {
        const m1Values = xValues.map(v => (rows.find(r => r[xKey] === v)?.[measureKeys[0]]) ?? 0);
        const m2Values = xValues.map(v => (rows.find(r => r[xKey] === v)?.[measureKeys[1]]) ?? 0);
        
        // Calculate scale difference to determine if dual-axis is needed
        const m1Max = Math.max(...m1Values);
        const m2Max = Math.max(...m2Values);
        const scaleRatio = Math.max(m1Max, m2Max) / Math.min(m1Max, m2Max);
        
        // Use dual Y-axis if scale difference is significant (>10x)
        if (scaleRatio > 10) {
          const data = [
            {
              type: 'bar',
              name: measureKeys[0],
              x: xValues,
              y: m1Values,
              yaxis: 'y',
              marker: { color: '#3182ce' }
            },
            {
              type: 'scatter',
              mode: 'lines+markers',
              name: measureKeys[1],
              x: xValues,
              y: m2Values,
              yaxis: 'y2',
              line: { color: '#e53e3e', width: 3 },
              marker: { color: '#e53e3e', size: 8 }
            }
          ];
          
          return createSafeFigure(data, {
              // Remove title to avoid duplication with ChartNode title
              xaxis: {
                title: { text: xKey, font: { size: 14, color: '#4a5568' } },
                tickangle: -45
              },
              yaxis: {
                title: { text: measureKeys[0], font: { size: 14, color: '#3182ce' } },
                side: 'left'
              },
              yaxis2: {
                title: { text: measureKeys[1], font: { size: 14, color: '#e53e3e' } },
                side: 'right',
                overlaying: 'y'
              },
              margin: { t: 20, b: 120, l: 80, r: 80 }, // Reduced top margin, increased bottom for legend
              plot_bgcolor: '#fafafa',
              paper_bgcolor: 'white',
              showlegend: true,
              legend: {
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.25, // Moved further down
                yanchor: 'top',
                bgcolor: 'rgba(255,255,255,0.8)',
                bordercolor: '#E2E8F0',
                borderwidth: 1,
                font: { size: 12 }
              }
          });
        }
      }
      
      // Fallback to grouped bar chart for single measure or similar scales
      const data = measureKeys.map(m => ({
        type: 'bar',
        name: m,
        x: xValues,
        y: xValues.map(v => (rows.find(r => r[xKey] === v)?.[m]) ?? 0)
      }));
      
      return createSafeFigure(data, { 
          // Remove title to avoid duplication with ChartNode title
          xaxis: {
            title: {
              text: xKey,
              font: { size: 14, color: '#4a5568' }
            },
            tickangle: -45
          },
          yaxis: {
            title: {
              text: measureKeys.length > 1 ? 'Values' : measureKeys[0],
              font: { size: 14, color: '#4a5568' }
            }
          },
          barmode: 'group', 
          margin: { t: 20, b: measureKeys.length > 1 ? 100 : 80, l: 80, r: 30 }, // Reduced top margin, more bottom space for legend
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: measureKeys.length > 1,
          legend: measureKeys.length > 1 ? {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E8F0',
            borderwidth: 1,
            font: { size: 12 }
          } : undefined
      });
    }
    
    // Strategy C: measure-by-dimension (1-variable fusion)
    if (payload.strategy?.type === 'measure-by-dimension') {
      const dims = payload.dimensions.filter(d => d !== 'count');
      const xKey = dims[0];
      const measures = payload.measures.filter(m => m !== 'count');
      const m = measures[0];
      const xValues = [...new Set(rows.map(r => r[xKey]))];
      const data = [{
        type: 'bar',
        name: m,
        x: xValues,
        y: xValues.map(v => (rows.find(r => r[xKey] === v)?.[m]) ?? 0)
      }];
      return createSafeFigure(data, {
          // Remove title to avoid duplication with ChartNode title
          xaxis: { title: { text: xKey, font: { size: 14, color: '#4a5568' } }, tickangle: -45 },
          yaxis: { title: { text: m, font: { size: 14, color: '#4a5568' } } },
          margin: { t: 20, b: 80, l: 80, r: 30 }, // Reduced top margin
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white'
      });
    }

    // Strategy B: same-measure-different-dimensions => STACKED BAR
    if (payload.strategy?.type === 'same-measure-different-dimensions-stacked') {
      // Check if a specific chart type is requested via the chart registry
      if (chartType && CHART_TYPES[chartType.toUpperCase()]) {
        const chartTypeConfig = CHART_TYPES[chartType.toUpperCase()];
        if (chartTypeConfig.isSupported(dims, measures)) {
          return chartTypeConfig.createFigure(rows, payload);
        }
      }
      
      // Default to stacked bar for 2D+1M
      return CHART_TYPES.STACKED_BAR.createFigure(rows, payload);
    }
    
    // Strategy B: same-measure-different-dimensions => multi-series line (fallback)
    if (payload.strategy?.type === 'same-measure-different-dimensions') {
      const groups = {};
      rows.forEach(r => {
        const g = r['DimensionType'];
        if (!groups[g]) groups[g] = [];
        groups[g].push(r);
      });
      const data = Object.entries(groups).map(([g, arr]) => ({
        type: 'scatter', 
        mode: 'lines+markers', 
        name: g,
        x: arr.map(a => a['DimensionValue']),
        y: arr.map(a => a['Value']),
        line: { width: 3 },
        marker: { size: 8 }
      }));
      
      return createSafeFigure(data, { 
          // Remove title to avoid duplication with ChartNode title
          xaxis: {
            title: {
              text: 'Categories',
              font: { size: 14, color: '#4a5568' }
            },
            tickangle: -45
          },
          yaxis: {
            title: {
              text: 'Value',
              font: { size: 14, color: '#4a5568' }
            }
          },
          margin: { t: 20, b: 100, l: 80, r: 30 }, // Reduced top margin, increased bottom for legend
          plot_bgcolor: '#fafafa',
          paper_bgcolor: 'white',
          showlegend: true,
          legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.2,
            yanchor: 'top',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#E2E8F0',
            borderwidth: 1,
            font: { size: 12 }
          }
      });
    }
    
    // Fallback: Use chart registry system for flexible chart types
    const keys = rows.length ? Object.keys(rows[0]) : [];
    
    // Respect chart configuration first (important for AI-generated charts!)
    const xKey = (payload.dimensions && payload.dimensions.length > 0) 
      ? payload.dimensions[0] 
      : keys.find(k => keys.indexOf(k) === 0 || !rows.some(r => typeof r[k] === 'number')) || 'Category';
      
    const numKey = (payload.measures && payload.measures.length > 0)
      ? payload.measures[0]  // Use first measure from chart configuration
      : keys.find(k => rows.some(r => typeof r[k] === 'number'));
    
    // Debug logging for AI-generated charts
    if (payload.is_ai_generated) {
      console.log('🤖 AI-generated chart detected:', {
        dimensions: payload.dimensions,
        measures: payload.measures,
        selectedXKey: xKey,
        selectedNumKey: numKey,
        availableKeys: keys
      });
    }
    
    // Determine chart type: explicit override > strategy > default
    let activeChartType;
    
    if (chartType && CHART_TYPES[chartType.toUpperCase()]) {
      // Explicit chart type requested
      activeChartType = CHART_TYPES[chartType.toUpperCase()];
    } else {
      // Get default chart type for this dimension/measure combination
      activeChartType = getDefaultChartType(dims, measures);
    }
    
    // Create standardized payload for chart type functions
    const standardPayload = {
      ...payload,
      dimensions: payload.dimensions || [xKey],
      measures: payload.measures || [numKey].filter(Boolean)
    };
    
    // Use chart registry to create figure
    return activeChartType.createFigure(rows, standardPayload);
  }

  // AI Exploration handler - defined after all dependencies (handleShowTable, updateChartAgg, figureFromPayload)
  const handleAIExplore = useCallback(async (chartId, aiResult) => {
    // AI exploration is now text-based and handled directly within ChartNode components
    // This callback is no longer used for creating chart nodes
    console.log('AI exploration (text-based):', aiResult);
  }, []);

  const uploadCSV = async (file) => {
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API}/upload`, { method: 'POST', body: fd });
      
      if (!res.ok) {
        throw new Error(`Upload failed: ${res.status} ${res.statusText}`);
      }
      
      const meta = await res.json();
      setDatasetId(meta.dataset_id);
      setCsvFileName(file.name); // Store the filename
      setAvailableDimensions(meta.dimensions || []);
      setAvailableMeasures(meta.measures || []);
      // Clear previous selections
      setSelectedDimension('');
      setSelectedMeasure('');
      
      console.log('CSV uploaded successfully:', meta);
    } catch (error) {
      console.error('Failed to upload CSV:', error);
      setCsvFileName(''); // Clear filename on error
      alert(`Failed to upload CSV: ${error.message}`);
    }
  };

  const createVisualization = async () => {
    if (!datasetId) return alert('Upload a CSV first.');
    
    // Validate selection - need at least one dimension or measure
    if (!selectedDimension && !selectedMeasure) {
      return alert('Please select at least one dimension or measure');
    }

    try {
      let id = `viz-${Date.now()}`;
      
      // Case 1: Two variables selected (Dimension + Measure)
      if (selectedDimension && selectedMeasure) {
        const body = { 
          dataset_id: datasetId, 
          dimensions: [selectedDimension], 
          measures: [selectedMeasure] 
        };
        const res = await fetch(`${API}/charts`, { 
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' }, 
          body: JSON.stringify(body) 
        });
        
        if (!res.ok) return alert('Create chart failed');
        const chart = await res.json();
        id = chart.chart_id;
        const figure = figureFromPayload(chart);
        
        setNodes(nds => nds.concat({ 
          id, 
          type: 'chart', 
          position: getViewportCenter(), 
          data: { 
            title: chart.title, 
            figure,
            selected: false,
            onSelect: handleChartSelect,
            onShowTable: handleShowTable,
            onAggChange: updateChartAgg,
            onAIExplore: handleAIExplore,
            agg: chart.agg || 'sum',
            dimensions: [selectedDimension],
            measures: [selectedMeasure],
            table: chart.table || [] // Add table data for chart type switching
          } 
        }));
      }
      
      // Case 2: Single Measure (Histogram)
      else if (selectedMeasure && !selectedDimension) {
        const res = await fetch(`${API}/histogram`, {
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_id: datasetId, measure: selectedMeasure })
        });
        
        if (!res.ok) throw new Error(await res.text());
        const { values, stats } = await res.json();
        
        // Register server-side chart for fusion
        try {
          const body = { 
            dataset_id: datasetId, 
            dimensions: [], 
            measures: [selectedMeasure], 
            agg: 'sum', 
            title: `Histogram: ${selectedMeasure}` 
          };
          const reg = await fetch(`${API}/charts`, { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify(body) 
          });
          if (reg.ok) {
            const chart = await reg.json();
            id = chart.chart_id;
          }
        } catch {}
        
        const figure = {
          data: [{ type: 'histogram', x: values, marker: { color: '#7c3aed' }, opacity: 0.85 }],
          layout: { 
            xaxis: { title: { text: selectedMeasure } }, 
            yaxis: { title: { text: 'Count' } }, 
            margin: { t: 20, b: 60, l: 60, r: 30 }, 
            plot_bgcolor: '#fafafa', 
            paper_bgcolor: 'white',
            showlegend: false,
            legend: undefined
          }
        };
        
        setNodes(nds => nds.concat({ 
          id, 
          type: 'chart', 
          position: getViewportCenter(), 
          data: { 
            title: `Histogram: ${selectedMeasure}`, 
            figure, 
            selected: false, 
            onSelect: handleChartSelect, 
            onShowTable: handleShowTable, 
            onAIExplore: handleAIExplore,
            stats, 
            agg: 'sum', 
            dimensions: [], 
            measures: [selectedMeasure], 
            onAggChange: updateChartAgg 
          } 
        }));
      }
      
      // Case 3: Single Dimension (Bar Chart)
      else if (selectedDimension && !selectedMeasure) {
        const res = await fetch(`${API}/dimension_counts`, {
          method: 'POST', 
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_id: datasetId, dimension: selectedDimension })
        });
        
        if (!res.ok) throw new Error(await res.text());
        const { labels, counts } = await res.json();
        
        // Register server-side chart for fusion
        try {
          const body = { 
            dataset_id: datasetId, 
            dimensions: [selectedDimension], 
            measures: [], 
            agg: 'count', 
            title: `Counts of ${selectedDimension}` 
          };
          const reg = await fetch(`${API}/charts`, { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify(body) 
          });
          if (reg.ok) {
            const chart = await reg.json();
            id = chart.chart_id;
          }
        } catch {}
        
        const figure = {
          data: [{ type: 'bar', x: labels, y: counts, marker: { color: '#0ea5e9' } }],
          layout: { 
            xaxis: { title: { text: selectedDimension } }, 
            yaxis: { title: { text: 'Count' } }, 
            margin: { t: 20, b: 80, l: 60, r: 30 }, 
            plot_bgcolor: '#fafafa', 
            paper_bgcolor: 'white',
            showlegend: false,
            legend: undefined
          }
        };
        
        setNodes(nds => nds.concat({ 
          id, 
          type: 'chart', 
          position: getViewportCenter(), 
          data: { 
            title: `Bar: ${selectedDimension} vs Count`, 
            figure, 
            selected: false, 
            onSelect: handleChartSelect, 
            onShowTable: handleShowTable, 
            onAIExplore: handleAIExplore,
            agg: 'count', 
            dimensions: [selectedDimension], 
            measures: ['count'], 
            onAggChange: updateChartAgg 
          } 
        }));
      }
    } catch (e) {
      alert('Visualization failed: ' + e.message);
    }
  };

  // Helper function to format user-friendly error messages
  const formatErrorMessage = (errorMessage) => {
    if (!errorMessage) return 'Unknown error occurred.';
    
    // Check for quota exceeded errors
    if (errorMessage.includes('quota') || errorMessage.includes('429') || errorMessage.includes('exceeded')) {
      return 'You exceeded your current quota, please check your plan and billing details.';
    }
    
    // Check for API key errors
    if (errorMessage.includes('API key not valid') || errorMessage.includes('401') || errorMessage.includes('403')) {
      return 'Invalid API key. Please check your API key and try again.';
    }
    
    // Check for rate limiting
    if (errorMessage.includes('rate limit') || errorMessage.includes('too many requests')) {
      return 'Rate limit exceeded. Please wait a moment and try again.';
    }
    
    // Check for network errors
    if (errorMessage.includes('Network error') || errorMessage.includes('fetch')) {
      return 'Network connection error. Please check your internet connection.';
    }
    
    // For other errors, try to extract a short meaningful message
    const lines = errorMessage.split('\n');
    const firstLine = lines[0];
    
    // If first line is too long, truncate it
    if (firstLine.length > 100) {
      return firstLine.substring(0, 100) + '...';
    }
    
    return firstLine;
  };

  // Settings panel component
  const SettingsPanel = () => {
    const handleTestConfiguration = async () => {
      if (!apiKey.trim()) {
        setConfigStatus('error');
        setConfigMessage('Please enter an API key');
        return;
      }

      setConfigStatus('testing');
      setConfigMessage('Testing configuration...');

      try {
        const response = await fetch(`${API}/test-config`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            api_key: apiKey,
            model: selectedModel
          })
        });

        const result = await response.json();
        
        if (result.success) {
          setConfigStatus('success');
          setConfigMessage('✅ Configuration successful! LLM is ready to use.');
          localStorage.setItem('gemini_api_key', apiKey);
          localStorage.setItem('gemini_model', selectedModel);
          setIsConfigLocked(true); // Lock the configuration after success
          
          // Track the test token usage
          if (result.token_usage) {
            updateTokenUsage(result.token_usage);
          }
        } else {
          setConfigStatus('error');
          setConfigMessage(`❌ ${formatErrorMessage(result.error)}`);
        }
      } catch (error) {
        setConfigStatus('error');
        setConfigMessage(`❌ ${formatErrorMessage(error.message)}`);
      }
    };

    const handleEditConfiguration = () => {
      setIsConfigLocked(false);
      setConfigStatus('idle');
      setConfigMessage('');
    };

    return (
      <div className="absolute top-12 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-4 w-80 z-50">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <Settings size={18} />
            AI Settings
          </h3>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSettings(false)}
            className="h-6 w-6 p-0"
          >
            <X size={14} />
          </Button>
        </div>

        <div className="space-y-4">
          {/* API Key Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Gemini API Key
            </label>
            <div className="relative">
              <Input
                type={showApiKey ? "text" : "password"}
                placeholder="Enter your Gemini API key"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                disabled={isConfigLocked}
                className={`w-full pr-10 ${isConfigLocked ? 'bg-gray-50 text-gray-500' : ''}`}
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
              >
                {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Get your free API key from{' '}
              <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">
                Google AI Studio
              </a>
            </p>
          </div>

          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model Selection
            </label>
            <Select value={selectedModel} onValueChange={setSelectedModel} disabled={isConfigLocked}>
              <SelectTrigger className={`w-full ${isConfigLocked ? 'bg-gray-50 text-gray-500' : ''}`}>
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="gemini-1.5-flash">Gemini 1.5 Flash</SelectItem>
                <SelectItem value="gemini-2.0-flash">Gemini 2.0 Flash</SelectItem>
                <SelectItem value="gemini-2.0-flash-exp">Gemini 2.0 Flash Experimental</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Test/Edit Configuration Button */}
          <Button 
            onClick={isConfigLocked ? handleEditConfiguration : handleTestConfiguration} 
            className={`w-full gap-2 ${isConfigLocked ? 'bg-orange-600 hover:bg-orange-700' : ''}`}
            disabled={configStatus === 'testing'}
          >
            {configStatus === 'testing' ? (
              <>
                <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                Testing...
              </>
            ) : isConfigLocked ? (
              <>
                <Edit size={16} />
                Edit Configuration
              </>
            ) : (
              <>
                <Check size={16} />
                Test Configuration
              </>
            )}
          </Button>

          {/* Status Message */}
          {configMessage && (
            <div className={`p-3 rounded-md text-sm break-words whitespace-normal leading-relaxed ${
              configStatus === 'success' 
                ? 'bg-green-50 text-green-800 border border-green-200' 
                : configStatus === 'error'
                ? 'bg-red-50 text-red-800 border border-red-200'
                : 'bg-blue-50 text-blue-800 border border-blue-200'
            }`}>
              {configMessage}
            </div>
          )}

          {/* Token Usage Display */}
          {tokenUsage.totalTokens > 0 && (
            <div className="border-t pt-4 mt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Token Usage (This Session)</h4>
              <div className="space-y-1 text-xs text-gray-600">
                <div className="flex justify-between">
                  <span>Input Tokens:</span>
                  <span>{tokenUsage.inputTokens.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Output Tokens:</span>
                  <span>{tokenUsage.outputTokens.toLocaleString()}</span>
                </div>
                <div className="flex justify-between font-medium">
                  <span>Total Tokens:</span>
                  <span>{tokenUsage.totalTokens.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-green-600 font-medium">
                  <span>Est. Cost:</span>
                  <span>${tokenUsage.estimatedCost.toFixed(4)}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="w-screen h-screen relative">
      {/* Floating Side Panel */}
      <div className="fixed top-4 left-4 w-80 max-h-[calc(100vh-8rem)] bg-white border border-gray-300 rounded-2xl shadow-lg z-40 overflow-hidden">
        <Card className="h-full border-0">
          <CardHeader className="pb-4">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <SquaresExclude size={40} className="text-blue-600 shrink-0" />
                <div className="flex flex-col">
                  <h1 className="text-xl font-bold text-gray-900">D.Fuse</h1>
                  <p className="text-sm text-muted-foreground">Data exploration notebook</p>
                </div>
              </div>
              
              <div className="pt-4">
                <FileUpload 
                  accept=".csv" 
                  onFileChange={(file) => uploadCSV(file)}
                >
                  {datasetId ? 'Replace CSV' : 'Choose CSV File'}
                </FileUpload>
                
                {(csvFileName || datasetId) && (
                  <div className="mt-2 border border-gray-200 rounded-lg p-3 bg-gray-50/30">
                    {csvFileName && (
                      <div className="text-sm text-gray-600 flex items-center gap-2 mb-2">
                        <File size={16} />
                        {csvFileName}
                      </div>
                    )}
                    
                    {datasetId && (
                      <>
                        <Badge variant="outline" className="w-fit mb-2">
                          Dataset: {datasetId.substring(0, 8)}...
                        </Badge>
                        <div className="flex gap-1 flex-wrap">
                          <Badge variant="secondary" className="text-xs">
                            {availableDimensions.length} dimensions
                          </Badge>
                          <Badge variant="secondary" className="text-xs">
                            {availableMeasures.length} measures
                          </Badge>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 overflow-y-auto max-h-[calc(100vh-12rem)]">

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">Dimensions</h3>
                </div>
                <RadioGroup
                  options={availableDimensions}
                  value={selectedDimension}
                  onChange={setSelectedDimension}
                  name="dimensions"
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium">Measures</h3>
                </div>
                <RadioGroup
                  options={availableMeasures}
                  value={selectedMeasure}
                  onChange={setSelectedMeasure}
                  name="measures"
                />
              </div>

              <div className="space-y-2">
                <Button 
                  className="w-full gap-2 bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 disabled:bg-gray-300 disabled:text-gray-500" 
                  onClick={createVisualization}
                  disabled={!selectedDimension && !selectedMeasure}
                >
                  <ChartColumn size={16} />
                  Visualise
                </Button>
              </div>
            </div>
            </CardContent>
        </Card>
      </div>

      {/* Main Canvas - Full Width */}
      <div className="w-full h-full absolute inset-0">
        <ReactFlow
          nodes={nodesWithSelection}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onPaneClick={onPaneClick}
          fitView
          style={{ cursor: activeTool === 'select' ? 'default' : 'crosshair' }}
          zoomOnScroll={!isHoveringChart}
          zoomOnPinch={!isHoveringChart}
          preventScrolling={isHoveringChart}
        >
          <MiniMap />
          <Controls style={{ position: 'absolute', bottom: '10px', right: '230px', left: 'auto' }} />
          <Background gap={16} />
        </ReactFlow>
        
        {/* Arrow preview line when creating arrow */}
        {activeTool === 'arrow' && arrowStart && (
          <div className="absolute top-4 left-4 bg-blue-100 text-blue-800 px-3 py-1 rounded-lg text-sm font-medium z-10">
            Click to set arrow end point
          </div>
        )}
        
        {/* Settings Button and Panel */}
        <div className="absolute top-4 right-4 z-20">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowSettings(!showSettings)}
            className="h-8 w-8 p-0 bg-white border border-gray-300 shadow-sm hover:bg-gray-50"
            title="AI Settings"
          >
            <Settings size={16} className="text-gray-600" />
          </Button>
          {showSettings && <SettingsPanel />}
        </div>
        
        {/* Tool status indicator - moved to avoid overlap with settings */}
        <div className="absolute top-4 right-16 bg-white border border-gray-300 rounded-lg px-3 py-1 text-sm text-gray-600 shadow-sm z-10">
          <span className="font-medium">Active Tool:</span> {activeTool === 'select' ? 'Select' : activeTool === 'arrow' ? 'Arrow' : 'Text Box'}
        </div>
        
        <Toolbar 
          activeTool={activeTool} 
          onToolChange={handleToolChange} 
          selectedCharts={selectedCharts}
          onMergeCharts={mergeSelectedCharts}
          onClearSelection={() => setSelectedCharts([])}
        />
      </div>
    </div>
  );
}

// Main App component with ReactFlowProvider
export default function App() {
  return (
    <ReactFlowProvider>
      <ReactFlowWrapper />
    </ReactFlowProvider>
  );
}
