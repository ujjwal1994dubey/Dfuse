# D.Fuse Frontend Architecture

## Overview
D.Fuse is a visual data exploration tool built with React and React Flow, enabling users to create, manipulate, and analyze data visualizations through an intuitive canvas-based interface.

---

## Tech Stack

### Core Technologies
- **React 18** - UI framework
- **React Flow** - Node-based canvas for visual data exploration
- **Plotly.js** - Interactive chart rendering
- **Tailwind CSS** - Styling framework
- **Tiptap** - Rich text editing for notes

### Key Libraries
- **Radix UI** - Accessible UI components (dropdowns, buttons)
- **Lucide React** - Icon library
- **Marked.js** - Markdown parsing for reports

---

## Application Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         D.Fuse Frontend                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Unified        │  │  Canvas          │  │  Report        │ │
│  │  Sidebar        │  │  (React Flow)    │  │  Panel         │ │
│  │  - Upload       │  │  - Chart Nodes   │  │  - Title       │ │
│  │  - Variables    │  │  - Text Nodes    │  │  - Sections    │ │
│  │  - Tools        │  │  - Expression    │  │  - Export      │ │
│  │  - Actions      │  │  - Arrows        │  │                │ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │            Top Bar (Settings, Token Usage, Report)         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
                    ┌──────────────────┐
                    │  Backend API     │
                    │  (FastAPI)       │
                    │  - Data Upload   │
                    │  - Charts        │
                    │  - AI Features   │
                    └──────────────────┘
```

---

## Component Hierarchy

### 1. **App Component** (Root)
```
App (ReactFlowProvider)
└── ReactFlowWrapper (Main Container)
    ├── UnifiedSidebar
    │   ├── Panel Toggles (Upload, Variables)
    │   ├── Tool Buttons (Select, Arrow, Text, Expression)
    │   └── Action Buttons (Merge, Arrange)
    │
    ├── SlidingPanel (Upload)
    │   └── FileUpload Component
    │
    ├── SlidingPanel (Variables)
    │   ├── Dimension RadioGroup
    │   ├── Measure RadioGroup
    │   └── Visualize Button
    │
    ├── Canvas (CustomReactFlow)
    │   ├── Background
    │   ├── Controls (Zoom, Pan)
    │   ├── MiniMap
    │   └── Nodes (ChartNode, TextBoxNode, ExpressionNode, ArrowNode)
    │
    ├── Top Bar
    │   ├── Settings Button & Panel
    │   ├── Token Usage Display
    │   └── Report Toggle Button
    │
    └── ReportPanel
        ├── Report Title (editable)
        ├── Report Subheading (editable)
        └── ReportSections[]
            ├── Chart Image
            ├── Markdown Content (editable)
            └── Action Buttons
```

---

## Node Types

### 1. **ChartNode** (Most Complex)
**Purpose:** Renders interactive Plotly charts with analysis features

**Features:**
- 📊 Multi-type chart rendering (bar, pie, scatter, line, histogram)
- 🔄 Aggregation switching (sum, avg, min, max, count)
- 🎨 Chart type conversion
- 📋 Data table view
- 🔍 Dimension filtering
- 🤖 AI Exploration (natural language queries)
- 💡 AI-generated insights (sticky notes)
- 📄 Add to Report functionality

**State Management:**
- Menu visibility, stats display, AI explore panel
- Filter states, aggregation method
- AI results and insights
- Loading states

### 2. **TextBoxNode**
**Purpose:** Rich text notes using Tiptap editor

**Features:**
- Bold, italic, headings, lists
- Click-to-edit mode
- Auto-save functionality
- Sticky note styling

### 3. **ExpressionNode**
**Purpose:** Mathematical expression calculator

**Features:**
- Field reference support (@Revenue.Sum)
- Real-time validation
- AI-powered calculation
- Dimension filtering
- Result display with formatting

### 4. **ArrowNode**
**Purpose:** Visual connectors between elements

**Features:**
- SVG-based rendering
- Arrowhead markers
- Start/end point coordinates

### 5. **TableNode**
**Purpose:** Display tabular data

**Features:**
- Scrollable table view
- Formatted numeric values
- Column headers

### 6. **InsightStickyNote**
**Purpose:** Draggable AI-generated insights

**Features:**
- Drag-and-drop positioning
- Resizable dimensions
- Close functionality

---

## State Management

### Global State (ReactFlowWrapper)
```javascript
├── Dataset State
│   ├── datasetId
│   ├── csvFileName
│   ├── availableDimensions
│   └── availableMeasures
│
├── Canvas State
│   ├── nodes (React Flow nodes array)
│   ├── edges (React Flow edges array)
│   └── selectedCharts
│
├── Tool State
│   ├── activeTool (select, arrow, textbox, expression)
│   ├── arrowStart
│   └── nodeIdCounter
│
├── UI State
│   ├── uploadPanelOpen
│   ├── variablesPanelOpen
│   ├── showSettings
│   └── reportPanelOpen
│
├── AI Configuration
│   ├── apiKey
│   ├── selectedModel
│   ├── configStatus
│   └── tokenUsage
│
└── Report State
    └── reportSections[]
```

### Local State (Node Level)
Each node type maintains its own local state:
- **ChartNode:** Menu state, filters, AI results, insights
- **TextBoxNode:** Edit mode, text content, height
- **ExpressionNode:** Expression, result, filters, loading

---

## Data Flow

### 1. **Data Upload Flow**
```
User uploads CSV
    ↓
POST /upload → Backend
    ↓
Receives: dataset_id, dimensions, measures
    ↓
Updates global state
    ↓
Variables panel populated
```

### 2. **Chart Creation Flow**
```
User selects dimension + measure
    ↓
POST /charts → Backend
    ↓
Receives: aggregated data
    ↓
Frontend generates Plotly figure
    ↓
Creates ChartNode on canvas
```

### 3. **AI Exploration Flow**
```
User enters natural language query
    ↓
POST /ai-explore → Backend
    ↓
Backend generates pandas code
    ↓
Executes on dataset
    ↓
Returns answer + token usage
    ↓
Updates node state + token counter
```

### 4. **Chart Fusion Flow**
```
User selects 2 charts + clicks Merge
    ↓
POST /fuse → Backend
    ↓
Backend detects fusion pattern
    ↓
Returns merged data + strategy
    ↓
Creates new fused ChartNode
```

### 5. **Report Generation Flow**
```
User clicks "Add to Report" on chart
    ↓
Captures chart as PNG (SVG → Canvas)
    ↓
POST /generate-report-section → Backend
    ↓
LLM generates insights
    ↓
Adds section to reportSections[]
    ↓
Auto-opens report panel
```

---

## Key Features

### 📊 Chart Capabilities
- **Types:** Bar, Pie, Scatter, Line, Histogram, Multi-Bar
- **Interactions:** Zoom, pan, box select, hover tooltips
- **Transformations:** Aggregation changes, type conversion
- **Analysis:** Statistical summaries, AI-powered insights

### 🤖 AI Features
- **Natural Language Queries:** Ask questions about data
- **Code Generation:** Gemini generates pandas code
- **Metric Calculation:** Text-to-number conversion
- **Insight Generation:** Automatic statistical summaries
- **Report Enhancement:** LLM-generated professional summaries

### 🛠️ Tools
- **Select:** Default selection and movement
- **Arrow:** Connect elements visually
- **Text:** Create rich text notes
- **Expression:** Calculate metrics with @Field.Agg syntax

### 📄 Report Mode
- **Document View:** Professional report layout
- **Editable Content:** Title, subheading, sections (markdown)
- **Chart Integration:** PNG images with insights
- **PDF Export:** Browser print to PDF

---

## Canvas Interactions

### Navigation
- **Two-finger scroll:** Pan canvas up/down/left/right
- **Pinch gesture:** Zoom in/out
- **Mouse wheel:** Pan (not zoom)

### Node Interactions
- **Drag:** Move nodes around canvas
- **Click:** Select/deselect nodes
- **Multi-select:** Shift + Click or box selection
- **Context Menu:** Right-click or menu button

### Tool Usage
- **Arrow Tool:** Click start → Click end → Creates arrow
- **Text Tool:** Click canvas → Creates text node
- **Expression Tool:** Click canvas → Creates calculator node

---

## Styling System

### Layout
- **Flexbox:** Main layout (sidebar + canvas + report)
- **Absolute Positioning:** UI overlays (settings, token counter)
- **Responsive:** Canvas resizes with panels

### Theme
- **Colors:** Blue primary (#3182ce, #0ea5e9), gray neutrals
- **Typography:** System fonts, sans-serif
- **Spacing:** Tailwind spacing scale (p-2, p-4, gap-2, etc.)

### Print Styles
- **Visibility Control:** Hide UI, show only report content
- **A4 Format:** 210mm width, 20mm margins
- **Page Breaks:** Avoid breaking sections
- **Clean Output:** Remove interactive elements

---

## Performance Optimizations

### React Optimizations
```javascript
// Memoized node types prevent re-rendering
const nodeTypes = useMemo(() => ({
  chart: (props) => <ChartNode {...props} />,
  // ... other types
}), [dependencies]);

// Callbacks prevent recreation
const handleAction = useCallback((params) => {
  // action logic
}, [dependencies]);
```

### Plotly Cleanup
- **Destroy on unmount:** Remove event listeners
- **Nullify internal refs:** `_hoverlayer`, `_fullLayout`, etc.
- **Prevent memory leaks:** Comprehensive cleanup strategy

### Canvas Performance
- **Node memoization:** Prevent unnecessary re-renders
- **Selective updates:** Only update changed nodes
- **Event propagation:** Stop bubbling for interactive elements

---

## Error Handling

### User-Facing Errors
- **API failures:** Alert messages with clear descriptions
- **Validation errors:** Inline error messages
- **Missing configuration:** Prompt to configure settings

### Silent Handling
- **Token estimation:** Fallback if usage_metadata unavailable
- **Plotly errors:** `onError` callback with console logging
- **Code execution:** Try-catch with error display

---

## Security Considerations

### Input Validation
- **File uploads:** CSV format only
- **Expression syntax:** Validated against allowed patterns
- **API requests:** Type-safe with Pydantic models

### Code Execution
- **Backend only:** Python code never executed in browser
- **Sandboxed environment:** Limited globals namespace
- **No file access:** Isolated execution context

### API Security
- **User-provided keys:** Stored in localStorage
- **No hardcoded credentials:** API keys from user settings
- **CORS enabled:** Development mode only

---

## Future Extensibility

### Plugin Architecture Potential
- **Custom node types:** Register new visualization types
- **Tool plugins:** Add custom tools to sidebar
- **Export formats:** Additional export handlers

### API Integration
- **Multiple backends:** Support different data sources
- **LLM providers:** Switch between AI providers
- **Collaboration:** Real-time sync capabilities

### Enhancement Areas
- **Undo/redo:** Canvas action history
- **Templates:** Pre-built analysis workflows
- **Saved workspaces:** Persist entire canvas state
- **Data connectors:** Direct database connections

---

## Development Guidelines

### Adding New Node Types
1. Create node component in `App.jsx`
2. Add to `nodeTypes` useMemo with dependencies
3. Implement `data` prop interface
4. Handle cleanup in useEffect
5. Add tool button to UnifiedSidebar
6. Update `onPaneClick` handler

### Adding AI Features
1. Define Pydantic model in `backend/app.py`
2. Create API endpoint with error handling
3. Add GeminiDataFormulator method if needed
4. Create frontend handler in ReactFlowWrapper
5. Add UI trigger (button, menu item)
6. Update token usage tracking

### Styling Conventions
- Use Tailwind utility classes
- Follow spacing scale (2, 4, 6, 8)
- Maintain color consistency
- Add print styles where needed
- Test responsive behavior

---

## Dependencies

### Production Dependencies
```json
{
  "react": "^18.x",
  "react-dom": "^18.x",
  "react-flow-renderer": "^10.x",
  "react-plotly.js": "^2.x",
  "plotly.js": "^2.x",
  "@tiptap/react": "^2.x",
  "@tiptap/starter-kit": "^2.x",
  "@radix-ui/react-*": "Multiple components",
  "lucide-react": "^0.x",
  "marked": "^9.x"
}
```

### Development Tools
- **Create React App** - Project bootstrapping
- **PostCSS** - CSS processing
- **Tailwind CSS** - Utility-first styling

---

## File Structure

```
frontend/
├── src/
│   ├── App.jsx                 # Main application component (4886 lines)
│   │   ├── Node Components     # ChartNode, TextBoxNode, etc.
│   │   ├── UI Components       # UnifiedSidebar, ReportPanel, etc.
│   │   ├── State Management    # All React hooks and state
│   │   └── Event Handlers      # Canvas interactions, API calls
│   │
│   ├── index.js                # React entry point
│   ├── index.css               # Global styles + print styles
│   ├── tiptap-styles.css       # Rich text editor styles
│   │
│   ├── components/ui/          # Reusable UI components
│   │   ├── button.jsx
│   │   ├── card.jsx
│   │   ├── input.jsx
│   │   ├── dropdown-menu.jsx
│   │   ├── select.jsx
│   │   └── ... (more components)
│   │
│   └── lib/
│       └── utils.js            # Utility functions (e.g., cn for classNames)
│
├── public/
│   └── index.html              # HTML template
│
├── package.json                # Dependencies
├── tailwind.config.js          # Tailwind configuration
└── components.json             # Shadcn UI configuration
```

---

## Conclusion

D.Fuse's frontend architecture is built around a flexible, node-based canvas system powered by React Flow, with deep integration of AI capabilities through Gemini LLM. The architecture emphasizes:

- **Modularity:** Each node type is self-contained
- **Extensibility:** Easy to add new features and node types
- **Performance:** Optimized rendering and cleanup
- **User Experience:** Intuitive interactions and visual feedback
- **AI-First:** Natural language as a primary interface

The combination of visual data exploration, AI-powered analysis, and professional report generation makes D.Fuse a powerful tool for interactive data analysis.

