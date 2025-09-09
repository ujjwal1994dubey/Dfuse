# D.fuse - Data Exploration Tool POC

D.fuse is a Proof of Concept for a data exploration tool that allows users to upload CSV files, create interactive charts on an infinite canvas, and fuse charts together based on intelligent fusion rules.

## Features

- **CSV Upload**: Upload and explore CSV datasets
- **Infinite Canvas**: Pan, zoom, and arrange charts on an infinite canvas using React Flow
- **Interactive Chart Creation**: Select dimensions and measures to create various chart types
- **Chart Fusion**: Connect two charts to create fused visualizations with intelligent rules:
  - **Same dimension + different measures** → Grouped bar charts
  - **Same measure + different dimensions** → Multi-series line charts
- **Real-time Visualization**: Charts powered by Plotly.js with interactive features

## Architecture

### Backend (Python + FastAPI)
- FastAPI web server with CORS support
- In-memory data storage for datasets and charts
- Pandas for data aggregation and processing
- RESTful API endpoints for upload, chart creation, and fusion

### Frontend (React + React Flow)
- React application with infinite canvas capabilities
- React Flow for node-based chart arrangement
- Plotly.js for interactive chart rendering
- Tailwind CSS for styling

## Project Structure

```
Dfuse 2/
├── backend/
│   ├── app.py              # FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── index.js        # React entry point
│   │   └── index.css       # Tailwind CSS imports
│   ├── package.json        # Node.js dependencies
│   ├── tailwind.config.js  # Tailwind configuration
│   └── postcss.config.js   # PostCSS configuration
├── README.md               # This file
└── .gitignore             # Git ignore rules
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

The backend API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend application will be available at `http://localhost:3000`

## Usage Guide

### 1. Upload CSV Data
- Click the file input in the sidebar
- Select a CSV file from your computer
- The dataset will be uploaded and column names will be displayed

### 2. Create Chart Nodes
- Select one or more **Dimensions** (categorical data for grouping)
- Select one or more **Measures** (numerical data for visualization)
- Click "Create Chart Node" to add a chart to the canvas

### 3. Explore the Canvas
- **Drag** nodes to rearrange them
- **Pan** by dragging the background
- **Zoom** using mouse wheel or controls
- Use the **MiniMap** for navigation
- Use **Controls** for zoom/fit options

### 4. Fuse Charts
- Create two or more chart nodes
- **Drag** from one node's connection point to another
- Fusion will only work if charts follow the fusion rules:
  - **Same dimension + different measures**: Creates grouped bar charts
  - **Same measure + different dimensions**: Creates multi-series line charts
- Invalid fusions will show an error message

### 5. Fusion Rules Example

**Valid Fusion - Same Dimension, Different Measures:**
- Chart 1: Sales by Region (Dimension: Region, Measure: Sales)
- Chart 2: Profit by Region (Dimension: Region, Measure: Profit)
- Fusion Result: Grouped bar chart showing both Sales and Profit by Region

**Valid Fusion - Same Measure, Different Dimensions:**
- Chart 1: Sales by Region (Dimension: Region, Measure: Sales)
- Chart 2: Sales by Product (Dimension: Product, Measure: Sales)
- Fusion Result: Multi-series line chart comparing Sales across Region and Product dimensions

## API Endpoints

### POST /upload
Upload a CSV file and receive dataset metadata.

**Request:** Multipart form data with file
**Response:**
```json
{
  "dataset_id": "uuid",
  "columns": ["col1", "col2", ...],
  "rows": 1000
}
```

### POST /charts
Create a new chart from a dataset.

**Request:**
```json
{
  "dataset_id": "uuid",
  "dimensions": ["column1"],
  "measures": ["column2", "column3"],
  "agg": "sum",
  "title": "My Chart"
}
```

**Response:**
```json
{
  "chart_id": "uuid",
  "dataset_id": "uuid",
  "dimensions": ["column1"],
  "measures": ["column2", "column3"],
  "title": "My Chart",
  "table": [...data rows...]
}
```

### POST /fuse
Fuse two existing charts.

**Request:**
```json
{
  "chart1_id": "uuid1",
  "chart2_id": "uuid2"
}
```

**Response:**
```json
{
  "chart_id": "uuid",
  "title": "Fusion: ...",
  "strategy": {
    "type": "same-dimension-different-measures",
    "suggestion": "grouped-bar | stacked-bar | dual-axis-line"
  },
  "table": [...fused data...]
}
```

### GET /charts/{chart_id}
Retrieve a specific chart by ID.

## Development Notes

- The application uses in-memory storage for simplicity - data will be lost when the backend restarts
- CORS is enabled for all origins in development mode
- The frontend assumes the backend is running on `http://localhost:8000`
- Charts are automatically positioned with a small offset to prevent overlap

## Future Enhancements

- Persistent data storage (database)
- More chart types (scatter plots, heatmaps, etc.)
- Advanced fusion strategies
- Chart export capabilities
- User authentication and workspace management
- Real-time collaboration features

## Troubleshooting

### Backend Issues
- Ensure all Python dependencies are installed: `pip install -r requirements.txt`
- Check that the FastAPI server is running: `curl http://localhost:8000`
- Verify CSV files have proper headers and data types

### Frontend Issues
- Ensure all Node.js dependencies are installed: `npm install`
- Check that the backend is accessible from the frontend
- Clear browser cache if experiencing display issues
- Verify that both servers are running on the correct ports

### Chart Fusion Issues
- Ensure charts come from the same dataset
- Verify fusion rules are met (same dimension + different measures OR same measure + different dimensions)
- Check that charts have at least one dimension or measure selected

## License

This is a Proof of Concept implementation for demonstration purposes.
