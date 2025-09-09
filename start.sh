#!/bin/bash

# D.fuse Startup Script
echo "🚀 Starting D.fuse POC..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

echo "📦 Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start backend server in background
echo "🐍 Starting FastAPI backend on http://localhost:8000..."
uvicorn app:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Setup frontend
cd ../frontend
echo "📦 Setting up frontend..."

# Install Node.js dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Wait a moment for backend to start
sleep 3

# Start frontend server
echo "⚛️  Starting React frontend on http://localhost:3000..."
echo ""
echo "🎉 D.fuse is starting up!"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers."

# Start frontend (this will block)
npm start

# Clean up: kill backend when frontend stops
echo "Stopping backend server..."
kill $BACKEND_PID 2>/dev/null
