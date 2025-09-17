#!/bin/bash

# Soccer Prediction Docker Runner Script

echo "Soccer Prediction Docker Setup"
echo "================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t soccer-prediction .

# Stop any existing container
echo "Stopping any existing container..."
docker stop soccer-app 2>/dev/null || true
docker rm soccer-app 2>/dev/null || true

if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
else
    echo "Failed to build Docker image"
    exit 1
fi

# Run the container
echo "Starting Soccer Prediction app..."
echo "App will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the app"
echo ""

docker run -p 8000:8000 --name soccer-app soccer-prediction
