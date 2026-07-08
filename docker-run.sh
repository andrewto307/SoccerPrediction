#!/bin/bash
# Soccer Prediction — build & run the API + UI with Docker Compose.
set -e

echo "Soccer Prediction — Docker (API + UI)"
echo "====================================="

# Docker must be running.
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# The API needs API_FOOTBALL_KEY, read from .env via docker-compose's env_file.
if [ ! -f .env ]; then
    echo "No .env found. Create one from the template and add your key:"
    echo "    cp .env.example .env    # then set API_FOOTBALL_KEY"
    exit 1
fi

echo "Building and starting services..."
echo "  UI  -> http://localhost:8501"
echo "  API -> http://localhost:8000  (docs at /docs)"
echo "Press Ctrl+C to stop."
echo ""

# Build both targets and start; --build ensures code changes are picked up.
docker compose up --build
