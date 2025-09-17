# 🐳 Docker Setup for Soccer Prediction System

This document explains how to run the Soccer Prediction System using Docker.

## 📋 Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (optional, for advanced usage)

## 🚀 Quick Start

### Option 1: Using the provided script (Recommended)

```bash
# Make sure you're in the SoccerPrediction directory
cd SoccerPrediction

# Run the Docker setup script
./docker-run.sh
```

### Option 2: Using Docker commands directly

```bash
# Build the Docker image
docker build -t soccer-prediction .

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data soccer-prediction
```

### Option 3: Using Docker Compose

```bash
# Start the application
docker-compose up

# Run in background
docker-compose up -d

# Stop the application
docker-compose down
```

## 🌐 Accessing the Application

Once the container is running, open your web browser and go to:

**http://localhost:8000**

## 📁 Data Persistence

The Docker setup includes your training data and models in the container:

- Training data: `/app/data/X_train.csv`, `/app/data/y_train.csv`
- Test data: `/app/data/X_test.csv`, `/app/data/y_test.csv`
- Models: Will be saved in the container's `/app/data` directory when trained
- Raw data: All season CSV files (07-08 to 19-20) are included

## 🔧 Development Mode

For development with live code changes:

```bash
# Using docker-compose with volume mounting
docker-compose up --build
```

This will:
- Mount your source code for live updates
- Rebuild the image when changes are detected
- Preserve data in the `data/` directory

## 🧪 Running Tests in Docker

```bash
# Run all tests
docker run --rm soccer-prediction python -m pytest tests/ -v

# Run specific test
docker run --rm soccer-prediction python -m pytest tests/test_data_loading.py -v
```

## 📊 Data Processing

To run data processing (cleaning and preprocessing):

```bash
# Using docker-compose profile
docker-compose --profile data-processing up data-processor

# Or directly
docker run --rm -v $(pwd)/data:/app/data soccer-prediction python src/main.py
```

## 🐛 Troubleshooting

### Port already in use
```bash
# Check what's using port 8000
lsof -i :8000

# Use a different port
docker run -p 8001:8000 soccer-prediction
```

### Permission issues
```bash
# Fix file permissions
sudo chown -R $USER:$USER data/
```

### Container won't start
```bash
# Check container logs
docker logs <container_id>

# Run container interactively for debugging
docker run -it soccer-prediction /bin/bash
```

## 📦 Image Details

- **Base Image**: Python 3.11-slim
- **Size**: ~1.5GB (includes all ML libraries)
- **Port**: 8000 (Streamlit default)
- **Health Check**: Built-in health monitoring

## 🔄 Updating the Application

```bash
# Rebuild with latest changes
docker build -t soccer-prediction .

# Or using docker-compose
docker-compose up --build
```

## 🚀 Production Deployment

For production deployment, consider:

1. **Environment Variables**: Set production configs
2. **Resource Limits**: Add memory/CPU constraints
3. **Security**: Use non-root user
4. **Monitoring**: Add logging and metrics
5. **Reverse Proxy**: Use nginx for production

Example production docker-compose.yml:
```yaml
version: '3.8'
services:
  soccer-prediction:
    build: .
    ports:
      - "80:8000"
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
    restart: always
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## 📝 Notes

- The application will automatically load data from the `data/` directory
- Models are trained on first run (may take a few minutes)
- All data and models persist between container restarts
- The container includes all necessary ML libraries (CatBoost, XGBoost, etc.)
