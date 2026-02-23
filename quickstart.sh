#!/bin/bash

# SYMBIOTIC-TWIN Quick Start Script
# This script automatically sets up everything needed to run the project

set -e

echo "🚀 SYMBIOTIC-TWIN Docker - Quick Start"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if Docker is running
echo -e "${BLUE}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop."
    exit 1
fi

if ! docker ps > /dev/null 2>&1; then
    echo "❌ Docker daemon is not running. Please start Docker Desktop."
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Create required directories
echo -e "${BLUE}Creating required directories...${NC}"
mkdir -p logs data
echo -e "${GREEN}✓ Created logs/ and data/${NC}"
echo ""

# Display current status
echo -e "${BLUE}Current configuration:${NC}"
echo "  Python:       3.10"
echo "  PyTorch:      2.1.0 (CPU)"
echo "  FastAPI:      0.104.1"
echo "  Streamlit:    1.28.1"
echo ""

# Build or use cache
echo -e "${BLUE}Building Docker images...${NC}"
echo "(This may take 5-10 minutes on first run)"
echo ""
docker-compose build --no-cache 2>&1 | tail -20
echo ""

# Start services
echo -e "${BLUE}Starting services...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Wait for services to be ready
echo -e "${BLUE}Waiting for services to become healthy...${NC}"
for i in {1..30}; do
    if docker-compose ps | grep -q "healthy"; then
        echo -e "${GREEN}✓ Server is healthy${NC}"
        break
    fi
    sleep 1
done
echo ""

# Display status
echo -e "${BLUE}Service Status:${NC}"
docker-compose ps
echo ""

# Display access information
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""
echo -e "${BLUE}Access your services:${NC}"
echo "  📖 API Documentation: http://localhost:8000/docs"
echo "  🔗 API Server:        http://localhost:8000"
echo "  💻 Dashboard:         http://localhost:8501"
echo "  💚 Health Check:      http://localhost:8000/health"
echo ""

echo -e "${BLUE}Useful commands:${NC}"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose stop"
echo "  View status:      docker-compose ps"
echo "  Execute command:  docker-compose exec server bash"
echo ""

echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Open http://localhost:8501 in your browser"
echo "  2. Check http://localhost:8000/docs for API"
echo "  3. View logs: docker-compose logs -f"
echo ""

echo -e "${BLUE}For more help:${NC}"
echo "  Setup guide:      cat DOCKER_SETUP_GUIDE.md"
echo "  Command reference: cat DOCKER_COMMANDS.md"
echo "  Changes summary:   cat CHANGES_SUMMARY.md"
echo ""
