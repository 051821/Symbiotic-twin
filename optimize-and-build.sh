#!/bin/bash
# SYMBIOTIC-TWIN Optimization & Build Script
# Run this after pulling the optimized code

set -e

PROJECT_ROOT="/Users/nitishkumar/Documents/Capstoneproject/Symbiotic-twin"
cd "$PROJECT_ROOT"

echo "======================================================"
echo "SYMBIOTIC-TWIN Docker Optimization & Build"
echo "======================================================"
echo ""

# Step 1: Clean up old artifacts
echo "Step 1: Cleaning up old artifacts..."
echo "  → Removing old images and containers..."
docker-compose down -v --remove-orphans 2>/dev/null || true

echo "  → Removing dangling images..."
docker image prune -af --filter="dangling=true" 2>/dev/null || true

echo "  → Cleaning build cache..."
docker builder prune -af 2>/dev/null || true

echo ""

# Step 2: Delete unnecessary local files
echo "Step 2: Deleting unnecessary local files..."
echo "  → Removing deprecated/duplicate files..."
rm -f requirements.txt
rm -f Dockerfile

echo "  → Removing documentation files (optional)..."
for doc in CHANGES_SUMMARY.md DOCKER_SETUP_GUIDE.md DOCKER_COMMANDS.md FIX_SUMMARY.md README_DOCKER.md INDEX.md; do
    rm -f "$doc" && echo "    ✓ $doc"
done

echo "  → Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""

# Step 3: Build optimized images
echo "Step 3: Building optimized Docker images..."
echo "  (This will take 2-3 minutes...)"
echo ""

docker-compose build --no-cache

echo ""
echo "======================================================"
echo "BUILD COMPLETE"
echo "======================================================"
echo ""

# Step 4: Show image sizes
echo "Final Docker Image Sizes:"
echo "======================================================"
docker images | grep symbiotic-twin || echo "No images found"
echo ""

# Step 5: Report disk usage
echo "Project Disk Usage:"
echo "======================================================"
du -sh . 2>/dev/null | awk '{print "Total: " $1}'
du -sh logs/ 2>/dev/null | awk '{print "Logs: " $1}'
du -sh data/ 2>/dev/null | awk '{print "Data: " $1}'
echo ""

echo "Docker System Usage:"
echo "======================================================"
docker system df
echo ""

# Step 6: Start the stack
echo "Step 4: Starting the stack..."
docker-compose up -d

echo ""
echo "======================================================"
echo "STACK RUNNING"
echo "======================================================"
echo ""
docker-compose ps
echo ""

echo "Quick health checks:"
echo "  Server:    curl http://127.0.0.1:8000/health"
echo "  Dashboard: curl http://127.0.0.1:8501/"
echo "  Docs:      curl http://127.0.0.1:8000/docs"
echo ""
echo "View logs:"
echo "  docker-compose logs -f server"
echo "  docker-compose logs -f edge1"
echo "  docker-compose logs -f dashboard"
echo ""
