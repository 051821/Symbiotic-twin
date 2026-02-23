#!/bin/bash

# SYMBIOTIC-TWIN Docker Configuration Verification Script
# Run this after downloading the updated files to ensure everything is correct

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SYMBIOTIC-TWIN Docker Verification${NC}"
echo -e "${BLUE}========================================${NC}\n"

ERRORS=0
WARNINGS=0

# Function to check file exists
check_file() {
    local file=$1
    local description=$2
    
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description (MISSING: $file)"
        ((ERRORS++))
        return 1
    fi
}

# Function to check directory exists
check_dir() {
    local dir=$1
    local description=$2
    
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description (MISSING: $dir)"
        ((ERRORS++))
        return 1
    fi
}

# Function to check file content
check_content() {
    local file=$1
    local search_string=$2
    local description=$3
    
    if grep -q "$search_string" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $description"
        return 0
    else
        echo -e "${RED}✗${NC} $description (NOT FOUND: \"$search_string\" in $file)"
        ((ERRORS++))
        return 1
    fi
}

# Function to extract version
get_version() {
    local file=$1
    local package=$2
    grep "^${package}==" "$file" | head -1 | cut -d'=' -f3
}

echo -e "${BLUE}Checking Files...${NC}\n"

# Check main files exist
check_file "Dockerfile.server" "Server Dockerfile"
check_file "Dockerfile.edge" "Edge Dockerfile"
check_file "Dockerfile.dashboard" "Dashboard Dockerfile"
check_file "docker-compose.yml" "Docker Compose file"
check_file "requirements-server.txt" "Server requirements"
check_file "requirements-edge.txt" "Edge requirements"
check_file "requirements-dashboard.txt" "Dashboard requirements"

echo ""
echo -e "${BLUE}Checking Python Version (should be 3.10)...${NC}\n"

check_content "Dockerfile.server" "python:3.10-slim" "Server uses Python 3.10"
check_content "Dockerfile.edge" "python:3.10-slim" "Edge uses Python 3.10"
check_content "Dockerfile.dashboard" "python:3.10-slim" "Dashboard uses Python 3.10"

echo ""
echo -e "${BLUE}Checking Version Pinning (no >= operators)...${NC}\n"

# Check for loose version specifiers
if ! grep -E ">=|<=|~=" requirements-server.txt > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Server requirements are fully pinned"
else
    echo -e "${YELLOW}⚠${NC} Server requirements may have loose versions"
    grep -E ">=|<=|~=" requirements-server.txt || true
fi

if ! grep -E ">=|<=|~=" requirements-edge.txt > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Edge requirements are fully pinned"
else
    echo -e "${YELLOW}⚠${NC} Edge requirements may have loose versions"
    grep -E ">=|<=|~=" requirements-edge.txt || true
fi

if ! grep -E ">=|<=|~=" requirements-dashboard.txt > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Dashboard requirements are fully pinned"
else
    echo -e "${YELLOW}⚠${NC} Dashboard requirements may have loose versions"
    grep -E ">=|<=|~=" requirements-dashboard.txt || true
fi

echo ""
echo -e "${BLUE}Checking Package Versions...${NC}\n"

# Check critical versions
echo "Server dependencies:"
check_content "requirements-server.txt" "fastapi==0.104.1" "  FastAPI 0.104.1"
check_content "requirements-server.txt" "uvicorn.*0.24.0" "  Uvicorn 0.24.0"

echo ""
echo "Edge dependencies:"
check_content "requirements-edge.txt" "torch==2.1.0" "  PyTorch 2.1.0 (CPU)"
check_content "requirements-edge.txt" "torchvision==0.16.0" "  TorchVision 0.16.0"
check_content "requirements-edge.txt" "numpy==1.24.3" "  NumPy 1.24.3"

echo ""
echo "Dashboard dependencies:"
check_content "requirements-dashboard.txt" "streamlit==1.28.1" "  Streamlit 1.28.1"
check_content "requirements-dashboard.txt" "plotly==5.17.0" "  Plotly 5.17.0"

echo ""
echo -e "${BLUE}Checking Docker Best Practices...${NC}\n"

check_content "Dockerfile.server" "PYTHONUNBUFFERED=1" "Server has PYTHONUNBUFFERED"
check_content "Dockerfile.edge" "PYTHONUNBUFFERED=1" "Edge has PYTHONUNBUFFERED"
check_content "Dockerfile.dashboard" "PYTHONUNBUFFERED=1" "Dashboard has PYTHONUNBUFFERED"

check_content "Dockerfile.server" "pip install --upgrade pip setuptools wheel" "Server upgrades pip/setuptools/wheel"
check_content "Dockerfile.edge" "pip install --upgrade pip setuptools wheel" "Edge upgrades pip/setuptools/wheel"
check_content "Dockerfile.dashboard" "pip install --upgrade pip setuptools wheel" "Dashboard upgrades pip/setuptools/wheel"

check_content "Dockerfile.edge" "download.pytorch.org/whl/cpu" "Edge uses PyTorch CPU index"

check_content "Dockerfile.server" "HEALTHCHECK" "Server has health check"

echo ""
echo -e "${BLUE}Checking Docker Compose Configuration...${NC}\n"

check_content "docker-compose.yml" "version: \"3.9\"" "Docker Compose 3.9 format"
check_content "docker-compose.yml" "service_healthy" "Proper healthcheck dependencies"
check_content "docker-compose.yml" "restart: unless-stopped" "Restart policies configured"
check_content "docker-compose.yml" "symbiotic-net" "Bridge network configured"

echo ""
echo -e "${BLUE}Checking Directory Structure...${NC}\n"

check_dir "server" "Server directory exists"
check_dir "edge" "Edge directory exists"
check_dir "dashboard" "Dashboard directory exists"
check_dir "shared" "Shared directory exists"
check_dir "config" "Config directory exists"

echo ""
echo -e "${BLUE}Checking Documentation (optional)...${NC}\n"

if [ -f "DOCKER_SETUP_GUIDE.md" ]; then echo -e "${GREEN}✓${NC} Docker Setup Guide"; else echo -e "${YELLOW}⚠${NC} Docker Setup Guide (optional)"; ((WARNINGS++)); fi
if [ -f "DOCKER_COMMANDS.md" ]; then echo -e "${GREEN}✓${NC} Docker Commands Reference"; else echo -e "${YELLOW}⚠${NC} Docker Commands Reference (optional)"; ((WARNINGS++)); fi
if [ -f "CHANGES_SUMMARY.md" ]; then echo -e "${GREEN}✓${NC} Changes Summary"; else echo -e "${YELLOW}⚠${NC} Changes Summary (optional)"; ((WARNINGS++)); fi

echo ""
echo -e "${BLUE}Checking Docker Ignore...${NC}\n"

if [ -f ".dockerignore" ]; then
    if grep -q "logs" ".dockerignore" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} .dockerignore excludes logs"
    else
        echo -e "${YELLOW}⚠${NC} .dockerignore doesn't explicitly exclude logs"
        ((WARNINGS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} .dockerignore file not found (optional but recommended)"
fi

echo ""
echo -e "${BLUE}Validating Docker Compose Syntax...${NC}\n"

if command -v docker-compose &> /dev/null; then
    if docker-compose config > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} docker-compose.yml syntax valid"
    else
        echo -e "${RED}✗${NC} docker-compose.yml has syntax errors"
        ((ERRORS++))
    fi
else
    echo -e "${YELLOW}⚠${NC} docker-compose not found in PATH (skipping validation)"
    echo "          Install Docker Desktop or docker-compose CLI to validate"
    ((WARNINGS++))
fi

echo ""
echo -e "${BLUE}========================================${NC}"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
fi

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s)${NC}"
fi

echo -e "${BLUE}========================================${NC}\n"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}Configuration Status: READY TO USE${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Create required directories: mkdir -p logs data"
    echo "2. Build images: docker-compose build --no-cache"
    echo "3. Start services: docker-compose up -d"
    echo "4. Check status: docker-compose ps"
    echo ""
    echo "For more information, see DOCKER_SETUP_GUIDE.md"
    echo ""
    exit 0
else
    echo -e "${RED}Configuration Status: NEEDS ATTENTION${NC}"
    echo ""
    echo "Please fix the errors listed above before proceeding."
    echo "For help, see DOCKER_SETUP_GUIDE.md"
    echo ""
    exit 1
fi
