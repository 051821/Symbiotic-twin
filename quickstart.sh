#!/bin/bash
# SYMBIOTIC-TWIN v2.0 — Quick Start
set -e
GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'

echo -e "${BLUE}🚀 SYMBIOTIC-TWIN v2.0 — Quick Start${NC}"
echo "======================================"

if ! docker ps > /dev/null 2>&1; then
  echo "❌ Docker is not running. Start Docker Desktop first."
  exit 1
fi
echo -e "${GREEN}✓ Docker running${NC}"

# Preprocess data if not done yet
if [ ! -f "data/processed/processed.csv" ]; then
  echo -e "${BLUE}Preprocessing data...${NC}"
  pip install pandas scikit-learn PyYAML joblib -q
  python -c "import sys; sys.path.insert(0,'.'); from data.preprocess import preprocess; preprocess()"
  echo -e "${GREEN}✓ Data preprocessed${NC}"
fi

mkdir -p logs data/partitions data/processed

echo -e "${BLUE}Building Docker images...${NC}"
docker-compose build --no-cache 2>&1 | tail -15

echo -e "${BLUE}Starting services...${NC}"
docker-compose up -d

echo ""
echo -e "${GREEN}🎉 Done!${NC}"
echo "  📊 Dashboard : http://localhost:8501"
echo "  🔗 API       : http://localhost:8000"
echo "  📖 API Docs  : http://localhost:8000/docs"
echo "  💚 Health    : http://localhost:8000/health"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f"
echo "  docker-compose ps"
echo "  docker-compose down"
