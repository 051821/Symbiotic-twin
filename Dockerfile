FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip safely
RUN pip install --upgrade pip \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# Copy requirements first
COPY requirements.txt .

# Install dependencies from PyPI + PyTorch CPU
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org

# Copy ENTIRE project (this fixes config module issue)
COPY . .

# Create runtime dirs
RUN mkdir -p logs data/processed data/partitions

CMD ["python", "-m", "server.main"]
