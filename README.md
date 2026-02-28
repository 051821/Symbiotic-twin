# 📘 SYMBIOTIC-TWIN
## Federated Multi-Agent Cognitive Digital Twin Framework for Secure Edge Intelligence

---

## 🚀 Project Overview

SYMBIOTIC-TWIN is a distributed federated learning framework that simulates intelligent IoT edge devices (Digital Twins) collaboratively training a global AI model **without sharing raw data**.

The system combines:

- Federated Learning (FedAvg)
- Multi-Agent Edge Simulation
- Cognitive Optimization
- Adaptive Aggregation
- Structured Logging
- Real-Time Performance Dashboard
- YAML-Based Configuration
- Dockerized Architecture

---

## 🏗 System Architecture

The system consists of:

- 3 Edge Nodes (Simulated IoT / Docker containers)
- 1 Federated Server
- 1 Streamlit Dashboard
- Shared Model Layer
- Metrics Tracking Layer
- Centralized YAML Configuration
- Structured Logging System

### High-Level Workflow

```
Edge 1  \
Edge 2   --->  Federated Server  --->  Aggregation  --->  Global Model
Edge 3  /
```

**Dashboard monitors:**
- Accuracy
- Latency
- Energy Consumption
- Aggregation Weights
- Node Activity Logs

---

## 🛠 Tools & Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| PyTorch | Model training and inference |
| FastAPI | Federated server API |
| Uvicorn | ASGI server runtime |
| Streamlit | Performance dashboard |
| Docker | Multi-node simulation |
| PyYAML | Configuration management |
| Matplotlib / Plotly | Visualization |
| Logging module | Structured logging |

---

## 🧠 Federated Learning Logic

**Each edge:**
1. Loads device-specific data
2. Trains model locally
3. Sends model weights to server
4. Receives updated global model
5. Repeats for N rounds

**The server:**
1. Collects local model updates
2. Computes aggregation weights
3. Applies weighted averaging
4. Updates global model
5. Broadcasts back to edges

---

## 📊 Federated Aggregation Weights

The federated averaging weight formula:

$$w_i = \frac{n_i}{\sum_{k=1}^{N} n_k}$$

Where:
- $n_i$ = number of samples in edge $i$
- $\sum n_k$ = total samples across all edges
- $w_i$ = contribution weight of edge $i$

The global model update:

$$\theta_{global} = \sum_{i=1}^{N} w_i \cdot \theta_i$$

Where $\theta_i$ = local model weights from edge $i$.

### 📈 Example From This Dataset

| Edge | Training Samples | Weight |
|------|-----------------|--------|
| Edge 1 | 149,960 | 0.46 |
| Edge 2 | 89,452 | 0.28 |
| Edge 3 | 84,734 | 0.26 |

$$w_1 = \frac{149960}{324146} \approx 0.46 \quad w_2 = \frac{89452}{324146} \approx 0.28 \quad w_3 = \frac{84734}{324146} \approx 0.26$$

This demonstrates:
- ✔ Non-IID distribution
- ✔ Weighted aggregation
- ✔ Edge contribution transparency

---

## 📂 Project Structure

```
symbiotic-twin/
│
├── config/
│   ├── config.yaml
│   ├── loader.py
│   └── logging_config.py
│
├── server/
│   ├── main.py
│   ├── routes.py
│   ├── aggregator.py
│   ├── reputation.py
│   ├── model_manager.py
│   └── utils.py
│
├── edge/
│   ├── main.py
│   ├── trainer.py
│   ├── model.py
│   ├── cognitive_layer.py
│   ├── communication.py
│   └── data_loader.py
│
├── dashboard/
│   └── app.py
│
├── metrics/
│   ├── accuracy.py
│   ├── latency.py
│   ├── energy.py
│   └── tracker.py
│
├── shared/
│   ├── model.py
│   ├── serialization.py
│   └── utils.py
│
├── data/
│   ├── preprocess.py
│   └── partition.py
│
├── scripts/
│   └── mock_feeder.py
│
├── logs/
├── data/                    # Dataset & partitions (created at runtime)
│
├── docker-compose.yml       # Orchestrates server, edge1–3, dashboard
├── Dockerfile.server        # Federated server image (FastAPI/Uvicorn)
├── Dockerfile.edge          # Edge node image (PyTorch, training)
├── Dockerfile.dashboard     # Streamlit dashboard image
├── requirements-server.txt  # Server dependencies
├── requirements-edge.txt   # Edge dependencies (PyTorch CPU, etc.)
├── requirements-dashboard.txt
├── .dockerignore
├── quickstart.sh            # One-command Docker setup
├── verify-docker-setup.sh   # Validate Docker config before run
├── optimize-and-build.sh    # Clean build & optional optimizations
└── README.md
```

---

## ⚙ Configuration System

### `config/config.yaml`

Central configuration file controlling:
- Number of edges
- Number of federated rounds
- Learning rate
- Batch size
- Aggregation settings
- Logging level
- Dataset paths

```yaml
system:
  num_edges: 3
  num_rounds: 10
  learning_rate: 0.001

aggregation:
  adaptive_weighting: true
```

### `config/loader.py`

Loads YAML configuration globally across all modules:

```python
config = load_config()
```

### `config/logging_config.py`

Provides centralized logging configuration:
- File-based logging
- Console logging
- Timestamped format
- Separate log file per node

---

## 📁 Module Responsibilities

### 🔹 `shared/`

Contains components shared by both edge nodes and the server.

| File | Responsibility |
|------|---------------|
| `model.py` | Defines neural network architecture — Input: 7 IoT features, Output: 3 classes (Normal / Warning / Critical) |
| `serialization.py` | Handles model `state_dict` conversion, tensor-to-dict transformation, and safe communication format |
| `utils.py` | Common utility functions |

---

### 🌐 `edge/`

Simulates Digital Twin IoT agents. Each edge container runs independently.

#### `edge/main.py`

Edge container entry point.

**Workflow:**
1. Load device partition
2. Train locally
3. Compute metrics
4. Send weights to server
5. Receive global model

#### `edge/trainer.py`

Implements:
- Forward pass
- Loss calculation
- Backpropagation
- Accuracy evaluation

#### `edge/data_loader.py`

Loads device-specific partition from the processed dataset using `get_edge_partition(device_id)`.

#### `edge/communication.py`

Handles HTTP communication with the server:
- `POST` model updates
- `GET` global model

#### `edge/cognitive_layer.py`

Implements intelligent adjustments:
- Learning rate tuning
- Energy-aware adaptation
- Multi-objective optimization

---

### 🧠 `server/`

Federated coordination layer.

#### `server/main.py`

Starts the FastAPI server and manages the global model lifecycle.

#### `server/routes.py`

Defines API endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/update` | Receive edge model weights |
| `/global-model` | Send aggregated global model |
| `/weights` | Return aggregation weights |

#### `server/aggregator.py`

Implements:
- Standard **FedAvg** algorithm
- Adaptive weighted aggregation
- Reputation-aware contribution

#### `server/model_manager.py`

Maintains:
- Global model state
- Model versioning
- State synchronization across rounds

#### `server/reputation.py`

Tracks:
- Edge contribution quality
- Trust scoring
- Potential anomaly detection

---

### 📊 `metrics/`

Responsible for system evaluation.

#### `metrics/accuracy.py`

$$\text{Accuracy} = \frac{\text{Correct}}{\text{Total}} \times 100$$

#### `metrics/latency.py`

$$\text{Latency (ms)} = \text{End Time} - \text{Start Time}$$

#### `metrics/energy.py`

$$\text{Energy} \propto \text{Computation Time} \times \text{Model Complexity}$$

#### `metrics/tracker.py`

Stores per-round metrics used by the Streamlit dashboard:
- Accuracy (global and per-edge)
- Latency
- Energy consumption
- Aggregation weights

---

### 📁 `data/`

Data processing layer.

#### `data/preprocess.py`

Performs:
- Timestamp conversion
- Label creation (Normal / Warning / Critical)
- Feature normalization
- Boolean conversion

#### `data/partition.py`

Partitions dataset by device ID, ensuring:
- ✔ Natural Non-IID distribution
- ✔ Edge-specific data isolation
- ✔ Train/test split
- ✔ PyTorch tensor conversion

---

### 📈 `dashboard/`

#### `dashboard/app.py`

Streamlit-based UI displaying:
- Model Accuracy (%) comparison
- Inference Latency graph
- Energy Consumption graph
- Federated Aggregation Weights & bar chart
- Training round progress
- Node health status

---

## 📝 Logging System

Each node generates its own log file:

```
logs/server.log
logs/edge1.log
logs/edge2.log
logs/edge3.log
```

**Log format:**

```
2026-02-18 14:32:11 | INFO | edge1 | Training started | Round: 3
```

Provides:
- Transparency across all nodes
- Debugging capability
- Demonstration clarity

---

## 🔄 Federated Learning Workflow

```
1. Server initializes global model
        ↓
2. Server broadcasts model to all edges
        ↓
3. Each edge trains locally + computes accuracy + sends weights
        ↓
4. Server aggregates weights → updates global model → logs performance
        ↓
5. Repeat for N rounds
```

---

## 📦 Installation

**Recommended: run with Docker** (see below). No local Python install required.

For local development without Docker, install dependencies per component:

```bash
# Server (API)
pip install -r requirements-server.txt

# Edge (training; use Python 3.10, PyTorch CPU)
pip install -r requirements-edge.txt

# Dashboard (Streamlit)
pip install -r requirements-dashboard.txt
```

Ensure directories exist: `mkdir -p logs data`. Preprocess and partition data as needed (see `data/`).

---

## 🐳 Run Using Docker

**Prerequisites:** Docker and Docker Compose installed and running (e.g. Docker Desktop).

The stack runs five containers: one **server** (FastAPI), three **edge** nodes (PyTorch CPU training), and one **dashboard** (Streamlit). Images are built from `Dockerfile.server`, `Dockerfile.edge`, and `Dockerfile.dashboard` (Python 3.10).

### Option 1: Quick start (recommended)

```bash
chmod +x quickstart.sh
./quickstart.sh
```

This checks Docker, creates `logs/` and `data/`, builds images, starts all services, and prints access URLs.

### Option 2: Manual steps

```bash
# Optional: verify Docker setup
./verify-docker-setup.sh

# Create directories
mkdir -p logs data

# Build and start
docker-compose up --build
```

Run in background: `docker-compose up -d --build`.

### Access URLs

| Service        | URL                        |
|----------------|----------------------------|
| API server     | http://localhost:8000       |
| API docs       | http://localhost:8000/docs |
| Health check   | http://localhost:8000/health |
| Dashboard      | http://localhost:8501      |

### Useful commands

```bash
docker-compose ps          # Status
docker-compose logs -f     # Follow logs
docker-compose stop        # Stop services
docker-compose down        # Stop and remove containers
```

---

## 📊 Performance Metrics

The system evaluates:

| Metric | Unit |
|--------|------|
| Model Accuracy | % |
| Inference Latency | ms |
| Energy Consumption | J |
| Convergence Speed | rounds |
| Aggregation Weights | ratio |
| Communication Overhead | bytes |

---

## 🎯 Key Features

| Feature | Status |
|---------|--------|
| Federated Learning (FedAvg) | ✔ |
| Real IoT Telemetry Dataset | ✔ |
| Device-Based Natural Partitioning | ✔ |
| Multi-Agent Edge Simulation | ✔ |
| Adaptive & Weighted Aggregation | ✔ |
| Aggregation Weight Visualization | ✔ |
| YAML-Based Configuration | ✔ |
| Structured Logging | ✔ |
| Real-Time Dashboard | ✔ |
| Dockerized Architecture | ✔ |

---

## 🎓 Academic Contribution

This project demonstrates how distributed cognitive digital twins can collaboratively learn at the edge while maintaining privacy, reducing latency, and improving energy efficiency. Specifically, it showcases:

- Realistic **Non-IID federated learning** using real IoT telemetry
- **Device-based data partitioning** preserving natural data heterogeneity
- **Weighted model aggregation** proportional to edge sample counts
- **Transparent contribution analysis** via the dashboard
- **Modular distributed architecture** suitable for production deployment