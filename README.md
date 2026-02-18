# 📘 SYMBIOTIC-TWIN
## Federated Multi-Agent Cognitive Digital Twin Framework for Secure Edge Intelligence

---

## 🚀 Project Overview

SYMBIOTIC-TWIN is a distributed AI framework that combines:

- Federated Learning
- Multi-Agent Edge Simulation
- Cognitive Optimization
- Adaptive Aggregation
- Structured Logging
- Performance Monitoring Dashboard

The system simulates edge-based digital twins that collaboratively train a global model without sharing raw data, ensuring privacy, scalability, and efficiency.

---

## 🏗 System Architecture

The system consists of:

- 1 Federated Server
- 3 Edge Nodes (Docker containers)
- 1 Streamlit Dashboard
- Centralized YAML Configuration
- Structured Logging System

### High-Level Workflow

```
Edge 1   \
Edge 2    --->  Federated Server  --->  Aggregation  ---> Global Model
Edge 3   /
```

Dashboard monitors:
- Accuracy
- Latency
- Energy Consumption
- Node activity logs

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
│   └── serialization.py
│
├── data/
├── logs/
├── docker-compose.yml
├── requirements.txt
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

**Example:**

```yaml
system:
  num_edges: 3
  num_rounds: 10
  learning_rate: 0.001

aggregation:
  adaptive_weighting: true
```

### `config/loader.py`

Loads YAML configuration globally:

```python
config = load_config()
```

Used across server, edge, and dashboard modules.

### `config/logging_config.py`

Provides centralized logging configuration:

- File-based logging
- Console logging
- Timestamped format
- Separate log file per node

---

## 🧠 Server Module

### `server/main.py`

Entry point for FastAPI server.

**Responsibilities:**
- Initialize API
- Start aggregation service
- Manage global model lifecycle

### `server/routes.py`

Defines API endpoints:

- `/update` → Receives model updates from edges
- `/global-model` → Sends aggregated model

### `server/aggregator.py`

Implements:
- FedAvg algorithm
- Adaptive weighted aggregation
- Reputation-aware contribution

### `server/reputation.py`

Maintains node trust score based on:
- Historical performance
- Contribution quality
- Model divergence

### `server/model_manager.py`

Handles:
- Global model storage
- Serialization/deserialization
- Version tracking

---

## 🌐 Edge Module

Each edge simulates a **Digital Twin Agent**.

### `edge/main.py`

Edge container entry point.

**Workflow:**
1. Load local data
2. Train local model
3. Send weights to server
4. Receive global model
5. Repeat for multiple rounds

### `edge/trainer.py`

Handles:
- Local training
- Backpropagation
- Accuracy computation

### `edge/model.py`

Defines neural network architecture.

### `edge/cognitive_layer.py`

Implements:
- Learning rate adjustment
- Multi-objective optimization
- Energy-aware tuning

### `edge/communication.py`

Handles HTTP communication with server:
- Send model updates
- Fetch global model

### `edge/data_loader.py`

Loads partitioned local dataset (Non-IID simulation).

---

## 📊 Metrics Module

### `metrics/accuracy.py`

Computes model accuracy:

$$\text{Accuracy} = \frac{\text{Correct}}{\text{Total}} \times 100$$

### `metrics/latency.py`

Measures inference time using `time.time()`:

$$\text{Latency (ms)} = \text{End Time} - \text{Start Time}$$

### `metrics/energy.py`

Simulates energy usage:

$$\text{Energy} \propto \text{Computation Time} \times \text{Model Complexity}$$

### `metrics/tracker.py`

Stores:
- Global accuracy
- Per-edge accuracy
- Latency per round
- Energy metrics

Used by the Streamlit dashboard.

---

## 📈 Dashboard Module

### `dashboard/app.py`

Streamlit-based UI showing:

- Model Accuracy Comparison
- Inference Latency Graph
- Energy Consumption Graph
- Federated Round Progress
- Node Health Status

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
- Transparency
- Debugging capability
- Demonstration clarity

---

## 🔄 Federated Learning Workflow

1. Server initializes global model
2. Server broadcasts model to edges
3. Each edge:
   - Trains locally
   - Computes accuracy
   - Sends weights
4. Server:
   - Aggregates weights
   - Updates global model
   - Logs performance
5. Repeat for N rounds

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🐳 Run Using Docker

```bash
docker-compose up --build
```

---

## 📊 Performance Metrics

The system evaluates:

- Model Accuracy (%)
- Inference Latency (ms)
- Energy Consumption (J)
- Convergence Speed
- Communication Overhead

---

## 🎯 Key Features

- ✔ Federated Learning Implementation
- ✔ Multi-Agent Edge Simulation
- ✔ YAML-Based Configuration
- ✔ Adaptive Aggregation
- ✔ Structured Logging
- ✔ Real-Time Dashboard
- ✔ Dockerized Architecture

---

## 🎓 Academic Contribution

This project demonstrates how distributed cognitive digital twins can collaboratively learn at the edge while maintaining privacy, reducing latency, and improving energy efficiency.