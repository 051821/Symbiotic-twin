# 🤖 SYMBIOTIC-TWIN
### Secure Multi-Agent Federated Learning Framework for IoT Digital Twins

> **8th Grade Capstone Project** — A production-grade distributed AI system that trains machine learning models across multiple IoT edge devices without ever moving raw sensor data to a central server.

---

## 📖 Table of Contents

1. [What This Project Actually Does](#what-this-project-actually-does)
2. [Why Federated Learning?](#why-federated-learning)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [Complete File Guide](#complete-file-guide)
6. [End-to-End Workflow](#end-to-end-workflow)
7. [The Temporal Sliding Window](#the-temporal-sliding-window)
8. [Security Layer](#security-layer)
9. [Multi-Agent System](#multi-agent-system)
10. [Dashboard](#dashboard)
11. [How to Run](#how-to-run)
12. [Key Academic Concepts Demonstrated](#key-academic-concepts-demonstrated)

---

## What This Project Actually Does

SYMBIOTIC-TWIN is a **federated learning system** for IoT environmental monitoring. It has:

- **3 IoT edge devices** (Raspberry Pi-style nodes), each collecting real sensor data
- **1 central server** that coordinates training but never sees raw data
- **1 dashboard** that visualizes everything in real time

Each edge device trains a neural network on its own local sensor readings, then sends only the **model weights** (not the data) to the server. The server combines all the weights into one improved global model using a technique called **FedAvg (Federated Averaging)**. This improved model is then sent back to all edges for the next round of training.

The model classifies every IoT reading into one of three risk classes:

| Class | Trigger | Meaning |
|-------|---------|---------|
| ✅ Normal (0) | All sensors within safe range | Environment is safe |
| ⚠️ Warning (1) | `temp > 90°F` OR `lpg > 0.007` | Potential hazard developing |
| 🚨 Critical (2) | `smoke > 0.10` OR `co > 0.005` | Immediate danger — fire/gas risk |

---

## Why Federated Learning?

Traditional ML requires all data to be uploaded to one server. This is a problem for IoT because:

- **Privacy** — sensor data from a building or home is sensitive
- **Bandwidth** — 405,000 readings × 3 devices = massive data transfer
- **Latency** — real-time danger detection can't wait for a round-trip to the cloud
- **Regulation** — data sovereignty laws may prohibit sending data off-premises

Federated learning solves this: **the data never leaves the device**. Only model weight updates (numbers describing what the model learned) are transmitted. SYMBIOTIC-TWIN takes this further by adding security, reputation scoring, and multi-agent intelligence on top.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SYMBIOTIC-TWIN System                     │
│                                                             │
│  ┌──────────┐   weights   ┌─────────────────────────────┐  │
│  │  Edge 1  │────────────▶│                             │  │
│  │b8:27:eb  │◀────────────│     Federated Server        │  │
│  └──────────┘  global     │  • FedAvg Aggregation       │  │
│                model      │  • Reputation Manager       │  │
│  ┌──────────┐             │  • Security Layer           │  │
│  │  Edge 2  │────────────▶│  • Multi-Agent Orchestrator │  │
│  │00:0f:00  │◀────────────│  • Live Classifier API      │  │
│  └──────────┘             │                             │  │
│                           └────────────┬────────────────┘  │
│  ┌──────────┐                          │                    │
│  │  Edge 3  │────────────▶             │ metrics.json       │
│  │1c:bf:ce  │◀────────────             ▼                    │
│  └──────────┘             ┌─────────────────────────────┐  │
│                           │     Streamlit Dashboard      │  │
│                           │  • Live Classifier           │  │
│                           │  • Federated Metrics         │  │
│                           │  • Agent Panel               │  │
│                           │  • Security Monitor          │  │
│                           │  • Window Visualizer         │  │
│                           └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

Everything runs in Docker containers on a single machine, connected via an internal bridge network called `symbiotic-net`.

---

## Dataset

**IoT Telemetry Data** — 405,184 timestamped sensor readings from 3 real IoT devices.

| Feature | Description | Unit |
|---------|-------------|------|
| `co` | Carbon monoxide concentration | ppm |
| `humidity` | Relative humidity | % |
| `light` | Light sensor | 0/1 (boolean) |
| `lpg` | Liquefied petroleum gas concentration | ppm |
| `motion` | Motion detected | 0/1 (boolean) |
| `smoke` | Smoke concentration | ppm |
| `temp` | Temperature | °F |

Each device has its own MAC address and contributes a different portion of the data — this is what makes the learning **Non-IID (Non-Independent and Identically Distributed)**, which is the realistic and challenging federated learning scenario.

| Device | MAC Address | Approx. Samples |
|--------|-------------|-----------------|
| Edge 1 | b8:27:eb:bf:9d:51 | ~149,960 |
| Edge 2 | 00:0f:00:70:91:0a | ~89,452 |
| Edge 3 | 1c:bf:ce:15:ec:4d | ~84,734 |

---

## Complete File Guide

### `config/`

#### `config/config.yaml`
The single source of truth for all settings. Every Python file reads from here instead of hardcoding values.

```yaml
system:
  num_rounds: 10          # How many federated training rounds to run
  learning_rate: 0.001    # Starting learning rate for Adam optimizer
  batch_size: 64          # Samples per gradient update
  epochs_per_round: 1     # Full passes over the window per round

data:
  window_strategy: "sliding"  # How data is sliced each round
  window_fraction: 0.3        # Each round trains on 30% of device data
  window_step: 0.1            # Window moves 10% forward each round
```

#### `config/loader.py`
Loads `config.yaml` once and caches it. Every other module calls `get_config()` to access settings. Uses a module-level singleton so the file is only read from disk once.

#### `config/logging_config.py`
Sets up Python logging so every module (server, edge1, edge2, edge3, etc.) writes to both the console and its own log file in `logs/`. Called as `setup_logger("edge1")` at the top of each module.

---

### `data/`

#### `data/preprocess.py`
**Run once** before training begins. Reads the raw CSV, cleans it, creates labels, normalizes features, and saves `processed.csv`.

Steps it performs:
1. Parses timestamps and sorts all 405,184 rows chronologically
2. Converts boolean columns (`light`, `motion`) to integers
3. **Creates 3-class labels** using `np.select` — Critical checked first, then Warning, then Normal as default
4. Applies `StandardScaler` to normalize all 7 features to zero mean and unit variance
5. Saves the scaler as `scaler.pkl` (needed later for inverse-transforming predictions)
6. Saves `processed.csv` to `data/processed/`

```python
# Label logic (order matters — Critical is checked first)
conditions = [
    (df["smoke"] > 0.10) | (df["co"] > 0.005),   # → 2 Critical
    (df["temp"]  > 90.0) | (df["lpg"] > 0.007),   # → 1 Warning
]
choices = [2, 1]
labels = np.select(conditions, choices, default=0)  # 0 = Normal
```

#### `data/partition.py`
**The most important data file.** Splits `processed.csv` by device MAC address so each edge only sees its own device's data (Non-IID partitioning). Also implements the **Temporal Sliding Window**.

Key function: `get_edge_partition(device_id, round_num)`

- Loads all rows for that device from `processed.csv`
- The **test set** is always the fixed last 20% of that device's data (held-out, never used for training)
- The **training window** shifts each round based on `window_strategy`:
  - `sliding` — a 30% window slides 10% forward each round
  - `expanding` — window grows from the start each round
  - `full` — uses all training data every round (original behaviour)
- Computes **class weights** for the current window's label distribution to fix the Normal/Warning/Critical imbalance

Returns: `(train_dataset, test_dataset, class_weights_tensor)`

---

### `shared/`
Code shared between edge nodes and the server — both need these.

#### `shared/model.py`
Defines the neural network architecture `IoTClassifier` — a 4-layer feedforward network:

```
Input (7 features)
  → Linear(7→64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64→128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128→64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64→3)
Output: 3 logits (Normal / Warning / Critical)
```

BatchNorm stabilizes training. Dropout prevents overfitting. The same architecture is used on every edge and on the server — this is required for federated averaging to work (you can only average weights from identical architectures).

#### `shared/serialization.py`
Converts PyTorch `state_dict` (model weights) to/from JSON-serializable Python lists so they can be sent over HTTP. `serialize_weights()` turns tensors → lists. `deserialize_weights()` turns lists → tensors.

#### `shared/utils.py`
Common utility functions: `set_seed()` for reproducibility, `ensure_dir()` for creating directories, `flatten_weights()` which concatenates all model weight tensors into one 1D numpy array (used by the reputation manager for cosine similarity).

---

### `edge/`
Everything that runs inside each edge container.

#### `edge/main.py`
The entry point for each edge node. Reads `EDGE_ID` from environment variable (set by Docker Compose), then runs the full federated training loop:

```
For each round 1..10:
  1. Load new temporal window for this round
  2. Update class weights in trainer
  3. Fetch latest global model from server
  4. Train on this round's window (1 epoch)
  5. Evaluate on fixed test set
  6. Cognitive layer adapts learning rate
  7. Send weight update to server
  8. Repeat
```

After all rounds complete, the edge exits cleanly (no infinite restart loop).

#### `edge/trainer.py`
`LocalTrainer` class handles the actual PyTorch training loop. Key features:

- Takes `class_weights` tensor in constructor and passes it to `CrossEntropyLoss` — this ensures the model doesn't just learn to predict "Normal" for everything
- `train(train_loader)` runs forward pass → compute loss → backprop → Adam optimizer step for every batch, every epoch
- `evaluate(test_loader)` runs inference on the fixed held-out test set with `torch.no_grad()` for efficiency
- Returns `(accuracy_%, latency_ms, energy_j)` for tracking

#### `edge/data_loader.py`
Thin wrapper around `data/partition.py`. Calls `get_edge_partition(device_id, round_num)` and wraps the resulting TensorDatasets in PyTorch `DataLoader` objects with shuffling for training and no shuffling for testing.

#### `edge/cognitive_layer.py`
`CognitiveLayer` — an adaptive intelligence layer that automatically adjusts the learning rate between rounds based on observed accuracy trends:

- **Plateau detection** — if accuracy doesn't change more than 0.5% over 3 rounds, multiplies LR by 0.7 (decay)
- **Improvement boost** — if accuracy improves more than 1% in one round, multiplies LR by 1.1
- **Energy throttling** — if total energy consumed exceeds budget, halves the LR
- Learning rate always clamped between `1e-6` and `0.05`

#### `edge/communication.py`
HTTP client for talking to the server. Two main functions:
- `send_update()` — POSTs serialized weights + metrics to `/update`
- `fetch_global_model()` — GETs the latest aggregated model from `/global-model`
- `check_server_health()` — pings `/health` to verify server is ready before starting

---

### `server/`
Everything that runs inside the server container.

#### `server/main.py`
FastAPI application factory. On startup:
1. Builds the global `IoTClassifier` model
2. Initializes reputation scores for all edges at 1.0
3. Calls `initialize_security()` to provision HMAC keys and JWT tokens for each edge
4. Registers all API routes
5. Starts uvicorn HTTP server on port 8000

#### `server/routes.py`
All API endpoints:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check — used by Docker and edges on startup |
| `/global-model` | GET | Returns current global model weights + version number |
| `/update` | POST | Receives weight update from an edge |
| `/classify` | POST | Live inference on 7 sensor values |
| `/agents` | GET | Returns latest multi-agent analysis results |
| `/security` | GET | Returns security event log and norm history |
| `/metrics` | GET | Returns full training metrics history |

The `/update` endpoint pipeline:
1. Rate limiter check (max 5 updates per 60s per edge)
2. HMAC signature verification
3. Stores weights in `_pending_updates`
4. When all 3 edges have submitted → triggers aggregation

The `_trigger_aggregation()` function:
1. Runs poisoning detector — excludes statistically anomalous edges
2. Runs FedAvg or adaptive weighted aggregation on clean edges only
3. Updates global model version
4. Computes **weighted global accuracy** = Σ(edge_acc × sample_count) / total_samples
5. Records metrics via `MetricsTracker`
6. Runs all 4 agents in parallel via `AgentOrchestrator`

#### `server/aggregator.py`
Pure aggregation math. Two strategies:

- **FedAvg** — weight each edge proportional to its sample count: `w_i = n_i / Σn_k`
- **Adaptive** — multiplies sample proportion by reputation score, then re-normalizes

The actual aggregation: `θ_global = Σ w_i · θ_i` — weighted sum of all edge weight tensors.

#### `server/reputation.py`
`ReputationManager` tracks trust scores for each edge (range 0.1 to 1.0). Updated every round using exponential moving average (α=0.3):

```
quality   = 0.5 × cosine_similarity(local, global) + 0.5 × (accuracy / 100)
new_score = 0.3 × quality + 0.7 × previous_score
```

Edges with consistently poor updates or diverging weights gradually lose reputation and contribute less to aggregation.

#### `server/model_manager.py`
`ModelManager` holds the single global `IoTClassifier` instance, tracks its version number, and saves a checkpoint `.pt` file to `logs/checkpoints/` after every aggregation.

---

### `agents/`

#### `agents/orchestrator.py`
Four specialized AI agents that run concurrently (in parallel threads) after every aggregation round, analyzing the federated learning process from different angles:

**AnalystAgent** — Monitors accuracy trends across rounds:
- Classifies trend as `improving`, `plateau`, or `regressing` based on last 3 rounds
- Detects which edges are diverging more than 10% from global accuracy
- Alerts if model is regressing

**AnomalyAgent** — Applies threshold rules to detected sensor readings:
- Counts `fire_risk` (smoke/CO over threshold), `gas_leak` (LPG over threshold), `temp_spike` readings
- Fires alerts for each anomaly type detected
- Maintains anomaly history across rounds

**PredictorAgent** — Linear extrapolation forecasting:
- Uses the trend from recent rounds to forecast next-round accuracy for global model and each edge
- Identifies predicted best and worst performing edges
- Warns if global accuracy is forecast to drop below 50%

**SecurityAgent** — Monitors for adversarial behaviour:
- Tracks L2 norm of each edge's weight update over time
- Flags edges whose norms deviate more than 0.3 from their own history
- Monitors HMAC failures and low reputation scores
- Reports whether the round was secure

The `AgentOrchestrator` runs all four in parallel threads, collects results, and stores them for the dashboard's `/agents` endpoint.

---

### `security/`

#### `security/security_layer.py`
Four security mechanisms:

**HMAC-SHA256 Signing** — Each edge signs its payload with a pre-shared secret key. The server verifies the signature and rejects payloads that are tampered with or replayed more than 60 seconds after signing.

**JWT Tokens** — The server issues a JWT (JSON Web Token) to each edge at startup. Tokens include edge ID, issue time, and expiry. The server can verify these to authenticate which edge is sending an update.

**PoisoningDetector** — After receiving all edge updates in a round, computes the L2 norm of each edge's weight vector and runs Z-score outlier detection. Edges whose norm deviates more than 2.5 standard deviations from the mean are flagged as potential Byzantine/poisoning attackers and **excluded from aggregation**.

**RateLimiter** — Token bucket limiting max 5 update requests per edge per 60-second window. Prevents flooding attacks.

---

### `metrics/`

#### `metrics/accuracy.py`
`compute_batch_accuracy(outputs, labels)` — returns `(correct_count, total_count)` for accumulating across batches. Called inside the training loop after every forward pass.

#### `metrics/latency.py`
`LatencyTimer` context manager — wraps a code block and records wall-clock time in milliseconds. Used in `trainer.py` to time the entire training phase.

#### `metrics/energy.py`
`EnergyMonitor` context manager — estimates energy consumption in Joules using: `Energy = Power × Time`, where power is scaled by the model's parameter count to simulate a real embedded device.

#### `metrics/tracker.py`
`MetricsTracker` — thread-safe store that records per-round metrics (global accuracy, per-edge accuracy/latency/energy, aggregation weights, reputation scores) and persists everything to `logs/metrics.json` after each round. The dashboard reads this file to populate its charts.

---

### `dashboard/`

#### `dashboard/app.py`
The Streamlit dashboard — a complete web application with 5 tabs:

**Tab 1 — Live Classifier**
- 7 sensor sliders + preset buttons (Normal / Warning / Critical / Random)
- Calls `/classify` endpoint in live mode, or uses rule-based local classifier in demo mode
- Shows prediction result with confidence, probability bars for all 3 classes, and reasoning
- Per-sensor threshold progress bars showing how close each reading is to triggering an alert
- Batch simulator: generates N random readings, shows class distribution pie chart and scatter plots

**Tab 2 — Federated Metrics**
- KPI cards for global accuracy and per-edge accuracy with round-over-round deltas
- Line chart: accuracy over rounds (global + all 3 edges)
- Pie chart: latest aggregation weights
- Stacked bar: weight evolution over all rounds
- Latency and energy bar/line charts per edge
- Reputation score history chart
- Node health status table

**Tab 3 — Agent Panel**
- Cards for each of the 4 agents showing status badge (DONE / RUNNING / ERROR)
- Per-agent findings displayed in human-readable form
- Color-coded alert pills for all alerts from the current round

**Tab 4 — Security Monitor**
- Reverse-chronological security event log with color coding (red=failure, yellow=warning, green=ok)
- Weight norm history line chart per edge (anomalies visible as spikes)
- KPI cards: HMAC status, poisoning event count, rate limit hit count

**Tab 5 — Window Visualizer**
- Interactive chart showing exactly which time slice of data each edge trains on each round
- Supports all 3 window strategies (sliding, expanding, full)
- Explanation cards for each strategy

---

### `scripts/`

#### `scripts/mock_feeder.py`
Simulates federated training rounds by writing fake-but-realistic metrics to `logs/metrics.json` at a regular interval. Use this to demo the dashboard without running the full Docker stack:

```bash
python scripts/mock_feeder.py
streamlit run dashboard/app.py
```

---

### Infrastructure Files

#### `docker-compose.yml`
Defines 5 services: `server`, `edge1`, `edge2`, `edge3`, `dashboard`. All connected via `symbiotic-net` bridge network. Edges wait for the server healthcheck to pass before starting. Edges use `restart: "no"` — they exit cleanly after all rounds complete. Server and dashboard use `restart: unless-stopped`.

#### `Dockerfile.server`
Multi-stage build for the server. Stage 1 installs Python packages. Stage 2 copies only the installed packages and source code (`server/`, `shared/`, `config/`, `metrics/`, `agents/`, `security/`). Exposes port 8000.

#### `Dockerfile.edge`
Multi-stage build for edge nodes. Uses `entrypoint-edge.sh` as the CMD — this auto-runs `preprocess.py` if `processed.csv` doesn't exist yet, then starts the edge. All 3 edges use the same image; `EDGE_ID` environment variable determines which edge each container becomes.

#### `Dockerfile.dashboard`
Simple build — installs Streamlit and dependencies. Copies only `dashboard/`. Exposes port 8501.

#### `entrypoint-edge.sh`
Shell script that guards against missing preprocessed data. If `data/processed/processed.csv` doesn't exist, it runs the preprocessing pipeline inside the container before starting the edge training loop.

---

## End-to-End Workflow

```
Step 1 — Startup
  docker-compose up
  → Server starts, initializes global model (random weights, v0)
  → Security layer provisions HMAC keys + JWT tokens for each edge
  → Dashboard starts (shows demo data until server responds)
  → Edges start, wait for server health check

Step 2 — Edge Initialization (each edge independently)
  → Reads EDGE_ID from environment
  → entrypoint-edge.sh checks for processed.csv
  → If missing: runs preprocess.py (sort, label, normalize, save)
  → Loads config, creates LocalTrainer and CognitiveLayer

Step 3 — Training Round (repeated 10 times)

  [Edge side]
  → data_loader.py calls get_edge_partition(device_id, round_num)
  → partition.py loads processed.csv, selects this round's time window
  → Computes class weights for this window's label distribution
  → Fetches latest global model weights from /global-model
  → Loads global weights into local model
  → Runs 1 epoch of training over the window (forward → loss → backprop → Adam step)
  → Loss function weighted by class_weights to handle imbalance
  → Evaluates on fixed test set
  → CognitiveLayer adapts learning rate based on accuracy trend
  → POSTs weights + metrics to /update

  [Server side — fires when all 3 edges have submitted]
  → Rate limiter check (max 5 req/60s per edge)
  → HMAC signature verification
  → PoisoningDetector computes L2 norms, runs Z-score outlier detection
  → Suspicious edges excluded from aggregation
  → ReputationManager updates trust scores
  → FedAvg / Adaptive aggregation on clean edge weights
  → Global model updated, version incremented, checkpoint saved
  → Weighted global accuracy computed
  → MetricsTracker records all metrics → writes logs/metrics.json
  → AgentOrchestrator runs 4 agents in parallel threads
  → Pending updates cleared, ready for next round

Step 4 — Dashboard
  → Reads logs/metrics.json every 3 seconds
  → Polls /classify, /agents, /security from server if in Live Mode
  → All 5 tabs update in real time

Step 5 — Completion
  → After round 10, all edges exit cleanly
  → Server and dashboard continue running
  → Final global model saved as logs/checkpoints/global_v10.pt
```

---

## The Temporal Sliding Window

This is one of the key academic contributions. Instead of training on the same data every round, each edge trains on a **different time slice** that advances each round:

```
Device data timeline: [====================================] 100%

Round 1:  [=========]                 (0%  – 30%)
Round 2:    [=========]               (10% – 40%)
Round 3:      [=========]             (20% – 50%)
Round 4:        [=========]           (30% – 60%)
...

Test set (fixed, never used for training): [====] (last 20%)
```

**Why this matters:**
- Simulates real IoT streaming — sensors continuously generate new data
- Prevents data leakage — training data is always temporally before test data
- Forces the model to generalize across different environmental conditions over time
- Class weights are recomputed for each window since Normal/Warning/Critical ratios shift

---

## Security Layer

```
Edge                           Server
 │                               │
 │  sign_payload(edge_id)        │
 │  → HMAC-SHA256(secret, body)  │
 │  → timestamp                  │
 │                               │
 │  POST /update                 │
 │  {weights, signature, ts} ───▶│
 │                               │  verify_signature()
 │                               │  → check timestamp (< 60s ago)
 │                               │  → recompute HMAC
 │                               │  → compare_digest()
 │                               │
 │                               │  PoisoningDetector.detect()
 │                               │  → compute L2 norms per edge
 │                               │  → Z-score outlier detection
 │                               │  → flag if |z| > 2.5σ
 │                               │
 │                               │  exclude suspicious edges
 │                               │  aggregate clean edges only
```

---

## Multi-Agent System

```
After each aggregation round:

AgentOrchestrator
  ├── Thread 1: AnalystAgent.run(context)
  │     reads: global_acc history, edge_acc history
  │     outputs: trend, delta, edge divergence alerts
  │
  ├── Thread 2: AnomalyAgent.run(context)
  │     reads: sensor_batch from current round
  │     outputs: fire_risk / gas_leak / temp_spike counts
  │
  ├── Thread 3: PredictorAgent.run(context)
  │     reads: global_acc history, edge_acc history
  │     outputs: linear forecast for next round
  │
  └── Thread 4: SecurityAgent.run(context)
        reads: weight_norms, reputations, hmac_failures
        outputs: suspicious edges, norm deviation alerts

All 4 results collected → stored in results_log
→ Available via GET /agents
→ Displayed in Dashboard Tab 3
```

---

## Dashboard

Access at **http://localhost:8501** after `docker-compose up`.

The dashboard has two modes controlled by the sidebar toggle:

- **🔴 Live Mode** — connects directly to the server API for real-time classification and agent results
- **Demo Mode** — uses mock data and local rule-based classifier, works without the full Docker stack

---

## How to Run

### Full Docker stack
```bash
# 1. Place your raw CSV at:
#    data/iot_telemetry_data.csv

# 2. Build and start everything
docker-compose build --no-cache
docker-compose up

# 3. Open dashboard
# http://localhost:8501

# 4. View API docs
# http://localhost:8000/docs

# 5. Shut down
docker-compose down
```

### Dashboard only (demo mode, no Docker needed)
```bash
pip install streamlit plotly pandas numpy requests
python scripts/mock_feeder.py &    # generates fake metrics in background
streamlit run dashboard/app.py
```

### Test the live classifier API
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"co":0.008,"humidity":30,"light":0,"lpg":0.012,"motion":1,"smoke":0.15,"temp":105}'

# Response:
# {"prediction": 2, "label": "Critical", "confidence": 94.2, ...}
```

---

## Key Academic Concepts Demonstrated

| Concept | Where in Code | Why It Matters |
|---------|--------------|----------------|
| Federated Learning (FedAvg) | `server/aggregator.py` | Privacy-preserving distributed ML |
| Non-IID Data | `data/partition.py` | Realistic: different devices see different data |
| Temporal Sliding Window | `data/partition.py` | Prevents data leakage in time-series ML |
| Class Imbalance Handling | `edge/trainer.py` | Normal >> Warning > Critical in real data |
| Reputation-Based Trust | `server/reputation.py` | Byzantine fault tolerance |
| HMAC Cryptographic Signing | `security/security_layer.py` | Data integrity and authentication |
| Poisoning Attack Detection | `security/security_layer.py` | Defending against adversarial edges |
| Multi-Agent AI Coordination | `agents/orchestrator.py` | Distributed intelligence and monitoring |
| Digital Twin Concept | Whole system | Virtual representation of physical IoT devices |
| Adaptive Learning Rate | `edge/cognitive_layer.py` | Self-tuning model training |

---

*SYMBIOTIC-TWIN — Federated Digital Twin Framework for IoT Environmental Intelligence*