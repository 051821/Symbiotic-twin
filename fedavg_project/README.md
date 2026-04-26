# FedAvg vs SYMBIOTIC-TWIN Dashboard

A simple Federated Learning (FedAvg) simulation with a Streamlit dashboard,
comparing performance against Centralized Training and the SYMBIOTIC-TWIN framework.

---

## Project Structure

```
fedavg_project/
├── fedavg_model.py     # FedAvg simulation model
├── dashboard.py        # Streamlit dashboard
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Requirements

- Python 3.9 or higher
- pip (Python package manager)

---

## Steps to Run

### Step 1 — Unzip the project

```bash
unzip fedavg_project.zip
cd fedavg_project
```

### Step 2 — (Optional) Create a virtual environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Streamlit dashboard

```bash
streamlit run dashboard.py --server.port 18501
```

### Step 5 — Open in browser

```
http://localhost:18501
```

---

## Dashboard Features

| Feature | Description |
|---|---|
| Metric Cards | Accuracy, Latency, Energy with delta vs Centralized |
| Bar Chart | All 3 models compared side by side |
| Convergence Charts | Accuracy / Latency / Energy over training rounds |
| Summary Table | Full comparison including Privacy and Adaptive columns |
| Round Detail Table | Per-round FedAvg metrics (expandable) |

### Sidebar Controls

| Control | Range | Default |
|---|---|---|
| Federated Rounds | 5 – 50 | 20 |
| Number of Edge Nodes | 2 – 8 | 4 |
| Random Seed | 0 – 999 | 42 |
| Run Simulation button | — | Re-runs simulation |

---

## Models Compared

| Model | Accuracy | Latency | Energy | Privacy | Adaptive |
|---|---|---|---|---|---|
| Centralized | 82% | 230 ms | 23 J | No | No |
| FedAvg (Simple) | ~76–89% | ~25–50 ms | ~4 J | Yes (FL) | No |
| SYMBIOTIC-TWIN | 94.3% | 150 ms | 1 J | Yes (FL+CDT) | Yes |

---

## Run Model Only (without dashboard)

```bash
python fedavg_model.py
```

Expected output:
```
FedAvg  → Accuracy: 76.25%  Latency: 29.6 ms  Energy: 4.31 J
```

---

## Troubleshooting

**Port already in use:**
```bash
streamlit run dashboard.py --server.port 18501
```

**Module not found error:**
```bash
pip install -r requirements.txt --upgrade
```

**Python version check:**
```bash
python --version   # should be 3.9+
```
