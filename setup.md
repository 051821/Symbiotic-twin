# Setup and Sequential Run Guide

This guide runs all projects in order on **Windows PowerShell** from the repository root:

```powershell
cd C:\Symbiotic-twin
```

## 1) Prerequisites

Install and verify:

- Python 3.10+ (`python --version`)
- pip (`python -m pip --version`)
- Docker Desktop (`docker --version`, `docker compose version`)

Create shared folders if needed:

```powershell
mkdir logs -ErrorAction SilentlyContinue
mkdir "main\data\processed" -ErrorAction SilentlyContinue
mkdir "main\data\partitions" -ErrorAction SilentlyContinue
```

## 2) Prepare Python environment for baseline scripts

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-baselines.txt
```

## 3) Run Baseline #1: FedAvg project

### 3.1 Run model simulation

```powershell
python .\fedavg_project\fedavg_model.py
```

This generates baseline metrics JSON in:

- `fedavg_project/metrics.json`

### 3.2 Run FedAvg dashboard

```powershell
streamlit run .\fedavg_project\dashboard.py --server.port 18501
```

Open: `http://localhost:18501`

Stop with `Ctrl + C` before moving to next project.

## 4) Run Baseline #2: Centralized project

### 4.1 Install project-specific extras (if missing)

```powershell
pip install -r ".\Centrelized symbiotic twin\requirements.txt"
```

### 4.2 Run centralized training script

```powershell
python ".\Centrelized symbiotic twin\main.py"
```

### 4.3 Run centralized dashboard

```powershell
streamlit run ".\Centrelized symbiotic twin\dashboard.py" --server.port 18503
```

Open: `http://localhost:18503`

Stop with `Ctrl + C` before moving to main system.

## 5) Run Main SYMBIOTIC-TWIN system (Docker)

The main stack runs:

- `server` on port `18000`
- `dashboard` on port `18502`
- `edge1`, `edge2`, `edge3` as training clients

From `main/`:

```powershell
cd .\main
docker compose build --no-cache
docker compose up -d
docker compose ps
```

Open:

- Dashboard: `http://localhost:18502`
- API docs: `http://localhost:18000/docs`
- Health: `http://localhost:18000/health`

Watch logs:

```powershell
docker compose logs -f
```

Stop stack:

```powershell
docker compose down
```

Return to repo root:

```powershell
cd ..
```

## 6) Optional local (non-Docker) dashboard demo for main app

If you only want to view dashboard visuals with mock data:

```powershell
python .\main\scripts\mock_feeder.py
```

In another terminal:

```powershell
streamlit run .\main\dashboard\app.py --server.port 18502
```

## 7) Troubleshooting

- **Port already in use**: choose a different free Streamlit port (example: `--server.port 18504`).
- **Docker not running**: start Docker Desktop and re-run `docker compose up -d`.
- **Python package errors**: run `pip install -r requirements-baselines.txt` again inside activated venv.
- **Execution policy blocks venv activation**:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 8) Sequential order recap

1. FedAvg model + dashboard
2. Centralized model + dashboard
3. Main Dockerized SYMBIOTIC-TWIN stack

## 9) Centralized comparison dashboard (all 3 models)

After running all three models so their metrics files exist, launch:

```powershell
streamlit run .\comparison_dashboard.py --server.port 18504
```

This app compares:

- `Centrelized symbiotic twin/metrics.json`
- `fedavg_project/metrics.json`
- `main/logs/metrics.json`
