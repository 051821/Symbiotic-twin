# SYMBIOTIC-TWIN Monorepo

This repository contains the main SYMBIOTIC-TWIN federated learning system and two baseline implementations used for comparison.

## Repository Layout

- `main/` - Primary SYMBIOTIC-TWIN project (FastAPI server, 3 edge clients, Streamlit dashboard, Docker setup).
- `fedavg_project/` - FedAvg baseline simulation and Streamlit comparison dashboard.
- `Centrelized symbiotic twin/` - Centralized baseline training and dashboard.
- `baseline_shared/` - Shared data and metric utilities used by baseline projects.
- `requirements-baselines.txt` - Combined dependency list for baseline flows.

## What Runs Where

- **Main system API**: `http://localhost:18000`
- **Main system dashboard**: `http://localhost:18502`
- **FedAvg dashboard**: `http://localhost:18501`
- **Centralized dashboard**: `http://localhost:18503`

## Quick Start

For complete, sequential setup and execution steps on Windows PowerShell, see:

- `setup.md`

## Verified Fix Applied

One runtime error was fixed in `fedavg_project/fedavg_model.py`:

- Replaced invalid variable `īX_local` with `X_local` in local prediction.

## Notes

- The main app is Docker-first.
- Baseline projects are Python/Streamlit scripts and can be run without Docker.
- Do not run multiple Streamlit apps on the same port at the same time.
