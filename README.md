# SYMBIOTIC-TWIN Monorepo

This monorepo includes the primary SYMBIOTIC-TWIN system and two baselines used for fair model comparison.

## Projects

- `main/` - SYMBIOTIC-TWIN federated stack (server, 3 edges, dashboard, security, agents).
- `fedavg_project/` - FedAvg baseline run + dashboard.
- `Centrelized symbiotic twin/` - Centralized baseline run + dashboard.
- `baseline_shared/` - Shared dataset/metrics utilities for baselines.
- `comparison_dashboard.py` - Unified comparison dashboard reading all three metrics files.

## Current comparison standard

All comparison outputs should follow these fairness rules:

- Same round/epoch budget across models.
- Same dataset source and split.
- Same energy estimation basis (`power * elapsed_time`).
- Same comparison horizon (common shared rounds).
- Balanced ranking uses normalized accuracy, inverted latency, and inverted energy.

## Recent improvements

To improve SYMBIOTIC-TWIN competitiveness while preserving features:

- Reduced partition/data loading overhead with caching.
- Removed duplicate per-round partition computation on edge.
- Improved local training loop efficiency.
- Tuned main config for lower per-round latency and energy.
- Updated comparison dashboard to enforce fair shared-round view and show balanced score ranking.
- Upgraded centralized dashboard visualization quality for cleaner trends.

## Expected outcome

SYMBIOTIC-TWIN should retain top-end accuracy and substantially reduce latency/energy gap, improving balanced score ranking versus FedAvg and centralized baselines after rerunning all three pipelines.

## How to generate fresh fair metrics

1. Run `Centrelized symbiotic twin/main.py`.
2. Run `fedavg_project/fedavg_model.py`.
3. Run SYMBIOTIC-TWIN (`main` server + edges).
4. Start `comparison_dashboard.py` with Streamlit.

## Service endpoints

- Main API: `http://localhost:18000`
- Main dashboard: `http://localhost:18502`
- FedAvg dashboard: `http://localhost:18501`
- Centralized dashboard: `http://localhost:18503`

## Setup reference

See `setup.md` for full environment and run steps.
