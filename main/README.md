# SYMBIOTIC-TWIN (Main Project)

Secure, multi-agent federated learning for IoT digital twins with edge privacy, adaptive aggregation, and live operational monitoring.

## What this project does

- Trains one shared model across multiple IoT edges without moving raw sensor data to the server.
- Keeps advanced features enabled: temporal windowing, class-imbalance handling, security checks, reputation scoring, and multi-agent analysis.
- Produces round-wise metrics used for cross-model comparison with centralized and FedAvg baselines.

## Architecture overview

- `edge/` handles local training, evaluation, and update submission.
- `server/` handles validation, aggregation, reputation, model versioning, and APIs.
- `agents/` runs analytical agents per round.
- `security/` runs poisoning/rate-limit/signature checks.
- `metrics/` records `global_acc`, edge latency, edge energy, and aggregation weights into `logs/metrics.json`.
- `dashboard/` renders real-time and historical federated performance.

## Performance optimization status

The latest tuning keeps model behavior/features intact while reducing overhead:

- Cached processed dataframe and per-device partitions to avoid repeated disk and filter work each round.
- Removed duplicate partition calls per round on edges.
- Improved local training efficiency (`zero_grad(set_to_none=True)`, non-blocking transfer, CUDA autocast path).
- Config tuning for lower cost per round:
  - `batch_size: 384`
  - `max_train_samples_per_round: 8000`
  - `dataloader_workers: 0` (reduces multiprocessing overhead on Windows environments)

## Fair comparison protocol

Use this protocol before claiming model ranking:

1. Same dataset and test split for all models.
2. Same round/epoch budget (`main/config/config.yaml -> system.num_rounds`).
3. Same energy formula (`energy_j = power_w * elapsed_time_s`) across all models.
4. Compare over common round budget (minimum shared rounds across metrics files).
5. Use balanced score from normalized accuracy, inverted latency, and inverted energy.

## Run order for fair results

1. Run centralized baseline (`Centrelized symbiotic twin/main.py`).
2. Run FedAvg baseline (`fedavg_project/fedavg_model.py`).
3. Run SYMBIOTIC-TWIN stack (`main/server` + all edges).
4. Open comparison dashboard (`comparison_dashboard.py` via Streamlit).

## Reliable Docker build (Windows)

If Docker builds fail due to low disk or take too long, use:

- `powershell -ExecutionPolicy Bypass -File build-and-run.ps1`

What it does:

- checks Docker availability
- checks free space on `C:`
- aborts early with cleanup commands if space is too low
- runs `docker compose build` then `docker compose up -d`

## Key files

- Config: `config/config.yaml`
- Edge loop: `edge/main.py`
- Trainer: `edge/trainer.py`
- Partitioning: `data/partition.py`
- Aggregation API: `server/routes.py`
- Aggregation math: `server/aggregator.py`
- Metrics persistence: `metrics/tracker.py`
- Dashboard: `dashboard/app.py`

## Goal of this main model

SYMBIOTIC-TWIN is intended to win under a **balanced** metric (accuracy + efficiency), not only raw accuracy. The optimization and fairness updates in this repo are designed to support that outcome.