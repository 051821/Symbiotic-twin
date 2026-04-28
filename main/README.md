# SYMBIOTIC-TWIN (Main Project)

Secure multi-agent federated learning for IoT digital twins with privacy-preserving edge training, adaptive aggregation, security hardening, and live operational dashboards.

## Core Features

- **Federated learning across 3 edge nodes** with a shared global model.
- **No raw data upload**: only model updates are sent to server.
- **Non-IID data support** by per-device partitioning.
- **Temporal windowing modes**: `sliding`, `expanding`, `full`.
- **Class-imbalance handling** using class-weighted cross-entropy.
- **FedProx stabilization** to reduce edge divergence in non-IID rounds.
- **Adaptive weighting + reputation** in aggregation.
- **Security layer** with HMAC verification, rate limiting, and poisoning outlier detection.
- **Multi-agent analysis panel** (analyst, anomaly, predictor, security agents).
- **Live IoT classifier UI** with confidence and explanation output.
- **Round-wise metrics tracking**: accuracy, latency, energy, aggregation weights, reputation.

## Architecture

- `edge/`: edge runtime (load windowed data, local train/eval, send update).
- `server/`: APIs, update intake, aggregation, reputation, model versioning.
- `data/`: preprocessing and partitioning logic.
- `security/`: signature/rate-limit/poisoning safeguards.
- `agents/`: round-level multi-agent insights.
- `metrics/`: writes `logs/metrics.json`.
- `dashboard/`: Streamlit monitoring and live classification app.

## Training And Optimization Features

- AMP training on CUDA when available.
- Efficient optimizer flow (`zero_grad(set_to_none=True)`, optional grad clipping).
- Data and device partition caching to reduce repeated preprocessing cost.
- Configurable per-round sample cap (`max_train_samples_per_round`) for latency/energy control.
- Configurable eval cadence (`eval_every_n_rounds`) to reduce overhead in fast mode.
- FedProx configurable via `model.fedprox_mu`.
- Cognitive layer adapts sample ratio, epochs, and learning rate based on performance/energy.

## Profiles (Paper vs Fast)

Use `config/config.yaml -> system.profile`:

- `paper`: prioritize strict per-round evaluation and reproducibility.
- `fast`: prioritize lower latency/energy with efficient defaults.

Current fast-oriented defaults:

- `batch_size: 512`
- `max_train_samples_per_round: 4096`
- `eval_every_n_rounds: 2`
- `learning_rate: 0.0008`
- `model.fedprox_mu: 0.0005`

If you need exact paper-style behavior, set:

- `eval_every_n_rounds: 1`
- increase `max_train_samples_per_round` to your reported value
- keep any hyperparameters exactly as used in your experiments

## Live Classifier Behavior

- `/classify` returns model prediction (`Normal/Warning/Critical`) plus probabilities.
- Explanation text now distinguishes:
  - **Threshold-triggered reasons** (hard rules exceeded), and
  - **Model-pattern reasons** (no hard rule exceeded, but model predicts risk from combined features).

This prevents contradictory output such as "Critical" with "all thresholds safe" explanation.

## Security Features

- HMAC signature verification for update payloads.
- Replay/time-window checks (via timestamp in payload).
- Per-edge rate limiting.
- Weight-norm outlier detection for poisoning attempts.
- Reputation penalty/exclusion behavior for suspicious edges.

## Metrics Captured Per Round

- Global weighted accuracy.
- Per-edge accuracy.
- Per-edge training latency (ms).
- Per-edge energy estimate (J).
- Aggregation weights.
- Reputation history.

## Docker Build And Runtime

Recommended:

- `docker compose build --parallel`
- `docker compose up -d`

APT and pip cache mounts are configured with locked sharing for stable concurrent builds.

Windows helper script:

- `powershell -ExecutionPolicy Bypass -File build-and-run.ps1`

## Important Files

- `config/config.yaml`: all training/system knobs
- `edge/main.py`: federated round loop and adaptive plan
- `edge/trainer.py`: local training, FedProx, evaluation
- `data/partition.py`: non-IID + temporal partitioning
- `server/routes.py`: API endpoints including `/update` and `/classify`
- `server/aggregator.py`: FedAvg/adaptive aggregation math
- `server/reputation.py`: reputation update logic
- `metrics/tracker.py`: metrics persistence
- `dashboard/app.py`: UI tabs for classifier, metrics, agents, security, windows

## Fair Comparison Protocol

For model comparisons (centralized vs FedAvg vs SYMBIOTIC-TWIN):

1. Keep same dataset and same test split.
2. Keep same total round/epoch budget.
3. Keep same energy equation across systems.
4. Compare over common rounds only.
5. Report balanced score (accuracy + latency + energy), not accuracy alone.