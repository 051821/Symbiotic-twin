# SYMBIOTIC-TWIN Main Project

SYMBIOTIC-TWIN is a secure, multi-agent federated learning system for IoT digital twins. It trains a shared global classifier across multiple edge devices without sending raw edge data to the server. Edge nodes train locally, send model updates, the server aggregates those updates, and a Streamlit dashboard shows live classifier output, training metrics, security status, and multi-agent analysis.

This README documents the full `main/` folder: architecture, runtime flow, configuration, APIs, dashboard, and every important class/function used in the SYMBIOTIC-TWIN implementation.

## Current Project Goal

The current tuned setup aims for realistic high accuracy, not artificial 100% accuracy:

- Target accuracy range: about 97-98% global/macro behavior after rerun.
- Reduced latency and energy using a smaller model, capped per-round samples, and one local epoch per round.
- Same federated functionality: server, 3 edge nodes, dashboard, metrics, security, and agents.

Important current choices:

- `data.test_split_mode: stratified`
- `model.rule_prior_enabled: true`
- `model.rule_prior_strength: 3.0`
- `model.rule_prior_margins: [0.00003, 0.00003, 0.00003, 0.3]`
- `system.max_train_samples_per_round: 4096`
- `system.epochs_per_round: 1`
- `model.hidden_size: 64`

## What The System Does

1. Loads IoT telemetry data.
2. Creates 3 risk labels:
   - `0 = Normal`
   - `1 = Warning`
   - `2 = Critical`
3. Splits data per IoT device to simulate non-IID edge nodes.
4. Each edge trains locally on its own windowed data.
5. Edge nodes send serialized model weights and metrics to the server.
6. Server verifies/update-checks incoming payloads.
7. Server aggregates updates with FedAvg plus optional reputation weighting.
8. Server records metrics and runs four analysis agents.
9. Dashboard visualizes classifier output, accuracy, latency, energy, reputation, aggregation weights, security, and temporal windows.

## Folder Layout

```text
main/
  agents/                Multi-agent round analysis
  config/                YAML config and logging setup
  dashboard/             Streamlit dashboard
  data/                  Preprocessing and edge partitioning
  edge/                  Edge-node runtime, local training, communication
  metrics/               Accuracy, latency, energy, JSON tracking
  scripts/               Demo/mock helpers
  security/              HMAC, JWT, rate limiting, poisoning detection
  server/                FastAPI server, aggregation, model manager
  shared/                Model, serialization, common utilities
  logs/                  Runtime logs, metrics, checkpoints
  docker-compose.yml     Full Docker runtime
  Dockerfile.app         Server/edge image
  Dockerfile.dashboard   Dashboard image
```

## Main Runtime Flow

### Server

Entry point: `server/main.py`

1. Loads config.
2. Seeds randomness.
3. Builds global model through `ModelManager`.
4. Initializes edge reputations.
5. Initializes security tokens/secrets.
6. Registers FastAPI routes.
7. Waits for edge updates.
8. Aggregates when all expected edges submit a round update.
9. Saves metrics and checkpoints.

### Edge Nodes

Entry point: `edge/main.py`

Each edge:

1. Reads its `EDGE_ID`.
2. Finds the matching device MAC in `config.yaml`.
3. Waits until server health endpoint is ready.
4. Loads local edge data for the current round.
5. Fetches the latest global model.
6. Trains locally with `LocalTrainer`.
7. Evaluates on its held-out test split.
8. Sends update to `/update`.
9. Repeats for `system.num_rounds`.

### Dashboard

Entry point: `dashboard/app.py`

Tabs:

- Live Classifier
- Federated Metrics
- Agent Panel
- Security Monitor
- Window Visualizer

The dashboard reads `logs/metrics.json` and, in live mode, calls the server APIs.

## Configuration Reference

Main config: `config/config.yaml`

### `system`

- `profile`: descriptive mode label, usually `fast` or `paper`.
- `num_edges`: number of edge clients expected by the server.
- `num_rounds`: federated rounds each edge runs.
- `learning_rate`: AdamW learning rate.
- `batch_size`: DataLoader batch size.
- `epochs_per_round`: local epochs per federated round.
- `max_train_samples_per_round`: cap on local train samples for latency/energy control.
- `eval_every_n_rounds`: evaluation cadence.
- `eval_max_samples_per_round`: optional evaluation cap; `0` means full test set.
- `allow_extra_epochs`: lets the cognitive layer add epochs when needed.
- `update_delivery_retries`: outer retry attempts from edge runtime.
- `update_delivery_wait_seconds`: wait between delivery retries.
- `dataloader_workers`: worker count. Current Windows-friendly value is `0`.
- `seed`: reproducibility seed.

### `server`

- `host`: service name/host used by edge clients.
- `port`: FastAPI server port inside container.

### `edges`

Each edge has:

- `id`: logical edge name, for example `edge1`.
- `device`: IoT device MAC address used for partitioning.
- `host`: local host value.
- `port`: reserved edge port.

### `model`

- `input_size`: number of input features, currently 7.
- `hidden_size`: neural network hidden width.
- `num_classes`: 3 labels.
- `dropout`: dropout probability.
- `weight_decay`: AdamW regularization.
- `label_smoothing`: CrossEntropy label smoothing.
- `grad_clip_norm`: optional gradient clipping.
- `fedprox_mu`: FedProx proximal regularization strength.
- `rule_prior_enabled`: enables soft domain-rule logits.
- `rule_prior_strength`: how strongly rule prior affects logits.
- `rule_prior_scaled_input`: whether model input is standardized.
- `rule_prior_margins`: uncertainty margin around rule thresholds.

Rule prior thresholds:

- Critical if smoke is above `0.10` or CO is above `0.005`.
- Warning if temperature is above `90.0` or LPG is above `0.007`.
- Near thresholds, the prior backs off so the neural model still matters.

### `aggregation`

- `strategy`: currently `fedavg`.
- `adaptive_weighting`: combines sample counts with reputation.
- `reputation_enabled`: enables reputation system behavior.
- `min_reputation`: lower bound for reputation.
- `reputation_similarity_weight`: weight of model similarity in reputation score.
- `reputation_ema_alpha`: how quickly reputation reacts.
- `aggregation_reputation_power`: controls reputation influence in aggregation weights.

### `data`

- `raw_path`: raw telemetry CSV path.
- `processed_path`: processed data folder.
- `partitions_path`: cached partition folder.
- `features_scaled`: whether processed features are standardized.
- `test_split`: test fraction.
- `test_split_mode`: `stratified` or `temporal`.
- `window_strategy`: `sliding`, `expanding`, or `full`.
- `window_fraction`: initial window fraction.
- `window_step`: per-round window movement/growth.
- `min_classes_per_window`: minimum classes desired in a training window.
- `min_samples_per_class_cap`: floor for class-aware capped sampling.
- `class_balance_cap_fraction`: class floor as a fraction of sample cap.
- `oversample_rare_classes`: whether to sample rare classes with replacement.
- `calibration_samples_per_class`: small global calibration sample added before cap.
- `class_weight_multipliers`: per-class loss multipliers.

### `security`

- `rate_limit_max_per_window`: max updates per edge per window.
- `rate_limit_window_seconds`: rate-limit window length.
- `update_max_retries`: HTTP retry attempts inside communication layer.
- `update_retry_backoff_seconds`: exponential backoff base.

### `logging`

Controls log level, log folder, format, and timestamp format.

### `dashboard`

- `refresh_interval`: dashboard auto-refresh interval.
- `metrics_path`: JSON metrics file path.

## Data And Labels

### Raw data

Expected raw CSV path:

```text
data/iot_telemetry_data.csv
```

Expected feature columns:

```text
co, humidity, light, lpg, motion, smoke, temp
```

Expected metadata columns include:

```text
ts, device
```

### Labels

Implemented in `data/preprocess.py:create_labels`.

Label rules:

- Critical (`2`): `smoke > 0.10` or `co > 0.005`
- Warning (`1`): `temp > 90.0` or `lpg > 0.007`
- Normal (`0`): otherwise

Critical is checked before Warning.

## How To Run

### Docker recommended

From `main/`:

```powershell
docker compose build --parallel
docker compose up
```

Services:

- Server: host port `18000`, container port `8000`
- Dashboard: host port `18502`, container port `8502`
- Edges: `edge1`, `edge2`, `edge3`

Dashboard URL:

```text
http://localhost:18502
```

Server health:

```text
http://localhost:18000/health
```

### Windows helper

```powershell
powershell -ExecutionPolicy Bypass -File build-and-run.ps1
```

### Local dashboard only

If dependencies are installed locally:

```powershell
streamlit run .\dashboard\app.py --server.port 18502
```

The root-level comparison dashboard is separate:

```powershell
streamlit run .\comparison_dashboard.py --server.port 18504
```

## API Endpoints

Server routes live in `server/routes.py`.

### `GET /health`

Returns:

- server status
- model version
- security status
- active agent names

### `GET /global-model`

Returns:

- global model version
- serialized global weights

Used by edges before local training.

### `POST /update`

Receives edge model update payload.

Payload fields:

- `edge_id`
- `round_num`
- `weights`
- `sample_count`
- `accuracy`
- `train_accuracy`
- `test_accuracy`
- `latency_ms`
- `energy_j`
- `signature`
- `timestamp`

Behavior:

- Drops stale/duplicate updates.
- Keeps updates grouped by coherent round.
- Applies rate limiting.
- Verifies HMAC if signature/timestamp are present.
- Stores pending update.
- Triggers aggregation when all expected edges submitted.

### `POST /classify`

Live IoT risk classification.

Input:

```json
{
  "co": 0.002,
  "humidity": 55.0,
  "light": 1.0,
  "lpg": 0.003,
  "motion": 0.0,
  "smoke": 0.02,
  "temp": 72.0
}
```

Output:

- prediction index
- label name
- confidence
- probabilities
- model version
- explanation reasons

### `GET /agents`

Returns latest multi-agent results and alerts.

### `GET /security`

Returns:

- recent security events
- poisoning detector norm history
- active pending edges

### `GET /weights`

Returns latest aggregation weights.

### `GET /metrics`

Returns persisted metrics from `logs/metrics.json`.

## Metrics File

Metrics are written to:

```text
logs/metrics.json
```

Tracked fields:

- `rounds`
- `global_acc`
- `edge_acc`
- `latency`
- `energy`
- `agg_weights`
- `reputation`

## Module And Function Reference

### `config/loader.py`

- `load_config(path=None)`: loads YAML config and caches it.
- `get_config()`: returns loaded config.
- `reload_config(path=None)`: clears cache and reloads config.

### `config/logging_config.py`

- `setup_logger(name)`: creates a file and console logger for a component.

### `shared/model.py`

#### `IoTClassifier`

Feedforward PyTorch classifier for IoT risk classes.

Network:

- Linear input layer
- LayerNorm
- GELU
- Dropout
- Wider hidden layer
- LayerNorm
- GELU
- Dropout
- Final classifier layer

Important methods:

- `_load_rule_margins()`: loads configurable rule uncertainty margins.
- `_load_scaled_rule_thresholds()`: returns label thresholds in model input space.
- `_rule_prior_logits(x)`: creates soft rule-based logits.
- `forward(x)`: neural logits plus optional rule prior logits.

#### `build_model()`

Builds an `IoTClassifier` from `config.yaml`.

### `shared/serialization.py`

- `serialize_weights(state_dict)`: converts tensors to JSON-safe lists.
- `deserialize_weights(serialized)`: converts JSON-safe lists back to tensors.
- `weights_to_numpy(state_dict)`: tensor weights to NumPy arrays.
- `numpy_to_weights(numpy_dict)`: NumPy arrays back to tensors.

### `shared/utils.py`

- `set_seed(seed)`: seeds Python, NumPy, and PyTorch.
- `get_device()`: returns CUDA if available, otherwise CPU.
- `ensure_dir(path)`: creates a directory if missing.
- `save_json(data, path)`: writes JSON.
- `load_json(path)`: reads JSON.
- `count_parameters(model)`: counts trainable model parameters.
- `flatten_weights(state_dict)`: flattens all weights into one NumPy vector.

### `data/preprocess.py`

- `create_labels(df)`: creates Normal/Warning/Critical labels from sensor thresholds.
- `preprocess(raw_path=None, out_dir=None)`: loads raw CSV, sorts by timestamp, creates labels, scales/records scaler, and writes processed output.

Note: current config uses `features_scaled: false` because the active processed file is raw-valued.

### `data/partition.py`

- `_expand_window_for_class_diversity(...)`: expands a temporal training window until enough classes are present.
- `_compute_class_weights(y)`: creates softened class weights with optional class multipliers.
- `get_edge_partition(...)`: returns train/test `TensorDataset`s and class weights for one device and round.
- `partition_all(round_num=0)`: returns sample counts for all configured edges.

Important behavior:

- Supports `temporal` and `stratified` test splits.
- Supports `sliding`, `expanding`, and `full` training windows.
- Adds optional calibration samples.
- Applies balanced sample capping for latency/energy control.

### `edge/data_loader.py`

- `load_edge_data(...)`: wraps `get_edge_partition` and returns train/test DataLoaders, class weights, and sample count.
- `get_sample_count(device_id, round_num=0)`: returns local training sample count.

### `edge/trainer.py`

#### `LocalTrainer`

Handles local edge training.

Important behavior:

- Uses AdamW.
- Uses CrossEntropyLoss with class weights.
- Supports label smoothing.
- Uses AMP on CUDA.
- Supports gradient clipping.
- Supports FedProx via a global reference model.
- Tracks latency and estimated energy.

Important methods:

- `set_global_reference(state_dict)`: stores global weights for FedProx.
- `update_class_weights(class_weights)`: updates loss for new round/window.
- `train(train_loader, epochs_override=None)`: trains locally and returns accuracy, latency, energy.
- `evaluate(test_loader, max_samples=None)`: evaluates local model.

### `edge/cognitive_layer.py`

#### `CognitiveLayer`

Adapts edge behavior based on accuracy and energy.

Important methods:

- `adapt(current_accuracy, energy_used_j)`: updates LR based on plateau, improvement, or energy use.
- `training_plan()`: returns per-round sample ratio and extra epochs.
- `should_skip_round()`: decides if edge should stop due to energy exhaustion.
- `status()`: returns current cognitive-layer state.

### `edge/communication.py`

- `_server_url()`: builds server base URL from config.
- `send_update(...)`: posts model weights and metrics to `/update`.
- `fetch_global_model(timeout=30)`: downloads global weights from `/global-model`.
- `check_server_health(timeout=5)`: calls `/health`.

### `edge/main.py`

- `run_edge(edge_id, device_id)`: full edge loop for all rounds.

Responsibilities:

- Wait for server.
- Build model.
- Load windowed data.
- Fetch global model.
- Train locally.
- Evaluate.
- Adapt LR through `CognitiveLayer`.
- Send update with retries.

### `server/main.py`

- `create_app()`: builds the FastAPI app, initializes model manager, reputation manager, security, and routes.

### `server/model_manager.py`

#### `ModelManager`

Maintains the global model.

Important methods:

- `get_weights()`: returns current PyTorch state dict.
- `get_serialized_weights()`: returns JSON-safe weights.
- `update_weights(aggregated)`: loads aggregated weights and increments version.
- `load_weights_from_dict(weights_dict)`: loads weights from serialized dict.
- `_save_checkpoint()`: saves global model checkpoint.
- `load_checkpoint(version=None)`: loads a saved checkpoint.

### `server/aggregator.py`

- `compute_fedavg_weights(sample_counts)`: standard FedAvg sample-proportional weights.
- `compute_adaptive_weights(sample_counts, reputations)`: combines sample count and reputation.
- `aggregate(local_weights, agg_weights)`: weighted parameter aggregation.
- `run_aggregation(local_weights, sample_counts, reputations=None)`: selects aggregation mode and returns aggregated model plus weights.

### `server/reputation.py`

#### `ReputationManager`

Tracks edge trust scores.

Important methods:

- `initialize(edge_ids)`: starts all reputations at 1.0.
- `update(edge_id, local_weights, global_weights, local_accuracy)`: updates score from similarity and accuracy.
- `get_scores()`: returns all current scores.
- `get_history(edge_id)`: returns score history for one edge.

### `server/routes.py`

Data models:

- `UpdatePayload`: edge update schema.
- `ClassifyPayload`: live classifier input schema.

Functions:

- `init_router(...)`: injects model manager, reputation manager, edge count, and tokens.
- `_get_scaler()`: loads saved scaler if scaled input is enabled.
- `health()`: health endpoint.
- `get_global_model()`: returns serialized global model.
- `receive_update(payload)`: receives and stores edge updates.
- `classify_sensor(payload)`: live sensor classification.
- `get_agent_results()`: latest agent output.
- `get_security_status()`: security event/norm data.
- `get_aggregation_weights()`: latest aggregation weights.
- `get_metrics()`: metrics JSON payload.
- `_log_sec(edge_id, etype, detail)`: appends security event.
- `_trigger_aggregation()`: runs poisoning detection, reputation update, aggregation, metric tracking, and agents.

### `security/security_layer.py`

Functions:

- `provision_edge(edge_id)`: creates edge secret.
- `sign_payload(edge_id)`: signs edge identity and timestamp.
- `verify_signature(edge_id, signature, timestamp)`: verifies HMAC and replay window.
- `_b64(data)`: URL-safe base64 helper.
- `_b64d(s)`: URL-safe base64 decode helper.
- `issue_token(edge_id, ttl=3600)`: creates JWT-like token.
- `verify_token(token)`: verifies token signature and expiry.
- `get_detector()`: singleton poisoning detector.
- `get_rate_limiter()`: singleton rate limiter.
- `initialize_security(edge_ids)`: provisions edges and returns tokens.

Classes:

- `PoisoningDetector`
  - `compute_norms(local_weights)`: computes L2 norms per edge.
  - `detect(local_weights)`: flags norm outliers.
- `RateLimiter`
  - `allow(edge_id)`: accepts or rejects update based on rate.

### `metrics/accuracy.py`

- `compute_accuracy(y_true, y_pred)`: computes percentage accuracy.
- `compute_batch_accuracy(outputs, targets)`: computes correct count and batch size from logits.

### `metrics/latency.py`

- `measure_latency(fn, *args, **kwargs)`: runs a function and measures latency.
- `measure_inference_latency(model, sample_input, runs=50)`: average inference latency.
- `LatencyTimer`: context manager for elapsed milliseconds.

### `metrics/energy.py`

- `estimate_energy(computation_time_s, model=None, power_w=...)`: estimates energy in Joules.
- `EnergyMonitor`: context manager for energy estimation.

Energy is simulated from elapsed compute time and model parameter count.

### `metrics/tracker.py`

#### `MetricsTracker`

Persists federated metrics to JSON.

Important methods:

- `record_round(...)`: stores global accuracy, edge metrics, aggregation weights, and reputations.
- `save()`: writes `logs/metrics.json`.
- `load()`: reads metrics JSON.
- `summary()`: returns text summary of latest round.

Function:

- `get_tracker()`: singleton metrics tracker.

### `agents/orchestrator.py`

Data structures:

- `AgentStatus`: `idle`, `running`, `done`, `error`.
- `AgentResult`: stores agent name, round, status, findings, alerts, timestamp.

Base class:

- `BaseAgent`
  - `run(context, round_num)`: wraps `_execute` with status/error handling.
  - `_execute(context, round_num)`: implemented by child agents.

Agents:

- `AnalystAgent`
  - Detects accuracy trend: improving, regressing, plateau.
  - Computes edge divergence from global accuracy.
- `AnomalyAgent`
  - Checks sensor batches for fire/gas/temp anomalies.
  - If no sensor batch is attached, returns `0 / 0` and no fake alerts.
- `PredictorAgent`
  - Uses simple linear forecast for next-round accuracy.
  - Handles ties instead of reporting same best/worst edge.
- `SecurityAgent`
  - Tracks weight norm deviations.
  - Reports low reputation or HMAC failures.

Orchestration:

- `AgentOrchestrator`
  - `run_round(context, round_num)`: runs all agents concurrently with threads.
  - `get_all_alerts(round_num)`: returns all alerts for a round.
  - `get_serializable_results(round_num)`: returns dashboard/API-safe output.
- `get_orchestrator()`: singleton orchestrator.

### `dashboard/app.py`

Important functions:

- `_layout(**kwargs)`: shared Plotly layout.
- `load_metrics()`: cached metrics JSON load.
- `server_up()`: checks server health.
- `api_get(path)`: GET helper.
- `api_post(path, data)`: POST helper.
- `classify_local(f)`: rule-based fallback classifier for demo mode.
- `mock_metrics(n_rounds=8)`: dashboard demo metrics.
- `prob_bars_html(probs)`: probability bar HTML.
- `kpi_html(val, lbl, delta=None, color=None)`: KPI card HTML.
- `alert_html(alerts)`: alert HTML.

Dashboard tabs:

- Tab 1: Live Classifier
- Tab 2: Federated Metrics
- Tab 3: Agent Panel
- Tab 4: Security
- Tab 5: Window Visualizer

### `scripts/mock_feeder.py`

- `load()`: loads or initializes demo metrics.
- `save(data)`: writes demo metrics.
- `step(data)`: simulates one federated round.
- `main(interval=3)`: continuously writes mock metrics for dashboard demos.

### `plot.py`

- `main()`: helper plotting entry point.

## Docker Files

### `docker-compose.yml`

Defines services:

- `server`
- `edge1`
- `edge2`
- `edge3`
- `dashboard`

Uses shared volumes:

- `./logs:/app/logs`
- `./data:/app/data`

### `Dockerfile.app`

Builds server and edge runtime image.

### `Dockerfile.dashboard`

Builds Streamlit dashboard image.

### `entrypoint-edge.sh`

Edge container entrypoint.

### Helper scripts

- `quickstart.sh`: quick Linux-style startup helper.
- `optimize-and-build.sh`: optimized Docker build helper.
- `verify-docker-setup.sh`: Docker setup validation helper.
- `build-and-run.ps1`: Windows build/run helper.

## Dashboard Notes

If dashboard still shows old values, it is reading old `logs/metrics.json`. Rerun the federated stack after config changes.

Expected current behavior after rerun:

- Global accuracy around 97-98%.
- Edge 3 around 97-98%.
- Edge 1 no longer artificially perfect because `test_split_mode` is stratified.
- Latency and energy lower than the larger `8192` sample / `hidden_size 96` setup.
- Agent panel no longer shows fake anomaly alerts for an empty sensor batch.

## Accuracy, Latency, And Energy Tuning

Current tuning balances accuracy and efficiency:

- Accuracy realism:
  - Stratified test split avoids a misleading one-class temporal test set.
  - Soft rule prior uses margins to avoid exact label copying.
- Latency:
  - `hidden_size: 64`
  - `max_train_samples_per_round: 4096`
  - `epochs_per_round: 1`
- Energy:
  - Smaller model and fewer samples reduce estimated joules.
  - `dataloader_workers: 0` avoids Windows multiprocessing overhead/errors.

## Fair Comparison Protocol

When comparing SYMBIOTIC-TWIN with Centralized or FedAvg baselines:

1. Use the same raw dataset.
2. Use the same test split mode.
3. Use the same total round/epoch budget.
4. Use the same energy equation.
5. Compare common rounds only.
6. Report accuracy, latency, energy, and a balanced score.

## Common Troubleshooting

### Dashboard shows old accuracy

Delete or ignore old `logs/metrics.json`, then rerun the Docker stack.

### Dashboard cannot connect to server

Inside Docker, dashboard uses:

```text
http://server:8000
```

From host machine, use:

```text
http://localhost:18000
```

### Python compileall fails on Windows

This can happen because Windows denies replacement of old `.pyc` files. Source parsing can still pass.

### Edge update stuck

Check:

- `/health`
- edge logs in `logs/`
- round synchronization in server logs
- rate limits

### Accuracy returns to 100%

Check:

- `model.rule_prior_strength`
- `model.rule_prior_margins`
- `data.test_split_mode`
- old `logs/metrics.json`

## Important Outputs

- Metrics: `logs/metrics.json`
- Logs: `logs/*.log`
- Checkpoints: `logs/checkpoints/global_v*.pt`

## Short Summary

SYMBIOTIC-TWIN is a complete federated IoT digital-twin pipeline:

- Edge devices train locally.
- Server aggregates securely.
- Reputation and poisoning checks protect aggregation.
- Agents analyze each round.
- Dashboard visualizes model, system, security, and anomaly state.
- Current tuning avoids misleading 100% accuracy and aims for realistic 97-98% with reduced latency and energy.
