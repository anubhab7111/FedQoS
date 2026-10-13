# FedQoS - Federated Learning for 5G QoS Prediction

A federated learning system that trains an LSTM model across distributed clients to predict Quality of Service (QoS) metrics — specifically **throughput** and **end-to-end delay** — in NR-V2X (New Radio Vehicle-to-Everything) communication networks, without sharing raw data between nodes.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running with Python Scripts](#running-with-python-scripts)
  - [Running with Docker](#running-with-docker)
  - [REST API](#rest-api)
  - [Exporting to ONNX](#exporting-to-onnx)
- [Notebooks](#notebooks)
- [Results](#results)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Overview

This project addresses QoS prediction in vehicular networks using **Federated Learning (FL)** with **FedAvg** as the aggregation strategy (via the [Flower](https://flower.ai/) framework). The core model is a multi-output LSTM that takes a sliding window of recent network measurements and predicts future throughput and delay values over a configurable forecast horizon.

Key design choices:
- **Privacy-preserving**: each client trains locally on its own slice of data; only model weights are shared with the server, never raw measurements.
- **Sequence-to-sequence forecasting**: the model ingests a window of past observations and outputs predictions for multiple future timesteps simultaneously.
- **Optional data augmentation**: Gaussian noise can be added to the training windows only (`noise_level`). It is off by default because the search found no benefit.
- **Run-aware data handling**: the dataset is many separate simulation runs back to back; smoothing, windows and splits never cross from one run into the next.
- **Leak-free evaluation**: every run is split chronologically into train / validation / test; the best round is chosen on validation loss and test metrics are only reported.
- **Search support**: `scripts/search_configs.py` runs repeated in-process federations (random search, grids, multi-seed reruns) and logs the results to CSV.

---

## Dataset

The dataset is sourced from:

> A. Reyhanoglu et al., "Machine Learning Aided NR-V2X Quality of Service Predictions," 2023 IEEE Vehicular Networking Conference (VNC), Istanbul, Türkiye, 2023, pp. 183–186. DOI: [10.1109/VNC57357.2023.10136346](https://ieeexplore.ieee.org/document/10136346)

Stored at `ML-Aided-NR-V2X-QoS-Predictions/dataset.csv`.

**Columns:**

| Column | Description |
|---|---|
| `throughput` | Data throughput (Mbps) |
| `mcs` | Modulation and Coding Scheme index |
| `distance` | Inter-vehicle distance (m) |
| `pktsize` | Packet size (bytes) |
| `delay` | End-to-end packet delay (ms) |
| `jitter` | Delay variation (ms) |
| `sinr` | Signal-to-Interference-plus-Noise Ratio (dB) |
| `power` | Transmission power (dBm) |
| `dlpathloss` | Downlink path loss (dB) |
| `pdr` | Packet Delivery Ratio (%) |

**Simulation runs**: `dataset.csv` is not one continuous recording. It is 49 separate simulation runs concatenated from 18 source files, each run with its own MCS, packet size and transmit power, and throughput and delay behave very differently from one run to the next. A new run starts wherever any of the columns in `run_cols` changes, or where the per-file row counter restarts (which separates repeated runs with identical settings). Everything below is done run by run.

**Preprocessing pipeline** (in `src/data/preprocessing.py`):
1. Load the file. The 221 exact duplicate rows are kept. They all sit in runs with constant throughput (a short trajectory logged three times, positions revisited within a run, rows shared with another run that has the same settings), and removing them would cut steps out of the middle of runs. The only copies that cross a split boundary are in three 48-row runs whose throughput and delay never change.
2. Label each row with its simulation run.
3. Drop end-of-log marker rows: 16 runs end with a row at a far-away position (further than `max_position_step` from the row before) that is not a time step.
4. Drop runs whose targets are all zero (two 102-row runs with no coverage at all).
5. Label each row train / validation / test by its position in its run (see below).
6. Impute missing `jitter` values via linear regression on `delay` and missing `distance` via linear regression on `pdr`. The regressions are fitted on training rows only.
7. Apply a rolling-mean smoothing filter (configurable window length) to the columns in `smooth_cols`, restarting at every run. The targets are left unsmoothed: smoothing them makes each future value largely a function of values already in the input window.
8. Subset to the configured feature set.

Two properties of the data are left as they are. In about a quarter of the rows nothing is received and delay is recorded as 0, so a delay of 0 means "no packets", not "instant". And inside many runs the vehicle position jumps once (for example at row 29 of the 102-row runs); the channel measurements change no more at those rows than they do between ordinary steps, so the run is not cut there.

**Window construction** (in `src/data/dataset.py`): inside each run, sliding windows of length `window_len` are extracted with stride `stride`; the target is the next `future_win_len` timesteps of `throughput` and `delay`. No window or target crosses a run boundary, and runs too short for one sample are skipped.

**Train / validation / test split**: the rows of each run are split chronologically, by default 60 / 20 / 20. (A 10 % validation slice was tried on an earlier version of the pipeline and rejected because its loss did not track test loss.) A sample belongs to the split that holds all of its target rows; samples whose targets straddle a split boundary are dropped. Training samples lie entirely within the training rows, so no validation or test row is ever seen in training. The input window of a validation or test sample may reach back into earlier rows of the same run, as it would in deployment. A noise-augmented copy of the training samples is added to the training data; validation and test samples are never augmented.

---

## Project Structure

```
.
├── ML-Aided-NR-V2X-QoS-Predictions/
│   └── dataset.csv              # Raw NR-V2X dataset
├── api/
│   ├── main.py                  # FastAPI inference server
│   └── schemas.py               # Pydantic request/response models
├── checkpoints/                 # Saved model weights & scalers (generated)
├── configs/
│   └── default.yaml             # Default hyperparameter configuration
├── docker/
│   ├── Dockerfile.server        # Server container
│   ├── Dockerfile.client        # Client container
│   └── docker-compose.yml       # Multi-container orchestration
├── notebooks/                   # Experimental & analysis notebooks
│   └── analysis/
├── plots/                       # EDA plots
├── results/                     # Search CSVs, server history, client logs (generated)
├── scripts/
│   ├── run_server.py            # Entry point: FL server
│   ├── run_client.py            # Entry point: FL client
│   ├── run_simulation.py        # Whole federation in one process (no gRPC)
│   ├── search_configs.py        # Random / grid search over configurations and seeds
│   └── export_model.py          # Export trained model to ONNX
├── src/
│   ├── config.py                # Config loading and validation
│   ├── data/
│   │   ├── preprocessing.py     # Data loading, run labelling, cleaning, split labels, imputation, smoothing
│   │   ├── dataset.py           # PyTorch Dataset, per-run windows, noise augmentation
│   │   ├── pipeline.py          # Config-driven pipeline shared by server and clients
│   │   └── partitioning.py      # Assigning runs to clients, scaling & DataLoader creation
│   ├── fl/
│   │   ├── client.py            # Flower NumPyClient implementation
│   │   ├── server.py            # FedAvg strategy factory, evaluator that saves best/final rounds
│   │   ├── simulation.py        # In-process FedAvg used by run_simulation and the search
│   │   └── utils.py             # Parameter serialisation helpers, seeding
│   ├── models/
│   │   └── lstm.py              # Multi-output LSTM definition
│   └── training/
│       └── train.py             # Local train & validate loops
├── tests/
│   ├── test_model.py            # LSTM forward-pass shape test
│   ├── test_data.py             # Run handling, cleaning, imputation, windowing, split and leakage tests
│   ├── test_training.py         # Loss averaging, repeatable simulation, config validation, strategy
│   └── test_api.py              # Inference API tests
├── pyproject.toml
├── requirements.txt             # Pinned runtime dependencies
└── requirements-dev.txt         # Adds pytest and httpx for the tests
```

---

## Architecture

### Model — `src/models/lstm.py`

A standard stacked LSTM followed by a two-layer fully-connected head:

```
Input  (batch, window_len, input_size)
  └─► LSTM (hidden_size, num_layers, optional bidirectional)
        └─► Last hidden state  →  Linear(hidden_size → 128)  →  ReLU
              └─► Linear(128 → future_win_len × num_targets)
                    └─► Reshape  →  (batch, future_win_len, num_targets)
```

Default targets: `[throughput, delay]` — i.e. `num_targets = 2`.

### Federated Setup — `src/fl/`

- **Strategy**: `FedAvg` from Flower. At each round, a fraction of clients (`fraction_fit`) are selected, train locally for `local_epochs` epochs using Adam + MSE loss, and send updated weights back to the server.
- **Aggregation**: the server averages client weights and broadcasts the global model.
- **Evaluation**: each client evaluates the global model on its local validation split; the server evaluates it on the pooled validation and test sets after every round and writes `results/server_history.csv`.
- **Checkpoints**: the server saves `global_model_best.pt` (the round with the lowest validation loss) and `global_model_final.pt` (the last round), plus `training_summary.json` with the metrics of both. The test set never influences which round is kept.
- **Logging**: each client writes per-round `train_loss` and the validation and test loss and per-target R² to `results/client_<id>_log_<timestamp>.csv` (one file per run). `train_loss` is empty for rounds in which the client was not selected for training.

### Data Partitioning — `src/data/partitioning.py`

Each client receives a contiguous group of whole simulation runs, chosen so that clients hold a similar number of training samples. Because every run is already split into train / validation / test, each client's three local sets come from the same runs, and clients differ from each other in the radio settings they have seen (non-IID). The pooled validation and test sets used by the server do not depend on how many clients there are. All clients share one feature `MinMaxScaler` and one target `MinMaxScaler`, fitted on the training windows of all clients (equivalent to the server aggregating per-client min/max); these are saved alongside the model checkpoint for inference.

---

## Installation

**Requirements**: Python ≥ 3.10 (developed and tested on 3.12), pip.

```bash
git clone <repo-url>
cd fed_l
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

The scripts in `scripts/` add the project root to the import path themselves. For anything else (e.g. `uvicorn api.main:app`), run from the project root.

---

## Configuration

All tuneable parameters live in `configs/default.yaml`:

```yaml
seed: 42                # Noise augmentation, weight init, batch shuffling

data:
  dataset_path: "./ML-Aided-NR-V2X-QoS-Predictions/dataset.csv"
  feature_cols: ["throughput", "sinr", "dlpathloss", "delay", "jitter"]
  target_cols:  ["throughput", "delay"]
  smooth_cols:  ["sinr", "dlpathloss", "jitter"]  # Smoothed columns (targets left raw)
  filter_len:   10      # Rolling-mean smoothing window
  run_cols:     ["mcs", "pktsize", "power"]  # A new simulation run starts where any of these change
  max_position_step: 60  # A run's last row is dropped if its X/Y jumps further than this
  drop_zero_runs: true  # Drop runs whose targets are all zero
  window_len:   30      # LSTM input sequence length
  future_win_len: 5     # Forecast horizon (timesteps)
  stride:       5       # Sliding window stride
  val_ratio:    0.2     # Share of each run used for validation (picks the best round)
  test_ratio:   0.2     # Share of each run used for testing (reporting only)
  noise_level:  0.0     # Training-only noise augmentation; 0 disables

model:
  input_size:   5
  hidden_size:  32
  num_layers:   1
  dropout:      0.3
  bidirectional: false

training:
  batch_size:     64
  learning_rate:  0.001
  local_epochs:   2

federation:
  num_clients:      7
  num_rounds:       200
  fraction_fit:     0.7
  min_fit_clients:  5

deployment:
  checkpoint_dir: "./checkpoints"
  results_dir:    "./results"
  inference_checkpoint: "global_model_best.pt"  # or global_model_final.pt
  api_host:       "0.0.0.0"
  api_port:       8000
```

Override by pointing `CONFIG_PATH` to a different YAML file. Every entry point loads the config through `src/config.py`, which rejects settings that would crash or hang later (for example `min_fit_clients` larger than `num_clients`).

`seed` makes the single-process run repeatable: two runs with the same seed give identical results, on CPU and GPU. Runs with separate client processes are seeded too, but their results may differ in the last digits from one run to the next.

---

## Usage

### Running with Python Scripts

**1. Start the FL server** (loads and partitions data, initialises global model, runs federation):

```bash
python scripts/run_server.py
```

The server listens on `0.0.0.0:8080` by default. Override with:

```bash
SERVER_ADDRESS=0.0.0.0:9090 python scripts/run_server.py
```

The server waits until `num_clients` clients are connected before starting. The shared feature/target scalers are written to `checkpoints/scalers.joblib` at startup; `checkpoints/global_model_best.pt` and `checkpoints/global_model_final.pt` are updated after every round.

**2. Start a client** (in a separate terminal for each client):

```bash
CLIENT_ID=0 SERVER_ADDRESS=0.0.0.0:8080 python scripts/run_client.py
CLIENT_ID=1 SERVER_ADDRESS=0.0.0.0:8080 python scripts/run_client.py
# ... repeat for CLIENT_ID 2 through num_clients-1
```

Clients may be started before the server; they wait until it accepts connections.

> **Note**: in this simulation setup, each client loads the full dataset and selects its own partition. In a real deployment, each client would hold only its own local data file.

### Running in a Single Process

To train without gRPC (same data pipeline, local training loop and checkpointing, clients run one after another):

```bash
python scripts/run_simulation.py
```

This is the fastest way to run experiments on one machine. The configuration search uses the same code:

```bash
# 60 random configurations from the search space defined at the top of the script
python scripts/search_configs.py --random 60 --tag random

# Re-run the 6 best of those with three new seeds (the search itself uses seed 0)
python scripts/search_configs.py --top 6 --from-tag random --seeds 1 2 3 --tag confirm

# Vary one setting on top of the config file
python scripts/search_configs.py --grid data.filter_len=1,5,10,20 --seeds 1 2 3 --tag filter_len
```

Each run stops once the validation loss has not improved for `--patience` rounds (20 by default, with at most `--max-rounds` = 300 rounds). Results are appended to `results/config_search.csv` (one row per run) and `results/config_search_history.csv` (one row per round), and finished runs are skipped when the command is repeated.

### Running with Docker

Build and start the server, seven clients (matching `num_clients` in the default config) and the inference API:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The compose file maps:
- `8080` → Flower gRPC port (`fl-server`)
- `8000` → FastAPI inference port (`api`)
- `./checkpoints` and `./results` as bind mounts

The number of client services must equal `federation.num_clients`, otherwise the server waits forever; add or remove `client-N` blocks if you change it. Clients that start before the server wait for it.

The images install the CPU-only build of PyTorch, so training in containers runs on the CPU; for GPU training run the scripts directly. Each container is limited to two CPU threads (`OMP_NUM_THREADS` in the Dockerfiles) so that the nine containers do not compete for every core. Files written to the bind mounts are owned by `root`. The `api` service reports `not_ready` on `/health` until training has written `global_model_best.pt` — restart it once training finishes.

### REST API

After training completes (so `checkpoints/global_model_best.pt` and `checkpoints/scalers.joblib` exist), start the inference API. It loads the checkpoint named by `deployment.inference_checkpoint`:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Health check:**

```
GET /health
→ {"status": "healthy", "model_loaded": true, "scalers_loaded": true}
```

**Predict:**

```
POST /predict
Content-Type: application/json

{
  "window": [
    [throughput, sinr, dlpathloss, delay, jitter],
    ...  // one row per time step, oldest first
  ]
}
```

Response:

```json
{
  "throughput": [t+1, t+2, t+3, t+4, t+5],
  "delay":      [t+1, t+2, t+3, t+4, t+5]
}
```

Send the raw measurements in original units; the API applies the same rolling mean to the `smooth_cols` columns as the training pipeline. The model uses the last `window_len` rows. Up to `filter_len - 1` earlier rows may be included (so `window_len` to `window_len + filter_len - 1` rows in total), which gives the rolling mean of the first rows its full history. Any other shape returns `400`. Every value must be present: the jitter imputation used in training is not available at inference. Predictions are in original units and are clipped at 0, because throughput and delay cannot be negative.

### Exporting to ONNX

```bash
python scripts/export_model.py
```

Exports the checkpoint named by `deployment.inference_checkpoint` (default `checkpoints/global_model_best.pt`) to a `.onnx` file of the same name, with a dynamic batch-size axis, compatible with ONNX Runtime.

---

## Notebooks

`config_search_and_final_run.ipynb` reports the results of the current `src/` pipeline. The other notebooks are self-contained experiments that do not use `src/`.

| Notebook | Purpose |
|---|---|
| `config_search_and_final_run.ipynb` | Configuration search and final-run results (current pipeline) |
| `eda.ipynb` | Exploratory data analysis, correlation map |
| `nr-v2x-qos-modified.ipynb` | Baseline centralised LSTM experiment |
| `centralised.ipynb` | Centralised training reference |
| `centralised-delay.ipynb` | Centralised delay-only prediction |
| `centralised-autoregressive-seq-to-one.ipynb` | Seq-to-one autoregressive variant |
| `federated-delay-v4.ipynb` | Federated delay prediction (v4) |
| `tp_and_delay-v3.3.2.ipynb` | Joint throughput + delay federated model |
| `jitter_and_delay-v2.1.ipynb` | Federated jitter + delay model |
| `hyp-fed-tune_delay.ipynb` | Hyperparameter sweep — delay |
| `hyp-fed-tune_tp.ipynb` | Hyperparameter sweep — throughput |
| `tp-delay-tuning.ipynb` | Joint tuning for throughput & delay |
| `gan_augmentation.ipynb` | GAN-based data augmentation experiment |
| `analysis/analysis.ipynb` | Post-hoc result analysis |

---

## Results

All numbers come from the `src/` pipeline described above: per-run windows and splits (60 / 20 / 20), shared scalers, raw (unsmoothed) targets. Configurations and rounds are chosen on validation loss; test metrics are only reported. Losses are the mean squared error per sample on scaled targets and R² is in original units, both on the pooled sets of all clients (303 validation and 303 test samples). Charts and full tables are in `notebooks/config_search_and_final_run.ipynb`.

### Reference: repeat-last-value baseline

Predicting that each target keeps its last observed value for all 5 future steps:

| Set | Loss | R² throughput | R² delay |
|---|---|---|---|
| Validation | 0.0185 | 0.920 | 0.040 |
| Test | 0.0299 | 0.795 | -0.245 |

The last 20 % of each run (test) is harder than the 20 % before it (validation), so each set is compared with its own baseline. In scaled units throughput has about 6 times the variance of delay, so the loss, and every choice made on it, is driven mostly by throughput.

### Default configuration

`python scripts/run_simulation.py` with `configs/default.yaml` (unidirectional LSTM, 1 layer, 32 hidden units, 30-step window, learning rate 0.001, batch size 64, 2 local epochs, 5 of 7 clients per round, 200 rounds, seed 42; about 30 seconds on an RTX 3050).

| Checkpoint | Round | Validation loss | Test loss | Test R² throughput | Test R² delay |
|---|---|---|---|---|---|
| `global_model_best.pt` | 118 | 0.0143 | 0.0247 | 0.804 | 0.129 |
| `global_model_final.pt` | 200 | 0.0146 | 0.0247 | 0.806 | 0.116 |
| Repeat last value | – | 0.0185 | 0.0299 | 0.795 | -0.245 |

The test loss is 17 % below the baseline. Throughput R² is slightly above the baseline (0.804 vs 0.795) and delay R² is clearly above it, but a delay R² of 0.13 is still only slightly better than always predicting the mean delay: delay five steps ahead is largely unpredictable from these inputs. With three other seeds the same configuration averages 0.0146 validation loss, 0.0248 test loss, 0.802 throughput R² and 0.126 delay R².

### Configuration search

207 runs in total (`scripts/search_configs.py`, about 5.8 hours of run time on an RTX 3050).

**Random search, 90 configurations, seed 0** over hidden size, layers, dropout, bidirectional, learning rate, batch size, local epochs, window length, smoothing window, noise level, fraction of clients per round and input features. Most configurations land close together: 85 of 90 beat the baseline on validation loss, and the middle half lies between 0.0154 and 0.0166 validation loss (0.0243 to 0.0264 test loss). Fewer layers, bidirectional models, a 30-step window, the five base inputs and at most 5 local epochs do slightly better. Validation and test loss have a rank correlation of 0.41 across configurations, so the search can discard poor configurations but cannot rank the good ones.

**Six best configurations, three new seeds each.** The top three are tied at 0.0146 validation loss (seed standard deviation 0.0002 to 0.0003). The simplest of the three became the default: one unidirectional layer of 32 units on the five base inputs. It also has the highest validation delay R² of the six.

**One setting at a time on the default, three seeds per value** (mean validation loss / test loss at the best round; the default value gives 0.0146 / 0.0248):

| Setting | Values tried | Result |
|---|---|---|
| Smoothing window `filter_len` | 1, 3, 5, 10, 15, 20, 30 | No gain from smoothing: 0.0145–0.0146 up to 15, slightly worse (0.0148–0.0149) at 20 and 30 |
| Input features | 5 base; base + MCS, packet size, power | Extra inputs are slightly worse: 0.0152 / 0.0251 |
| Local epochs | 1, 2, 5, 10, 20 | No effect on the loss (0.0146–0.0148); more local epochs need fewer rounds (209, 104, 44, 26, 10) |
| Number of clients | 1, 3, 5, 7, 10 | Small cost of federation on validation only: 0.0139 / 0.0245 with one client (centralised), 0.0146 / 0.0248 with 7, 0.0148 / 0.0249 with 10; more clients need more rounds (23 with one, 104 with 7, 139 with 10) |
| Window length | 10, 15, 30 | Validation and test disagree: 0.0157 / 0.0238 at 10, 0.0153 / 0.0239 at 15, 0.0146 / 0.0248 at 30 |
| Hidden size | 32, 64, 128, 256, 512, 1024 | Smaller is better: 0.0146 / 0.0248 at 32, rising to 0.0161 / 0.0275 at 1024 |
| LSTM layers | 1, 2, 3 | One layer is clearly best: 0.0146, 0.0156 and 0.0190 for 1, 2 and 3 layers |
| Bidirectional | no, yes | No difference for one layer: 0.0146 vs 0.0145 |

The seed-to-seed standard deviation of the validation loss is about 0.0002, so differences of that size are noise.

`filter_len` is left at 10; `filter_len: 1` (no smoothing) performs the same.

### Limits

- Validation and test are later parts of the same simulation runs used for training, so the numbers measure forecasting under radio settings the model has already seen, not generalisation to new settings.
- Nothing was tuned for delay specifically, because the loss is dominated by throughput.
- More than a third of the samples (107 of the 303 test samples) come from stretches where throughput and delay do not change at all. The model and the baseline both get these right, so the scores say less about the varying runs than they appear to.
- The window length is the one setting where the choice made on validation (30 steps) does not carry over to the test set, which prefers 10 or 15.
- The final-run table is a single seed; the three-seed averages are given above.

### Result files

| File | Contents |
|---|---|
| `results/config_search.csv` | One row per search run: settings, best-round and final-round metrics |
| `results/config_search_history.csv` | Per-round metrics for every search run |
| `results/final_run_history.csv`, `results/final_run_summary.json` | Snapshot of the default-configuration run above |
| `results/server_history.csv` | Per-round metrics of the most recent run (not tracked) |
| `results/client_<id>_log_<timestamp>.csv` | Per-client loss and R² per round when running separate clients (not tracked) |
| `checkpoints/training_summary.json` | Best-round and final-round metrics of the most recent run (not tracked) |

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The tests cover:

- **Data** (`test_data.py`): run labelling (settings changes and new source files), end-marker and all-zero-run removal, duplicate rows being kept, split labels, imputation fitted on training rows only, smoothing and windows staying inside one run, no validation or test rows in training data, clients holding whole runs, augmentation on training data only.
- **Training** (`test_training.py`): losses averaged per sample, a small end-to-end federated simulation that must give identical results when repeated, config validation, and the strategy waiting for all clients before sampling.
- **API** (`test_api.py`): health check, prediction shape, accepted and rejected input shapes, non-negative predictions.
- **Model** (`test_model.py`): output shape `(batch_size, future_win_len, num_targets)`.

---

## Dependencies

Versions are pinned in `requirements.txt` (and mirrored in `pyproject.toml`) to the ones the project is developed and tested with, on Python 3.12. The Docker images install the same versions, with the CPU build of PyTorch.

| Package | Purpose |
|---|---|
| `torch 2.7.0` | LSTM model, training |
| `flwr 1.18.0` | Federated learning framework (Flower) |
| `fastapi 0.143.0` | REST inference API |
| `uvicorn 0.54.0` | ASGI server for FastAPI |
| `scikit-learn 1.6.1` | MinMaxScaler, R² metric, linear regression imputation |
| `pandas 2.3.1` | Data loading and logging |
| `numpy 2.1.0` | Array operations |
| `pyyaml 6.0.2` | Configuration parsing |
| `joblib 1.6.0` | Scaler serialisation |
| `onnx 1.20.1` | Model export format |
| `onnxruntime 1.31.0` | ONNX inference runtime |
| `pydantic 2.11.7` | API schema validation |

`flwr 1.18.0` requires `protobuf` 4.x. An environment that also contains a recent TensorFlow (which needs `protobuf` 5 or later) shows a pip warning about this; training and the tests run correctly there, but a clean install from `requirements.txt` avoids the conflict.