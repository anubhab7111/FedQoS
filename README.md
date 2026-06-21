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
- **Data augmentation**: Gaussian noise is added to training windows to improve generalisation.
- **Hyperparameter sweep support**: results from grid searches over learning rate, hidden size, number of layers, and window parameters are logged to CSV for analysis.

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

**Preprocessing pipeline** (in `src/data/preprocessing.py`):
1. Drop duplicates and unnamed index columns.
2. Impute missing `jitter` values via linear regression on `delay`; impute missing `distance` via linear regression on `pdr`.
3. Apply a rolling-mean smoothing filter (configurable window length) to the feature columns.
4. Subset to the configured feature set.

**Window construction**: sliding windows of length `window_len` are extracted with stride `stride`; the target is the next `future_win_len` timesteps of `throughput` and `delay`. A noise-augmented copy of each window set is concatenated to the training data.

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
├── client_results/              # Per-client prediction outputs (generated)
├── configs/
│   └── default.yaml             # Default hyperparameter configuration
├── docker/
│   ├── Dockerfile.server        # Server container
│   ├── Dockerfile.client        # Client container
│   └── docker-compose.yml       # Multi-container orchestration
├── notebooks/                   # Experimental & analysis notebooks
│   └── analysis/
├── plots/                       # Saved training/evaluation plots
│   ├── throughput/
│   ├── delay/
│   ├── jitter/
│   └── tp/
├── results/                     # Hyperparameter sweep CSVs (generated)
├── scripts/
│   ├── run_server.py            # Entry point: FL server
│   ├── run_client.py            # Entry point: FL client
│   └── export_model.py          # Export trained model to ONNX
├── src/
│   ├── data/
│   │   ├── preprocessing.py     # Data loading, imputation, smoothing
│   │   ├── dataset.py           # PyTorch Dataset & sliding-window augmentation
│   │   └── partitioning.py      # Federated data splitting & DataLoader creation
│   ├── fl/
│   │   ├── client.py            # Flower NumPyClient implementation
│   │   ├── server.py            # FedAvg strategy factory
│   │   └── utils.py             # Parameter serialisation helpers
│   ├── models/
│   │   └── lstm.py              # Multi-output LSTM definition
│   └── training/
│       └── train.py             # Local train & validate loops
├── tests/
│   └── test_model.py            # LSTM forward-pass shape test
├── pyproject.toml
└── requirements.txt
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
- **Evaluation**: each client evaluates the global model on its local held-out set; the server runs an additional centralised evaluation on a combined global test set.
- **Logging**: each client appends per-round `train_loss`, `val_loss`, and per-target R² scores to `results/client_<id>_log.csv`.

### Data Partitioning — `src/data/partitioning.py`

The full dataset is split into `num_clients` contiguous chunks (preserving time order). Each chunk is further split 80/20 (chronological) into local train/test sets. Input features and targets are scaled independently per client using `MinMaxScaler`; scalers are saved alongside the model checkpoint for inference.

---

## Installation

**Requirements**: Python ≥ 3.10, pip.

```bash
git clone <repo-url>
cd fed_l
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

Set the Python path so imports resolve correctly:

```bash
export PYTHONPATH=$(pwd)
```

---

## Configuration

All tuneable parameters live in `configs/default.yaml`:

```yaml
data:
  dataset_path: "./ML-Aided-NR-V2X-QoS-Predictions/dataset.csv"
  feature_cols: ["throughput", "sinr", "dlpathloss", "delay", "jitter"]
  target_cols:  ["throughput", "delay"]
  filter_len:   10      # Rolling-mean smoothing window
  window_len:   15      # LSTM input sequence length
  future_win_len: 5     # Forecast horizon (timesteps)
  stride:       5       # Sliding window stride

model:
  input_size:   5
  hidden_size:  1024
  num_layers:   3
  dropout:      0.3
  bidirectional: false

training:
  batch_size:     64
  learning_rate:  0.001
  local_epochs:   100

federation:
  num_clients:      7
  num_rounds:       50
  fraction_fit:     0.7
  min_fit_clients:  5

deployment:
  checkpoint_dir: "./checkpoints"
  results_dir:    "./results"
  api_host:       "0.0.0.0"
  api_port:       8000
```

Override by pointing `CONFIG_PATH` to a different YAML file.

---

## Usage

### Running with Python Scripts

**1. Start the FL server** (loads and partitions data, initialises global model, runs federation):

```bash
export PYTHONPATH=$(pwd)
python scripts/run_server.py
```

The server listens on `0.0.0.0:8080` by default. Override with:

```bash
SERVER_ADDRESS=0.0.0.0:9090 python scripts/run_server.py
```

After all rounds complete, the global model is saved to `checkpoints/global_model.pt` and the feature/target scalers to `checkpoints/scalers.joblib`.

**2. Start a client** (in a separate terminal for each client):

```bash
CLIENT_ID=0 SERVER_ADDRESS=0.0.0.0:8080 python scripts/run_client.py
CLIENT_ID=1 SERVER_ADDRESS=0.0.0.0:8080 python scripts/run_client.py
# ... repeat for CLIENT_ID 2 through num_clients-1
```

> **Note**: in this simulation setup, each client loads the full dataset and selects its own partition. In a real deployment, each client would hold only its own local data file.

### Running with Docker

Build and start the server plus two clients:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The compose file maps:
- `8080` → Flower gRPC port
- `8000` → FastAPI inference port
- `./checkpoints` and `./results` as named volumes

Add more client services to `docker-compose.yml` by copying the `client-1` block and incrementing `CLIENT_ID`.

### REST API

After training completes (so `checkpoints/global_model.pt` and `checkpoints/scalers.joblib` exist), start the inference API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Health check:**

```
GET /health
→ {"status": "healthy", "model_loaded": true}
```

**Predict:**

```
POST /predict
Content-Type: application/json

{
  "window": [
    [throughput, sinr, dlpathloss, delay, jitter],
    ...  // window_len rows × input_size columns
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

All values are in the original (unscaled) units.

### Exporting to ONNX

```bash
python scripts/export_model.py
```

Exports `checkpoints/global_model.pt` → `checkpoints/global_model.onnx` with a dynamic batch-size axis, compatible with ONNX Runtime.

---

## Notebooks

Exploratory and experimental notebooks are in `notebooks/`. Notable ones:

| Notebook | Purpose |
|---|---|
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

Hyperparameter sweep results are written to `results/` as timestamped CSVs. Each row records the configuration and the best R² score achieved for throughput and delay, along with the round it was achieved in.

Representative best results from sweeps (federated, 50 rounds):

| Hidden Size | Layers | LR | Best R² Throughput | Best R² Delay |
|---|---|---|---|---|
| 512 | 2 | 0.001 | ~0.85 | ~0.38 |
| 256 | 2 | 0.001 | ~0.83 | ~0.38 |

Per-client training logs (loss and R² per round) are stored in `results/client_<id>_log.csv`.

---

## Testing

A basic forward-pass shape test is included:

```bash
pytest tests/
```

Or run directly:

```bash
python tests/test_model.py
```

The test verifies that the LSTM produces output of shape `(batch_size, future_win_len, num_targets)` for a random input.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch >= 2.0` | LSTM model, training |
| `flwr >= 1.18` | Federated learning framework (Flower) |
| `fastapi >= 0.100` | REST inference API |
| `uvicorn >= 0.20` | ASGI server for FastAPI |
| `scikit-learn >= 1.3` | MinMaxScaler, R² metric, linear regression imputation |
| `pandas >= 2.0` | Data loading and logging |
| `numpy >= 1.24` | Array operations |
| `pyyaml >= 6.0` | Configuration parsing |
| `joblib >= 1.3` | Scaler serialisation |
| `onnx >= 1.14` | Model export format |
| `onnxruntime >= 1.16` | ONNX inference runtime |
| `pydantic >= 2.0` | API schema validation |