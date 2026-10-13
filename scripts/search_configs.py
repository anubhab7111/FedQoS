import os
import sys

# Ensure project root is on sys.path so `from src...` imports work
# when running this script directly (e.g. `python scripts/search_configs.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import copy
import itertools
import json
import random
import time
import yaml
import pandas as pd
from src.config import load_config, validate_config
from src.fl.simulation import run_simulation

BASE_FEATURES = ["throughput", "sinr", "dlpathloss", "delay", "jitter"]
FEATURE_SETS = {
    "base": BASE_FEATURES,
    # Radio settings of the run, known in advance at inference time
    "settings": BASE_FEATURES + ["mcs", "pktsize", "power"],
}

# Values tried by the random search. Keys are dotted config paths, except
# "features", which names an entry of FEATURE_SETS. Larger models (e.g. hidden
# size 1024) take 20-60 minutes per run; test those with --grid instead.
SPACE = {
    "model.hidden_size": [32, 64, 128, 256, 512],
    "model.num_layers": [1, 2, 3],
    "model.dropout": [0.0, 0.3],
    "model.bidirectional": [False, True],
    "training.learning_rate": [0.0003, 0.001, 0.003],
    "training.batch_size": [32, 64],
    "training.local_epochs": [1, 2, 5, 10, 20],
    "data.window_len": [10, 15, 30],
    "data.filter_len": [1, 5, 10, 20],
    "data.noise_level": [0.0, 0.02, 0.05],
    "federation.fraction_fit": [0.7, 1.0],
    "features": ["base", "settings"],
}

def parse_value(text):
    return yaml.safe_load(text)

def apply_overrides(base_config, overrides):
    """Return a copy of the config with dotted-key overrides applied."""
    config = copy.deepcopy(base_config)
    for key, value in overrides.items():
        if key == "features":
            config["data"]["feature_cols"] = list(FEATURE_SETS[value])
            config["model"]["input_size"] = len(FEATURE_SETS[value])
            continue
        section, name = key.split(".")
        config[section][name] = value
    # Never ask for more clients per round than exist
    config["federation"]["min_fit_clients"] = min(config["federation"]["min_fit_clients"], config["federation"]["num_clients"])
    return validate_config(config)

def main():
    parser = argparse.ArgumentParser(description="Search model, training, data and federation settings with in-process FedAvg.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--random", type=int, metavar="N", help="Sample N configurations from SPACE.")
    mode.add_argument("--grid", nargs="+", metavar="KEY=V1,V2", help="Try every combination of the listed values.")
    mode.add_argument("--top", type=int, metavar="K", help="Re-run the K best configurations (by validation loss) of --from-tag.")
    parser.add_argument("--from-tag", help="Tag to take the best configurations from (with --top).")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Overrides applied to every run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--search-seed", type=int, default=0, help="Seed for sampling configurations (with --random).")
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20, help="Stop a run after this many rounds without a better validation loss.")
    parser.add_argument("--tag", default="search", help="Label stored with each run.")
    parser.add_argument("--out", default=None, help="Output CSV prefix (default: <results_dir>/config_search).")
    args = parser.parse_args()

    base_config = load_config()

    out_prefix = args.out or os.path.join(base_config["deployment"]["results_dir"], "config_search")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    summary_path = f"{out_prefix}.csv"
    history_path = f"{out_prefix}_history.csv"
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame(columns=["tag", "overrides", "seed"])

    fixed = dict(item.split("=", 1) for item in args.set)
    fixed = {k: parse_value(v) for k, v in fixed.items()}

    if args.random is not None:
        rng = random.Random(args.search_seed)
        candidates = [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(args.random)]
    elif args.grid is not None:
        axes = dict(item.split("=", 1) for item in args.grid)
        axes = {k: [parse_value(v) for v in values.split(",")] for k, values in axes.items()}
        candidates = [dict(zip(axes, combo)) for combo in itertools.product(*axes.values())]
    else:
        source = summary[summary["tag"] == args.from_tag]
        ranked = source.groupby("overrides")["best_val_loss"].mean().sort_values()
        candidates = [json.loads(o) for o in ranked.index[:args.top]]

    # Resume: skip runs that are already in the summary file
    done = set(map(tuple, summary[["tag", "overrides", "seed"]].values.tolist()))

    for number, candidate in enumerate(candidates, start=1):
        overrides = {**candidate, **fixed}
        overrides_key = json.dumps(overrides, sort_keys=True)
        for seed in args.seeds:
            if (args.tag, overrides_key, seed) in done:
                print(f"[{number}/{len(candidates)}] seed={seed} skipped (already done)")
                continue

            config = apply_overrides(base_config, overrides)
            config["seed"] = seed
            config["federation"]["num_rounds"] = args.max_rounds

            start = time.time()
            result, history = run_simulation(config, verbose=False, patience=args.patience)

            # Record the full set of searched settings, whether overridden or not
            settings = {"features": overrides.get("features", "base")}
            for key in SPACE:
                if key != "features":
                    section, name = key.split(".")
                    settings[key] = config[section][name]
            settings["federation.num_clients"] = config["federation"]["num_clients"]
            run_info = {"tag": args.tag, "overrides": overrides_key, "seed": seed, **settings}

            # The best round is chosen on validation loss; its test metrics are the reported ones.
            row = dict(run_info)
            row.update({f"best_{k}": v for k, v in result["best"].items()})
            row.update({f"final_{k}": v for k, v in result["final"].items()})
            row["seconds"] = round(time.time() - start, 1)
            pd.DataFrame([row]).to_csv(summary_path, mode="a", header=not os.path.exists(summary_path), index=False)

            history_df = pd.DataFrame(history)
            for k in ("seed", "overrides", "tag"):
                history_df.insert(0, k, run_info[k])
            history_df.to_csv(history_path, mode="a", header=not os.path.exists(history_path), index=False)

            print(f"[{number}/{len(candidates)}] seed={seed} {overrides_key}: best round {row['best_round']}/{row['final_round']}, "
                  f"val_loss={row['best_val_loss']:.5f}, test_loss={row['best_test_loss']:.5f} ({row['seconds']}s)", flush=True)

if __name__ == "__main__":
    main()
