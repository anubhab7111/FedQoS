import os
import sys

# Ensure project root is on sys.path so `from src...` imports work
# when running this script directly (e.g. `python scripts/run_simulation.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import load_config
from src.fl.simulation import run_simulation
from src.fl.server import BEST_CHECKPOINT, FINAL_CHECKPOINT

def main():
    """Train the federated model in one process (no gRPC), saving the best and final rounds."""
    config = load_config()

    checkpoint_dir = config["deployment"]["checkpoint_dir"]
    results_dir = config["deployment"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    summary, _ = run_simulation(
        config, checkpoint_dir=checkpoint_dir,
        history_path=os.path.join(results_dir, "server_history.csv")
    )
    print(f"Best round {summary['best']['round']} saved to {os.path.join(checkpoint_dir, BEST_CHECKPOINT)}: {summary['best']}")
    print(f"Final round {summary['final']['round']} saved to {os.path.join(checkpoint_dir, FINAL_CHECKPOINT)}: {summary['final']}")

if __name__ == "__main__":
    main()
