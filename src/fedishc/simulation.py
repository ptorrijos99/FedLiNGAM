import flwr as fl
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.utils.data_generation import SyntheticDataGenerator
from experiments.utils.evaluation import count_accuracy_combined
from fedishc.client import CumulantClient
from fedishc.strategy import FedISHCStrategy

def run_simulation():
    print("="*80)
    print("FedISHC Simulation Run")
    print("="*80)
    
    # 1. Generate Synthetic Data
    gen = SyntheticDataGenerator(seed=42)
    n_samples = 10000 
    p = 5
    X, B_true, true_order = gen.generate(n_samples, p, graph_density=0.6, dist_type='exponential')
    
    # Standardize Data (Unit Variance)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    print(f"Generated Data: {n_samples} samples, {p} variables")
    print(f"True Order: {true_order}")
    print(f"True Adjacency (B):\n{B_true}")
    
    # 2. Partition Data (Horizontal Split)
    n_clients = 3
    # Simple split
    split_indices = np.array_split(np.arange(n_samples), n_clients)
    partitions = [X[indices] for indices in split_indices]
    
    # 3. Define Client Function
    def client_fn(cid: str) -> fl.client.Client:
        idx = int(cid)
        return CumulantClient(partitions[idx])
    
    # 4. Define Strategy
    strategy = FedISHCStrategy(min_available_clients=n_clients)
    
    # 5. Run Simulation
    print("\nStarting Flower Simulation...")
    # FedISHC is one-shot, so 1 round is enough.
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        client_resources={"num_cpus": 1} 
    )
    
    # 6. Evaluate Results
    print("\nSimulation Complete.")
    B_est = strategy.result_B
    est_order = strategy.result_order
    
    print(f"Estimated Order: {est_order}")
    
    if est_order is None:
        print("Error: No result obtained from strategy.")
        return

    # Metrics
    metrics = count_accuracy_combined(B_true, B_est, true_order, est_order)
    
    print("\nMetrics:")
    print(f"  F1 (Order): {metrics['f1_order']:.4f}")
    print(f"  Recall (Order): {metrics['recall_order']:.4f}")
    print(f"  SHD: {metrics['shd']}")
    print(f"  F1 (Edges): {metrics['f1_edges']:.4f}")
    
    # Check strict requirements
    if metrics['f1_order'] == 1.0:
        print("\nSUCCESS: Perfect Causal Order Recovery! 🎉")
    else:
        print("\nWARNING: Order recovery not perfect.")

if __name__ == "__main__":
    run_simulation()
