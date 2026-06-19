# ==============================================================================
# Quantangle-SAT: A Quantum SAT Solver Based on Entanglement and Equivalence Checking
# Authors: Shang-Wei LIN, Ji-Qing YAN, Yean-Ru CHEN, Zhe HOU, David Sanán
# Source Code for SAT 2026 Submission
# ==============================================================================

import os
import json
import argparse
import pandas as pd
import numpy as np
import itertools
import random
import time
from datetime import datetime
from typing import Iterator, List, Tuple
from src.quantangle_sat import Solver

def generate_solution_diagonals(n_total: int, k_solutions: int, num_to_generate: int = -1) -> Iterator[Tuple[int, List[int]]]:
    """
    Generates diagonal matrices with k solution entries (-1).
    
    Args:
        n_total: Total dimension (2^n).
        k_solutions: Number of -1 entries (solutions).
        num_to_generate: Number of samples to yield. If -1, yields all combinations.
    """
    if num_to_generate == -1:
        # Exhaustive: All possible combinations
        index_source = itertools.combinations(range(n_total), k_solutions)
    else:
        # Sampling: Generate random samples on the fly
        index_source = (random.sample(range(n_total), k_solutions) for _ in range(num_to_generate))

    for i, solution_indices in enumerate(index_source):
        diag = [1] * n_total
        for idx in solution_indices:
            diag[idx] = -1
        yield i, diag

def run_experiment(args):
    """
    Core experiment controller for Quantangle-SAT.
    Executes parameter sweeps or random instance sampling and saves results.
    """
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    start_total = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"{timestamp}_{args.exp_id}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save metadata for reproducibility
    with open(os.path.join(exp_dir, "experiment_metadata.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    solver = Solver()
    n_total = 2**args.n
    s_list = args.s if isinstance(args.s, list) else [args.s]
    L_list = args.L if isinstance(args.L, list) else [args.L]
    all_dfs = []

    print("-" * 60)
    print(f"Quantangle-SAT Multi-Parameter Experiment: {args.exp_id}")
    print(f"System: n={args.n} (N={n_total}), m={args.m}")
    print(f"Evaluating Confidence Levels (L): {L_list}")
    print(f"Evaluating Sampling Sizes (s): {s_list}")
    print(f"Error Bound Method: {args.bound_method}")
    print("-" * 60)

    # --- Task Preparation ---
    if getattr(args, 'binom_k_trials', 0) > 0:
        # Uniform Random Boolean Function model analysis 
        print(f"[MODE] Random Instance Model: K ~ Binomial(N, 0.5) for {args.binom_k_trials} trials.")
        experiment_tasks = [(np.random.binomial(n_total, 0.5), 1) for _ in range(args.binom_k_trials)]
    else:
        # Targeted solution-count scan
        print(f"[MODE] Specific K-Solution Scan.")
        target_ks = sorted(args.K_list) if args.K_list else list(range(args.K_min, args.K_max + 1))
        experiment_tasks = []
        for k in target_ks:
            reps = args.reps_high if k <= args.reps_threshold else args.reps
            experiment_tasks.append((k, reps))

    methods_to_run = ["exact", "hoeffding"] if args.bound_method == "compare" else [args.bound_method]
    task_total = len(experiment_tasks)

    # --- Execution Loop ---
    for method in methods_to_run:
        for L_val in L_list:
            for s_val in s_list:
                raw_results = []
                s_label = f"s={s_val}" if s_val != -1 else "Adaptive"
                print(f"\n[BATCH START] L={L_val}, {s_label}")
                print("-" * 65)
                
                for step_idx, (k, current_reps) in enumerate(experiment_tasks):
                    progress_str = f"[{step_idx + 1}/{task_total}]"
                    print(f"{progress_str:8} Testing K={k:3d} | reps={current_reps} ...", end=" ", flush=True)
                    
                    start_k = time.perf_counter()
                    k_correct_count = 0
                    
                    # Generate matrix with K solutions
                    gen = generate_solution_diagonals(n_total, k, current_reps)
                    
                    for idx, diag in gen:
                        # Execute Solver based on Equivalence Checking 
                        ans, ratio, t_stop, s_actual, history = solver.solve(diag, args.m, L_val, s_val, bound_method=method)
                        expected = "no" if k == 0 else "yes"

                        if ans == expected:
                            status_code = 1
                        elif ans == "uncertain":
                            status_code = -1
                        else:
                            status_code = 0

                        if status_code == 1:
                            k_correct_count += 1

                        raw_results.append({
                            "bound_method": method,
                            "n": args.n, "K_actual": k, "m": args.m,
                            "s_init": s_val, "s_actual": s_actual,
                            "confidence_level": L_val, "is_correct": status_code,
                            "t_stop": t_stop,
                            "history": history
                        })
                    
                    elapsed_k = time.perf_counter() - start_k
                    accuracy = (k_correct_count / current_reps) * 100
                    print(f" Success: {accuracy:6.2f}% | Time: {elapsed_k:6.2f}s")
                
                current_df = pd.DataFrame(raw_results)
                all_dfs.append(current_df)

    # --- Data Persistence ---
    full_df = pd.concat(all_dfs)
    # Save main results: one row per trial
    full_df.to_csv(os.path.join(exp_dir, "results.csv"), index=False)

    # Flatten history for diagnostic analysis
    diag_records = []
    for _, row in full_df.iterrows():
        for step in row['history']:
            diag_records.append({
                "K_actual": row['K_actual'],
                "s_actual": row['s_actual'],
                "t": step['t'],
                "bf": step['bf'],
                "is_lean_correct": step['is_lean_correct']
            })
    
    pd.DataFrame(diag_records).to_csv(os.path.join(exp_dir, "diagnostics.csv"), index=False)
    print(f"\n[Export] Results saved to: {exp_dir}")

    print("\n" + "-" * 60)
    print(f"Experiment Completed. Total Execution Time: {time.perf_counter() - start_total:.2f}s")
    print("-" * 60)

def get_args():
    """Parses command line arguments for the Quantangle-SAT experiment."""
    parser = argparse.ArgumentParser(
        description="Quantangle-SAT: A Quantum SAT Solver based on Entanglement and Equivalence Checking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- System Specification ---
    parser.add_argument("--n", type=int, default=4, 
                        help="Number of Boolean variables (System size N=2^n)")
    parser.add_argument("--m", type=int, default=2, 
                        help="Number of measurement types (observables) for QEC")
    
    # --- Instance Generation (K-Solution Distribution) ---
    parser.add_argument("--binom_k_trials", type=int, default=0, 
                        help="Number of trials for the 'Uniform Random Boolean Function' model. "
                             "Generates instances where K follows a Binomial(N, 0.5) distribution.")
    parser.add_argument("--K_min", type=int, default=0, 
                        help="Lower bound of solutions (K) for range scanning")
    parser.add_argument("--K_max", type=int, default=4, 
                        help="Upper bound of solutions (K) for range scanning")
    parser.add_argument("--K_list", type=int, nargs='+', default=None, 
                        help="Specific list of K values to evaluate (overrides K_min/max)")
    
    # --- Statistical Reliability ---
    parser.add_argument("--reps", type=int, default=100, 
                        help="Standard number of repetitions for each K")
    parser.add_argument("--reps_high", type=int, default=None, 
                        help="Increased repetitions for critical regions (e.g., near K=0/1)")
    parser.add_argument("--reps_threshold", type=int, default=-1, 
                        help="Threshold K below which 'reps_high' is used")
    
    # --- Algorithm Parameters (L and s) ---
    parser.add_argument("--L", type=float, nargs='+', default=[0.95], 
                        help="Confidence levels for the SAT decision (e.g., 0.95 0.99)")
    parser.add_argument("--s", type=int, nargs='+', default=[-1], 
                        help="Sampling size for QEC. Use -1 for Adaptive Mode (based on finite-sample normal approximation)")
    
    parser.add_argument("--bound_method", type=str, default="exact", 
                        choices=["exact", "hoeffding", "compare"], 
                        help="Error bound calculation method (exact, hoeffding, or compare)")

    # --- Metadata and IO ---
    parser.add_argument("--exp_id", type=str, default="Exp", 
                        help="Experiment identifier for the output directory")

    args = parser.parse_args()

    # Apply default value for high-repetition count if not specified
    if args.reps_high is None:
        args.reps_high = args.reps
        
    return args

if __name__ == "__main__":
    # Initialize arguments
    args = get_args()
    
    # Execute the primary Quantangle-SAT experiment
    run_experiment(args)