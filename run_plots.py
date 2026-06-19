# ==============================================================================
# Quantangle-SAT: Data Analysis and Visualization Suite
# Authors: Shang-Wei LIN, Ji-Qing YAN, Yean-Ru CHEN, Zhe HOU, David Sanán
# Used for generating figures and complexity metrics for SAT 2026
# ==============================================================================

import os
import pandas as pd
import argparse
import numpy as np
from scipy.special import comb
from src.visualizer import plot_success_probability, plot_expansion_complexity, plot_combined_efficiency, plot_evolutionary_accuracy, plot_uncertainty_probability

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot results from Quantangle-SAT experiments.")
    parser.add_argument("folder", type=str, help="Path to the experiment output directory.")
    args = parser.parse_args()

    res_path = os.path.join(args.folder, "results.csv")
    if not os.path.exists(res_path):
        print(f"Error: Primary data file not found at {res_path}")
        return

    full_df = pd.read_csv(res_path)
    if 'bound_method' not in full_df.columns:
        full_df['bound_method'] = 'Exact'
    n = int(full_df['n'].iloc[0])
    N = 2**n

    unique_s = sorted(full_df['s_actual'].unique())
    unique_L = sorted(full_df['confidence_level'].unique())
    unique_bounds = sorted(full_df['bound_method'].unique())
    is_compare_mode = len(unique_bounds) > 1

    if not is_compare_mode:
        # Case 1: Fixed confidence level L, varying sample size s
        for l_val in unique_L:
            df_for_l = full_df[full_df['confidence_level'] == l_val]
            df_list = [group for _, group in df_for_l.groupby('s_actual')]
            df_list = sorted(df_list, key=lambda x: x['s_actual'].iloc[0])
            
            if df_list:
                print(f"\n[ANALYSIS] Statistical Report: Fixed Confidence Level L = {l_val}")
                print("-" * 65)
                for df in df_list:
                    s_val = int(df['s_actual'].iloc[0])
                    
                    # Compute success probability for UNSAT (K=0) instances
                    k0_df = df[df['K_actual'] == 0]
                    if not k0_df.empty:
                        p_suc_k0 = k0_df['is_correct'].mean()
                        count_k0 = len(k0_df)
                        print(f"📍 s={s_val:3d} | UNSAT P_suc: {p_suc_k0:.2%} | Trials: {count_k0}")

                    # 2. Complexity Metrics (Uniform vs. Weighted)
                    # Group by K to find average iterations t for each K
                    k_group = df.groupby('K_actual')['t_stop'].mean()
                    
                    # Metric A: Arithmetic Mean (Uniform average across observed K)
                    arithmetic_mean_t = k_group.mean()
                    
                    # Metric B: Weighted Expectation E[t] based on Random Boolean Function model
                    weighted_t_sum = 0
                    total_prob = 0
                    for k_val, t_avg in k_group.items():
                        # Use log-space to prevent overflow with large N
                        try:
                            log_prob = np.log(comb(N, k_val)) - (N * np.log(2))
                            prob_k = np.exp(log_prob)
                        except:
                            prob_k = 0
                            
                        weighted_t_sum += t_avg * prob_k
                        total_prob += prob_k

                    expected_t_val = weighted_t_sum / total_prob if total_prob > 0 else 0
                    
                    print(f"   - Arithmetic Mean t: {arithmetic_mean_t:.4f}")
                    print(f"   - Expected Complexity E[t]: {expected_t_val:.4f} (Weighted)")
                    print(f"   - Unique K states observed: {len(k_group)}")
                    print("-" * 65)

                    # [OPTIONAL] Evolutionary Accuracy Plot
                    # Uncomment the line below to generate Bayesian Diagnostic Accuracy plots for each s
                    if 'history' in df.columns and not df.empty:
                        actual_s = int(df['s_actual'].iloc[0])
                        actual_L = df['confidence_level'].iloc[0]
                        new_suffix = f"_n{n}_s{actual_s}_L{actual_L}"
                        plot_evolutionary_accuracy(df, n, args.folder, suffix=new_suffix)

                suffix = f"_fixed_L{l_val}"
                plot_success_probability(df_list, n, args.folder, suffix=suffix)
                plot_expansion_complexity(df_list, n, args.folder, suffix=suffix)
                plot_combined_efficiency(df_list, n, args.folder, suffix=suffix)

        # Case 2: Fixed sample size s, varying confidence level L
        for s_val in unique_s:
            df_for_s = full_df[full_df['s_actual'] == s_val]
            df_list = [group for _, group in df_for_s.groupby('confidence_level')]
            df_list = sorted(df_list, key=lambda x: x['confidence_level'].iloc[0])
            
            if df_list:
                print(f"\n[ANALYSIS] Statistical Report: Fixed Sampling Size s = {s_val}")
                print("-" * 65)
                for df in df_list:
                    l_val = df['confidence_level'].iloc[0]
                    k0_df = df[df['K_actual'] == 0]
                    if not k0_df.empty:
                        p_suc_k0 = k0_df['is_correct'].mean()
                        print(f"📍 L={l_val:.2f} | UNSAT P_suc: {p_suc_k0:.2%}")
                        
                print("-" * 65)
                suffix = f"_fixed_s{s_val}"
                plot_success_probability(df_list, n, args.folder, suffix=suffix)
                plot_expansion_complexity(df_list, n, args.folder, suffix=suffix)
                plot_combined_efficiency(df_list, n, args.folder, suffix=suffix)

    else:
        # Case 3: Compare different error bound methods (Exact vs Hoeffding)
        if len(unique_bounds) > 1:
            for l_val in unique_L:
                for s_val in unique_s:
                    # Filter data for specific L and s parameters
                    df_fixed = full_df[(full_df['confidence_level'] == l_val) & (full_df['s_actual'] == s_val)]
                    
                    if df_fixed.empty: continue
                    
                    # Group data by bound_method (Exact vs Hoeffding)
                    df_list = [group for _, group in df_fixed.groupby('bound_method')]
                    df_list = sorted(df_list, key=lambda x: str(x['bound_method'].iloc[0]))
                    
                    if len(df_list) > 1:
                        print(f"\n[ANALYSIS] Bound Method Comparison: Fixed s = {s_val}, L = {l_val}")
                        print("-" * 65)
                        for df in df_list:
                            method_name = df['bound_method'].iloc[0]
                            k0_df = df[df['K_actual'] == 0]
                            if not k0_df.empty:
                                p_suc_k0 = k0_df['is_correct'].mean()
                                print(f"📍 Method: {method_name:9s} | UNSAT P_suc: {p_suc_k0:.2%}")
                        print("-" * 65)
                        
                        # Append suffix to prevent overwriting previous plots
                        suffix = f"_compare_s{s_val}_L{l_val}"
                        plot_success_probability(df_list, n, args.folder, suffix=suffix)
                        plot_expansion_complexity(df_list, n, args.folder, suffix=suffix)
                        plot_uncertainty_probability(df_list, n, args.folder, suffix=suffix)

    print(f"\nSuccess: Visualization plots generated in directory: {args.folder}")

if __name__ == "__main__":
    main()