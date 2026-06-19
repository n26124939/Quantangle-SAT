import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import ast
from scipy.special import comb

FONT_SIZE = 20 
MARKER_SIZE = 10 
LINE_WIDTH = 2.0

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", 
    "axes.linewidth": 1.2,
    "xtick.direction": "in", 
    "ytick.direction": "in",
})

SAT_COLORS = [
    '#0072B2', '#D55E00', '#009E73', '#E69F00', 
    '#56B4E9', '#CC79A7', '#000000'
]

# Standard marker set for distinct plot lines
SAT_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'P', '*']

def save_paper_plot(output_path, filename):
    """Saves high-quality vector PDF for submission and PNG for preview."""
    plot_dir = os.path.join(output_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    pdf_path = os.path.join(plot_dir, f"{filename}.pdf")
    png_path = os.path.join(plot_dir, f"{filename}.png")
    
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

def get_param_str(df, n):
    """Extracts common parameters into a standard LaTeX label."""
    m = df['m'].iloc[0]
    L = df['confidence_level'].iloc[0]
    return f"$n={n}, m={m}, L={L}$"

def get_adaptive_labels(df_list):
    """Dynamically generates legend labels based on varying parameters."""
    labels = []
    all_L = [df['confidence_level'].iloc[0] for df in df_list]
    all_s = [int(df['s_actual'].mean()) for df in df_list]
    
    has_bound = all('bound_method' in df.columns for df in df_list)
    all_bounds = [df['bound_method'].iloc[0] if has_bound else 'exact' for df in df_list]
    
    L_constant = all(l == all_L[0] for l in all_L)
    s_constant = all(s == all_s[0] for s in all_s)
    bound_constant = all(b == all_bounds[0] for b in all_bounds)

    for df in df_list:
        L_val = df['confidence_level'].iloc[0]
        s_val = int(df['s_actual'].mean())
        bound_val = df['bound_method'].iloc[0] if has_bound else 'exact'
        
        if not bound_constant:
            if bound_val == "exact":
                display_name = "Analytical"
            elif bound_val == "hoeffding":
                display_name = "Hoeffding"
            else:
                display_name = str(bound_val).capitalize()
                
            labels.append(f"Method: {display_name}")
            
        elif not L_constant and s_constant:
            labels.append(f"$L = {L_val}$")
        elif L_constant and not s_constant:
            labels.append(f"$s = {s_val}$")
        else:
            labels.append(f"$L={L_val}, s={s_val}$")
            
    return labels

def get_subtitle_info(df_list, n):
    """Helper to generate the parameter subtitle for the plots."""
    m_val = df_list[0]['m'].iloc[0]
    all_L = [df['confidence_level'].iloc[0] for df in df_list]
    all_s = [int(df['s_actual'].mean()) for df in df_list]
    
    param_parts = [f"n={n}", f"m={m_val}"]
    if all(l == all_L[0] for l in all_L): param_parts.append(f"L={all_L[0]}")
    if all(s == all_s[0] for s in all_s): param_parts.append(f"s={all_s[0]}")
    
    return "$" + ", ".join(param_parts) + "$"

# --- 1. Function: Success Probability (P_succ) ---

def plot_success_probability(df_list, n, output_path, suffix=""):
    """
    Plots the Success Probability P_succ with an axis break (truncation) logic.
    """
    if isinstance(df_list, pd.DataFrame): df_list = [df_list]
    
    param_info = get_subtitle_info(df_list, n)
    labels = get_adaptive_labels(df_list)

    plt.figure(figsize=(8, 6))
    all_acc_values = []

    for i, df in enumerate(df_list):
        summary = df.groupby("K_actual")["is_correct"].apply(lambda x: (x == 1).mean())
        all_acc_values.extend(summary.values)
        
        plt.plot(summary.index, summary.values, 
                 label=labels[i],
                 color=SAT_COLORS[i % len(SAT_COLORS)], 
                 marker=SAT_MARKERS[i % len(SAT_MARKERS)], 
                 markersize=MARKER_SIZE, linestyle='None', 
                 markeredgecolor='black', markeredgewidth=0.8)

    plt.title(r"Success Probability $P_{\mathrm{succ}}$" + "\n", fontsize=FONT_SIZE, fontweight='bold')
    plt.text(0.5, 1.02, param_info, transform=plt.gca().transAxes, ha='center', fontsize=FONT_SIZE, style='italic')
    plt.xlabel(r"Number of Solutions $K$", fontsize=FONT_SIZE)
    plt.ylabel(r"Probability $P_{\mathrm{succ}}$", fontsize=FONT_SIZE)

    # --- Axis Break (Truncation) Logic ---
    ax = plt.gca()
    y_true_min = min(0.8, min(all_acc_values) - 0.01) if all_acc_values else 0.8
    y_fake_zero = y_true_min - 0.05
    plt.ylim(y_fake_zero, 1.02)

    normal_ticks = np.arange(y_true_min, 1.05, 0.05)
    ax.set_yticks([y_fake_zero] + list(normal_ticks))
    ax.set_yticklabels(['0'] + [f"{t:.2f}" for t in normal_ticks])

    # Add the "approx" symbol to indicate broken axis
    ax.text(0, (y_fake_zero + y_true_min) / 2, r"$\approx$", 
            transform=ax.get_yaxis_transform(), 
            fontsize=FONT_SIZE+4, ha='center', va='center', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', pad=2), clip_on=False)

    if len(df_list) > 1:
        plt.legend(fontsize=FONT_SIZE-2, loc='lower right', frameon=True, 
                   edgecolor='black', facecolor='white', framealpha=1.0, fancybox=False)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tick_params(labelsize=FONT_SIZE-2)

    save_paper_plot(output_path, f"psucc_k{suffix}")
    plt.close()

# --- 2. Function: Average Shannon Expansions (t_bar) ---

def plot_expansion_complexity(df_list, n, output_path, suffix=""):
    """
    Plots the average number of Shannon expansions t_bar required to reach a decision.
    """
    if isinstance(df_list, pd.DataFrame): df_list = [df_list]
    
    param_info = get_subtitle_info(df_list, n)
    labels = get_adaptive_labels(df_list)

    plt.figure(figsize=(8, 6))
    
    for i, df in enumerate(df_list):
        summary = df.groupby("K_actual")["t_stop"].mean()
        plt.plot(summary.index, summary.values, 
                 label=labels[i], 
                 color=SAT_COLORS[i % len(SAT_COLORS)], 
                 marker=SAT_MARKERS[i % len(SAT_MARKERS)], 
                 markersize=MARKER_SIZE, linestyle='None', 
                 markeredgecolor='black', markeredgewidth=0.8)

    plt.title(r"Average Shannon Expansions $\bar{t}$" + "\n", fontsize=FONT_SIZE, fontweight='bold')
    plt.text(0.5, 1.02, param_info, transform=plt.gca().transAxes, ha='center', fontsize=FONT_SIZE, style='italic')
    plt.xlabel(r"Number of Solutions $K$", fontsize=FONT_SIZE)
    plt.ylabel(r"Avg. Expansions $\bar{t}$", fontsize=FONT_SIZE)

    if len(df_list) > 1:
        plt.legend(fontsize=FONT_SIZE-2, loc='upper right', frameon=True, 
                   edgecolor='black', facecolor='white', framealpha=1.0, fancybox=False)
                   
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tick_params(labelsize=FONT_SIZE-2)

    save_paper_plot(output_path, f"tbar_k{suffix}")
    plt.close()

def plot_evolutionary_accuracy(df, n, output_path, suffix=""):
    """
    Visualizes the convergence of Bayesian diagnostic accuracy.
    
    This plot demonstrates how the probability of correct satisfiability 
    classification increases as a function of Shannon expansion steps (t).
    """
    rows = []
    for _, row in df.iterrows():
        history_data = row['history']
        
        # Parse serialized history. Restricted eval environment handles 
        # non-standard floating point literals (inf, nan) from simulation logs.
        if isinstance(history_data, str):
            try:
                safe_env = {"inf": float('inf'), "nan": float('nan'), "np": np}
                history_list = eval(history_data, {"__builtins__": None}, safe_env)
            except Exception:
                continue
        else:
            history_list = history_data
        
        for step in history_list:
            rows.append({
                "K_actual": row['K_actual'],
                "t": step['t'],
                "is_lean_correct": step['is_lean_correct']
            })
    
    diag_df = pd.DataFrame(rows)
    if diag_df.empty:
        return

    plt.figure(figsize=(9, 7))
    
    # Filter representative solution counts (K) for visual clarity
    target_ks = sorted(diag_df['K_actual'].unique())
    if len(target_ks) > 5:
        target_ks = [target_ks[0], target_ks[1], target_ks[len(target_ks)//2], target_ks[-1]]

    for i, k in enumerate(target_ks):
        subset = diag_df[diag_df['K_actual'] == k]
        convergence = subset.groupby("t")["is_lean_correct"].mean()
        
        plt.plot(convergence.index, convergence.values, 
                 label=f"$K={k}$",
                 color=SAT_COLORS[i % len(SAT_COLORS)], 
                 marker=SAT_MARKERS[i % len(SAT_MARKERS)], 
                 markersize=MARKER_SIZE, 
                 linestyle='None', 
                 markeredgecolor='black', markeredgewidth=0.8)

    # Reference baseline for random classification (p = 0.5)
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label="Random Guess")

    # Metadata and axis formatting
    cur_s, cur_L, cur_m = int(df['s_actual'].iloc[0]), df['confidence_level'].iloc[0], df['m'].iloc[0]
    param_info = f"$n={n}, m={cur_m}, L={cur_L}, s={cur_s}$"

    plt.title(r"Bayesian Diagnostic Accuracy" + "\n", fontsize=FONT_SIZE, fontweight='bold')
    plt.text(0.5, 1.02, param_info, transform=plt.gca().transAxes, ha='center', fontsize=FONT_SIZE, style='italic')
    
    plt.xlabel(r"Expansion Steps $t$", fontsize=FONT_SIZE)
    plt.ylabel(r"Correct Lean Probability", fontsize=FONT_SIZE)
    
    plt.ylim(-0.05, 1.05) 
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(fontsize=FONT_SIZE-4, loc='lower right', frameon=True, edgecolor='black', fancybox=False)
    
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tick_params(labelsize=FONT_SIZE-2)

    save_paper_plot(output_path, f"bayes_trend{suffix}")
    plt.close()

def plot_uncertainty_probability(df_list, n, output_path, suffix=""):
    """
    Plots P_uncertain with automatic Y-axis scaling and peak value annotation.
    Strictly follows: is_correct == -1 is 'uncertain'.
    """
    if isinstance(df_list, pd.DataFrame): df_list = [df_list]
    param_info = get_subtitle_info(df_list, n)
    labels = get_adaptive_labels(df_list)

    plt.figure(figsize=(8, 6))
    
    global_max_val = 0
    peak_point = (0, 0)

    # 1. 繪圖與數據分析
    for i, df in enumerate(df_list):
        # 🌟 嚴格遵守你的定義：-1 代表 Uncertain
        summary = df.groupby("K_actual")["is_correct"].apply(lambda x: (x == -1).mean())
        
        # 尋找全局最高點用於標記
        if not summary.empty:
            current_max = summary.max()
            if current_max > global_max_val:
                global_max_val = current_max
                peak_point = (summary.idxmax(), current_max)

        plt.plot(summary.index, summary.values, 
                 label=labels[i],
                 color=SAT_COLORS[i % len(SAT_COLORS)], 
                 marker=SAT_MARKERS[i % len(SAT_MARKERS)], 
                 markersize=MARKER_SIZE, 
                 linestyle='None',   # 純散點
                 alpha=1.0,          # 不透明
                 markeredgecolor='black', 
                 markeredgewidth=0.8)

    # 2. 標記最高百分比 (只標記全圖最高的那一個，通常是 Hoeffding)
    if global_max_val > 0:
        plt.annotate(f'{global_max_val:.1%}', 
                     xy=peak_point, 
                     xytext=(5, 10), # 向上偏移 10 points
                     textcoords='offset points', 
                     ha='center', 
                     va='bottom',
                     fontsize=FONT_SIZE-2,
                    #  fontweight='bold',
                     color='black') # 用紅色標出最高值，超顯眼

    plt.title(r"Uncertainty Probability $P_{\mathrm{uncertain}}$" + "\n", fontsize=FONT_SIZE, fontweight='bold')
    plt.text(0.5, 1.02, param_info, transform=plt.gca().transAxes, ha='center', fontsize=FONT_SIZE, style='italic')
    plt.xlabel(r"Number of Solutions $K$", fontsize=FONT_SIZE)
    plt.ylabel(r"Probability $P_{\mathrm{uncertain}}$", fontsize=FONT_SIZE)

    # 3. 🌟 自動 Y 軸縮放：0 到最高值的 1.2 倍（留白給標籤）
    y_limit = max(global_max_val * 1.3, 0.1) # 至少顯示到 0.1，避免全 0 時出錯
    plt.ylim(-0.01, min(y_limit, 1.05)) # 但不要超過 1.05
    
    if len(df_list) > 1:
        plt.legend(fontsize=FONT_SIZE-2, loc='upper right', frameon=True, 
                   edgecolor='black', facecolor='white', framealpha=1.0, fancybox=False)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tick_params(labelsize=FONT_SIZE-2)

    save_paper_plot(output_path, f"punc_k{suffix}")
    plt.close()

def plot_combined_efficiency(df_list, n, output_path, suffix=""):
    """Overlays Average Shannon Expansions with the theoretical P(K) distribution."""
    if isinstance(df_list, pd.DataFrame): df_list = [df_list]
    N = 2**n
    K_range = np.arange(0, N + 1)
    probs = np.exp([np.log(comb(N, k)) - N * np.log(2) for k in K_range])

    fig, ax1 = plt.subplots(figsize=(8, 6))
    # ax2 = ax1.twinx()
    
    # Background: Distribution P(K)
    # ax2.bar(K_range, probs, color='gray', alpha=0.55, width=1.0, label=r'$P(K)$')
    # ax2.set_ylabel(r'Probability $P(K)$', color='gray', fontsize=FONT_SIZE)
    # ax2.tick_params(axis='y', labelcolor='gray', labelsize=FONT_SIZE-4)

    labels = get_adaptive_labels(df_list)

    # Foreground: Experimental t_bar
    for i, df in enumerate(df_list):
        s_val = int(df['s_actual'].mean())
        summary = df.groupby('K_actual')['t_stop'].mean().reset_index()
        ax1.plot(summary['K_actual'], summary['t_stop'], 
                 marker=SAT_MARKERS[i % len(SAT_MARKERS)], linestyle='None', 
                 label=labels[i],
                 color=SAT_COLORS[i % len(SAT_COLORS)],
                 markersize=MARKER_SIZE, markeredgecolor='black')

    param_info = get_param_str(df_list[0], n)
    ax1.set_title(r"Average Shannon Expansions $\bar{t}$" + "\n", fontsize=FONT_SIZE, fontweight='bold')
    ax1.text(0.5, 1.02, param_info, transform=ax1.transAxes, ha='center', fontsize=FONT_SIZE, style='italic')
    ax1.set_xlabel(r"Number of Solutions $K$", fontsize=FONT_SIZE)
    ax1.set_ylabel(r"Avg. Expansions $\bar{t}$", fontsize=FONT_SIZE)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # # Merged Legend
    # h1, l1 = ax1.get_legend_handles_labels()
    # h2, l2 = ax2.get_legend_handles_labels()
    # ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=FONT_SIZE-4)

    # Only ax1 
    ax1.legend(loc='upper right', fontsize=FONT_SIZE-4)

    plt.tight_layout()
    save_paper_plot(output_path, f"combined_tbar_dist{suffix}")
    plt.close()