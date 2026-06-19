#!/bin/bash

# ==============================================================================
# Quantangle-SAT: Experiment Automation Script
# Authors: Shang-Wei LIN, Ji-Qing YAN, Yean-Ru CHEN, Zhe HOU, David Sanán
# Purpose: Streamline parameter sweeps and random model evaluations
# ==============================================================================

# 1. --- 🎯 Mode Selection ---
# Set TRIALS > 0 to use the 'Uniform Random Boolean Function' model (Binomial K)
# Set TRIALS = 0 to perform a standard K-solution sweep (for plotting curves)
TRIALS_RANDOM_MODEL=0

# 2. --- 🏗️ Problem Specification ---
N_VARS=7               # Number of Boolean variables (n)
EXP_NAME="n7_m2_L0.95_s-1_rep100_Hoeffding_vs_Exact"

# 3. --- 🎯 K-Solution Selection (Conditional) ---
# Option A: Specific K values (e.g., "0 1 5 10"). Leave empty to use Range mode.
K_LIST=""         

# Option B: Continuous Range (Effective only if K_LIST is empty)
K_START=0              
K_END=128              

# 4. --- 📊 Reliability & Measurement Settings ---
S_SAMPLES="-1"   # Sampling sizes (s). Use -1 for Adaptive Mode.
L_CONFIDENCE="0.95"    # Target Confidence Level (L)
M_SETTINGS=2           # Number of measurement types (m)

# Error Bound Calculation Mode
# "exact"     : Use the exact error bound derived from the population standard deviation.
# "hoeffding" : Use the Hoeffding's error bound.
# "compare"   : Run both error bounds simultaneously and compare the results.
BOUND_METHOD="compare"

# 5. --- 🛠️ Quality Control & Repetitions ---
REPS_STD=100           # Standard repetitions per K
REPS_HIGH=1000         # High-quality repetitions for critical regions
REPS_THRESHOLD=0       # If K <= Threshold, use REPS_HIGH (usually for K=0,1)

# ==============================================================================
# Execution Logic
# ==============================================================================

# Determine K arguments based on user selection
if [ -n "$K_LIST" ]; then
    K_ARGS="--K_list $K_LIST"
    TARGET_DISPLAY="Specific K List: ($K_LIST)"
else
    K_ARGS="--K_min $K_START --K_max $K_END"
    TARGET_DISPLAY="K Range: $K_START to $K_END"
fi

echo "----------------------------------------------------------"
echo "🚀 Starting Quantangle-SAT Experiment: $EXP_NAME"
echo "System: n=$N_VARS ($((2**N_VARS)) states), $TARGET_DISPLAY"
echo "Config: s=($S_SAMPLES), L=$L_CONFIDENCE, m=$M_SETTINGS"
echo "----------------------------------------------------------"

# Execute the main experiment controller
python main.py \
    --n $N_VARS \
    --binom_k_trials $TRIALS_RANDOM_MODEL \
    $K_ARGS \
    --reps $REPS_STD \
    --reps_high $REPS_HIGH \
    --reps_threshold $REPS_THRESHOLD \
    --m $M_SETTINGS \
    --L $L_CONFIDENCE \
    --s $S_SAMPLES \
    --bound_method $BOUND_METHOD \
    --exp_id "$EXP_NAME"