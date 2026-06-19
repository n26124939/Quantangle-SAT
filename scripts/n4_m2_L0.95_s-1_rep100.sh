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
N_VARS=4               # Number of Boolean variables (n)
EXP_NAME="n4_m2_L0.95_rep100"

# 3. --- 🎯 K-Solution Selection (Conditional) ---
# Option A: Specific K values (e.g., "0 1 5 10"). Leave empty to use Range mode.
K_LIST=""         

# Option B: Continuous Range (Effective only if K_LIST is empty)
K_START=0              
K_END=16              

# 4. --- 📊 Reliability & Measurement Settings ---
L_CONFIDENCE="0.95"    # Target Confidence Level (L)
M_SETTINGS=2           # Number of measurement types (m)
S_SAMPLES="-1"   # Sampling sizes (s). Use -1 for Adaptive Mode.

# 5. --- 🛠️ Quality Control & Repetitions ---
REPS_STD=100          # Standard repetitions per K
REPS_HIGH=100         # High-quality repetitions for critical regions
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
python3 main.py \
    --n $N_VARS \
    --binom_k_trials $TRIALS_RANDOM_MODEL \
    $K_ARGS \
    --reps $REPS_STD \
    --reps_high $REPS_HIGH \
    --reps_threshold $REPS_THRESHOLD \
    --m $M_SETTINGS \
    --L $L_CONFIDENCE \
    --s $S_SAMPLES \
    --exp_id "$EXP_NAME"

echo "----------------------------------------------------------"
echo "✅ Experiment Completed Successfully!"
echo "Data saved in: experiments/ (under timestamped directory)"
echo "----------------------------------------------------------"