# Quantangle-SAT

Quantangle-SAT is a quantum-inspired solver for the Boolean Satisfiability (SAT) problem based on entanglement and equivalence checking.

---

## Installation

Ensure you have **Python 3.10 or newer** installed. We provide a setup script to automate environment configuration and dependency installation.

```bash
# Grant execution permission
chmod +x setup.sh

# Run setup (creates venv and installs requirements)
./setup.sh

# Activate the virtual environment
source quantangle_venv/bin/activate
```

---

## Running Experiments

We provide several pre-configured scripts. All scripts generate data for both **Success Probability** and **Average Shannon Expansion counts**.

### Prerequisite: Activate Environment

Before running any experiments or plotting scripts, **always ensure your virtual environment is activated**:

```bash
source quantangle_venv/bin/activate
```

### 1. Quick Verification (~1 minute)

A small-scale test ($n=4$) to verify the installation and core solver logic.

```bash
bash scripts/n4_m2_L0.95_s-1_rep100.sh
```

### 2. High-Precision Reproduction (Figure 8 - All $K$ @ 1000 reps)

This is the standard baseline experiment. It runs 1,000 repetitions for **every** $K$ value to produce the smoothest curves for success rates and complexity.

```bash
bash scripts/n8_m2_L0.95_s14_28_56_rep1000.sh
```

### 3. Efficient Focused Sweep ($K=0$ Focused)

An optimized version of the $n=8$ sweep. It runs **1,000 repetitions for $K=0$ (UNSAT)** to ensure high-density verification of the false-positive rate, while using 100 repetitions for other $K$ values to save time.

```bash
bash scripts/n8_m2_L0.95_s14_28_56_rep100_K0rep1k.sh
```

### 4. Expected Complexity Proof

Evaluates the $O(1)$ scaling claim using the Uniform Random Boolean Function model (Binomial $K$) for $n=8$.

```bash
bash scripts/n8_m2_L0.95_s-1_bino1000.sh
```

### 5. Confidence Level Sensitivity Analysis ($L$ Comparison)

Evaluates how different target confidence levels ($L = 0.9, 0.95, 0.99$) impact the adaptive sampling size and success probability for $n=5$.

```bash
bash scripts/n5_m2_L0.9_0.95_0.99_s-1_K0rep1k.sh
```

### 6. Custom Experiments

To test your own parameters, modify the variables inside the general entry script in the root directory:

```bash
bash run_exp.sh
```

---

## Visualization

After the experiments are complete, generate **publication-quality** plots (PDF and PNG) using the provided visualization tool:

```bash
# Replace <folder_name> with the actual timestamped folder in experiments/
python run_plots.py experiments/<folder_name>
```

---

## Project Structure

* **`src/`**: Core logic, quantum algorithms, and visualization logic.
* **`scripts/`**: Automation scripts for reproduction and testing.
* **`main.py`**: The primary experiment controller.
* **`run_plots.py`**: A tool to analyze experimental data and generate all figures.
* **`run_exp.sh`**: A flexible wrapper script to launch custom experiment batches.
* **`requirements.txt`**: List of Python dependencies.
* **`setup.sh`**: Automated environment and dependency setup.

---

## Configuration & Parameter Logic

The `run_exp.sh` script serves as the primary configuration hub. Below are the key parameters:

### A. Mode Selection

* **`TRIALS_RANDOM_MODEL`**:
* Set to `0` for **Standard Sweep Mode** (evaluates specific $K$ values).
* Set to `> 0` for **Random Model Mode** (uses Binomial distribution $K \sim B(n, 0.5)$ to simulate Uniform Random Boolean Functions).

### B. Problem Specification

* **`N_VARS` ($n$)**: Number of Boolean variables. The state space is $2^n$.
* **`EXP_NAME`**: Unique identifier for the experiment; results will be saved in `experiments/<timestamp>_<EXP_NAME>`.

### C. K-Solution Selection (Priority Logic)

The solver determines which $K$ values to test based on the following priority:

1. `K_LIST`: If this string is not empty (e.g., `"0 1 5 10"`), the solver **only** tests these values and **ignores** the range settings.
2. `K_START` / `K_END`: If `K_LIST` is empty, the solver performs a continuous sweep from `K_START` to `K_END`.

### D. Sampling & Confidence (Cross-Product Sweep)

Both $L$ and $s$ support multiple values. The solver automatically executes the **Cartesian product** of all specified settings (e.g., 2 $L$ values $\times$ 3 $s$ values = 6 experiments).

* **`S_SAMPLES` ($s$)**: Number of quantum measurements.
* Supports multiple values (e.g., `"14 28 56"`).
* **Adaptive Mode ($s=-1$)**: Uses **Lemma 9** to calculate the sampling size $\hat{s}$ required to satisfy the **Asymptotic Normality**.

* **`L_CONFIDENCE` ($L$)**: Target confidence level. Supports multiple values (e.g., `"0.95 0.99"`).
* **`M_SETTINGS` ($m$)**: Number of measurement types (default: 2).

### E. Quality Control (Dynamic Repetitions)

To ensure statistical density in critical regions (like UNSAT or near-UNSAT):

* **`REPS_STD`**: Default repetitions for most $K$ values.
* **`REPS_HIGH`**: Increased repetitions for specific $K$ values.
* **`REPS_THRESHOLD`**: If $K \le \text{Threshold}$, the solver uses `REPS_HIGH`; otherwise, it uses `REPS_STD`.