"""
Quantangle-SAT Solver System
---------------------
This module implements a SAT solving algorithm based on the principles of 
Quantum Equivalence Checking (QEC), utilizing optimal CGLMP measurement statistics.

The core principle is to treat satisfiability testing as an equivalence checking task:
the problem instance is statistically compared against a reference UNSAT baseline.

- Equivalent: High statistical overlap with the UNSAT baseline => Classified as UNSAT.
- Not Equivalent: Significant statistical deviation => Classified as SAT.
"""

import math
import logging
from typing import List

import os, time
import numpy as np
from scipy.stats import norm
from qiskit import qasm3, transpile, QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full
from qiskit.circuit.library import DiagonalGate
from qiskit_aer import AerSimulator

import cudaq
from qbraid.transpiler.conversions import openqasm3_to_cudaq

# Configure logging (Set root logger to WARNING to suppress third-party clutter)
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Only keep our custom Solver logs visible

# Force silence noisy third-party modules (Qiskit internal compiler & package extensions)
logging.getLogger("qiskit").setLevel(logging.CRITICAL)
logging.getLogger("qiskit_aer").setLevel(logging.CRITICAL)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)

class Solver:
    """
    Quantangle-SAT Solver based on Quantum Equivalence Checking.
    Supports optional GPU acceleration via NVIDIA CUDA-Q.
    """

    def __init__(self, mode: str = "cpu"):
        """
        Args:
            mode: "cpu" to run on Qiskit AerSimulator, 
                  "gpu" to run on NVIDIA CUDA-Q (requires physical NVIDIA GPU and environment).
        """
        self.mode = mode.lower()
        self._warned_params = set()
        
        if self.mode == "gpu":
            import cudaq
            # Set the execution target explicitly to physical NVIDIA GPUs
            cudaq.set_target("nvidia")
            logger.info("CUDA-Q GPU Target ('nvidia') initialized successfully.")
        else:
            self.backend = AerSimulator()
            logger.info("Qiskit AerSimulator (CPU) initialized.")

    def _calculate_alpha_k(self, k: int, d: int, m: int) -> float:
        tan_term = np.tan(np.pi / (2 * m))
        denom_arg = np.pi * (k + 0.5 / m) / d
        cot_term = 1.0 / np.tan(denom_arg)
        return (1.0 / (2 * d)) * tan_term * cot_term

    def _calculate_epsilon(self, model: str, s: int, delta: float, sigma: float) -> float:
        if s <= 0:
            return float('inf')

        if model == "H": # Hoeffding bound
            return math.sqrt((8 * math.log(2 / delta)) / s)
        elif model == "Z": # Normal approximation
            z_score = norm.ppf(1 - delta / 2)
            return (sigma / math.sqrt(s)) * z_score
        else:
            raise ValueError(f"Unknown model: {model}")

    def _apply_shannon_expansion(self, F: List[int]) -> List[int]:
        r"""Reduces variable set via Shannon expansion: $F_{new} = F_{v=0} \lor F_{v=1}$."""
        arr = np.array(F, dtype=np.int8)
        mid = len(arr) // 2
        # Logical OR via element-wise min for -1 (SAT) / 1 (UNSAT) encoding
        return np.minimum(arr[:mid], arr[mid:]).tolist()

    def _get_measurement_configs(self, m: int):
        configs = []
        for i in range(1, m + 1):
            configs.append({'x': i, 'y': i, 'formula': 'a-b', 'adjust_a': False})
            configs.append({'x': 1 if i == m else i + 1, 'y': i, 
                            'formula': 'b-a', 'adjust_a': True if i == m else False})
        return configs

    def _QEC(self, oracle_gate: DiagonalGate, s: int, m: int) -> float:
        n = oracle_gate.num_qubits
        d = 2**n
        gpu_basis = ['u', 'p', 'cx', 'id', 'h', 'x', 'y', 'z', 'rz']
        base_qc = QuantumCircuit(n * 2, n * 2)
        base_qc.h(range(n))
        base_qc.cx(range(n), range(n, n * 2))
        base_qc.append(oracle_gate, range(n))

        if self.mode == "gpu":
            start_time = time.perf_counter() 
            
            base_qc_transpiled = transpile(
                base_qc, 
                basis_gates=gpu_basis, 
                optimization_level=1,
                layout_method='trivial',
                routing_method='none'
            )
            
            end_time = time.perf_counter() 
            elapsed_time = end_time - start_time 
            
            print(f"base_qc_transpiled consume: {elapsed_time:.4f} sec")
            
        qft_inst = synth_qft_full(n)
        iqft_inst = synth_qft_full(n).inverse()

        configs = self._get_measurement_configs(m)
        s_dist = np.random.multinomial(s, [1.0/len(configs)]*len(configs))

        precompiled_circuits = {}
        precompiled_cudaq_kernels = {}
        total_gpu_compile_time = 0
        for idx, shots in enumerate(s_dist):
            if shots <= 0: continue
            cfg = configs[idx]
            qec_qc = QuantumCircuit(n * 2, n * 2)
            
            alpha_x = (cfg['x'] - 0.5) / m
            beta_y = cfg['y'] / m
            
            # --- ALICE: Phase Shift + Inverse Quantum Fourier Transform (IQFT) ---
            for j in range(n):
                phase_a = (2 * np.pi / d) * alpha_x * (2**j)
                qec_qc.p(phase_a, j)
            qec_qc = qec_qc.compose(iqft_inst, range(n))
            
            # --- BOB: Phase Shift + Forward Quantum Fourier Transform (QFT) ---
            for j in range(n):
                phase_b = -(2 * np.pi / d) * beta_y * (2**j)
                qec_qc.p(phase_b, n + j)
            qec_qc = qec_qc.compose(qft_inst, range(n, n * 2))
            qec_qc.measure(range(n * 2), range(n * 2))
            qc_flattened = base_qc.compose(qec_qc)

            if self.mode == "gpu":
                t_start = time.perf_counter()
                qc_flattened = base_qc_transpiled.compose(qec_qc)
                qasm3_str = qasm3.dumps(qc_flattened)
                precompiled_cudaq_kernels[idx] = openqasm3_to_cudaq(qasm3_str)
                total_gpu_compile_time += (time.perf_counter() - t_start)
            else:
                precompiled_circuits[idx] = qc_flattened

        if self.mode == "gpu":
            print(f"qasm3_str + precompiled_cudaq_kernels consume: {total_gpu_compile_time:.4f} s")

        # 3. Execution and Processing Loop
        total_val = 0.0
        total_gpu_sample_time = 0
        for idx, shots in enumerate(s_dist):
            if shots <= 0: continue
            cfg = configs[idx]
        
            # --- EXECUTION BRANCH: GPU (CUDA-Q) vs CPU (Qiskit Aer) ---
            if self.mode == "gpu":
                t_start = time.perf_counter()
                cudaq_kernel = precompiled_cudaq_kernels[idx]
                cudaq_result = cudaq.sample(cudaq_kernel, shots_count=int(shots))
                counts = dict(cudaq_result.items())
                total_gpu_sample_time += (time.perf_counter() - t_start)
            else:
                qc_decomposed = precompiled_circuits[idx]
                counts = self.backend.run(qc_decomposed, shots=int(shots)).result().get_counts()

            # --- PROCESS RESULTS ---
            for b_str, count in counts.items():
                res = b_str.replace(" ", "")

                if self.mode == "gpu":
                    # Critical fix for CUDA-Q Big-Endian vs Qiskit Little-Endian string order
                    res = res[::-1]

                a, b = int(res[n:], 2), int(res[:n], 2)
                
                if cfg['adjust_a']: a = (a + 1) % d
                diff = (a - b) % d if cfg['formula'] == 'a-b' else (b - a) % d
                total_val += count * 2 * self._calculate_alpha_k(diff, d, m)
            
        if self.mode == "gpu":
            print(f"cudaq_result consume: {total_gpu_sample_time:.4f} s\n")

        return total_val / s

    def _calculate_sigma(self, d: int, m: int, K: int) -> float:
        """Theoretical standard deviation for SAT (K=1) and UNSAT (K=0) """
        tan = np.tan(np.pi / (2 * m))
        if K == 0: return (1.0 / d) * np.sqrt((d**2 - 1) / 3.0) * tan
        term1 = (d**2 * (d**2 - 1) / 3.0) * (tan**2)
        term2 = 4.0 * (d - 1) * ((d - 2)**2)
        return (1.0 / d**2) * np.sqrt(term1 + term2)

    def _calculate_delta(self, L: float, V_count: int) -> float:
        """Calculates the Bonferroni-corrected error rate delta."""
        return (1.0 - L) / (V_count + 1)

    def _calculate_s_hat(self, delta: float, m: int) -> int:
        """
        Lemma 9: Calculates the lower bound s_hat required for 
        asymptotic normality (CLT requirement).
        """
        z_score = norm.ppf(1 - delta / 2)
        tan_val = np.tan(np.pi / (2 * m))
        s_hat = (64 / (9 * (tan_val**2))) * (z_score**2)
        return math.ceil(s_hat)

    def _calculate_s_prime(self, delta: float, m: int) -> int:
        """
        Lemma 10: Calculates the sampling size s' required to 
        guarantee an empty uncertain region in the final iteration.
        """
        z_score = norm.ppf(1 - delta / 2)
        tan_val = np.tan(np.pi / (2 * m))
        s_prime = (tan_val**2) * (z_score**2)
        return math.ceil(s_prime)

    def _diagnose_uncertain_state(self, mu_t: float, s: int, d: int, m: int, W_size: int):
        """
        Computes the Bayes Factor (BF10) to quantify evidence for SAT (H1) 
        vs. UNSAT (H0) during uncertain states.
        Scale based on Kass & Raftery (1995).
        """
        # Theoretical expectations
        exp_h0 = 1.0
        exp_h1 = (1 - 1.0 / (2**W_size))**2
        
        # Theoretical standard deviations
        sig0 = self._calculate_sigma(d, m, 0)
        sig1 = self._calculate_sigma(d, m, 1)

        def _log_likelihood(mu, exp, sig, s_val):
            # Log-normal PDF: -0.5 * log(2*pi*var) - (s*(mu-exp)^2 / (2*sig^2))
            variance = (sig**2) / s_val
            return -0.5 * np.log(2 * np.pi * variance) - ((mu - exp)**2) / (2 * variance)

        log_p0 = _log_likelihood(mu_t, exp_h0, sig0, s)
        log_p1 = _log_likelihood(mu_t, exp_h1, sig1, s)
        
        # BF10 = p(mu|H1) / p(mu|H0) -> exp(log_p1 - log_p0)
        # Clip to avoid overflow/underflow
        bf_10 = np.exp(np.clip(log_p1 - log_p0, -700, 700))
        
        # Interpret evidence strength
        is_sat = bf_10 > 1
        val = bf_10 if is_sat else 1.0 / bf_10 if bf_10 != 0 else float('inf')
        
        if val <= 3: strength = "Weak"
        elif val <= 20: strength = "Positive"
        elif val <= 150: strength = "Strong"
        else: strength = "Very Strong"
        
        target = "SAT (H1)" if is_sat else "UNSAT (H0)"
        diag_msg = f"{strength} evidence for {target} [BF10={bf_10:.2f}]"
        
        return bf_10, diag_msg

    def _record_history(self, history, t, mu, s, d, m, W, expected_ans):
        bf_10, _ = self._diagnose_uncertain_state(mu, s, d, m, W_size=W)
        lean = "yes" if bf_10 > 1 else "no"
        history.append({
            "t": t,
            "bf": bf_10,
            "lean": lean,
            "is_lean_correct": 1 if lean == expected_ans else 0
        })

    def solve(self, truth_table: List[int], m: int, L: float, s: int, bound_method: str = "exact"):
        F = list(truth_table)
        V_count = len(F).bit_length() - 1
        W_size = V_count
        delta = self._calculate_delta(L, V_count)
        s_hat = self._calculate_s_hat(delta, m)
        s_prime = self._calculate_s_prime(delta, m)

        param_key = (L, m, V_count, s)

        if s == -1:
            current_s = s_hat
            if param_key not in self._warned_params:
                logger.info(f"Adaptive Mode: Setting s = s_hat ({s_hat}) [Lemma 9].")
                self._warned_params.add(param_key)
        else:
            current_s = s
            if param_key not in self._warned_params:
                if current_s < s_prime:
                    logger.warning(f"Critical sampling size: s={s} is below s'={s_prime} [Lemma 10]. "
                                "The solver might fail to provide a conclusive result.")
                elif current_s < s_hat:
                    logger.warning(f"Low sampling size: s={s} is below s_hat={s_hat} [Lemma 9]. "
                                "The statistical confidence levels may be compromised.")
                self._warned_params.add(param_key)
                
        t = 0
        history = []
        expected_ans = "no" if truth_table.count(-1) == 0 else "yes"
        mu_0 = None

        while W_size >= 0:
            mu_t = self._QEC(DiagonalGate([1] * len(F) + F), current_s, m)
            if mu_0 is None: mu_0 = mu_t
            
            d = 2**(W_size + 1)
            self._record_history(history, t, mu_t, current_s, d, m, W_size, expected_ans)

            eps_model = "H" if bound_method == "hoeffding" else "Z"
            e0 = self._calculate_epsilon(eps_model, current_s, delta, self._calculate_sigma(d, m, 0))
            e1 = self._calculate_epsilon(eps_model, current_s, delta, self._calculate_sigma(d, m, 1))
            
            if mu_t >= (1 - 1.0/(2**W_size))**2 + e1: return "no", 0, t, current_s, history
            if mu_t <= 1 - e0: return "yes", 1 - np.sqrt(np.clip(mu_0, 0, 1)), t, current_s, history

            if W_size > 0:
                F = self._apply_shannon_expansion(F)
                W_size = len(F).bit_length() - 1
                t += 1
            elif W_size == 0:
                # Lemma 10 Failure Case (occurs if s < s')
                logger.warning("Final iteration inconclusive. Returning uncertain.")
                return "uncertain", None, t, current_s, history