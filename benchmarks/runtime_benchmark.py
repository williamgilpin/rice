#!/usr/bin/env python3
import os
import sys
import numpy as np
import time
from memory_profiler import memory_usage

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from rice.rice.models import CausalDetection

def main(seed=0):
    # Generate 15 logarithmically‐spaced values between 10 and 10 000
    n_samples_list  = np.logspace(np.log10(10), np.log10(10000), num=15, dtype=int)
    n_features_list = np.logspace(np.log10(10), np.log10(2000), num=15, dtype=int)

    # Create data array
    np.random.seed(seed)
    X0 = np.random.randn(10000, 2000)

    # Preallocate result arrays
    mem_usages = np.zeros((len(n_samples_list), len(n_features_list)))
    runtimes   = np.zeros_like(mem_usages)

    # Baseline process memory (MiB)
    baseline_mem = memory_usage(max_usage=True)

    # Instantiate model
    model = CausalDetection(
        d_embed=3, 
        neighbors="simplex", 
        forecast="smap", 
        ensemble=True, 
    )

    for i, n_samples in enumerate(n_samples_list):
        for j, n_features in enumerate(n_features_list):

            print(f"samples={n_samples:5d}, features={n_features:5d}", flush=True)
            
            X = X0[:n_samples, :n_features].copy()

            # Wrapper to measure both time and peak memory
            def benchmark():
                t0 = time.perf_counter()
                model.fit(X)
                return time.perf_counter() - t0

            # memory_usage with retval returns (peak_mem, runtime)
            peak_mem, duration = memory_usage((benchmark, (), {}), max_usage=True, retval=True)

            # Subtract baseline to get model‐specific memory
            mem_usages[i, j] = peak_mem - baseline_mem
            runtimes[i, j]   = duration

            print(f"samples={n_samples:5d}, features={n_features:5d} → "
                  f"Δmem={mem_usages[i,j]:6.2f} MiB, time={runtimes[i,j]:.4f} s")

    # Save results
    np.save(f"memory_usage_{seed}.npy", mem_usages)
    np.save(f"runtimes_{seed}.npy", runtimes)

## optionally take seed as argument
if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 0
    main(seed)
