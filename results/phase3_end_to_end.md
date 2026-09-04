> Measured on an idle GPU (2% utilization, 350 MiB in use). 3 runs after 2 discarded warmups; spread (max-min)/median 8.9%, cv (stdev/mean) 4.5%.

| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |
|---|---|---|---|
| 1 | 25.1 | 58.6 | **2.34x** |
| 4 | 97.6 | 225.9 | **2.32x** |
| 16 | 376.8 | 905.1 | **2.40x** |
| 32 | 745.1 | 1739.1 | **2.33x** |
