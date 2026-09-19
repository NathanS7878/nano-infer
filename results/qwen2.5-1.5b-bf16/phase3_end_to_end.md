> Measured on an idle GPU (1% utilization, 131 MiB in use). 3 runs after 2 discarded warmups; spread (max-min)/median 4.9%, cv (stdev/mean) 2.5%.

| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |
|---|---|---|---|
| 1 | 21.1 | 49.0 | **2.32x** |
| 4 | 84.4 | 192.5 | **2.28x** |
| 16 | 332.0 | 732.1 | **2.20x** |
| 32 | 639.5 | 1410.2 | **2.21x** |
