> Measured on an idle GPU (0% utilization, 259 MiB in use). 3 runs after 2 discarded warmups; spread (max-min)/median 3.0%, cv (stdev/mean) 1.7%.

| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |
|---|---|---|---|
| 1 | 24.7 | 57.7 | **2.33x** |
| 4 | 95.7 | 223.7 | **2.34x** |
| 16 | 388.7 | 858.0 | **2.21x** |
| 32 | 748.0 | 1694.4 | **2.27x** |
