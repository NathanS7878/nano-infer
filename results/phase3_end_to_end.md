> Measured on an idle GPU (1% utilization, 354 MiB in use). 3 runs after 2 discarded warmups; spread (max-min)/median 6.3%, cv (stdev/mean) 3.3%.

| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |
|---|---|---|---|
| 1 | 23.6 | 53.4 | **2.26x** |
| 4 | 93.4 | 200.8 | **2.15x** |
| 16 | 359.2 | 842.5 | **2.35x** |
| 32 | 718.8 | 1570.9 | **2.19x** |
