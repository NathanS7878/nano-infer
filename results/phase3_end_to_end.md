> **Provisional — contended measurement.** The GPU was at 42% utilization from other processes when this ran, and run-to-run spread reached 31.1% against this project's 3% bar. The A/B ratio is more robust than the absolute tok/s, since both sides shared the same contention, but neither is a headline number until this is re-run on an idle GPU.

| Batch | PyTorch tok/s | Custom kernels tok/s | Speedup |
|---|---|---|---|
| 1 | 6.1 | 15.2 | **2.48x** |
| 4 | 22.5 | 60.6 | **2.69x** |
| 16 | 88.0 | 257.6 | **2.93x** |
| 32 | 180.4 | 417.2 | **2.31x** |
