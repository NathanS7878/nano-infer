| batch | precision | engine | decode ms/step | vs fp16 (same engine) | weight GB/s | % of peak |
|---|---|---|---|---|---|---|
| 1 | fp16 | paged | 20.04 | 1.00x | 49.3 | 11.0% |
| 1 | fp16 | graph | 3.67 | 1.00x | 269.3 | 60.1% |
| 1 | int8 | paged | 17.95 | 1.12x | 35.1 | 7.8% |
| 1 | int8 | graph | 2.72 | 1.35x | 231.9 | 51.8% |
| 1 | int4 | paged | 17.20 | 1.16x | 26.7 | 6.0% |
| 1 | int4 | graph | 2.59 | 1.42x | 177.8 | 39.7% |
| 4 | fp16 | paged | 20.85 | 1.00x | 47.4 | 10.6% |
| 4 | fp16 | graph | 3.91 | 1.00x | 252.4 | 56.3% |
| 4 | int8 | paged | 17.54 | 1.19x | 36.0 | 8.0% |
| 4 | int8 | graph | 3.58 | 1.09x | 176.0 | 39.3% |
| 4 | int4 | paged | 17.49 | 1.19x | 26.3 | 5.9% |
| 4 | int4 | graph | 4.77 | 0.82x | 96.4 | 21.5% |
| 32 | fp16 | paged | 21.89 | 1.00x | 45.1 | 10.1% |
| 32 | fp16 | graph | 4.39 | 1.00x | 225.1 | 50.3% |
| 32 | int8 | paged | 20.35 | 1.08x | 31.0 | 6.9% |
| 32 | int8 | graph | 16.54 | 0.27x | 38.1 | 8.5% |
| 32 | int4 | paged | 32.37 | 0.68x | 14.2 | 3.2% |
| 32 | int4 | graph | 28.18 | 0.16x | 16.3 | 3.6% |

GPU at start: 26% util, 1317 MiB. **Contended: ratios robust (round-robin), absolutes not publishable.**
