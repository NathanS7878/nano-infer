| batch | precision | engine | decode ms/step | vs fp16 (same engine) | weight GB/s | % of peak |
|---|---|---|---|---|---|---|
| 1 | fp16 | paged | 17.91 | 1.00x | 55.2 | 12.3% |
| 1 | fp16 | graph | 3.43 | 1.00x | 288.3 | 64.4% |
| 1 | int8 | paged | 15.59 | 1.15x | 40.4 | 9.0% |
| 1 | int8 | graph | 2.55 | 1.34x | 247.0 | 55.1% |
| 1 | int4 | paged | 15.88 | 1.13x | 28.9 | 6.5% |
| 1 | int4 | graph | 2.42 | 1.42x | 190.0 | 42.4% |
| 4 | fp16 | paged | 19.09 | 1.00x | 51.7 | 11.5% |
| 4 | fp16 | graph | 3.70 | 1.00x | 266.7 | 59.5% |
| 4 | int8 | paged | 15.95 | 1.20x | 39.5 | 8.8% |
| 4 | int8 | graph | 3.37 | 1.10x | 187.1 | 41.8% |
| 4 | int4 | paged | 16.23 | 1.18x | 28.3 | 6.3% |
| 4 | int4 | graph | 4.53 | 0.82x | 101.5 | 22.7% |
| 32 | fp16 | paged | 20.11 | 1.00x | 49.1 | 11.0% |
| 32 | fp16 | graph | 4.15 | 1.00x | 238.0 | 53.1% |
| 32 | int8 | paged | 19.08 | 1.05x | 33.0 | 7.4% |
| 32 | int8 | graph | 15.65 | 0.27x | 40.3 | 9.0% |
| 32 | int4 | paged | 29.87 | 0.67x | 15.4 | 3.4% |
| 32 | int4 | graph | 26.40 | 0.16x | 17.4 | 3.9% |

GPU at start: 1% util, 882 MiB. **Contended: ratios robust (round-robin), absolutes not publishable.**
