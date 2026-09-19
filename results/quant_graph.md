| batch | precision | engine | decode ms/step | vs fp16 (same engine) | weight GB/s | % of peak |
|---|---|---|---|---|---|---|
| 1 | fp16 | paged | 16.83 | 1.00x | 58.7 | 13.1% |
| 1 | fp16 | graph | 3.34 | 1.00x | 295.5 | 66.0% |
| 1 | int8 | paged | 14.29 | 1.18x | 44.1 | 9.8% |
| 1 | int8 | graph | 2.50 | 1.34x | 252.4 | 56.3% |
| 1 | int4 | paged | 14.44 | 1.17x | 31.8 | 7.1% |
| 1 | int4 | graph | 2.38 | 1.40x | 192.8 | 43.0% |
| 4 | fp16 | paged | 17.23 | 1.00x | 57.3 | 12.8% |
| 4 | fp16 | graph | 3.60 | 1.00x | 274.8 | 61.3% |
| 4 | int8 | paged | 14.20 | 1.21x | 44.4 | 9.9% |
| 4 | int8 | graph | 3.30 | 1.09x | 191.4 | 42.7% |
| 4 | int4 | paged | 14.31 | 1.20x | 32.1 | 7.2% |
| 4 | int4 | graph | 4.41 | 0.82x | 104.2 | 23.3% |
| 32 | fp16 | paged | 18.76 | 1.00x | 52.7 | 11.8% |
| 32 | fp16 | graph | 4.03 | 1.00x | 245.2 | 54.7% |
| 32 | int8 | paged | 18.10 | 1.04x | 34.8 | 7.8% |
| 32 | int8 | graph | 15.24 | 0.26x | 41.4 | 9.2% |
| 32 | int4 | paged | 28.72 | 0.65x | 16.0 | 3.6% |
| 32 | int4 | graph | 25.81 | 0.16x | 17.8 | 4.0% |

GPU at start: 0% util, 140 MiB. Idle GPU.
