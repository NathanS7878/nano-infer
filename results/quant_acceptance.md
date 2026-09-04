| Precision | Model size | Compression | bits/wt | tok/s (batch 1) | tok/s (batch 32) | Peak VRAM | Perplexity | vs fp16 |
|---|---|---|---|---|---|---|---|
| fp16 | 988 MB | 1.00x | 16.00 | 57.9 | 1646.0 | 1030 MiB | 22.42 | +0.00% |
| int8 | 631 MB | 1.57x | 8.01 | 67.9 | 1274.9 | 682 MiB | 22.29 | -0.55% |
| int4 | 460 MB | 2.15x | 4.19 | 67.1 | 829.1 | 519 MiB | 27.15 | +21.10% |
