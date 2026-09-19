| Precision | Model size | Compression | bits/wt | tok/s (batch 1) | tok/s (batch 32) | Peak VRAM | Perplexity | vs fp16 |
|---|---|---|---|---|---|---|---|---|
| fp16 | 988 MB | 1.00x | 16.00 | 57.8 | 1679.5 | 1030 MiB | 22.42 | +0.00% |
| int8 | 631 MB | 1.57x | 8.01 | 69.0 | 1325.1 | 682 MiB | 22.28 | -0.59% |
| int4 | 460 MB | 2.15x | 4.19 | 67.7 | 840.3 | 519 MiB | 27.12 | +20.97% |
