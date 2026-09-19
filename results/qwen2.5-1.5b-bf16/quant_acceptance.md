| Precision | Model size | Compression | bits/wt | tok/s (batch 1) | tok/s (batch 32) | Peak VRAM | Perplexity | vs bf16 |
|---|---|---|---|---|---|---|---|---|
| bf16 | 3087 MB | 1.00x | 16.00 | 48.9 | 1388.2 | 3103 MiB | 15.11 | +0.00% |
| int8 | 1779 MB | 1.74x | 8.01 | 57.0 | 401.7 | 1855 MiB | 15.13 | +0.17% |
| int4 | 1153 MB | 2.68x | 4.19 | 52.8 | 272.9 | 1290 MiB | 17.81 | +17.90% |
