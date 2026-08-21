| Engine | Batch | Tokens/sec | TTFT (ms) | Inter-token (ms) | Variance | Stable |
|---|---|---|---|---|---|---|
| HF generate() | 1 | 19.5 | 63.03 | 51.26 | 2.3% | yes |
| HF generate() | 4 | 76.9 | 59.93 | 51.94 | 1.2% | yes |
| HF generate() | 16 | 313.3 | 67.25 | 50.94 | 2.3% | yes |
| HF generate() | 32 | 621.5 | 68.33 | 51.36 | 1.0% | yes |
