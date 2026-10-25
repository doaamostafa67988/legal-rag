100 questions, ONNX Runtime CPU, median of 5 runs; ms per question (lower is better)

| Batch size | as they arrive | grouped by length | speed-up vs batch 1 |
|---|---|---|---|
| 1 | 26.4 | 18.8 | 1.00x |
| 2 | 17.0 | 17.0 | 1.56x |
| 4 | 16.1 | 15.6 | 1.64x |
| 8 | 24.9 | 15.8 | 1.06x |
| 16 | 20.4 | 17.2 | 1.29x |
| 32 | 23.9 | 18.3 | 1.11x |
| 100 | 29.3 | 28.5 | 0.90x |

Fastest batch size as the questions arrive: 4.
