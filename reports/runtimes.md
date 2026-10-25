100 questions, 2 CPU threads, single-question latency (p50 / p95 of each call) and 4 questions per call; accuracy against eager PyTorch

| Runtime | p50 (ms) | p95 (ms) | per question, 4 per call (ms) | speed-up vs eager PyTorch (p50) | cosine to reference (mean / min) | hit@3 | MRR |
|---|---|---|---|---|---|---|---|
| eager PyTorch | 31.3 | 50.2 | 21.4 | 1.00x | 1.0000 / 1.0000 | 1.000 | 0.817 |
| ONNX Runtime, graph optimisation off | 20.1 | 34.6 | 17.3 | 1.56x | 1.0000 / 1.0000 | 1.000 | 0.817 |
| ONNX Runtime, graph optimisation on | 20.8 | 34.4 | 18.0 | 1.50x | 1.0000 / 1.0000 | 1.000 | 0.817 |
| OpenVINO FP32 | 19.9 | 28.3 | 16.9 | 1.57x | 1.0000 / 1.0000 | 1.000 | 0.817 |
| OpenVINO INT8 | 34.4 | 56.7 | 32.0 | 0.91x | 0.9982 / 0.9967 | 0.933 | 0.826 |
