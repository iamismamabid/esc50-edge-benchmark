# ESC-50 Edge vs. Cloud Audio Classification Benchmark

This repository contains the benchmarking scripts, server configuration, and experimental results evaluating environmental sound classification (ESC-50) across local edge environments (Node.js CPU and Browser WebGPU/WASM) and serverless cloud deployment (Google Cloud Run).

## Repository Contents

- **`batch_benchmark.ts`**: Runs local batch inference benchmarks comparing `fp32` and `q8` quantized models on CPU.
- **`cloud_benchmark.ts`**: Evaluates end-to-end latency, cold-start overhead, and throughput against the Cloud Run API endpoint.
- **`browser_v2.html`**: Client-side browser benchmark interface testing WebGPU/WASM execution performance.
- **`server.ts` & `Dockerfile`**: Containerized inference server deployed on Google Cloud Run (`asia-southeast1`).
- **`local_results.csv` & `cloud_results.csv`**: Raw latency, memory usage, and model prediction outputs recorded across test runs.

## Hardware & Test Environments

- **Local Edge Hardware**: Intel Core i5-8365U @ 1.60GHz, Intel UHD Graphics 620 (Integrated GPU)
- **Cloud Environment**: Google Cloud Run serverless container

## Running the Benchmarks

1. **Install dependencies:**
   ```bash
   npm install
