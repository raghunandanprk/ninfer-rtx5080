# RTX 5080 GSQ/RCO Port Status: Complete & Verified

## Target Hardware & Runtime

- **GPU:** NVIDIA GeForce RTX 5080 Laptop GPU (16,303 MiB, Blackwell GB203, 175W)
- **CUDA Architecture:** `sm_120a` native compilation
- **Host:** Intel Core Ultra 9 275HX, 32 GB RAM, Windows 11 Home
- **Engine:** Ryan-gsq NInfer v3 (`runtime-v3\engine\ninfer-serve.exe`, Release `sm_120a`)
- **Model:** RentedNoodle Qwen3.8-27B OrcaRouter GSQ/RCO IQ3_XXS v2.1 (preserves embedded trunk GSQ/RCO blocks and S1 MTP head)

---

## Validation Status on Physical Hardware

All target validation criteria have been measured, benchmarked, and verified on the physical RTX 5080 Laptop GPU:

| # | Validation Item | Status | Verified Result |
| :---: | :--- | :---: | :--- |
| **1** | **Artifact load & memory-plan validation** | **Passed** | Strict dedicated VRAM allocation: zero memory spills into host RAM at max context (15,706 MiB peak usage). |
| **2** | **Prefill scaling (8K → 229K)** | **Passed** | 8K prefill ~880 tok/s; full 229K window prefill verified (711 tok/s). |
| **3** | **MTP3 decode & acceptance** | **Passed** | MTP3 decode reaches ~106.4 tok/s on short prompts (0.584 acceptance); sustains 80.2 tok/s @ 99K and 60.5 tok/s @ 229K. |
| **4** | **DFlash2 decode & tuning** | **Passed** | DFlash2 width calibrated to K=5 (118.9 tok/s short decode); K=7 drops to 102 tok/s on Blackwell. |
| **5** | **Long-context needle retrieval** | **Passed** | 100% pass: successfully retrieved 3 hidden needle codes planted at 25%, 55%, and 85% depth across a 229,375-token prompt. |
| **6** | **Vision projector probe** | **Passed** | VRAM-resident vision window at 147,456 tokens (~1.4s encode); Video RAM overlay at 169,984 tokens. |
| **7** | **Quality & regression checks** | **Passed** | Greedy-lossless verification and deterministic prompt outputs matched between MTP and base trunk. |
| **8** | **Sustained generation stability** | **Passed** | Zero CUDA errors or driver crashes across 40+ consecutive bisection and stress-test server starts. |

---

## Performance Summary

- **Max Single-Stream Context:** **231,424 tokens (226K)** under `rk4v4` KV and MTP3.
- **Fast Chat Decode:** **118.9 tok/s** under `rk4v4` KV and DFlash2 K=5.
- **Multimodal Context:** **147,456 tokens** (Image) / **169,984 tokens** (Video).
- **Prefix Reuse TTFT:** **0.38 – 0.68 s** on multi-turn conversations.

Full calibration methodology, raw data, and tuning bisection logs are documented in [`inf-cmd/benchmark-summary.md`](inf-cmd/benchmark-summary.md).
