# RTX 5080 GSQ3 port status

## Target

- GPU: RTX 5080 Laptop 16 GB / Blackwell GB203
- CUDA target: `sm_120a`
- model: Qwen3.8-27B GSQ3
- weight scheme: `Q3G128_F16S`, 3.125 bpw text body
- artifact size: 13,330,776,576 bytes (12.41 GiB)
- KV: `rk4v4-e8`
- speculation: MTP3 and DFlash2 K=7

The upstream `roofkid/ninfer-4080` fork already contains the GSQ3 codec, Q3 kernels,
rotated/E8 KV modes, MTP3, DFlash2, and inherited `sm_120a` code paths. The published
4080 fork is blocked on Blackwell primarily by the top-level CMake gate that forces `sm_89`.

## Validation required on the real RTX 5080 Laptop

1. artifact load and memory-plan validation
2. 8K / 32K / 64K / ~100K prefill
3. MTP3 decode and acceptance
4. DFlash2 K=7 decode and greedy-lossless checks
5. long-context retrieval
6. vision probe
7. quick perplexity / MBPP regression
8. sustained-generation CUDA error check

RTX 4080 performance figures are reference data only and must not be presented as 5080 measurements.
