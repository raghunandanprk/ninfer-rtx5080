# NInfer RTX 5080 GSQ3

Blackwell (`sm_120a`) port targeting **RTX 5080 / RTX 5070 Ti 16 GB** for Qwen3.8-27B GSQ3 inference.

This repository is being assembled from:
- the validated Blackwell/Windows 16 GB port: https://github.com/YukinoKaorisuna/ninfer-5070ti
- the GSQ3 + DFlash2 RTX 4080 work: https://github.com/roofkid/ninfer-4080

Primary target:
- Qwen3.8-27B
- GSQ3 / `Q3G128_F16S`
- 16 GB Blackwell
- long context with `rk4v4-e8`
- MTP3 and DFlash2
- native Windows first, Linux/WSL2 second

Status: port in progress.
