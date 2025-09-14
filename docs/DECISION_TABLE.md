# Engine Decision Table

This table summarizes when to prefer each engine and the trade‑offs involved.

- GFPGAN (default)
  - Strengths: Natural outputs, good identity retention on most inputs, fast with Torch 2.x and `torch.compile`.
  - Use when: Mixed quality inputs, portraits, multi‑face scenes.
  - Notes: Background upsampling via RealESRGAN improves whole‑image perception.

- CodeFormer
  - Strengths: Robust on severely degraded faces (blur, compression); fidelity control from 0..1.
  - Use when: Very low‑quality inputs, surveillance‑like frames.
  - Notes: Higher fidelity values can preserve more original structure at the cost of fine detail.

- RestoreFormer / RestoreFormer++
  - Strengths: Identity‑faithful restoration; stable on higher‑quality inputs.
  - Use when: Faces are mostly sharp; you want minimal identity drift.
  - Notes: "++" variant uses updated weights when available.

- GFPGAN (ONNX Runtime)
  - Strengths: Portable runtime; can leverage CUDA/TensorRT/DirectML/CoreML providers.
  - Use when: You need deployment flexibility or lower latency with hardware‑accelerated EPs.
  - Notes: Provide `--model-path-onnx` or JobSpec `model_path_onnx`.

- DiffBIR / HYPIR (experimental)
  - Strengths: Diffusion‑prior and heavy models for challenging scenes.
  - Use when: Research or advanced users; expect longer runtimes.
  - Notes: Behind feature flags; not included in CI.

Trade‑offs
- Identity vs Detail: CodeFormer (with fidelity) and RestoreFormer++ bias toward identity; GFPGAN often yields more natural detail.
- Speed vs Quality: Use the Quality preset to map tiling/precision; ORT providers can improve latency. The optimizer can evaluate several weights and pick the best under a time budget.

Metrics
- ArcFace cosine (identity): Higher is better; used for identity lock and optimizer.
- LPIPS/DISTS (perceptual): Lower is better; optional in fast/full metrics modes.
- NIQE/BRISQUE (no‑ref): Optional for auto backend heuristics.

