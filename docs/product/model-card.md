# Model Card: GFPGAN

## Model Overview

**Model Name**: GFPGAN (Generative Facial Prior GAN)  
**Version**: 1.4  
**Model Type**: Generative Adversarial Network for face restoration  
**License**: Apache 2.0 (weights bundled under the upstream TencentARC terms)

## Intended Use

### Primary Use Cases

- **Photo restoration**: enhance damaged, blurred, or low-quality face photos
- **Image enhancement**: improve facial details in compressed or degraded images
- **Historical photo recovery**: restore old or deteriorated photographs
- **Content creation**: enhance facial quality in digital media

### Intended Users

- **Photographers**: professional and amateur photo enhancement
- **Archivists**: digital preservation of historical photographs
- **Content creators**: video and image post-production
- **Researchers**: computer vision and image processing studies

### Out-of-Scope Uses

❌ **Not intended for:**

- Real-time video processing (performance limitations)
- Non-facial image enhancement (specialised for faces)
- Identity modification or deepfake creation
- Medical diagnosis or analysis
- Surveillance or law-enforcement identification

## Model Details

### Architecture

- **Base Model**: StyleGAN2 generator with facial priors
- **Training Framework**: PyTorch with custom perceptual and adversarial losses
- **Input Resolution**: 512×512 pixels (faces automatically detected and cropped)
- **Output Resolution**: 512×512 to 2048×2048 (depending on upscale factor)

### Model Versions

| Version | Release Date | Key Features | Model Size |
|---------|--------------|--------------|------------|
| v1.4    | 2024-Q4      | Best identity preservation | ~348 MB |
| v1.3    | 2024-Q2      | Improved texture quality | ~348 MB |
| v1.2    | 2024-Q1      | Enhanced stability | ~348 MB |

### Training Data

- **Dataset composition**: synthetic degradations applied to FFHQ, CelebA-HQ, and curated real-world portraits released by TencentARC.  The training pipeline augments with noise, blur, compression, and colour drift to mimic scanned photographs.
- **Data sources**: FFHQ (Creative Commons BY-NC-SA 4.0), CelebA-HQ (CelebA license), and TencentARC internal datasets; see the upstream GFPGAN paper for detailed counts.
- **Number of images**: ~70k base portraits with heavy augmentation.
- **Demographics**: dominated by Western facial datasets (FFHQ/CelebA).  Subsequent internal datasets add more age ranges but remain skewed toward lighter skin tones.
- **Geographic coverage**: primarily North America and Europe; augmentation adds limited global variance.

## Performance and Limitations

### Model Performance

#### Quantitative Metrics

We maintain a benchmark suite under `bench/` that computes LPIPS, DISTS, and ArcFace cosine for each backend.  Representative values (GFPGAN v1.4, CUDA, batch of 32 images) are tracked in the `bench/out/` reports; regenerate via `restoria bench --metrics full`.  Typical results on FFHQ-grade portraits:

| Metric | GFPGAN v1.4 | CodeFormer | Notes |
|--------|-------------|-----------|-------|
| ArcFace Cosine ↑ | 0.83 | 0.87 | Higher is better for identity retention |
| LPIPS ↓ | 0.165 | 0.148 | Lower is better for perceptual similarity |
| DISTS ↓ | 0.082 | 0.079 | Lower is better |
| Runtime (s/img) ↓ | 0.42 | 0.95 | Measured on RTX 4090 |

Regenerate benchmarks whenever dependencies change and commit the CSV/HTML artefacts for traceability.

#### Qualitative Assessment

✅ **Strengths:**

- Excellent identity preservation with realistic textures
- Robust to common degradations (motion blur, JPEG artefacts, mild scratches)
- Flexible backends (CodeFormer, RestoreFormer++) for alternative trade-offs

⚠️ **Limitations:**

- Struggles with extreme degradation or heavy occlusion (faces < 32 px)
- Requires GPU acceleration for production throughput; CPU-only runs are slow
- Outputs can hallucinate details if the input lacks structure

### Known Biases and Fairness

- Performance is best on the demographics represented in FFHQ/CelebA (lighter skin tones, adult faces).  Accuracy and aesthetic quality may differ for underrepresented groups.
- Limited coverage of non-Western attire and accessories can lead to inconsistent reconstruction of cultural garments.

#### Mitigation Strategies

- Track ArcFace and LPIPS metrics by demographic slices when evaluation data is available.
- Solicit community-sourced evaluation sets and publish bias reports each release.
- When in doubt, fall back to the original image alongside the restoration to preserve user agency.

### Technical Limitations

#### Hardware Requirements

- **Recommended GPU**: NVIDIA with ≥8 GB VRAM for production; Apple Silicon (M1+) works for smaller batches
- **CPU fallback**: available but 10–30× slower depending on photo count

#### Input Constraints

- **Face size**: best for faces ≥ 64 px in height
- **Image formats**: JPG, PNG, WebP, BMP
- **Maximum resolution**: 6K × 6K (planner auto-tiles large images)

## Ethical Considerations

See the bias and out-of-scope sections above.  Users must ensure they have the right to process images, especially in archival or forensic scenarios.

## Evaluation and Validation

- **Benchmark harness**: `restoria bench` + `notebooks/Restoria_Benchmark.ipynb`
- **Recommended datasets**: FFHQ validation split, the public subset of CelebA-HQ, internal “Restoria Stress” set (synthetic blur/noise)
- **Release gating**: every release must publish the benchmark CSV/HTML artefact and attach the SBOM diff described in the release playbook.

For more context on operational practices see `docs/governance/release-playbook.md`.
