# Face Enhancement

This guide walks you through enhancing faces using Restoria.

- What you’ll learn:
  - Choosing a backend (GFPGAN, CodeFormer, RestoreFormer++)
  - Tuning options (upscale, strength)
  - Batch vs single-image workflows

Quick start:

```bash
restoria run --input inputs/whole_imgs --output out/ --backend gfpgan
```

Notes:

- For device selection, keep `--device auto` (default) and Restoria will choose.
- Identity metrics can be enabled with `--metrics fast` or
  `--metrics full` when available.

TODO:

- Add screenshots and recommended settings per backend.
