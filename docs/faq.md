# FAQ

- Which Python/Torch should I use?
  - Python 3.11 with Torch 2.x is recommended. See docs/COMPATIBILITY.md for details.

- Why is background upsampling disabled on CPU?
  - Real-ESRGAN is heavy on CPU and often slower than desired; this fork disables it by default on CPU for a smoother UX. Use GPU for best results.

- How can I run inference without downloads?
  - Use `--no-download` and pass `--model-path /path/to/GFPGANv1.4.pth`.

- How do I quickly validate the CLI works locally?
  - `gfpgan-infer --dry-run -v 1.4` parses arguments and exits.

- Where is the docs site?
  - Once enabled via GitHub Pages on gh-pages, the site will be available at the repo Pages URL.
