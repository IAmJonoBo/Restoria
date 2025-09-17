# Guided backend (experimental, opt-in)

The guided backend lets you provide a reference image to bias restoration
toward preserving identity.

- Enable:
	`gfpup run --backend guided --reference path/to/ref.jpg --input <img> --output out/`
- Defaults remain unchanged unless you opt in.

What it does:

- Computes an identity similarity (ArcFace) between the input and the
	reference, when available.
- If similarity is low, it slightly lowers the restoration weight to retain
	more identity.
- Internally runs GFPGAN with the adjusted weight.

Notes:

- If ArcFace or the reference image is unavailable, it gracefully falls back
	to normal GFPGAN settings.
- The plan and metrics record `guided.reference`, `guided.arcface_cosine`, and
	`guided.adjusted_weight` when applicable.
- This is a lightweight heuristic, not face swapping. No model weights are
	changed.

Troubleshooting:

- Install metrics extras for ArcFace support if you want identity similarity:
	`pip install -e ".[metrics,arcface]"`
- You can continue using `--metrics off` and the backend will still run, just
	without the guidance tweak.
