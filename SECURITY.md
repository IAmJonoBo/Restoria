# Security

This repository provides local-first tools for image and video restoration. For production deployments:

- Do not expose Gradio with `share=True` on public networks.
- Run the FastAPI service behind an authenticated reverse proxy.
- Scrub EXIF metadata on upload to avoid leaking camera/location data.
- Keep dependencies updated; CI should run OSV scans and generate SBOMs.

Report vulnerabilities privately via the repository maintainers.

