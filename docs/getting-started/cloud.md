# Run in the Cloud

Restoria ships with ready-to-use notebooks and UIs for hosted environments.  This page collects the supported options.

## Google Colab

- **Notebook**: [Restoria Colab](https://colab.research.google.com/github/IAmJonoBo/Restoria/blob/main/notebooks/Restoria_Colab.ipynb)
- **Features**: upload images, fetch from URLs, adjust backend, preview results, download ZIP archives.
- **GPU costs**: Colab free tier provides K80/T4/V100 on a best-effort queue.  Pro tiers offer A100/TPU at an hourly rate—remember to stop the runtime when finished.
- **Headless mode**: the final notebook section shows how to call the CLI directly (`!restoria run ...`) so you can script runs without using the UI cells.

## HuggingFace Spaces (coming soon)

A Spaces deployment is in progress and will reuse the same Gradio blocks as the local UI.  When it launches, you will be able to:

- Upload images through the browser and stream progress from the FastAPI backend
- Compare backends via preset cards and download manifests/metrics
- Opt into anonymous telemetry (`RESTORIA_FEEDBACK=1`) to share non-identifying performance data with the maintainers

Bookmark this page for updates; the link will be added once the public Space is live.

## Self-hosted Gradio UI

For air-gapped or custom deployments you can run the included Gradio app locally:

```bash
pip install -e .[web]
python -m gfpgan.gradio_app --share
```

Passing `--share` creates a temporary public link.  See `docs/UI_GUIDE.md` for a tour and advanced configuration (weight caching, GPU selection, background upsampling).

## FastAPI Service

For automation or integration with your own web front-end:

```bash
pip install -e .[api]
uvicorn services.api.main:app --reload --port 8000
```

Use `/docs` for the OpenAPI UI and `docs/usage/api.md` for request examples.  Combine with the planner (`auto_backend=true`) to offload backend selection to the server.

## Cost awareness checklist

| Environment    | Free tier | Paid tier | Notes |
| -------------- | --------- | --------- | ----- |
| Google Colab   | Yes (queue-based) | Pro/Pro+ subscriptions | Charges apply when using high-end GPUs for extended sessions |
| HuggingFace Spaces | Free CPU | GPU grants or pay-as-you-go | GPU Spaces accrue hourly charges; configure auto-shutdown |
| Self-hosted VM | N/A | Cloud provider billing | Track egress/storage costs when exporting large result sets |

For long-lived workloads consider using the benchmark harness locally and uploading only curated outputs to the cloud.
