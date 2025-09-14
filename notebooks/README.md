# Notebooks Alignment (GFPP)

This repository includes a legacy GFPGAN Colab notebook and a new runtime under `src/gfpp`.
To align notebooks with the current project:

- Use the new CLI `gfpup` for quick local runs:
  - `!pip install -e ".[metrics]"`
  - `!gfpup run --input inputs/whole_imgs --backend gfpgan --metrics fast --output /tmp/out --dry-run`
- Or start the API service in Colab and interact over HTTP:
  - `!pip install -e ".[api,metrics]"`
  - `!uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload &`
  - Submit a job: `requests.post("http://127.0.0.1:8000/jobs", json={"input":"inputs/whole_imgs","backend":"gfpgan","output":"results","dry_run":True}).json()`

Notes:
- Set `NB_CI_SMOKE=1` in the environment to enable smoke-friendly behavior in tests and notebooks.
- For Apple Silicon or CPU-only environments, prefer `--dry-run` for demos and use smaller inputs.
- The legacy `notebooks/GFPGAN_Colab.ipynb` still works for upstream workflows; for the new toolkit, prefer the CLI/API paths above.

