# GFPP Quickstart

Install (editable) with optional extras:

```
pip install -e ".[dev,metrics,arcface,codeformer,restoreformerpp,ort,web]"
```

Run the new CLI (GFPGAN baseline):

```
gfpup run --input inputs/whole_imgs --backend gfpgan --metrics fast --output out/
```

API dev server:

```
uvicorn services.api.main:app --reload
```

Web (stub):

```
pnpm --filter apps/web dev
```

