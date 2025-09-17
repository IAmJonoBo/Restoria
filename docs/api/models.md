# API: Models

Model- and backend-related APIs and configuration.

- Listing available backends
- Resolving weights via centralized logic
- Notes on per-backend options (overview)

## Backends and availability

```python
from gfpp.core.registry import list_backends, get

avail = list_backends()  # {'gfpgan': True, 'codeformer': False, ...}

# Include experimental ones
avail_all = list_backends(include_experimental=True)

# Resolve a backend class (imports on first use)
RestorerClass = get('gfpgan')
restorer = RestorerClass()
```

Aliases: some names map to canonical implementations, e.g.,
`restoreformer -> restoreformerpp`.

## Weight resolution

Use the IO helper to ensure a model weight is present locally and obtain its
path and optional sha256. This delegates to the centralized resolver to
avoid duplication.

```python
from gfpp.io.weights import ensure_weight

path, sha256 = ensure_weight("GFPGANv1.4")
print(path, sha256)
```

Notes:

- If `no_download=True`, this will not attempt network access and may raise
  from the underlying resolver if the weight is missing.
- The central resolver supports local caches and may fetch from configured
  sources when permitted.

## Per-backend options (overview)

- gfpgan: `weight` (blend), `upscale`, optional background upsampler
- codeformer: fidelity `weight` typically >= 0.6 for strong restoration
- restoreformerpp: similar to GFPGAN with different artifact trade-offs
- ensemble: combines outputs; expect higher latency

Tip: When using the orchestrator, backend parameters may be normalized for
determinism (e.g., weight defaults to 0.6 for moderate degradation).
