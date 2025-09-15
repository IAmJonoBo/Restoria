from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def apply_security(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
    )
    # TODO: rate limiting (slowapi) and temp-file sandbox
    # TODO: EXIF scrubbing on upload paths
