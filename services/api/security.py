from __future__ import annotations

from typing import Callable, TypeVar

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:  # pragma: no cover - slowapi is optional
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
except ImportError:  # pragma: no cover
    Limiter = None  # type: ignore[assignment]
    RateLimitExceeded = None  # type: ignore[assignment]
    SlowAPIMiddleware = None  # type: ignore[assignment]
    get_remote_address = None  # type: ignore[assignment]


_DEFAULT_LIMITS = ["240/minute", "20/second"]
_limiter = Limiter(key_func=get_remote_address, default_limits=_DEFAULT_LIMITS) if (Limiter and get_remote_address) else None

_Fn = TypeVar("_Fn", bound=Callable)


def _rate_limit_handler(request, exc):  # pragma: no cover - exercised via integration tests
    return JSONResponse(status_code=429, content={"error": "rate_limited", "detail": str(exc)})


def apply_security(app: FastAPI) -> None:
    """Attach baseline security middleware.

    - Enables permissive CORS for local development (front-end proxy)
    - Registers rate limiting middleware when ``slowapi`` is available
    - Keeps hooks extensible for future security hardening
    """

    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
    )

    if _limiter and SlowAPIMiddleware and RateLimitExceeded:
        app.state.limiter = _limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
        app.add_middleware(SlowAPIMiddleware)



def rate_limit(rule: str) -> Callable[[Callable], Callable]:
    """Return a rate-limit decorator that degrades gracefully when slowapi is absent."""

    if not _limiter:
        def _noop(func: _Fn) -> _Fn:
            return func

        return _noop
    return _limiter.limit(rule)


__all__ = ["apply_security", "_limiter", "rate_limit"]
