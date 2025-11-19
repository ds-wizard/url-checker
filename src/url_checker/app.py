import asyncio
import time
from typing import List, Optional, Literal

import httpx
import jwt
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from url_checker import config

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------


class UrlCheck(BaseModel):
    urls: List[str]


ErrorCode = Literal[
    "TIMEOUT",
    "CONNECT_TIMEOUT",
    "INVALID_URL",
    "HTTP_ERROR",
    "NOT_HTML",
    "NETWORK_ERROR",
    "UNKNOWN_ERROR",
    None,  # means no error
]


class UrlResult(BaseModel):
    url: str
    ok: bool
    error_code: ErrorCode = None
    error_message: Optional[str] = None

    # Diagnostics (always present when available, even on errors)
    http_status: Optional[int] = None
    final_url: Optional[str] = None
    content_type: Optional[str] = None  # e.g. "text/html"
    reason: Optional[str] = None  # "html-by-header" | "html-by-sniff" | "not-html"
    elapsed_ms: float


class UrlCheckResponse(BaseModel):
    ok_count: int
    error_count: int
    results: List[UrlResult]


# ---------- Core checker ----------

HTML_MIME_PREFIXES = ("text/html", "application/xhtml+xml")


def looks_like_html(sample: bytes) -> bool:
    s = sample.lower()
    return any(tag in s for tag in (b"<!doctype html", b"<html", b"<head", b"<body"))


async def check_one(client: httpx.AsyncClient, url: str, sniff_bytes: int = 1024) -> UrlResult:
    t0 = time.perf_counter()
    try:
        async with client.stream("GET", url) as resp:
            final_url = str(resp.url)
            status = resp.status_code
            ctype = resp.headers.get("content-type", "")
            cmain = ctype.split(";")[0].strip().lower() if ctype else ""

            # If not a 2xx status, fail regardless of content
            if status < 200 or status >= 300:
                return UrlResult(
                    url=url, ok=False, error_code="HTTP_ERROR",
                    error_message=f"HTTP status {status} is not 2xx.",
                    http_status=status, final_url=final_url,
                    content_type=cmain or "unknown",
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                )

            # 1) decide by header
            if any(cmain.startswith(p) for p in HTML_MIME_PREFIXES):
                return UrlResult(
                    url=url, ok=True, http_status=status,
                    final_url=final_url, content_type=cmain,
                    reason="html-by-header", elapsed_ms=(time.perf_counter() - t0) * 1000.0
                )

            # 2) sniff small chunk
            sample = b""
            async for chunk in resp.aiter_bytes():
                sample += chunk
                if len(sample) >= sniff_bytes:
                    break
            if looks_like_html(sample):
                return UrlResult(
                    url=url, ok=True, http_status=status,
                    final_url=final_url, content_type=cmain or "unknown",
                    reason="html-by-sniff", elapsed_ms=(time.perf_counter() - t0) * 1000.0
                )

            # not HTML
            return UrlResult(
                url=url, ok=False, error_code="NOT_HTML",
                error_message="Final response is not HTML.",
                http_status=status, final_url=final_url,
                content_type=cmain or "unknown",
                reason="not-html", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            )

    except httpx.ReadTimeout:
        return UrlResult(url=url, ok=False, error_code="TIMEOUT",
                         error_message="Read timeout.",
                         elapsed_ms=(time.perf_counter() - t0) * 1000.0)
    except httpx.ConnectTimeout:
        return UrlResult(url=url, ok=False, error_code="CONNECT_TIMEOUT",
                         error_message="Connect timeout.",
                         elapsed_ms=(time.perf_counter() - t0) * 1000.0)
    except httpx.InvalidURL:
        return UrlResult(url=url, ok=False, error_code="INVALID_URL",
                         error_message="Invalid URL.",
                         elapsed_ms=(time.perf_counter() - t0) * 1000.0)
    except httpx.HTTPError as e:
        return UrlResult(url=url, ok=False, error_code="NETWORK_ERROR",
                         error_message=f"HTTP error: {e.__class__.__name__}",
                         elapsed_ms=(time.perf_counter() - t0) * 1000.0)
    except Exception as e:
        return UrlResult(url=url, ok=False, error_code="UNKNOWN_ERROR",
                         error_message=f"{e.__class__.__name__}: {e}",
                         elapsed_ms=(time.perf_counter() - t0) * 1000.0)


async def check_many(urls: List[str]) -> List[UrlResult]:
    """Reasonable defaults."""
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
    timeout = httpx.Timeout(connect=10.0, read=10.0, write=10.0, pool=None)
    headers = {
        "User-Agent": "URLChecker/1.0",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            verify=True,
            limits=limits,
            timeout=timeout,
            http2=False,
            trust_env=True,
    ) as client:
        sem = asyncio.Semaphore(100)
        results: List[Optional[UrlResult]] = [None] * len(urls)

        async def bounded(i: int, u: str):
            async with sem:
                results[i] = await check_one(client, u)

        await asyncio.gather(*(bounded(i, u) for i, u in enumerate(urls)))
        return results  # type: ignore


# ---------- Core checker ----------

def validate_jwt_token(token):
    if not config.WIZARD_JWT_PUBLIC_KEY:
        return True

    try:
        jwt.decode(
            token,
            config.WIZARD_JWT_PUBLIC_KEY,
            algorithms=["RS256"],
        )
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False


# ---------- FastAPI endpoint ----------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/url-check", response_model=UrlCheckResponse)
async def url_check(payload: UrlCheck, token: str = Depends(oauth2_scheme)):
    if not validate_jwt_token(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not payload.urls:
        return UrlCheckResponse(ok_count=0, error_count=0, results=[])

    results = await check_many(payload.urls)
    ok_count = sum(1 for r in results if r.ok)
    return UrlCheckResponse(
        ok_count=ok_count,
        error_count=len(results) - ok_count,
        results=results
    )
