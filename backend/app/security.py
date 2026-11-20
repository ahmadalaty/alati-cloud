"""
Security helpers: response headers and a small in-memory rate limiter.

The service runs as a single process on Render, so a per-process limiter is
enough to blunt password guessing and email flooding. It is not a substitute
for an edge firewall, and the counters reset when the service restarts.
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import Request

# Pages use inline <script> and <style> blocks, so 'unsafe-inline' is allowed
# for both; everything else - fonts, images, fetch calls - must come from this
# site. frame-ancestors blocks clickjacking; object-src and base-uri close two
# classic injection routes.
CONTENT_SECURITY_POLICY = "; ".join([
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=31536000",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}

# Responses that carry account or clinical data must not be kept by browsers
# or proxies.
NO_STORE_PREFIXES = ("/scan/", "/user/", "/admin/", "/auth/")

_lock = threading.Lock()
_hits = defaultdict(deque)


def client_ip(request: Request) -> str:
    """Best guess at the caller's address behind Cloudflare and Render's proxy."""
    for header in ("cf-connecting-ip", "x-forwarded-for"):
        value = request.headers.get(header, "")
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limited(bucket: str, key: str, limit: int, window_seconds: int) -> bool:
    """Record one attempt; True if this key has used up its allowance for the window."""
    now = time.monotonic()
    k = (bucket, key)
    with _lock:
        q = _hits[k]
        while q and now - q[0] > window_seconds:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        if len(_hits) > 50000:          # keep memory bounded under a flood of new keys
            for old in [x for x, v in _hits.items() if not v][:10000]:
                _hits.pop(old, None)
        return False


def reset_rate_limits() -> None:
    """For tests."""
    with _lock:
        _hits.clear()
