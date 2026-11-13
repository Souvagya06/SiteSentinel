"""
SiteSentinel — Raspberry Pi Cloudflare Tunnel API Client
Connects the Render hosted backend (or local CV services) to the permanent Raspberry Pi tunnel:
  https://pi-api.sitesentinel.site
Secured via:
- Cloudflare Access Service Token (CF-Access-Client-Id & CF-Access-Client-Secret)
- Application-Level Device API Key (X-SiteSentinel-Device-Key)
"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

PI_API_BASE_URL = os.getenv("PI_API_BASE_URL", "https://pi-api.sitesentinel.site").rstrip("/")
CF_ACCESS_CLIENT_ID = os.getenv("CF_ACCESS_CLIENT_ID", "").strip()
CF_ACCESS_CLIENT_SECRET = os.getenv("CF_ACCESS_CLIENT_SECRET", "").strip()
SITESENTINEL_DEVICE_API_KEY = os.getenv("SITESENTINEL_DEVICE_API_KEY", "").strip()


def get_pi_headers() -> dict:
    """Builds authentication headers for Cloudflare Access and application API key."""
    headers = {}
    if CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET:
        headers["CF-Access-Client-Id"] = CF_ACCESS_CLIENT_ID
        headers["CF-Access-Client-Secret"] = CF_ACCESS_CLIENT_SECRET
    if SITESENTINEL_DEVICE_API_KEY:
        headers["X-SiteSentinel-Device-Key"] = SITESENTINEL_DEVICE_API_KEY
    return headers


def is_tunnel_configured() -> bool:
    """Returns True if a valid permanent tunnel base URL is configured."""
    return bool(PI_API_BASE_URL and "sitesentinel.site" in PI_API_BASE_URL)


def pi_request(endpoint: str, method: str = "GET", **kwargs) -> requests.Response:
    """
    Sends an authenticated request to the Raspberry Pi permanent tunnel endpoint.
    Automatically merges Cloudflare Service Token and SiteSentinel Device Key headers.
    """
    clean_endpoint = endpoint.lstrip("/")
    url = f"{PI_API_BASE_URL}/{clean_endpoint}"

    custom_headers = kwargs.pop("headers", {}) or {}
    headers = get_pi_headers()
    headers.update(custom_headers)

    timeout = kwargs.pop("timeout", (5, 20))
    return requests.request(
        method=method,
        url=url,
        headers=headers,
        timeout=timeout,
        **kwargs
    )

