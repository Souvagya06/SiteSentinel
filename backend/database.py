import os
import requests
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

TURSO_URL = os.getenv("TURSO_DATABASE_URL").replace("libsql://", "https://")
TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

def execute(sql, args=[]):
    response = requests.post(
        f"{TURSO_URL}/v2/pipeline",
        headers={"Authorization": f"Bearer {TURSO_TOKEN}"},
        json={"requests": [
            {"type": "execute", "stmt": {"sql": sql, "args": args}},
            {"type": "close"}
        ]}
    )
    response.raise_for_status()
    return response.json()

def query_one(sql, args=[]):
    """Returns first row as a dict, or None."""
    result = execute(sql, args)
    try:
        cols = [c["name"] for c in result["results"][0]["response"]["result"]["cols"]]
        rows = result["results"][0]["response"]["result"]["rows"]
        if not rows:
            return None
        return dict(zip(cols, [v["value"] for v in rows[0]]))
    except (KeyError, IndexError):
        return None