import os
from pathlib import Path

import ee
import streamlit as st


def _load_dotenv():
    """Load a .env file from the project root into os.environ (if not already set)."""
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


@st.cache_resource
def _initialize_ee():
    service_account = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    project_id = os.environ.get("GEE_PROJECT_ID", "")
    if not service_account or not project_id:
        raise RuntimeError(
            "GEE_SERVICE_ACCOUNT and GEE_PROJECT_ID must be set in .env or the environment."
        )
    credentials = ee.ServiceAccountCredentials(service_account, "key.json")
    ee.Initialize(credentials, project=project_id)
    return True


def ensure_ee_initialized():
    _initialize_ee()
