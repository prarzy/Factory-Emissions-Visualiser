import os

import ee
import streamlit as st


@st.cache_resource
def _initialize_ee():
    service_account = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    project_id = os.environ.get("GEE_PROJECT_ID", "")
    if not service_account or not project_id:
        raise RuntimeError(
            "GEE_SERVICE_ACCOUNT and GEE_PROJECT_ID environment variables must be set."
        )
    credentials = ee.ServiceAccountCredentials(service_account, "key.json")
    ee.Initialize(credentials, project=project_id)
    return True


def ensure_ee_initialized():
    _initialize_ee()
