import ee
import streamlit as st


@st.cache_resource
def _initialize_ee():
    credentials = ee.ServiceAccountCredentials(
        "imsukudu24@gmail.com", "key.json"
    )
    ee.Initialize(credentials, project="careful-drummer-462304-u9")
    return True


def ensure_ee_initialized():
    _initialize_ee()
