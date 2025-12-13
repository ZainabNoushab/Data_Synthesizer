# -*- coding: utf-8 -*-
"""Synthetic Data Generator App with Logos"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
import sys
from scipy.stats import ks_2samp

# ---- Handle SDV version compatibility ----
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
except ModuleNotFoundError:
    from sdv.tabular import CTGAN as CTGANSynthesizer
    from sdv.tabular import GaussianCopula as GaussianCopulaSynthesizer
    SingleTableMetadata = None
    st.warning("Using legacy SDV version (tabular). Some metadata features may be limited.")

# ---- Helper function for PyInstaller path ----
def resource_path(relative_path):
    """Get absolute path to resource, works for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---- Load logos ----
ecliptica_logo_path = resource_path("ecliptica_logo.png")
rss_logo_path = resource_path("rss_logo.png")
ecliptica_logo = Image.open(ecliptica_logo_path)
rss_logo = Image.open(rss_logo_path)

# ---- Streamlit page setup ----
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# ---- Display Logos ----
st.image(ecliptica_logo, width=150)
st.title("🔬 Synthetic Data Generator")
st.sidebar.image(rss_logo, width=80)
st.sidebar.header("Navigation")

# ---- Sidebar Pages ----
pages = ["Home", "About", "Generate", "Post-Processing"]
page = st.sidebar.radio("Go to", pages)
page = page.lower().replace("-", "_")

# ---- Initialize session state ----
for key, value in {
    'df': None,
    'synthetic_df': None,
    'processed_df': None,
    'combined_df': None,
    'metadata': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---- Pages ----
if page == "home":
    st.header("Welcome")
    st.markdown("Generate privacy-preserving synthetic datasets using CTGAN or GaussianCopula.")

elif page == "about":
    st.header("About This App")
    st.markdown("""
    This app uses SDV to generate synthetic data.
    - **CTGAN**: deep learning model for complex data.
    - **GaussianCopula**: statistical model for smaller datasets.
    """)

elif page == "generate":
    st.header("Generate Synthetic Data")
    uploaded_file = st.file_uploader("Upload CSV/XLSX file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.dataframe(df.head())

        # Drop unsuitable columns
        df = df.drop(columns=[c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()], errors='ignore')
        df = df.dropna()
        st.session_state.df = df

        model_type = st.selectbox("Select Synthesizer Model", ["CTGAN", "GaussianCopula"])
        if st.button("Generate Synthetic Data"):
            metadata = SingleTableMetadata() if SingleTableMetadata else None
            try:
                if metadata:
                    metadata.detect_from_dataframe(df)
                if model_type == "CTGAN":
                    synthesizer = CTGANSynthesizer(metadata, epochs=300)
                    synthesizer.fit(df)
                else:
                    synthesizer = GaussianCopulaSynthesizer(metadata)
                    synthesizer.fit(df)

                synthetic_df = synthesizer.sample(len(df))
                st.session_state.synthetic_df = synthetic_df
                st.success("Synthetic data generated!")

                # Combine
                combined_df = pd.concat([df, synthetic_df], ignore_index=True)
                combined_df['source'] = ['original'] * len(df) + ['synthetic'] * len(synthetic_df)
                st.session_state.combined_df = combined_df
                st.dataframe(combined_df.head())

            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

elif page == "post_processing":
    st.header("Post-Processing")
    if st.session_state.combined_df is not None:
        st.dataframe(st.session_state.combined_df.head())
    else:
        st.warning("Generate synthetic data first.")

