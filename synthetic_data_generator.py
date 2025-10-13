# -*- coding: utf-8 -*-
"""Synthetic_Data_Generator"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from PIL import Image

# ---- Handle SDV version compatibility ----
try:
    # New SDV versions (>=1.0)
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    st.write("Using modern SDV (single_table).")
except ModuleNotFoundError:
    # Old SDV versions (<1.0)
    from sdv.tabular import CTGAN as CTGANSynthesizer
    from sdv.tabular import GaussianCopula as GaussianCopulaSynthesizer
    SingleTableMetadata = None
    st.warning("Using legacy SDV version (tabular). Some metadata features may be limited.")

# ---- Streamlit page setup ----
st.set_page_config(page_title="Synthetic Data Generation App", layout="wide")

# Initialize session state
for key, value in {
    'view': 'home',
    'df': None,
    'synthetic_df': None,
    'processed_df': None,
    'metadata': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---- Header ----
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1f77b4;">🔬 Synthetic Data App</h1>
        <p style="font-size: 14px; color: #666;">Powered by CTGAN & GaussianCopula</p>
    </div>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("Navigation")
pages = ["Home", "About", "Generate", "Post-Processing"]
for page in pages:
    if st.sidebar.button(page, key=page, help=f"Go to {page} page"):
        st.session_state.view = page.lower().replace("-", "_")
        st.rerun()
st.sidebar.markdown("---")
st.sidebar.info("Select a page to explore the app.")

# ---- Home ----
if st.session_state.view == 'home':
    st.header("Welcome to Synthetic Data Generation App")
    st.markdown("""
    Generate secure, realistic synthetic datasets using deep learning (CTGAN) or statistical (GaussianCopula) models.
    """)

    st.subheader("Key Features")
    features = [
        "Upload, preview, and summarize datasets",
        "Choose CTGAN or Statistical model for generation",
        "Advanced comparison and visual validation",
        "Built-in post-processing tools",
        "Download refined datasets easily"
    ]
    for f in features:
        st.markdown(f"- {f}")
    st.info("Navigate using the sidebar → Start with **Generate**!")

# ---- About ----
elif st.session_state.view == 'about':
    st.header("About This App")
    st.markdown("""
    This app uses **SDV (Synthetic Data Vault)** to generate privacy-preserving synthetic data.
    You can pick between:
    - **CTGAN** — a deep learning model for complex data
    - **GaussianCopula** — a statistical model that’s faster for small datasets
    """)

# ---- Generate ----
elif st.session_state.view == 'generate':
    st.header("Synthetic Data Generation")
    st.info("Upload your dataset and generate synthetic data below.")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.df = df

        st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)

        # ---- Model Selection ----
        st.subheader("Model Selection")

        model_type = st.selectbox(
            "Select Synthesizer Model",
            ["CTGAN (Deep Learning)", "GaussianCopula (Statistical)"],
            help="Choose between a neural network or a statistical model."
        )

        if "CTGAN" in model_type:
            st.subheader("CTGAN Configuration")
            data_type = st.selectbox(
                "Data Type",
                ["Mixed", "Numerical", "Categorical"],
                help="Helps optimize preprocessing for data type."
            )
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Epochs", 50, 1000, 300)
                batch_size = st.number_input("Batch Size", 100, 2000, 500)
            with col2:
                generator_decay = st.number_input("Generator Decay", 1e-7, 1e-3, 1e-6, step=1e-7, format="%.1e")
                embedding_dim = st.number_input("Embedding Dim (Accuracy)", 10, 200, 128)
        else:
            st.subheader("Statistical Model Configuration")
            st.info("GaussianCopulaSynthesizer requires no tuning — just click generate!")

        # ---- Generate Button ----
        if st.button("Generate Synthetic Data"):
            df = st.session_state.df
            try:
                # Detect metadata
                metadata = SingleTableMetadata() if SingleTableMetadata else None
                if metadata:
                    metadata.detect_from_dataframe(df)

                # Choose model
                if "CTGAN" in model_type:
                    synthesizer = CTGANSynthesizer(
                        metadata,
                        epochs=epochs,
                        batch_size=batch_size,
                        generator_decay=generator_decay,
                        embedding_dim=embedding_dim
                    )
                    with st.spinner("Training CTGAN..."):
                        synthesizer.fit(df)
                else:
                    synthesizer = GaussianCopulaSynthesizer(metadata)
                    with st.spinner("Fitting GaussianCopula model..."):
                        synthesizer.fit(df)

                synthetic_df = synthesizer.sample(len(df))
                st.session_state.synthetic_df = synthetic_df
                st.success(f"Synthetic data generated using {model_type}!")

                # ---- Comparison ----
                st.subheader("Original vs Synthetic")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data**")
                    st.dataframe(df.head())
                with col2:
                    st.write("**Synthetic Data**")
                    st.dataframe(synthetic_df.head())

                # ---- Visualization ----
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    col_name = numeric_cols[0]
                    fig = px.histogram(
                        pd.DataFrame({
                            "Value": pd.concat([df[col_name], synthetic_df[col_name]]),
                            "Type": ["Original"] * len(df) + ["Synthetic"] * len(synthetic_df)
                        }),
                        x="Value", color="Type", barmode="overlay",
                        title=f"Distribution Comparison — {col_name}"
                    )
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

    else:
        st.info("Upload a dataset to begin.")

# ---- Post Processing ----
elif st.session_state.view == 'post_processing':
    st.header("Post-Processing & Validation")
    if st.session_state.synthetic_df is None:
        st.warning("No synthetic data available yet. Generate data first.")
    else:
        synthetic_df = st.session_state.synthetic_df
        st.dataframe(synthetic_df.head())

        fix_option = st.selectbox(
            "Missing Value Handling",
            ["Keep Missing", "Interpolate Missing", "Manual Replace"]
        )

        if st.button("Apply Fixes"):
            processed_df = synthetic_df.copy()
            if fix_option == "Interpolate Missing":
                processed_df.fillna(method='ffill', inplace=True)
                processed_df.fillna(method='bfill', inplace=True)
                st.success("Interpolated missing values.")
            elif fix_option == "Manual Replace":
                col = st.selectbox("Select Column", processed_df.columns)
                value = st.text_input("Replacement Value", "Unknown")
                processed_df[col] = processed_df[col].fillna(value)
                st.success(f"Replaced missing values in {col}.")
            else:
                st.info("Missing values kept as-is.")
            st.session_state.processed_df = processed_df

        if st.session_state.processed_df is not None:
            csv_buffer = io.StringIO()
            st.session_state.processed_df.to_csv(csv_buffer, index=False)
            st.download_button("Download Final Data", csv_buffer.getvalue(), "synthetic_data_final.csv")
