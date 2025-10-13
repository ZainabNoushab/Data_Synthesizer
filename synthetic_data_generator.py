# -*- coding: utf-8 -*-
"""Synthetic_Data_Generator"""

# ---- Install dependencies (optional when deployed) ----
# You can keep these for local runs; Streamlit Cloud uses requirements.txt
# !pip install sdv streamlit plotly pillow openpyxl pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from PIL import Image

# ---- Handle SDV version compatibility ----
try:
    # New SDV versions (>=1.0)
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    st.write("Using modern SDV (single_table).")
except ModuleNotFoundError:
    # Old SDV versions (<1.0)
    from sdv.tabular import CTGAN as CTGANSynthesizer
    SingleTableMetadata = None
    st.warning("Using legacy SDV version (tabular). Some metadata features may be limited.")

# ---- Streamlit page setup ----
st.set_page_config(page_title="Synthetic Data Generation App", layout="wide")

# Initialize session state
if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'synthetic_df' not in st.session_state:
    st.session_state.synthetic_df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None

# Logo (replace with your logo URL or upload)
st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #1f77b4;">🔬 Synthetic Data App</h1>
        <p style="font-size: 14px; color: #666;">Powered by CTGAN</p>
    </div>
""", unsafe_allow_html=True)

# If you have a logo image: st.image("logo.png", width=200) or URL: st.image("https://your-logo-url.com/logo.png")
# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "About", "Generate", "Post-Processing"]
for page in pages:
    if st.sidebar.button(page, key=page, help=f"Navigate to {page} page"):
        st.session_state.view = page.lower().replace("-", "_")
        st.rerun()
st.sidebar.markdown("---")
st.sidebar.info("Select a page to explore the app.")

# Main Content Based on View
if st.session_state.view == 'home':
    # Home Page
    st.header("Welcome to Synthetic Data Generation App")
    st.markdown("""
    This app enables secure and realistic synthetic data creation using CTGAN,
    ideal for privacy-preserving analytics and machine learning.
    """)

    st.subheader("Key Features")
    features = [
        "Seamless file upload and data preview",
        "Advanced CTGAN synthesis with customizable parameters",
        "Comprehensive data summaries and comparisons",
        "Post-processing for data quality and validation",
        "Easy download of refined datasets"
    ]
    for feature in features:
        st.markdown(f"- {feature}")
        st.markdown("---")
    st.info("Navigate using the sidebar. Start with **Generate** to upload your data!")
elif st.session_state.view == 'about':
    # About Page
    st.header("About This App")
    st.markdown("""
    ### Overview
    Built for synthetic data generation, this app leverages the **CTGAN (Conditional Tabular GAN)** model from the SDV library to create
    high-fidelity synthetic datasets that mimic real data distributions without compromising privacy.

    ### Why CTGAN?
    - Handles mixed data types (numerical, categorical).
    - Captures complex dependencies and correlations.
    - Customizable for accuracy and speed.

    ### Team Contributions
    - **UI & Navigation**: Clean sidebar-based multi-page design with tooltips.
    - **Data Summary**: Detailed previews and stats.
    - **Generation**: Integrated CTGAN with param controls.
    - **Post-Processing**: Validation and fixing options.

    ### Technologies
    - Streamlit for UI
    - SDV (Synthetic Data Vault) for CTGAN
    - Pandas & Plotly for data handling and visuals

    For issues or contributions, contact the development team.
    """)

    st.subheader("Quick Start Guide")
    st.write("""
    1. Go to **Generate** → Upload your dataset.
    2. Customize CTGAN parameters.
    3. Generate and compare synthetic data.
    4. Move to **Post-Processing** for refinements.
    5. Download your final dataset!
    """)

elif st.session_state.view == 'generate':
    # Generate Page
    st.header("Synthetic Data Generation")
    st.info("Upload your dataset and generate synthetic data using CTGAN.")

    # File Upload
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload XLSX, XLS, or CSV (max 200MB)"
    )

    if uploaded_file is not None:
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("File too large! Limit: 200MB.")
        else:
            # Read data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df
            st.success(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            # Data Summary & Display
            st.subheader("Data Summary")

            # Info: Rows, Columns, Missing
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                missing_total = df.isnull().sum().sum()
                st.metric("Missing Values", missing_total)

            # Preview
            st.subheader("Preview (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)

            # Numerical & Categorical Summary
            st.subheader("Detailed Summary")
            tab1, tab2 = st.tabs(["Numerical", "Categorical"])

            with tab1:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe().round(2), use_container_width=True)
                else:
                    st.warning("No numerical columns found.")

            with tab2:
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    for col in categorical_cols:
                        st.write(f"**{col} Value Counts:**")
                        st.dataframe(df[col].value_counts().head(10), use_container_width=True)
                else:
                    st.warning("No categorical columns found.")

            # CTGAN Parameters
            st.subheader("CTGAN Configuration")
            data_type = st.selectbox(
                "Data Type",
                ["Mixed", "Numerical", "Categorical"],
                help="Select to guide preprocessing (auto-detected otherwise)."
            )

            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Epochs", min_value=50, max_value=1000, value=300, help="Training iterations (higher = better quality).")
                batch_size = st.number_input("Batch Size", min_value=100, max_value=2000, value=500, help="Samples per training batch.")
            with col2:
                generator_decay = st.number_input("Generator Decay", min_value=1e-7, max_value=1e-3, value=1e-6, step=1e-7, format="%.1e", help="Weight decay for generator.")
                accuracy_threshold = st.number_input("Embedding Dim (Accuracy)", min_value=10, max_value=200, value=128, help="Higher for better accuracy in distributions.")
            # Generate Button
            if st.button("Generate Synthetic Data", help="Train CTGAN and create synthetic dataset"):
                df = st.session_state.df
                try:
                    # Metadata
                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(df)

                    # Adjust for data type
                    if data_type == "Categorical":
                        for col in df.select_dtypes(include=['object']).columns:
                            metadata.update_column(col, sdtype="categorical")
                    elif data_type == "Numerical":
                        for col in df.select_dtypes(include=['object']).columns:
                            metadata.update_column(col, sdtype="numerical")  # Treat as num if possible

                    st.session_state.metadata = metadata
                    # CTGAN Synthesizer
                    synthesizer = CTGANSynthesizer(
                        metadata,
                        epochs=epochs,
                        batch_size=batch_size,
                        generator_decay=generator_decay,
                        embedding_dim=accuracy_threshold
                    )

                    with st.spinner("Training CTGAN... This may take a few minutes."):
                        synthesizer.fit(df)

                    # Generate (same size)
                    synthetic_df = synthesizer.sample(len(df))
                    st.session_state.synthetic_df = synthetic_df
                    st.success("Synthetic data generated successfully!")

                    # Side-by-Side Comparison
                    st.subheader("Original vs Synthetic Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Original Data**")
                        st.dataframe(df.head(), use_container_width=True)
                    with col2:
                        st.write("**Synthetic Data**")
                        st.dataframe(synthetic_df.head(), use_container_width=True)

                    # Quick Visual (if numeric cols)
                    numeric_cols = df.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        col_name = numeric_cols[0]
                        fig = px.histogram(
                            pd.DataFrame({
                                'Value': pd.concat([df[col_name], synthetic_df[col_name]]),
                                'Type': ['Original'] * len(df) + ['Synthetic'] * len(synthetic_df)
                            }),
                            x='Value', color='Type', barmode='overlay',
                            title=f'Distribution: {col_name}'
                        )
                        st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"Generation failed: {str(e)}. Check data types or reduce params for large files.")
    else:
        st.info("Upload a file to begin generation.")
elif st.session_state.view == 'post_processing':
    # Post-Processing Page
    st.header("Post-Processing & Validation")
    st.info("Refine your synthetic data and validate quality.")

    if st.session_state.synthetic_df is None:
        st.warning("No synthetic data found. Generate data first on the **Generate** page.")
    else:
        synthetic_df = st.session_state.synthetic_df
        st.subheader("Current Synthetic Data Preview")
        st.dataframe(synthetic_df.head())

        # Detect Issues (Categorical Contradictions/Missing)
        categorical_cols = synthetic_df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("Categorical Validation")

            # Missing in Categoricals (Before)
            missing_before = synthetic_df[categorical_cols].isnull().sum()
            st.write("**Missing Values Before Processing:**")
            st.dataframe(missing_before.to_frame(), use_container_width=True)

            # Options for Fixing
            fix_option = st.selectbox(
                "Fix Option",
                ["1) Interpolate Missing Data", "2) Keep Missing", "3) Manual Edit"],
                help="Choose how to handle missing or contradictory categoricals."
            )

            if st.button("Apply Post-Processing", help="Fix issues based on selected option"):
                processed_df = synthetic_df.copy()

                if fix_option == "1) Interpolate Missing Data":
                    # Interpolate: Forward-fill for categoricals (mode per group or simple fill)
                    for col in categorical_cols:
                        processed_df[col] = processed_df[col].fillna(method='ffill').fillna(method='bfill')
                        if processed_df[col].isnull().sum() > 0:  # Fallback to mode
                            mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else 'Unknown'
                            processed_df[col] = processed_df[col].fillna(mode_val)
                    st.success("Interpolated missing values in categoricals.")
                elif fix_option == "2) Keep Missing":
                    # No change
                    st.info("Missing values kept as-is.")

                elif fix_option == "3) Manual Edit":
                    # Simple manual: User inputs replacement for a selected column
                    if len(categorical_cols) > 0:
                        col_to_edit = st.selectbox("Select Column to Edit", categorical_cols)
                        replacement = st.text_input(f"Replacement value for missing in {col_to_edit}", "Unknown")
                        mask = processed_df[col_to_edit].isnull()
                        processed_df.loc[mask, col_to_edit] = replacement
                        st.success(f"Manually replaced missing in {col_to_edit} with '{replacement}'.")
                    else:
                        st.warning("No categorical columns to edit.")
                # Contradiction Check (Simple: Flag unseen values vs original)
                original_df = st.session_state.df
                contradictions = {}
                for col in categorical_cols:
                    orig_unique = set(original_df[col].dropna().unique())
                    synth_unique = set(processed_df[col].dropna().unique())
                    contradictions[col] = len(synth_unique - orig_unique)
                if any(contradictions.values()):
                    st.warning(f"Potential contradictions detected: {contradictions}")
                else:
                    st.success("No contradictions found.")

                # After Missing
                missing_after = processed_df[categorical_cols].isnull().sum()
                st.write("**Missing Values After Processing:**")
                st.dataframe(missing_after.to_frame(), use_container_width=True)

                # Comparison Chart
                comparison_data = pd.DataFrame({
                    'Column': list(missing_before.index) + list(missing_after.index),
                    'Missing Count': list(missing_before.values) + list(missing_after.values),
                    'Stage': ['Before'] * len(missing_before) + ['After'] * len(missing_after)
                })
                fig = px.bar(comparison_data, x='Column', y='Missing Count', color='Stage', barmode='group',
                             title="Missing Values Comparison")
                st.plotly_chart(fig)

                st.session_state.processed_df = processed_df
                st.success("Post-processing applied!")

        # Download Final Dataset
        if st.session_state.processed_df is not None:
            final_df = st.session_state.processed_df
            csv_buffer = io.StringIO()
            final_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Final Dataset",
                csv_buffer.getvalue(),
                "final_synthetic_data.csv",
                help="Download the processed synthetic dataset as CSV."
            )
        elif st.session_state.synthetic_df is not None:
            # Fallback download if no processing
            csv_buffer = io.StringIO()
            st.session_state.synthetic_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Synthetic Data (Unprocessed)",
                csv_buffer.getvalue(),
                "synthetic_data.csv",
                help="Download without post-processing."
            )
