# -*- coding: utf-8 -*-
"""Synthetic_Data_Generator_Improved"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# ---- Handle SDV version compatibility ----
try:
    # New SDV versions (>=1.0)
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality, get_column_plot
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
    'combined_df': None,
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

    # Step 1: Input
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        # Step 2: Reads data according to the category (file type)
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.session_state.df = df

        st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head(10), use_container_width=True)

        # Step 3: Categorize the data (auto-detect via SDV metadata)
        metadata = SingleTableMetadata() if SingleTableMetadata else None
        if metadata:
            metadata.detect_from_dataframe(df)
            st.write("**Detected Column Types:**", metadata.columns)
        else:
            st.warning("Metadata detection not available in legacy SDV.")

        # Step 4: Drop data not suitable for synthesizing like date and time
        unsuitable_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                unsuitable_cols.append(col)
        if unsuitable_cols:
            df = df.drop(columns=unsuitable_cols)
            st.warning(f"Dropped unsuitable columns: {unsuitable_cols}")
        st.session_state.df = df

        # Step 5: Drop missing values (or handle them)
        missing_threshold = st.slider("Drop columns with missing values > %", 0, 100, 50)
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() / len(df) * 100 > missing_threshold]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            st.warning(f"Dropped columns with high missing values: {cols_to_drop}")
        df = df.dropna()  # Drop rows with any remaining NaNs
        st.success(f"After preprocessing: {df.shape[0]} rows and {df.shape[1]} columns")
        st.session_state.df = df

        # ---- Model Selection ----
        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Select Synthesizer Model",
            ["CTGAN (Deep Learning)", "GaussianCopula (Statistical)"],
            help="Choose between a neural network or a statistical model."
        )

        if "CTGAN" in model_type:
            st.subheader("CTGAN Configuration")
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

        # Step 6: Apply CTGAN for synthesizing data (or GaussianCopula)
        if st.button("Generate Synthetic Data"):
            df = st.session_state.df
            try:
                if metadata:
                    metadata.detect_from_dataframe(df)  # Re-detect after preprocessing

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

                # Step 7: Check for column syns and connection (quality evaluation)
                if 'evaluate_quality' in globals():
                    quality_report = evaluate_quality(df, synthetic_df, metadata)
                    st.subheader("Synthesis Quality Report")
                    st.write(quality_report)
                    # Check for column connections (e.g., correlations)
                    st.write("**Column Relationships (Original Correlations):**")
                    st.dataframe(df.corr(numeric_only=True))
                    st.write("**Column Relationships (Synthetic Correlations):**")
                    st.dataframe(synthetic_df.corr(numeric_only=True))
                else:
                    st.warning("Quality evaluation not available in legacy SDV.")

                # Step 8: Combine synthetic and non synthetic data to form combined synthetic data
                combined_df = pd.concat([df, synthetic_df], ignore_index=True)
                combined_df['source'] = ['original'] * len(df) + ['synthetic'] * len(synthetic_df)
                st.session_state.combined_df = combined_df
                st.success("Combined original and synthetic data.")

                # Step 9: Perform statistical calculation like mean, std deviation and correlation
                st.subheader("Statistical Calculations")
                stats_original = df.describe()
                stats_synthetic = synthetic_df.describe()
                st.write("**Original Data Stats:**")
                st.dataframe(stats_original)
                st.write("**Synthetic Data Stats:**")
                st.dataframe(stats_synthetic)
                st.write("**Combined Correlations:**")
                st.dataframe(combined_df.corr(numeric_only=True))

                # Step 10: Validation and graphical representation
                st.subheader("Validation & Graphical Representations")
                numeric_cols = df.select_dtypes(include=np.number).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns

                # For numerical: KS plot and KDE
                if len(numeric_cols) > 0:
                    col_name = st.selectbox("Select Numerical Column for Validation", numeric_cols)
                    # KS Test
                    ks_stat, p_value = ks_2samp(df[col_name].dropna(), synthetic_df[col_name].dropna())
                    st.write(f"**KS Test for {col_name}**: Statistic={ks_stat:.4f}, p-value={p_value:.4f}")
                    # KDE Plot
                    fig, ax = plt.subplots()
                    sns.kdeplot(df[col_name], label='Original', ax=ax)
                    sns.kdeplot(synthetic_df[col_name], label='Synthetic', ax=ax)
                    ax.legend()
                    st.pyplot(fig)

                # For categorical: Histogram
                if len(categorical_cols) > 0:
                    cat_col = st.selectbox("Select Categorical Column for Histogram", categorical_cols)
                    fig = px.histogram(combined_df, x=cat_col, color='source', barmode='group', title=f"Categorical Histogram for {cat_col}")
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"❌ Generation failed: {e}")

    else:
        st.info("Upload a dataset to begin.")

# ---- Post Processing ----
elif st.session_state.view == 'post_processing':
    st.header("Post-Processing & Validation")
    if st.session_state.combined_df is None:
        st.warning("No combined data available yet. Generate data first.")
    else:
        combined_df = st.session_state.combined_df
        st.dataframe(combined_df.head())

        fix_option = st.selectbox(
            "Missing Value Handling",
            ["Keep Missing", "Interpolate Missing", "Manual Replace"]
        )

        if st.button("Apply Fixes"):
            processed_df = combined_df.copy()
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
