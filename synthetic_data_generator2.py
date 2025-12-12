# -*- coding: utf-8 -*-
"""Synthetic Data Generator Improved — Fully Working Version"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ks_2samp

# SDV imports with safe fallback
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
    SDV_MODERN = True
except:
    from sdv.tabular import CTGAN as CTGANSynthesizer
    from sdv.tabular import GaussianCopula as GaussianCopulaSynthesizer
    SingleTableMetadata = None
    evaluate_quality = None
    SDV_MODERN = False

# --- Streamlit setup ---
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# --- Session state init ---
for key in ['df', 'synthetic_df', 'combined_df', 'metadata', 'processed_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Safe generation wrapper ---
def safe_generate_synthetic(model, n_samples):
    if model is None:
        st.error("Generation failed: No trained model found. Train a model first.")
        return None
    try:
        synthetic = model.sample(n_samples)
    except Exception as e:
        st.error(f"Generation failed during sampling: {e}")
        return None
    if synthetic is None:
        st.error("Generation failed: model returned None. Training likely failed or dataset invalid.")
        return None
    return synthetic

# --- App UI ---
tabs = st.tabs(["Upload", "Preprocess", "Generate", "Validate", "Post-process"])

with tabs[0]:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv','xlsx','xls'])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} cols")
        st.dataframe(df.head(10))

with tabs[1]:
    st.header("Preprocessing")
    df = st.session_state.df
    if df is not None:
        # Drop datetime-like columns
        unsuitable_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
        if unsuitable_cols:
            df = df.drop(columns=unsuitable_cols)
            st.warning(f"Dropped datetime/time columns: {unsuitable_cols}")
        # Drop missing
        df = df.dropna()
        st.session_state.df = df
        st.success(f"Preprocessed dataset: {df.shape[0]} rows x {df.shape[1]} cols")
        st.dataframe(df.head(10))

with tabs[2]:
    st.header("Generate Synthetic Data")
    df = st.session_state.df
    if df is not None and df.shape[0]>0:
        model_type = st.selectbox("Choose Synthesizer", ["CTGAN", "GaussianCopula"])
        if model_type == "CTGAN":
            epochs = st.number_input("Epochs", 50, 1000, 300)
            batch_size = st.number_input("Batch Size", 100, 2000, 500)
            embedding_dim = st.number_input("Embedding Dim", 10, 200, 128)
            generator_decay = st.number_input("Generator Decay", 1e-7, 1e-3, 1e-6, format="%.1e")

        if st.button("Generate"):
            metadata = SingleTableMetadata() if SingleTableMetadata else None
            if metadata:
                metadata.detect_from_dataframe(df)
            try:
                if model_type=="CTGAN":
                    synthesizer = CTGANSynthesizer(metadata, epochs=int(epochs), batch_size=int(batch_size), embedding_dim=int(embedding_dim), generator_decay=float(generator_decay))
                else:
                    synthesizer = GaussianCopulaSynthesizer(metadata)
                with st.spinner("Training..."):
                    synthesizer.fit(df)
                synthetic_df = safe_generate_synthetic(synthesizer, len(df))
                if synthetic_df is not None:
                    st.session_state.synthetic_df = synthetic_df
                    st.success("Synthetic data generated")
                    st.dataframe(synthetic_df.head(10))
                    # Combine
                    combined = pd.concat([df.reset_index(drop=True), synthetic_df.reset_index(drop=True)], ignore_index=True)
                    combined['source'] = ['original']*len(df) + ['synthetic']*len(synthetic_df)
                    st.session_state.combined_df = combined
            except Exception as e:
                st.error(f"Generation failed: {e}")

with tabs[3]:
    st.header("Validate")
    combined = st.session_state.combined_df
    df = st.session_state.df
    synthetic_df = st.session_state.synthetic_df
    if combined is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Select numeric column", numeric_cols)
            ks_stat, p_val = ks_2samp(df[col], synthetic_df[col])
            st.write(f"KS Test: stat={ks_stat:.4f}, p={p_val:.4f}")
            fig, ax = plt.subplots()
            sns.kdeplot(df[col], label='Original', ax=ax)
            sns.kdeplot(synthetic_df[col], label='Synthetic', ax=ax)
            ax.legend()
            st.pyplot(fig)
        if cat_cols:
            cat_col = st.selectbox("Select categorical column", cat_cols)
            fig = px.histogram(combined, x=cat_col, color='source', barmode='group')
            st.plotly_chart(fig)

with tabs[4]:
    st.header("Post-process & Download")
    combined = st.session_state.combined_df
    if combined is not None:
        processed = combined.fillna(method='ffill').fillna(method='bfill')
        st.session_state.processed_df = processed
        st.dataframe(processed.head(10))
        csv_buf = io.StringIO()
        processed.to_csv(csv_buf, index=False)
        st.download_button("Download CSV", csv_buf.getvalue(), "synthetic_data_final.csv")
