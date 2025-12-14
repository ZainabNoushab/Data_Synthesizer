# -*- coding: utf-8 -*-
"""Synthetic Data Generator — With Complete Step 9 Statistical Summary"""

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
    SDV_MODERN = True
except:
    from sdv.tabular import CTGAN as CTGANSynthesizer
    from sdv.tabular import GaussianCopula as GaussianCopulaSynthesizer
    SingleTableMetadata = None
    SDV_MODERN = False

# ---------------- Streamlit setup ----------------
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")
st.title("🔬 Synthetic Data Generator")

# ---------------- Session state ----------------
for key in ['df', 'synthetic_df', 'combined_df', 'processed_df']:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- Safe generation ----------------
def safe_generate_synthetic(model, n_samples):
    try:
        return model.sample(n_samples)
    except Exception as e:
        st.error(f"Synthetic generation failed: {e}")
        return None

# ---------------- Sidebar navigation ----------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Upload", "Preprocess", "Generate", "Validate", "Post-process"])

# ==================================================
# PAGE 1: UPLOAD
# ==================================================
if page == "Upload":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))

# ==================================================
# PAGE 2: PREPROCESS
# ==================================================
elif page == "Preprocess":
    st.header("Preprocessing")

    df = st.session_state.df
    if df is not None:
        unsuitable_cols = [
            c for c in df.columns
            if 'date' in c.lower()
            or 'time' in c.lower()
            or pd.api.types.is_datetime64_any_dtype(df[c])
        ]

        if unsuitable_cols:
            df = df.drop(columns=unsuitable_cols)
            st.warning(f"Dropped datetime/time columns: {unsuitable_cols}")

        df = df.dropna()
        st.session_state.df = df

        st.success(f"Preprocessed dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head(10))

# ==================================================
# PAGE 3: GENERATE
# ==================================================
elif page == "Generate":
    st.header("Generate Synthetic Data")

    df = st.session_state.df
    if df is not None and len(df) > 0:

        model_type = st.sidebar.selectbox("Choose Synthesizer", ["CTGAN", "GaussianCopula"])

        if model_type == "CTGAN":
            epochs = st.sidebar.number_input("Epochs", 50, 1000, 300)
            batch_size = st.sidebar.number_input("Batch Size", 100, 2000, 500)
            embedding_dim = st.sidebar.number_input("Embedding Dim", 10, 200, 128)
            generator_decay = st.sidebar.number_input(
                "Generator Decay", 1e-7, 1e-3, 1e-6, format="%.1e"
            )

        if st.button("Generate"):
            metadata = SingleTableMetadata() if SingleTableMetadata else None
            if metadata:
                metadata.detect_from_dataframe(df)

            try:
                if model_type == "CTGAN":
                    synthesizer = CTGANSynthesizer(
                        metadata,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        embedding_dim=int(embedding_dim),
                        generator_decay=float(generator_decay)
                    )
                else:
                    synthesizer = GaussianCopulaSynthesizer(metadata)

                with st.spinner("Training model..."):
                    synthesizer.fit(df)

                synthetic_df = safe_generate_synthetic(synthesizer, len(df))

                if synthetic_df is not None:
                    st.session_state.synthetic_df = synthetic_df
                    st.success("Synthetic data generated successfully")
                    st.dataframe(synthetic_df.head(10))

                    combined = pd.concat(
                        [df.reset_index(drop=True), synthetic_df.reset_index(drop=True)],
                        ignore_index=True
                    )
                    combined['source'] = ['original'] * len(df) + ['synthetic'] * len(synthetic_df)
                    st.session_state.combined_df = combined

            except Exception as e:
                st.error(f"Generation failed: {e}")

# ==================================================
# PAGE 4: VALIDATE (STEP 9 COMPLETE)
# ==================================================
elif page == "Validate":
    st.header("Validation & Statistical Analysis")

    df = st.session_state.df
    synthetic_df = st.session_state.synthetic_df
    combined = st.session_state.combined_df

    if df is not None and synthetic_df is not None:

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # ---------- KS TEST ----------
        if numeric_cols:
            st.subheader("Distribution Validation (KS Test)")
            col = st.selectbox("Select numeric column", numeric_cols)

            ks_stat, p_val = ks_2samp(df[col], synthetic_df[col])
            st.write(f"**KS Statistic:** {ks_stat:.4f} | **p-value:** {p_val:.4f}")

            fig, ax = plt.subplots()
            sns.kdeplot(df[col], label="Original", ax=ax)
            sns.kdeplot(synthetic_df[col], label="Synthetic", ax=ax)
            ax.legend()
            st.pyplot(fig)

        # ---------- STATISTICAL SUMMARY ----------
        st.subheader("Statistical Summary")

        summary_original = df[numeric_cols].describe().T
        summary_synthetic = synthetic_df[numeric_cols].describe().T

        summary_original['Dataset'] = 'Original'
        summary_synthetic['Dataset'] = 'Synthetic'

        summary_table = pd.concat([summary_original, summary_synthetic])
        st.dataframe(summary_table)

        # ---------- MEAN & STD ----------
        st.subheader("Mean & Standard Deviation Comparison")

        stats_table = []
        for col in numeric_cols:
            stats_table.append({
                "Column": col,
                "Original Mean": df[col].mean(),
                "Synthetic Mean": synthetic_df[col].mean(),
                "Original Std": df[col].std(),
                "Synthetic Std": synthetic_df[col].std()
            })

        st.dataframe(pd.DataFrame(stats_table))

        # ---------- CORRELATION ----------
        if len(numeric_cols) > 1:
            st.subheader("Correlation Difference Matrix")

            corr_diff = (df[numeric_cols].corr() - synthetic_df[numeric_cols].corr()).abs()
            st.dataframe(corr_diff)

        # ---------- CATEGORICAL ----------
        if cat_cols:
            st.subheader("Categorical Distribution Comparison")
            cat_col = st.selectbox("Select categorical column", cat_cols)

            fig = px.histogram(combined, x=cat_col, color="source", barmode="group")
            st.plotly_chart(fig)

# ==================================================
# PAGE 5: POST-PROCESS
# ==================================================
elif page == "Post-process":
    st.header("Post-process & Download")

    combined = st.session_state.combined_df
    if combined is not None:
        processed = combined.fillna(method='ffill').fillna(method='bfill')
        st.session_state.processed_df = processed

        st.dataframe(processed.head(10))

        buffer = io.StringIO()
        processed.to_csv(buffer, index=False)

        st.download_button(
            label="Download Final Dataset (CSV)",
            data=buffer.getvalue(),
            file_name="synthetic_data_final.csv",
            mime="text/csv"
        )