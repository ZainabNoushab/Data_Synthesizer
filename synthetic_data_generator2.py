# -*- coding: utf-8 -*-
"""Improved Synthetic Data Generation Streamlit App

Features added/changed from original:
- Streamlit tabs UI (no sidebar-triggered reruns)
- Cleaner modular structure with functions
- Better error handling and user feedback
- Options for imputation instead of always dropna
- Ability to set sample size for synthesizer
- Caching of fitted synthesizer where appropriate
- Packaging-friendly structure (main() entrypoint)
- Logging and lightweight progress indications
- Support for both modern and legacy SDV with graceful fallbacks

Run with: streamlit run synthetic_data_app_improved.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
from typing import Optional, Tuple
from scipy.stats import ks_2samp
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---- SDV imports with safe fallbacks ----
try:
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    from sdv.evaluation.single_table import evaluate_quality
    SDV_MODERN = True
except Exception:
    # Legacy fallback
    try:
        from sdv.tabular import CTGAN as CTGANSynthesizer  # type: ignore
        from sdv.tabular import GaussianCopula as GaussianCopulaSynthesizer  # type: ignore
        SingleTableMetadata = None
        evaluate_quality = None
        SDV_MODERN = False
    except Exception:
        CTGANSynthesizer = None
        GaussianCopulaSynthesizer = None
        SingleTableMetadata = None
        evaluate_quality = None
        SDV_MODERN = False

# ---- Logging ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("synthetic_data_app")

# ---- Streamlit page config ----
st.set_page_config(page_title="Synthetic Data Generator", layout="wide")

# ---- Utility functions ----

def detect_metadata(df: pd.DataFrame) -> Optional[object]:
    """Return a SingleTableMetadata-like object if available; otherwise None."""
    if SingleTableMetadata is None:
        return None
    try:
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(df)
        return meta
    except Exception as e:
        logger.warning("Metadata detection failed: %s", e)
        return None


def safe_read_file(uploaded_file) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame safely."""
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")


def drop_datetime_like_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Drop columns that are very likely date/time and return dropped list."""
    unsuitable = []
    for col in df.columns:
        try:
            dtype = df[col].dtype
            name = str(col).lower()
            if pd.api.types.is_datetime64_any_dtype(dtype) or 'date' in name or 'time' in name:
                unsuitable.append(col)
        except Exception:
            continue
    if unsuitable:
        df = df.drop(columns=unsuitable)
    return df, unsuitable


def smart_missing_handling(df: pd.DataFrame, threshold_pct: float, strategy: str) -> Tuple[pd.DataFrame, list]:
    """Drop columns above missing threshold and apply chosen row-level strategy.

    strategy: one of 'drop_rows', 'ffill_bfill', 'none'
    """
    cols_to_drop = [
        col for col in df.columns if df[col].isnull().sum() / max(1, len(df)) * 100 > threshold_pct
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    if strategy == 'drop_rows':
        df = df.dropna()
    elif strategy == 'ffill_bfill':
        df = df.fillna(method='ffill').fillna(method='bfill')
    # else keep as-is
    return df, cols_to_drop


@st.cache_data
def sample_count_choices(max_rows: int) -> list:
    # Provide reasonable choices for sample size
    choices = [max_rows]
    if max_rows >= 100:
        choices = [50, 100, 200, max_rows]
    return sorted(list(set([c for c in choices if c > 0])))


@st.cache_resource
def cached_synthesizer_factory(kind: str, metadata):
    """Return a synthesizer class instance factory; not fitted."""
    if kind == 'ctgan':
        if CTGANSynthesizer is None:
            raise RuntimeError("CTGAN synthesizer not available (SDV not installed).")
        return CTGANSynthesizer
    else:
        if GaussianCopulaSynthesizer is None:
            raise RuntimeError("GaussianCopula synthesizer not available (SDV not installed).")
        return GaussianCopulaSynthesizer


# ---- UI components broken down as functions ----

def sidebar_info():
    st.sidebar.title("About")
    st.sidebar.info("Improved synthetic data generator. Use the tabs to progress: Upload → Preprocess → Generate → Validate → Post-process → Download")


def upload_tab():
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        try:
            df = safe_read_file(uploaded_file)
            st.success(f"Loaded dataset: {uploaded_file.name} — {df.shape[0]} rows x {df.shape[1]} cols")
            st.dataframe(df.head(10), use_container_width=True)
            st.session_state['df'] = df
        except Exception as e:
            st.error(e)
    else:
        st.info("Please upload a dataset to continue.")


def preprocess_tab():
    st.header("Preprocessing")
    df = st.session_state.get('df')
    if df is None:
        st.warning("No dataset loaded. Upload a dataset first.")
        return

    st.subheader("Automatic metadata detection")
    metadata = detect_metadata(df)
    if metadata is not None:
        st.success("Metadata detected")
        # show a compact view of types
        try:
            types = {c: metadata.get_column_type(c) for c in df.columns}
            st.write(types)
            st.session_state['metadata'] = metadata
        except Exception:
            st.write("Metadata available but preview failed — continuing.")
    else:
        st.info("Metadata not available (legacy SDV or detection failed).")

    st.subheader("Drop datetime-like columns")
    if st.button("Detect & Drop Datetime/Time columns"):
        new_df, dropped = drop_datetime_like_columns(df)
        if dropped:
            st.warning(f"Dropped columns: {dropped}")
            st.session_state['df'] = new_df
            df = new_df
        else:
            st.info("No datetime-like columns detected.")

    st.subheader("Missing values handling")
    missing_threshold = st.slider("Drop columns with missing values > (%)", 0, 100, 50)
    strategy = st.selectbox("Row-level missing handling", ('none', 'drop_rows', 'ffill_bfill'), help="none: keep rows; drop_rows: drop remaining rows with any NaN; ffill_bfill: forward/backward fill")

    if st.button("Apply Missing Handling"):
        processed, dropped_cols = smart_missing_handling(df.copy(), missing_threshold, strategy)
        if dropped_cols:
            st.warning(f"Dropped columns due to missing threshold: {dropped_cols}")
        st.session_state['df'] = processed
        st.success(f"After processing: {processed.shape[0]} rows x {processed.shape[1]} cols")
        st.dataframe(processed.head(10), use_container_width=True)

    # allow manual column type override
    st.subheader("Manual type overrides (optional)")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.write(f"Detected numeric columns: {numeric_cols}")


def generate_tab():
    st.header("Model & Generation")
    df = st.session_state.get('df')
    if df is None or df.shape[0] == 0:
        st.warning("No usable data found. Ensure you uploaded and preprocessed a dataset.")
        return

    model_choice = st.radio("Choose synthesizer", ('CTGAN (deep)', 'GaussianCopula (stat)'))

    # model params
    if 'CTGAN' in model_choice:
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Epochs", min_value=50, max_value=5000, value=300, step=50)
            batch_size = st.number_input("Batch size", min_value=1, max_value=5000, value=500)
        with col2:
            embedding_dim = st.number_input("Embedding dim", min_value=8, max_value=512, value=128)
            generator_decay = st.number_input("Generator decay", min_value=1e-8, max_value=1e-2, value=1e-6, format="%.1e")
        kind = 'ctgan'
    else:
        st.info("GaussianCopula has minimal hyperparameters. It's usually faster for small datasets.")
        kind = 'gaussian'

    max_rows = df.shape[0]
    choices = sample_count_choices(max_rows)
    sample_size = st.selectbox("Sample size for synthetic dataset", choices, index=len(choices)-1)

    # Generate action
    if st.button("Generate Synthetic Data"):
        # prepare metadata if available
        metadata = st.session_state.get('metadata')
        try:
            SynthClass = cached_synthesizer_factory(kind, metadata)
        except Exception as e:
            st.error(e)
            return

        try:
            with st.spinner("Fitting synthesizer — this can take time depending on data/model..."):
                # instantiate and fit
                if kind == 'ctgan':
                    synthesizer = SynthClass(metadata, epochs=int(epochs), batch_size=int(batch_size), embedding_dim=int(embedding_dim), generator_decay=float(generator_decay))
                else:
                    synthesizer = SynthClass(metadata)

                synthesizer.fit(df)

            with st.spinner("Sampling synthetic data..."):
                synthetic_df = safe_generate_synthetic(synthesizer, len(df))

            st.session_state['synthetic_df'] = synthetic_df
            st.success(f"Synthetic dataset generated: {synthetic_df.shape[0]} rows x {synthetic_df.shape[1]} cols")
            st.dataframe(synthetic_df.head(10))

            # combine and tag
            combined = pd.concat([df.reset_index(drop=True), synthetic_df.reset_index(drop=True)], ignore_index=True)
            combined['source'] = ['original'] * len(df) + ['synthetic'] * len(synthetic_df)
            st.session_state['combined_df'] = combined

            # optional quality evaluation
            if evaluate_quality is not None:
                try:
                    st.subheader("Quality evaluation (this may take a moment)")
                    qreport = evaluate_quality(df, synthetic_df, metadata)
                    st.write(qreport)
                except Exception as e:
                    st.warning(f"Quality evaluation failed: {e}")

        except Exception as e:
            st.error(f"Generation failed: {e}")


def validate_tab():
    st.header("Validation & Visuals")
    combined = st.session_state.get('combined_df')
    df = st.session_state.get('df')
    synthetic_df = st.session_state.get('synthetic_df')

    if combined is None:
        st.warning("No combined dataset available. Generate synthetic data first.")
        return

    st.subheader("Statistical summaries")
    st.write("Original data stats:")
    st.dataframe(df.describe())
    st.write("Synthetic data stats:")
    st.dataframe(synthetic_df.describe())

    st.subheader("Correlation comparison (numeric columns)")
    try:
        st.write("Original correlations:")
        st.dataframe(df.corr(numeric_only=True))
        st.write("Synthetic correlations:")
        st.dataframe(synthetic_df.corr(numeric_only=True))
    except Exception as e:
        st.warning(f"Correlation calculation failed: {e}")

    st.subheader("Graphical validation")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_cols:
        num_choice = st.selectbox("Choose numeric column for KS+KDE", numeric_cols)
        if st.button("Run KS & Draw KDE"):
            try:
                a = df[num_choice].dropna()
                b = synthetic_df[num_choice].dropna()
                ks_stat, p_value = ks_2samp(a, b)
                st.write(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

                fig, ax = plt.subplots()
                sns.kdeplot(a, label='Original', ax=ax)
                sns.kdeplot(b, label='Synthetic', ax=ax)
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Plotting failed: {e}")

    if cat_cols:
        cat_choice = st.selectbox("Choose categorical column for histogram", cat_cols)
        if st.button("Draw categorical histogram"):
            try:
                fig = px.histogram(combined, x=cat_choice, color='source', barmode='group', title=f"Histogram: {cat_choice}")
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Categorical plotting failed: {e}")


def postprocess_tab():
    st.header("Post-processing & Export")
    processed_df = st.session_state.get('processed_df')
    combined = st.session_state.get('combined_df')

    if combined is None:
        st.warning("No combined dataset available. Generate synthetic data first.")
        return

    st.subheader("Missing value fixes")
    option = st.selectbox("Missing value strategy", ("Keep Missing", "Interpolate (ffill/bfill)", "Manual Replace"))
    if st.button("Apply Fixes"):
        processed = combined.copy()
        if option == "Interpolate (ffill/bfill)":
            processed = processed.fillna(method='ffill').fillna(method='bfill')
            st.success("Applied interpolation.")
        elif option == "Manual Replace":
            col = st.selectbox("Select column to replace missing in", processed.columns)
            val = st.text_input("Replacement value", "Unknown")
            processed[col] = processed[col].fillna(val)
            st.success(f"Replaced missing in {col}.")
        else:
            st.info("Keeping missing values as-is.")

        st.session_state['processed_df'] = processed
        st.dataframe(processed.head(10))

    st.subheader("Download options")
    final_df = st.session_state.get('processed_df') or combined
    if st.button("Prepare CSV for download"):
        csv_buf = io.StringIO()
        final_df.to_csv(csv_buf, index=False)
        st.download_button("Download final CSV", csv_buf.getvalue(), file_name="synthetic_data_final.csv")

# --- Safe Synthetic Generation Wrapper ---
def safe_generate_synthetic(model, n_samples):
    import streamlit as st
    if model is None:
        st.error("Generation failed: No trained model found. Train a model first.")
        return None

    try:
        synthetic = model.sample(n_samples)
    except Exception as e:
        st.error(f"Generation failed during sampling: {e}")
        return None

    if synthetic is None:
        st.error("Generation failed: model returned None. Training likely failed or dataset was invalid.")
        return None

    return synthetic

def main():
    sidebar_info()
    st.title("🔬 Synthetic Data Generator — Improved")

    tabs = st.tabs(["Upload", "Preprocess", "Generate", "Validate", "Post-process", "Help"])
    with tabs[0]:
        upload_tab()
    with tabs[1]:
        preprocess_tab()
    with tabs[2]:
        generate_tab()
    with tabs[3]:
        validate_tab()
    with tabs[4]:
        postprocess_tab()
    with tabs[5]:
        st.header("Help & Notes")
        st.markdown("""
        - This app supports CTGAN and GaussianCopula from SDV. If SDV is not installed, generation won't work.
        - For CTGAN large datasets and many epochs can take long — consider smaller epochs or smaller sample_size for testing.
        - Use the preprocessing tab to avoid dropping important columns accidentally.
        """)


if __name__ == '__main__':
    main()



