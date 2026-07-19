import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import os
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
from tensorflow.keras.models import load_model
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
# -------------------------
# CONFIG
# -------------------------
BASE_DATASET_PATH = BASE_DIR / "datasets"
fs = 51200
window_size = 2048
overlap = 1024
step = window_size - overlap

label_map = {0: "B", 1: "I", 2: "O", 3: "IB", 4: "IO", 5: "OB", 6: "N"}

# -------------------------
# LOAD MODELS
# -------------------------
models = {
    "Random Forest": joblib.load(MODEL_DIR / "random_forest_model.pkl"),
    "SVM": joblib.load(MODEL_DIR / "svm_model.pkl"),
    "KNN": joblib.load(MODEL_DIR / "knn_model.pkl"),
}

nn_model = load_model(MODEL_DIR / "tf_nn_model.keras")

scalers = {
    "SVM": joblib.load(MODEL_DIR / "svm_scaler.pkl"),
    "KNN": joblib.load(MODEL_DIR / "knn_scaler.pkl"),
    "NN": joblib.load(MODEL_DIR / "tf_scaler.pkl"),
}

# -------------------------
# UI STYLE
# -------------------------
st.set_page_config(layout="wide")

st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
}

.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 15px;
    transition: transform 0.3s ease;
}

.card:hover {
    transform: scale(1.03);
}

h1, h2, h3, h4 {
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# FUNCTIONS
# -------------------------
def segment_signal(signal):
    return [
        signal[i : i + window_size] for i in range(0, len(signal) - window_size, step)
    ]


def extract_features(x):
    rms = np.sqrt(np.mean(x**2))
    kurt_val = kurtosis(x)

    max_val = np.max(x)
    min_val = np.min(x)
    ptp = max_val - min_val

    abs_mean = np.mean(np.abs(x)) + 1e-12

    crest_factor = max_val / (rms + 1e-12)
    impulse_factor = max_val / abs_mean

    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)

    spectral_variance = np.var(fft_vals)
    spectral_skewness = skew(fft_vals)
    spectral_kurtosis = kurtosis(fft_vals)

    prob = fft_vals / (np.sum(fft_vals) + 1e-12)
    spectral_entropy = -np.sum(prob * np.log2(prob + 1e-12))

    peak_freq = freqs[np.argmax(fft_vals)]
    low_band_power = np.sum(fft_vals[freqs < 1000])

    return [
        rms,
        kurt_val,
        max_val,
        min_val,
        ptp,
        crest_factor,
        impulse_factor,
        spectral_variance,
        spectral_skewness,
        spectral_kurtosis,
        spectral_entropy,
        peak_freq,
        low_band_power,
    ]


# -------------------------
# SIDEBAR (CONTROL PANEL)
# -------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    file = st.file_uploader("Upload .mat file", type=["mat"])

    selected_models = st.multiselect(
        "Select Models",
        ["Random Forest", "SVM", "KNN", "Neural Network"],
        default=["Random Forest"],
    )

    range_val = st.slider("Signal View Range", 100, 5000, 1000)

    show_features = st.checkbox("Show Features Table")

# -------------------------
# MAIN TITLE
# -------------------------
st.title("⚙️ Bearing Fault Analyzer Pro")

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🧠 Models", "📁 Export"])

# -------------------------
# MAIN LOGIC
# -------------------------
if file:

    with st.spinner("Processing signal..."):
        mat = loadmat(file)

        signal = None
        for key in mat:
            if not key.startswith("__"):
                signal = mat[key].flatten()
                break

        # Optional normal signal
        normal_signal = None

        if BASE_DATASET_PATH.exists():
            filename = file.name
            number = "".join(filter(str.isdigit, filename))
            normal_path = BASE_DATASET_PATH / "N" / f"N{number}.mat"

            if normal_path.exists():
                mat_n = loadmat(normal_path)
                for key in mat_n:
                    if not key.startswith("__"):
                        normal_signal = mat_n[key].flatten()
                        break

        segments = segment_signal(signal)
        features = np.array([extract_features(s) for s in segments])

    # -------------------------
    # TAB 1: ANALYSIS
    # -------------------------
    with tab1:

        col1, col2 = st.columns([2, 1])

        # LEFT SIDE (PLOTS)
        with col1:

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=signal[:range_val], name="Fault"))

            if normal_signal is not None:
                fig.add_trace(go.Scatter(y=normal_signal[:range_val], name="Normal"))

            fig.update_layout(template="plotly_dark", title="Time Domain")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("FFT Spectrum"):
                fft_vals = np.abs(np.fft.rfft(signal))
                freqs = np.fft.rfftfreq(len(signal), 1 / fs)

                fig_fft = go.Figure()
                fig_fft.add_trace(go.Scatter(x=freqs, y=fft_vals))
                fig_fft.update_layout(template="plotly_dark")
                st.plotly_chart(fig_fft)

        # RIGHT SIDE (KPIs)
        with col2:

            st.markdown('<div class="card">', unsafe_allow_html=True)

            st.metric("Segments", len(segments))
            st.metric("Signal Length", len(signal))

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # TAB 2: MODELS
    # -------------------------
    with tab2:

        results = []

        for name, model in models.items():
            if name not in selected_models:
                continue

            X = features.copy()

            if name in scalers:
                X = scalers[name].transform(X)

            probs = model.predict_proba(X)
            avg = np.mean(probs, axis=0)

            pred = np.argmax(avg)
            conf = np.max(avg)

            results.append((name, pred, conf))

        if "Neural Network" in selected_models:
            X_nn = scalers["NN"].transform(features)
            probs = nn_model.predict(X_nn, verbose=0)
            avg = np.mean(probs, axis=0)
            results.append(("Neural Network", np.argmax(avg), np.max(avg)))

        # DISPLAY CARDS
        for name, pred, conf in results:
            st.markdown(
                f"""
            <div class="card">
                <h4>{name}</h4>
                <p><b>{label_map[pred]}</b></p>
                <p>Confidence: {conf:.3f}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # FINAL
        preds = [r[1] for r in results]
        final = max(set(preds), key=preds.count)
        avg_conf = np.mean([r[2] for r in results])

        st.subheader("Final Diagnosis")
        st.success(label_map[final])

        st.metric("Overall Confidence", f"{avg_conf:.2f}")

    # -------------------------
    # TAB 3: EXPORT
    # -------------------------
    with tab3:

        df = pd.DataFrame(results, columns=["Model", "Prediction", "Confidence"])
        df["Prediction"] = df["Prediction"].map(label_map)

        st.dataframe(df)

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "Download Results", data=csv, file_name="results.csv", mime="text/csv"
        )

    # -------------------------
    # FEATURES TABLE
    # -------------------------
    if show_features:
        st.subheader("Extracted Features")
        st.dataframe(features[:10])
