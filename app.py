import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
import pickle
import io

# Set Streamlit page config
st.set_page_config(
    page_title="Alzheimer's Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for dark theme
st.markdown("""
    <style>
        body {
            color: #e0e0e0;
            background-color: #0e1117;
        }
        .stMetric {
            border-radius: 0.5rem;
            padding: 1rem;
            background-color: #1e222d;
            margin-bottom: 10px;
        }
        .stDataFrame {
            background-color: #1e222d !important;
        }
        .metric-container {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #333;
            border-radius: 8px;
            background-color: #1e1e1e;
        }
        .css-1n76uvr {
            background-color: #1e222d;
        }
        .css-1n76uvr:hover {
            background-color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('logistic_regression_model.pkl'),
        'SVM': joblib.load('svm_model.pkl'),
        'Decision Tree': joblib.load('dt_model.pkl'),
        'Random Forest': joblib.load('rf_model.pkl'),
        'AdaBoost': joblib.load('boost_model.pkl')
    }
    scaler = joblib.load('scaler.pkl')
    return models, scaler

models, scaler = load_models()

# Sidebar inputs

# Utility function to sync slider and text input
def synced_slider_input(label, min_val, max_val, default_val, step=0.1, format_func=None, help_text=""):
    key_slider = f"{label}_slider"
    key_input = f"{label}_input"

    # Initialize session state only once
    if key_slider not in st.session_state:
        st.session_state[key_slider] = default_val
    if key_input not in st.session_state:
        st.session_state[key_input] = str(default_val)

    # Update session state when input box changes
    def on_input_change():
        try:
            val = float(st.session_state[key_input])
            if min_val <= val <= max_val:
                st.session_state[key_slider] = val
        except:
            pass

    # Update session state when slider changes
    def on_slider_change():
        st.session_state[key_input] = str(st.session_state[key_slider])

    # Layout: Label, Input box, Slider, Help text in a column (each on a new line)
    st.sidebar.markdown(f"**{label}:**")
    st.sidebar.text_input(
        label="",
        key=key_input,
        on_change=on_input_change,
        label_visibility="collapsed"
    )
    st.sidebar.slider(
        label="",
        min_value=float(min_val),  # Ensure min_value is a float
        max_value=float(max_val),  # Ensure max_value is a float
        value=float(st.session_state[key_slider]),  # Ensure value is a float
        step=float(step),  # Ensure step is a float
        key=key_slider,
        on_change=on_slider_change,
        format=format_func,
        label_visibility="collapsed"
    )
    st.sidebar.markdown(f"ℹ️ {help_text}", unsafe_allow_html=True)

    return st.session_state[key_slider]

st.sidebar.header("🧠 Patient Details")

gender = st.sidebar.radio("Gender", ['Male', 'Female'], index=0)

age = synced_slider_input("Age", 50, 100, 70, help_text="Age of the patient in years")
educ = synced_slider_input("Education Years", 1, 23, 12, help_text="Total years of formal education")
ses = synced_slider_input("Socioeconomic Status", 1, 5, 3, help_text="SES level (1=high, 5=low)")
mmse = synced_slider_input("MMSE Score", 15, 30, 25, help_text="Mini-Mental State Examination score")
etiv = synced_slider_input("eTIV", 800, 2100, 1500, help_text="Estimated Total Intracranial Volume (mm³)")
nwbv = synced_slider_input("nWBV", 0.6, 0.9, 0.75, step=0.01, help_text="Normalized Whole Brain Volume")
asf = synced_slider_input("ASF", 0.8, 1.5, 1.0, step=0.01, help_text="Atlas Scaling Factor")

input_data = {
    'M/F': gender,
    'Age': int(age),
    'EDUC': int(educ),
    'SES': int(ses),
    'MMSE': float(mmse),
    'eTIV': int(etiv),
    'nWBV': float(nwbv),
    'ASF': float(asf)
}
# Preprocess input
input_df = pd.DataFrame([input_data])
input_df['M/F'] = input_df['M/F'].map({'Male': 1, 'Female': 0})
X_scaled = scaler.transform(input_df)

# Main title
st.title("🧬 Alzheimer's Disease Detection System")
# Short description
st.markdown("**Alzheimer's disease** is a progressive neurological disorder that causes brain cells to shrink and die, leading to memory loss, cognitive decline, and behavioral changes.")
st.markdown("This web application uses machine learning models to predict the likelihood of Alzheimer's disease based on various patient MRI features.")
st.subheader("Choose a model and view predictions:")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🧪 Prediction Results", "📊 Model Comparison", "📉 Confusion Matrices"])

with tab1:
    st.header("🔬 Patient Prediction")
    
    # Expander for input table
    with st.expander("📋 View Patient Input"):
        st.dataframe(input_df.style.set_properties(**{
            'background-color': '#1e1e1e',
            'color': 'white',
            'border-color': 'gray'
        }), use_container_width=True)

    # Create columns for model predictions
    left_col, right_col = st.columns(2)

    for idx, (model_name, model) in enumerate(models.items()):
        try:
            if model_name == 'Logistic Regression':
                with open('logistic_regression_model.pkl', 'rb') as f:
                    model_dict = pickle.load(f)
                model = model_dict['model']
                optimal_threshold = model_dict.get('optimal_threshold', 0.5)
                probas = model.predict_proba(X_scaled)
                pred = (probas[:, 1] > optimal_threshold).astype(int)
                proba = probas[:, 1][0]
            elif model_name == 'SVM':
                probas = model.predict_proba(X_scaled)
                pred = (probas[:, 1] > 0.5).astype(int)
                proba = probas[:, 1][0]
            else:
                pred = model.predict(X_scaled)
                proba = model.predict_proba(X_scaled)[0][1]

            # Result
            tag = "🟢 Non-Demented" if pred[0] == 0 else "🔴 Demented"
            container = left_col if idx % 2 == 0 else right_col
            with container:
                with st.container():
                    st.markdown(f"### 🔎 {model_name}")
                    st.metric("Prediction", tag, f"{proba:.2%} confidence")
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")

    # CSV Export
    export_data = []

    for name, model in models.items():
        if isinstance(model, dict):
            actual_model = model['model']
        else:
            actual_model = model

        pred = actual_model.predict(X_scaled)[0]
        status = "Demented" if pred == 1 else "Non-Demented"
        row = input_df.copy()
        row["Model"] = name
        row["Prediction"] = status
        export_data.append(row)

    # Combine all rows into a single DataFrame
    export_df = pd.concat(export_data, ignore_index=True)

    # Export to CSV
    export_csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("📁 Download CSV of Results", export_csv, file_name="alzheimer_prediction_results.csv", mime="text/csv")

with tab2:
    st.header("📈 Model Performance Comparison")

    model_metrics = {
        'Model': ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'AdaBoost'],
        'Accuracy': [0.805, 0.815, 0.815, 0.868, 0.868],
        'Recall': [0.75, 0.70, 0.65, 0.80, 0.65],
        'Precision': [0.77, 0.75, 0.70, 0.85, 0.72],
        'F1-Score': [0.76, 0.72, 0.67, 0.82, 0.68],
        'AUC': [0.750, 0.822, 0.825, 0.872, 0.825]
    }
    metrics_df = pd.DataFrame(model_metrics)

    # Display table first
    st.markdown("### 📋 Comparison Table")
    st.dataframe(metrics_df.style.set_properties(**{
        'background-color': '#1e1e1e',
        'color': 'white',
        'border-color': 'gray'
    }), use_container_width=True)

    # Interactive Plotly Chart
    st.markdown("### 📊 Interactive Metrics Chart")
    fig = px.bar(metrics_df, x='Model', y=['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC'],
                 title='Model Performance Comparison',
                 labels={'value': 'Score', 'Model': 'Algorithms'},
                 barmode='group')

    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='white'),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig)

with tab3:
    st.header("🧮 Confusion Matrices")
    conf_matrices = {
        'Logistic Regression': [[8, 2], [1, 7]],
        'SVM': [[9, 1], [2, 6]],
        'Decision Tree': [[7, 3], [1, 8]],
        'Random Forest': [[10, 0], [1, 8]],
        'AdaBoost': [[8, 2], [2, 6]]
    }

    cols = st.columns(2)
    for idx, (model_name, matrix) in enumerate(conf_matrices.items()):
        with cols[idx % 2]:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
            ax_cm.set_title(model_name, color='white')
            ax_cm.set_xlabel("Predicted", color='white')
            ax_cm.set_ylabel("Actual", color='white')
            ax_cm.tick_params(colors='white')
            fig_cm.patch.set_facecolor('#0e1117')
            ax_cm.set_facecolor('#1e1e1e')
            st.pyplot(fig_cm)
