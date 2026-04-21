import streamlit as st
import pandas as pd
import numpy as np
import datetime
import warnings
from utils import TriageInference
from pathlib import Path

# Suppress transformers __path__ deprecation warnings  
warnings.filterwarnings('ignore', message=".*Accessing `__path__` from.*")
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Page Config
st.set_page_config(
    page_title="Medical Triage Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #ff4b4b;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .urgent { background-color: #d32f2f; }
    .emergent { background-color: #f57c00; }
    .non-urgent { background-color: #388e3c; }
    </style>
    """, unsafe_allow_html=True)

# Cache Inference Object
@st.cache_resource
def get_inference():
    inf = TriageInference()
    inf.load_models()
    return inf

inf = get_inference()

# Header
st.title("🏥 Medical Triage Assistant")
st.markdown("---")

# Sidebar - Help and About
with st.sidebar:
    st.header("About")
    st.info("This AI assistant helps clinicians determine patient triage acuity (ESI) based on vitals, history, and chief complaint.")
    st.markdown("### Triage Levels:")
    st.markdown("- **🔴 Urgent (0)**: Critical, immediate care needed (ESI 1-2)")
    st.markdown("- **🟠 Emergent (1)**: High risk, needs prompt evaluation (ESI 3)")
    st.markdown("- **🟢 Non-Urgent (2)**: Stable, routine care (ESI 4-5)")

# Main Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Information")
    
    with st.expander("Chief Complaint", expanded=True):
        chief_complaint = st.text_area("Chief Complaint Text", placeholder="e.g., Patient reports severe chest pain and shortness of breath since morning.")

    with st.expander("Vital Signs", expanded=True):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 0, 120, 45)
        sex = c2.selectbox("Sex", ["Female", "Male"])
        temp = c3.number_input("Temp (°F)", 90.0, 110.0, 98.6)
        
        c4, c5, c6 = st.columns(3)
        sys_bp = c4.number_input("Systolic BP", 40, 250, 120)
        dias_bp = c5.number_input("Diastolic BP", 30, 150, 80)
        heart_rate = c6.number_input("Heart Rate", 30, 250, 75)
        
        c7, c8, c9 = st.columns(3)
        resp_rate = c7.number_input("Resp Rate", 8, 60, 16)
        spo2 = c8.number_input("SpO2 (%)", 50, 100, 98)
        pain_score = c9.slider("Pain Score", 0, 10, 0)

    with st.expander("Medical History"):
        ha = st.checkbox("Alzheimer's / Dementia", key="hist_alzheimers")
        ast = st.checkbox("Asthma", key="hist_asthma")
        can = st.checkbox("Cancer", key="hist_cancer")
        strk = st.checkbox("Stroke", key="hist_stroke")
        ckd = st.checkbox("CKD", key="hist_ckd")
        copd = st.checkbox("COPD", key="hist_copd")
        chf = st.checkbox("CHF", key="hist_chf")
        cad = st.checkbox("CAD", key="hist_cad")
        dep = st.checkbox("Depression", key="hist_depression")
        d1 = st.checkbox("Diabetes Type 1", key="hist_diabetes_t1")
        d2 = st.checkbox("Diabetes Type 2", key="hist_diabetes_t2")
        htn = st.checkbox("Hypertension", key="hist_hypertension")

    with st.expander("Context"):
        c1, c2 = st.columns(2)
        arrival_date = c1.date_input("Arrival Date", datetime.date.today())
        arrival_time = c2.time_input("Arrival Time", datetime.datetime.now().time())
        ems = st.checkbox("EMS Arrival")
        seen_72h = st.checkbox("Seen in last 72h")

with col2:
    st.subheader("📊 Triage Analysis")
    
    if st.button("Generate Triage Prediction"):
        if not chief_complaint:
            st.warning("Please enter a chief complaint.")
        else:
            with st.spinner("Analyzing data and text..."):
                # Prepare input
                input_data = {
                    'chief_complaint_text': chief_complaint,
                    'age': age,
                    'sex': sex, # 'Female' or 'Male' from selectbox
                    'temp': temp,
                    'sys_bp': sys_bp,
                    'dias_bp': dias_bp,
                    'heart_rate': heart_rate,
                    'resp_rate': resp_rate,
                    'spo2': spo2,
                    'pain_score': pain_score,
                    'ems_arrival': 'Yes' if ems else 'No',
                    'seen_last_72h': int(seen_72h),
                    'visit_month': arrival_date.month,
                    'day_of_week': arrival_date.weekday() + 1, # Map 0-6 (Mon-Sun) to 1-7
                    'arrival_time': arrival_time.hour * 100 + arrival_time.minute,
                    'injury_cause_text': 'None', # Placeholder for injury cause
                    'episode': 'Initial visit to this ED', 
                    'is_injury_poison': 'No injury',
                    'hist_alzheimers': int(ha),
                    'hist_asthma': int(ast),
                    'hist_cancer': int(can),
                    'hist_stroke': int(strk),
                    'hist_ckd': int(ckd),
                    'hist_copd': int(copd),
                    'hist_chf': int(chf),
                    'hist_cad': int(cad),
                    'hist_depression': int(dep),
                    'hist_diabetes_t1': int(d1),
                    'hist_diabetes_t2': int(d2),
                    'hist_diabetes_unspec': 0,
                    'hist_esrd': 0,
                    'hist_pe': 0,
                    'hist_hiv': 0,
                    'hist_high_cholesterol': 0,
                    'hist_hypertension': int(htn),
                    'hist_obesity': 0,
                    'hist_sleep_apnea': 0,
                    'hist_osteoporosis': 0,
                    'hist_substance_abuse': 0,
                }
                
                # Get Prediction
                results = inf.predict(input_data)
                
                # Display Result Card
                cls = results['final_class']
                prob = results['final_probs']
                
                labels = ["Urgent", "Emergent", "Non-Urgent"]
                colors = ["#d32f2f", "#f57c00", "#388e3c"]
                classes = ["urgent", "emergent", "non-urgent"]
                
                st.markdown(f"""
                    <div class="status-card {classes[cls]}">
                        <h2>Triage Level: {labels[cls]}</h2>
                        <h1 style="font-size: 3em;">{'🔴' if cls==0 else '🟠' if cls==1 else '🟢'}</h1>
                        <p>Confidence: {prob[cls]*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probabilities Chart
                st.write("### Urgency Distribution")
                for i in range(3):
                    col_p1, col_p2 = st.columns([1, 4])
                    col_p1.write(labels[i])
                    col_p2.progress(float(prob[i]))
                    
                # Model Breakdown
                with st.expander("Model Ensemble Details"):
                    st.write("Base Model Probability Consensus:")
                    base_probs = results['base_probs']
                    
                    # Build dataframe with ordered models
                    model_order = ['xgb_reg', 'lgb_reg', 'xgb_cls', 'lgb_cls', 'nlp']
                    model_labels = ["XGB Reg", "LGB Reg", "XGB Cls", "LGB Cls", "DistilBERT"]
                    
                    base_data = {
                        "Model": model_labels,
                        "Urgent": [base_probs[k][0, 0] for k in model_order],
                        "Emergent": [base_probs[k][0, 1] for k in model_order],
                        "Non-Urgent": [base_probs[k][0, 2] for k in model_order]
                    }
                    base_df = pd.DataFrame(base_data)
                    st.dataframe(base_df.style.highlight_max(axis=1, color="#262730"))
                    
                    st.write("Notes: Meta-Learner integrates these predictions with learned weights to provide final triage recommendatons.")

    else:
        st.info("Fill out the patient details on the left and click 'Generate Triage Prediction'.")

st.markdown("---")
st.caption("Disclaimer: This tool is for demonstration purposes only and should not be used for clinical decision making. Always consult with a qualified medical professional.")
