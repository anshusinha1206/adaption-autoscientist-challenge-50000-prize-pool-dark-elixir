import re
import streamlit as st
import pandas as pd
import pickle

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(page_title="CardioRisk AI", page_icon="🫀", layout="wide")

# =========================================================================
# LOAD MODEL
# =========================================================================
@st.cache_resource
def load_components():
    with open("heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_components()
model = model_data["model"]
scaler = model_data["scaler"]
expected_features = model_data["features"]

# =========================================================================
# DESIGN TOKENS
# =========================================================================
# Palette — a clinical chart-review palette (cool paper, deep navy ink,
# muted teal for "in range", amber for borderline, brick red for elevated).
BG = "#F1F5F4"
CARD = "#FFFFFF"
INK = "#152530"
MUTED = "#5C6D71"
PRIMARY = "#1F3B4D"
SAFE = "#2F6F62"
WARN = "#C1852B"
RISK = "#A9433D"
BORDER = "#DCE4E2"

def render_html(html_str: str) -> None:
    """Render a block of custom HTML via st.markdown.

    Streamlit's markdown parser treats any blank (or whitespace-only) line
    followed by an indented line as an indented CODE block instead of raw
    HTML — which is exactly what produced the boxed/"copy" widget instead
    of the intended layout. This collapses those stray blank lines and
    strips leading indentation so the HTML renders as intended.
    """
    cleaned = re.sub(r"\n[ \t]*\n", "\n", html_str)
    st.markdown(cleaned.strip(), unsafe_allow_html=True)


st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {INK};
    }}
    .stApp {{
        background-color: {BG};
    }}
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY};
    }}
    [data-testid="stSidebar"] * {{
        color: #EAF0EE !important;
    }}
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {{
        padding-top: 4px;
    }}
    .stButton > button {{
        background-color: {RISK};
        color: white;
        border: none;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .stButton > button:hover {{
        background-color: #8C3630;
        color: white;
    }}
    h1, h2, h3 {{
        font-family: 'Lora', serif;
        color: {PRIMARY};
    }}

    /* ---- Hero / monitor strip ---- */
    .monitor-strip {{
        background: {PRIMARY};
        border-radius: 10px;
        padding: 18px 28px;
        display: flex;
        align-items: center;
        gap: 24px;
        overflow: hidden;
        margin-bottom: 6px;
    }}
    .monitor-strip svg {{ flex-shrink: 0; }}
    .ecg-trace {{
        stroke: #7FD9C4;
        stroke-width: 2.2;
        fill: none;
        stroke-dasharray: 6 4;
        animation: ecg-scan 2.2s linear infinite;
    }}
    @keyframes ecg-scan {{
        to {{ stroke-dashoffset: -400; }}
    }}
    .monitor-text h1 {{
        color: white;
        margin: 0;
        font-size: 1.7rem;
    }}
    .monitor-text p {{
        color: #C7D7D2;
        margin: 2px 0 0 0;
        font-size: 0.92rem;
    }}

    /* ---- Info / disclaimer badge ---- */
    .disclaimer {{
        background: #FBF3E6;
        border: 1px solid #E9D3A6;
        color: #6B4E1E;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.85rem;
        margin: 14px 0;
    }}

    /* ---- Fact cards row ---- */
    .fact-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 18px;
    }}
    .fact-card .label {{
        color: {MUTED};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .fact-card .value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
        color: {PRIMARY};
        margin-top: 2px;
    }}

    /* ---- Result card ---- */
    .result-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 26px 30px;
    }}
    .result-verdict {{
        font-family: 'Lora', serif;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 4px;
    }}
    .risk-readout {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.1rem;
        font-weight: 600;
    }}

    /* ---- Reference range bar (lab-report style) ---- */
    .range-bar-wrap {{ margin: 18px 0 6px 0; }}
    .range-bar {{
        position: relative;
        height: 14px;
        border-radius: 7px;
        background: linear-gradient(
            to right,
            {SAFE} 0%, {SAFE} 33%,
            {WARN} 33%, {WARN} 66%,
            {RISK} 66%, {RISK} 100%
        );
    }}
    .range-marker {{
        position: absolute;
        top: -9px;
        width: 0; height: 0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        border-top: 10px solid {INK};
        transform: translateX(-50%);
    }}
    .range-labels {{
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: {MUTED};
        letter-spacing: 0.05em;
        margin-top: 4px;
        text-transform: uppercase;
    }}

    /* ---- Contributing factor rows ---- */
    .factor-row {{ margin-bottom: 10px; }}
    .factor-name {{ font-size: 0.88rem; color: {INK}; margin-bottom: 3px; }}
    .factor-track {{
        background: {BORDER};
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
    }}
    .factor-fill {{
        height: 100%;
        border-radius: 4px;
        background: {PRIMARY};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
# HELPER: reference-range bar for the risk probability
# =========================================================================
def render_range_bar(probability: float) -> str:
    pct = round(probability * 100, 1)
    return f"""
    <div class="range-bar-wrap">
        <div class="range-bar">
            <div class="range-marker" style="left:{pct}%;"></div>
        </div>
        <div class="range-labels">
            <span>Low</span><span>Moderate</span><span>Elevated</span>
        </div>
    </div>
    """

# =========================================================================
# HELPER: ECG waveform SVG for the hero strip
# =========================================================================
ECG_PATH = (
    "M0,30 L40,30 L52,30 L58,10 L66,50 L72,18 L78,30 L110,30 "
    "L150,30 L162,30 L168,10 L176,50 L182,18 L188,30 L220,30 "
    "L260,30 L272,30 L278,10 L286,50 L292,18 L298,30 L330,30"
)
ECG_SVG = (
    f'<svg width="240" height="60" viewBox="0 0 330 60" '
    f'xmlns="http://www.w3.org/2000/svg">'
    f'<path class="ecg-trace" d="{ECG_PATH}"></path></svg>'
)

# =========================================================================
# SIDEBAR — CLINICAL INTAKE FORM
# =========================================================================
with st.sidebar:
    render_html(
        """
        <svg width="46" height="46" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 21s-7.5-4.6-10-9.3C.4 8.1 2.1 4.5 5.6 4.1c2-.2 3.9.8 5 2.4 1.1-1.6 3-2.6 5-2.4 3.5.4 5.2 4 3.6 7.6C19.5 16.4 12 21 12 21z"
                  stroke="#7FD9C4" stroke-width="1.6" fill="none"/>
            <path d="M3 12h4l2 5 3-10 2 6h5" stroke="#EAF0EE" stroke-width="1.4" fill="none"/>
        </svg>
        """
    )
st.sidebar.title("Patient Intake")
st.sidebar.caption("Fill in the fields below, then run the analysis.")
st.sidebar.markdown("---")

with st.sidebar.expander("🧍 Demographics", expanded=True):
    age = st.slider("Age", min_value=18, max_value=100, value=45)
    gender = st.selectbox("Gender", ["Female", "Male"])
    # NOTE: verify this 0/1 mapping matches how "Gender" was encoded in your
    # original training CSV — flip if your data used the opposite convention.
    gender_val = 1.0 if gender == "Male" else 0.0

with st.sidebar.expander("❤️ Presenting Symptoms", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        chest_pain = st.checkbox("Chest pain / pressure")
        shortness_of_breath = st.checkbox("Shortness of breath")
        fatigue = st.checkbox("Unusual fatigue")
        palpitations = st.checkbox("Palpitations")
    with c2:
        dizziness = st.checkbox("Dizziness / lightheaded")
        swelling = st.checkbox("Swelling (legs/ankles)")
        pain_arms_jaw_back = st.checkbox("Pain: arm / jaw / back")
        cold_sweats_nausea = st.checkbox("Cold sweats / nausea")

with st.sidebar.expander("⚠️ Risk Factors & History", expanded=True):
    c3, c4 = st.columns(2)
    with c3:
        high_bp = st.checkbox("High blood pressure")
        high_cholesterol = st.checkbox("High cholesterol")
        diabetes = st.checkbox("Diabetes")
        smoking = st.checkbox("Smoker (current/former)")
    with c4:
        obesity = st.checkbox("Obesity (BMI ≥ 30)")
        sedentary_lifestyle = st.checkbox("Sedentary lifestyle")
        family_history = st.checkbox("Family history of heart disease")
        chronic_stress = st.checkbox("Chronic stress")

st.sidebar.markdown("---")
analyze_button = st.sidebar.button(
    "Run Diagnostic Analysis", type="primary", use_container_width=True
)

# =========================================================================
# MAIN — HERO
# =========================================================================
render_html(
    f"""
    <div class="monitor-strip">
        {ECG_SVG}
        <div class="monitor-text">
            <h1>🫀 CardioRisk AI</h1>
            <p>Cardiovascular risk stratification from clinical intake data</p>
        </div>
    </div>
    <div class="disclaimer">
        ⚕️ <b>Educational prototype only.</b> This tool is not a certified
        diagnostic device and has not been clinically validated. It must never
        replace evaluation by a licensed clinician — if you or someone else may
        be having a heart attack, call emergency services immediately.
    </div>
    """
)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    render_html(
        f'<div class="fact-card"><div class="label">Model</div>'
        f'<div class="value">Random Forest</div></div>'
    )
with fc2:
    render_html(
        f'<div class="fact-card"><div class="label">Clinical Inputs</div>'
        f'<div class="value">{len(expected_features)} parameters</div></div>'
    )
with fc3:
    render_html(
        f'<div class="fact-card"><div class="label">Status</div>'
        f'<div class="value">Ready</div></div>'
    )

st.write("")

# =========================================================================
# INFERENCE ENGINE
# =========================================================================
if analyze_button:
    with st.spinner("Analyzing patient vitals..."):

        raw_inputs = {
            "Age": float(age),
            "Gender": gender_val,
            "Chest_Pain": 1.0 if chest_pain else 0.0,
            "Shortness_of_Breath": 1.0 if shortness_of_breath else 0.0,
            "Fatigue": 1.0 if fatigue else 0.0,
            "Palpitations": 1.0 if palpitations else 0.0,
            "Dizziness": 1.0 if dizziness else 0.0,
            "Swelling": 1.0 if swelling else 0.0,
            "Pain_Arms_Jaw_Back": 1.0 if pain_arms_jaw_back else 0.0,
            "Cold_Sweats_Nausea": 1.0 if cold_sweats_nausea else 0.0,
            "High_BP": 1.0 if high_bp else 0.0,
            "High_Cholesterol": 1.0 if high_cholesterol else 0.0,
            "Diabetes": 1.0 if diabetes else 0.0,
            "Smoking": 1.0 if smoking else 0.0,
            "Obesity": 1.0 if obesity else 0.0,
            "Sedentary_Lifestyle": 1.0 if sedentary_lifestyle else 0.0,
            "Family_History": 1.0 if family_history else 0.0,
            "Chronic_Stress": 1.0 if chronic_stress else 0.0,
        }

        # Baseline of zeros for any feature the model expects that we didn't
        # explicitly collect (keeps this robust if the .pkl is retrained with
        # a slightly different feature list).
        input_dict = {feature: 0.0 for feature in expected_features}
        input_dict.update(
            {k: v for k, v in raw_inputs.items() if k in input_dict}
        )

        input_df = pd.DataFrame([input_dict])[expected_features]
        input_df[["Age"]] = scaler.transform(input_df[["Age"]])

        prediction = model.predict(input_df)[0]
        probability = float(model.predict_proba(input_df)[0][1])

    st.subheader("Diagnostic Report")

    left, right = st.columns([1.3, 1])

    with left:
        verdict_color = RISK if prediction == 1.0 else SAFE
        verdict_text = (
            "Elevated Risk Pattern Detected"
            if prediction == 1.0
            else "Nominal Risk Pattern"
        )
        render_html(
            f"""
            <div class="result-card">
                <div class="result-verdict" style="color:{verdict_color};">
                    {verdict_text}
                </div>
                <div class="risk-readout" style="color:{verdict_color};">
                    {probability*100:.1f}%
                </div>
                <div style="color:{MUTED}; font-size:0.85rem;">
                    Modeled probability of a cardiovascular-disease pattern
                </div>
                {render_range_bar(probability)}
            </div>
            """
        )
        if prediction == 1.0:
            st.error(
                "The input pattern is consistent with elevated cardiovascular "
                "risk in this model. Recommend prompt clinical follow-up."
            )
        else:
            st.success(
                "The input pattern does not resemble the elevated-risk cases "
                "this model was trained on. Continued routine care is still "
                "recommended."
            )

    with right:
        st.markdown("**Top contributing factors**")
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            active = [
                (feat, imp)
                for feat, imp in zip(expected_features, importances)
                if input_dict.get(feat, 0.0) == 1.0 and feat not in ("Gender",)
            ]
            active.sort(key=lambda x: x[1], reverse=True)
            max_imp = max(importances) if len(importances) else 1.0

            if active:
                for feat, imp in active[:6]:
                    pct = min(100, round((imp / max_imp) * 100))
                    render_html(
                        f"""
                        <div class="factor-row">
                            <div class="factor-name">{feat.replace('_', ' ')}</div>
                            <div class="factor-track">
                                <div class="factor-fill" style="width:{pct}%;"></div>
                            </div>
                        </div>
                        """
                    )
            else:
                st.caption(
                    "No symptom or risk-factor flags were selected — the "
                    "result is driven mainly by age and gender."
                )
        else:
            st.caption("Feature importances aren't available for this model.")

    st.caption(
        "This report reflects patterns in the training data only and carries "
        "no clinical certification. Always confirm findings with a qualified "
        "healthcare professional."
    )

else:
    st.info(
        "👈 Enter patient details in the sidebar — demographics, presenting "
        "symptoms, and risk-factor history — then click **Run Diagnostic "
        "Analysis**."
    )