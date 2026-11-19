import os
import sys
import streamlit as st

# Ensure local modules are importable
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from predict import load_model, predict_message, validate_input

# ─── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detection System",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Model loading (cached) ──────────────────────────────────────────────────
MODEL_PATH = os.path.join(APP_DIR, 'models', 'spam_detection_model.pkl')


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


pipeline = get_model()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
    .subtitle   { font-size: 1.05rem; color: #555; margin-bottom: 1.5rem; }
    .result-box { padding: 1.2rem; border-radius: 10px; margin-top: 1rem; }
    .spam-result { background: #fff0f0; border-left: 5px solid #e74c3c; }
    .ham-result  { background: #f0fff4; border-left: 5px solid #27ae60; }
    .prob-note   { font-size: 0.82rem; color: #888; font-style: italic; margin-top: 0.4rem; }
    footer { text-align: center; color: #999; font-size: 0.78rem; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📧 About")
    st.info(
        "This application uses a machine learning model trained on a "
        "real SMS/email spam dataset to classify messages as **SPAM** or **HAM (Not Spam)**."
    )
    st.markdown("### 🤖 Model Info")
    if pipeline is not None:
        model_name = type(pipeline.named_steps.get(
            'clf', pipeline.steps[-1][1]
        )).__name__
        st.success(f"Model loaded: **{model_name}**")
    else:
        st.error("Model not found. Run `python train_model.py` first.")

    st.markdown("### 📋 How to Use")
    st.markdown("""
1. Paste your email or message in the text box.
2. Click **🔍 Detect Spam**.
3. View the prediction result.
4. Use **🗑️ Clear** to reset.
    """)
    st.markdown("### ⚠️ Disclaimer")
    st.caption(
        "ML models are not 100% accurate. Always use your own judgement."
    )

# ─── Session state ────────────────────────────────────────────────────────────
if 'message_input' not in st.session_state:
    st.session_state['message_input'] = ''
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'clear_flag' not in st.session_state:
    st.session_state['clear_flag'] = False

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📧 Email Spam Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste any email or message below and let the AI decide — '
    'Spam or Not Spam.</p>',
    unsafe_allow_html=True
)
st.markdown("---")

# ─── Input area ───────────────────────────────────────────────────────────────
if st.session_state['clear_flag']:
    st.session_state['message_input'] = ''
    st.session_state['result'] = None
    st.session_state['clear_flag'] = False

user_input = st.text_area(
    label="📨 Enter your message:",
    value=st.session_state['message_input'],
    height=200,
    placeholder="Paste your email or message here to check whether it is spam...",
    key='message_input',
)

# ─── Buttons ─────────────────────────────────────────────────────────────────
col1, col2, _ = st.columns([2, 2, 3])
with col1:
    detect_clicked = st.button("🔍 Detect Spam", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state['clear_flag'] = True
    st.rerun()

# ─── Prediction logic ─────────────────────────────────────────────────────────
if detect_clicked:
    if pipeline is None:
        st.error(
            "🚫 **Model not found.**\n\n"
            "Please train the model first by running:\n\n"
            "```\npython train_model.py\n```"
        )
    elif not validate_input(user_input):
        st.warning("⚠️ Please enter a message to analyse.")
    else:
        with st.spinner("Analysing your message..."):
            result = predict_message(pipeline, user_input)
        st.session_state['result'] = result

# ─── Result display ───────────────────────────────────────────────────────────
result = st.session_state.get('result')
if result is not None:
    st.markdown("---")
    label = result['label']
    probability = result['probability']
    confidence = result['confidence']

    if label == 'SPAM':
        st.error(
            "🚨 **SPAM DETECTED**\n\n"
            "This message appears to be **spam**. Be cautious — "
            "do not click links or share personal information."
        )
    else:
        st.success(
            "✅ **NOT SPAM (HAM)**\n\n"
            "This message appears to be **legitimate**. "
            "It does not show typical spam characteristics."
        )

    if probability is not None:
        spam_pct = probability * 100
        st.markdown(f"**Spam Probability:** {spam_pct:.1f}%")
        st.progress(min(probability, 1.0))
        st.caption(
            f"Confidence level: **{confidence}** — "
            "This is a model estimate, not a guarantee."
        )
    else:
        st.info(
            "Spam probability is not available for this model "
            "(LinearSVC does not natively output probabilities). "
            "The prediction is still reliable."
        )

    with st.expander("📊 What does this mean?"):
        if label == 'SPAM':
            st.markdown("""
            The model identified patterns in your message that are commonly found in spam emails:
            - Unusual promotional language or urgency cues
            - Suspicious links or phone numbers
            - Requests for personal information

            **Recommendation:** Do not reply, click links, or provide any personal data.
            """)
        else:
            st.markdown("""
            The model did not find significant spam patterns in your message.
            It resembles normal, legitimate communication.

            **Note:** No ML model is 100% accurate. If a message looks suspicious,
            trust your instincts regardless of this result.
            """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<footer>Built with Streamlit & Scikit-learn · '
    'Machine learning predictions may be incorrect · '
    'For educational purposes only</footer>',
    unsafe_allow_html=True
)
