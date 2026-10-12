"""
AI Programming Error Analyzer - Streamlit Web Application
Bi-LSTM + Word2Vec error classifier with a rule-based cause/fix knowledge base.
"""
import json
import os
import streamlit as st
import pandas as pd
from PIL import Image

from common import ART_DIR
from predict import ErrorAnalyzer
from knowledge_base import RULES

st.set_page_config(
    page_title="DeepTrace AI — Neural Error Classifier",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Modern Dark Glassmorphism UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main Layout Containers */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(16px);
        position: relative;
        overflow: hidden;
    }
    .header-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC, #F472B6);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 40%, #C084FC 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 0.96rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem 1rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        transform: translateY(-3px);
        box-shadow: 0 12px 30px -5px rgba(99, 102, 241, 0.25);
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }
    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        font-weight: 600;
    }
    
    /* Category Badges */
    .badge-syntax { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-runtime { background: rgba(239, 68, 68, 0.18); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-type { background: rgba(168, 85, 247, 0.18); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-dependency { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-database { background: rgba(56, 189, 248, 0.18); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-network { background: rgba(96, 165, 250, 0.18); color: #60A5FA; border: 1px solid rgba(96, 165, 250, 0.4); }
    .badge-api { background: rgba(244, 114, 182, 0.18); color: #F472B6; border: 1px solid rgba(244, 114, 182, 0.4); }
    .badge-unable { background: rgba(148, 163, 184, 0.18); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.4); }
    
    .cat-pill {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.02em;
    }
    
    /* Result Info Cards */
    .info-box {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.6) 100%);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-left: 5px solid #6366F1;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.4);
    }
    .cause-box { border-left-color: #F59E0B; }
    .fix-box { border-left-color: #10B981; }
    
    .box-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .cause-title { color: #FBBF24; }
    .fix-title { color: #34D399; }
    
    /* Code Pill */
    .token-pill {
        display: inline-block;
        background: rgba(51, 65, 85, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #F1F5F9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        padding: 4px 11px;
        border-radius: 8px;
        margin: 4px;
        transition: background 0.2s ease;
    }
    .token-pill:hover {
        background: rgba(99, 102, 241, 0.3);
        border-color: rgba(99, 102, 241, 0.5);
    }
    
    /* Custom Styling for Text Area */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.92rem !important;
        background-color: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B1120;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# Check model artifact existence
if not os.path.exists(os.path.join(ART_DIR, "bilstm.keras")):
    st.error("⚠️ Model artifacts not found. Please run `python train.py` first to generate the models.")
    st.stop()

# Cache analyzer instances per model
@st.cache_resource
def get_analyzer(model_file="bilstm.keras"):
    return ErrorAnalyzer(model_file=model_file)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric-header/100/code-fork.png", width=64)
    st.title("⚙️ Control Panel")
    st.markdown("---")
    
    st.subheader("🤖 Model Selection")
    model_choice = st.selectbox(
        "Classifier Architecture",
        ["Bi-LSTM (Recommended)", "LSTM", "CNN"],
        index=0,
        help="Bi-LSTM offers bidirectional contextual reading, optimal for error tracebacks."
    )
    
    model_filename_map = {
        "Bi-LSTM (Recommended)": "bilstm.keras",
        "LSTM": "lstm.keras",
        "CNN": "cnn.keras"
    }
    selected_model_file = model_filename_map[model_choice]
    
    st.markdown("---")
    st.subheader("🎯 Sensitivity Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.70,
        step=0.05,
        format="%.0f%%",
        help="Predictions below this confidence threshold will display 'Unable to Predict' and be marked as Unreliable."
    )
    
    st.markdown("---")
    st.subheader("📊 System Specs")
    st.caption("• **Vocab Size**: ~300+ tokens")
    st.caption("• **Vector Space**: Word2Vec 100D")
    st.caption("• **Sequence Length**: Max 40 tokens")
    st.caption("• **Supported Classes**: Syntax, Runtime, Type, Dependency, Database, Network, API")


# --- HEADER ---
col_head1, col_head2 = st.columns([4, 1])
with col_head1:
    st.markdown("""
    <div class="header-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1 class="header-title">🛠️ DeepTrace AI</h1>
            <div class="status-badge"><span class="status-dot"></span> MODEL ACTIVE</div>
        </div>
        <p class="header-subtitle">Deep Neural Error Classifier (Bi-LSTM / Word2Vec) & Contextual Root Cause Remediation Engine</p>
    </div>
    """, unsafe_allow_html=True)

# Main Navigation Tabs
tab_analyzer, tab_benchmarks, tab_rules = st.tabs([
    "🔍 Diagnostic Studio",
    "📊 Model Performance & Benchmarks",
    "📚 Knowledge Base Rules"
])

# Sample presets organized by language/category
PRESETS = {
    "🗄️ DB Connection Failure": "psycopg2.OperationalError: could not connect to server: Connection refused (0x0000274D/10061)",
    "🗄️ DB SQLite No Table": "sqlite3.OperationalError: no such table: users",
    "🌐 Network Connection Timeout": "requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.example.com', port=443): Max retries exceeded",
    "🌐 Network CORS Blocked": "Access to fetch at 'https://api.domain.com/data' from origin 'http://localhost:3000' has been blocked by CORS policy",
    "🔑 API 401 Unauthorized": "requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://api.openai.com/v1/chat/completions",
    "🔑 API 429 Rate Limit": "AxiosError: Request failed with status code 429 (Too Many Requests)",
    "🐍 Python ModuleNotFoundError": "ModuleNotFoundError: No module named 'pandas'",
    "☕ Java NullPointerException": 'java.lang.NullPointerException: Cannot invoke "String.length()" because "name" is null',
}

# --- TAB 1: DIAGNOSTIC STUDIO ---
with tab_analyzer:
    st.markdown("### 📝 Input Error Log")
    
    # Preset Selector buttons
    st.caption("Quick Load Example Error Traces:")
    preset_cols = st.columns(4)
    selected_preset = None
    
    preset_keys = list(PRESETS.keys())
    for idx, key in enumerate(preset_keys):
        col_idx = idx % 4
        if preset_cols[col_idx].button(key, key=f"btn_{idx}", width="stretch"):
            selected_preset = PRESETS[key]

    default_text = selected_preset if selected_preset else PRESETS["🗄️ DB Connection Failure"]

    error_input = st.text_area(
        "Paste error log, traceback, or compiler exception below:",
        value=default_text,
        height=130,
        placeholder="e.g. java.lang.NullPointerException or ModuleNotFoundError..."
    )

    analyze_btn = st.button("🚀 Analyze Error Trace", type="primary", width="stretch")

    if analyze_btn or selected_preset:
        clean_input = error_input.strip()
        if not clean_input:
            st.warning("Please enter a valid error message.")
        else:
            with st.spinner("Analyzing traceback with neural pipeline..."):
                analyzer = get_analyzer(selected_model_file)
                res = analyzer.analyze(clean_input, conf_threshold=conf_threshold)
            
            if "error" in res:
                st.warning(f"⚠️ {res['error']}")
            else:
                is_uncertain = res["uncertain"]
                
                st.markdown("---")
                st.markdown("### 📊 Diagnostic Results")
                
                # Prominent alert if confidence is below threshold
                if is_uncertain:
                    st.warning("⚠️ **Unable to Predict — Please provide a more detailed error log.**")
                
                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                if is_uncertain:
                    display_cat = "Unable to Predict"
                    badge_class = "badge-unable"
                else:
                    display_cat = res["error_type"]
                    badge_class = f"badge-{display_cat.lower()}"
                
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Category</div>
                        <div class="metric-value"><span class="cat-pill {badge_class}">{display_cat}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col2:
                    conf_pct = res['confidence'] * 100
                    conf_color = "#34D399" if conf_pct >= conf_threshold * 100 else ("#FBBF24" if conf_pct >= 50 else "#F87171")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Prediction Confidence</div>
                        <div class="metric-value" style="color: {conf_color};">{conf_pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col3:
                    match_status = "Direct Rule Match" if res["kb_match"] else "No Rule Match"
                    match_color = "#818CF8" if res["kb_match"] else "#94A3B8"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Knowledge Base Status</div>
                        <div class="metric-value" style="color: {match_color}; font-size: 1.1rem; margin-top: 0.5rem;">{match_status}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with m_col4:
                    status_text = f"Reliable (≥ {conf_threshold:.0%})" if not is_uncertain else f"Unreliable (< {conf_threshold:.0%})"
                    status_clr = "#34D399" if not is_uncertain else "#F87171"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Reliability Assessment</div>
                        <div class="metric-value" style="color: {status_clr}; font-size: 1.1rem; margin-top: 0.5rem;">{status_text}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Separate model debugging display when prediction is unreliable
                if is_uncertain:
                    with st.expander("🛠️ Debugging: Model Raw Prediction (Unreliable)", expanded=False):
                        st.info(f"**Raw Model Prediction**: `{res['raw_error_type']}`  \n**Model Confidence**: `{res['confidence']:.1%}` (Required threshold: `{conf_threshold:.0%}`)  \n*Note: This raw prediction was suppressed from the final diagnosis because confidence is below the {conf_threshold:.0%} threshold.*")

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Cause & Fix Row
                c_col1, c_col2 = st.columns(2)
                
                with c_col1:
                    st.markdown(f"""
                    <div class="info-box cause-box">
                        <div class="box-title cause-title">🔍 Identified Root Cause</div>
                        <div style="font-size: 1.05rem; color: #F1F5F9; line-height: 1.5;">{res['cause']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with c_col2:
                    st.markdown(f"""
                    <div class="info-box fix-box">
                        <div class="box-title fix-title">🛠️ Recommended Remediation</div>
                        <div style="font-size: 1.05rem; color: #F1F5F9; line-height: 1.5;">{res['fix']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probabilities & Tokens Layout
                p_col1, p_col2 = st.columns([3, 2])
                
                with p_col1:
                    st.markdown("#### 📈 Probability Distribution Across Classes")
                    prob_df = pd.DataFrame({
                        "Category": list(res["probabilities"].keys()),
                        "Probability (%)": [v * 100 for v in res["probabilities"].values()]
                    }).set_index("Category")
                    
                    st.bar_chart(prob_df, color="#6366F1")
                    
                with p_col2:
                    st.markdown("#### 🔤 NLP Preprocessing Pipeline")
                    st.caption("Tokens after CamelCase split, normalization, stop-word filtering & Porter Stemming:")
                    
                    token_html = "".join([f'<span class="token-pill">{t}</span>' for t in res["tokens"]])
                    st.markdown(f'<div style="background: rgba(15, 23, 42, 0.85); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); min-height: 140px;">{token_html}</div>', unsafe_allow_html=True)


# --- TAB 2: MODEL PERFORMANCE & BENCHMARKS ---
with tab_benchmarks:
    st.markdown("### 📊 Model Evaluation & Metrics")
    st.caption("Comparison between Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Bidirectional LSTM (Bi-LSTM)")
    
    results_path = os.path.join(ART_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            benchmarks = json.load(f)
        
        bench_df = pd.DataFrame(benchmarks).T
        bench_df.columns = [c.upper() for c in bench_df.columns]
        
        col_b1, col_b2 = st.columns([2, 3])
        with col_b1:
            st.markdown("#### 📋 Performance Table")
            st.dataframe(bench_df.style.format("{:.2%}"), width="stretch")
            st.info("💡 Bi-LSTM handles sequential bi-directional context best, preserving word order semantics in complex tracebacks.")
            
        with col_b2:
            st.markdown("#### 📈 Accuracy & F1 Comparison")
            st.bar_chart(bench_df[["ACCURACY", "F1"]])
            
    st.markdown("---")
    st.markdown("### 🖼️ Diagnostic Plots")
    
    img_cols = st.columns(2)
    
    cm_path = os.path.join(ART_DIR, "confusion_matrix.png")
    tc_path = os.path.join(ART_DIR, "training_curves.png")
    mc_path = os.path.join(ART_DIR, "model_comparison.png")
    roc_path = os.path.join(ART_DIR, "roc_curves.png")
    
    if os.path.exists(cm_path):
        with img_cols[0]:
            st.markdown("##### Confusion Matrix")
            st.image(Image.open(cm_path), width="stretch")
            
    if os.path.exists(tc_path):
        with img_cols[1]:
            st.markdown("##### Training & Validation Curves")
            st.image(Image.open(tc_path), width="stretch")

    img_cols2 = st.columns(2)
    if os.path.exists(mc_path):
        with img_cols2[0]:
            st.markdown("##### Model Architecture Comparison")
            st.image(Image.open(mc_path), width="stretch")
            
    if os.path.exists(roc_path):
        with img_cols2[1]:
            st.markdown("##### ROC / AUC Curves")
            st.image(Image.open(roc_path), width="stretch")


# --- TAB 3: KNOWLEDGE BASE RULES ---
with tab_rules:
    st.markdown("### 📚 Pattern Matching Knowledge Base Rules")
    st.caption("Regex patterns evaluated when an error is classified to provide precise, contextual fixes.")
    
    search_query = st.text_input("🔍 Search rules by pattern or keyword:", "")
    
    rule_data = []
    for pat, cause, fix in RULES:
        if search_query.lower() in pat.lower() or search_query.lower() in cause.lower() or search_query.lower() in fix.lower():
            rule_data.append({
                "Regex Pattern": pat,
                "Identified Cause": cause,
                "Suggested Remediation": fix
            })
            
    rule_df = pd.DataFrame(rule_data)
    st.dataframe(rule_df, width="stretch", height=500)
    st.caption(f"Showing {len(rule_df)} out of {len(RULES)} total regex rules in the knowledge base.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.85rem; padding: 1rem 0;">
    DeepTrace AI • Built with Streamlit, TensorFlow (Bi-LSTM), Word2Vec & NLTK
</div>
""", unsafe_allow_html=True)
