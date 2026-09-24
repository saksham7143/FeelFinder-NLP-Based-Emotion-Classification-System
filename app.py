"""
FeelFinder - AI-Powered NLP Emotion Detection Web Application
Built with Streamlit & Scikit-Learn
"""

import streamlit as st
import time
from emotion_model import (
    check_model_files,
    load_model_artifacts,
    predict_emotion,
    EMOTION_CONFIG
)

# Page configuration
st.set_page_config(
    page_title="FeelFinder - AI Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container tweaks */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1.5rem 1.5rem 1.5rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }
    
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    
    .tech-badge {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #CBD5E1;
        padding: 0.35rem 0.85rem;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    
    /* Card Styles */
    .custom-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Result Primary Card */
    .result-card {
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        position: relative;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        transition: transform 0.3s ease;
    }
    
    .emotion-emoji {
        font-size: 4.5rem;
        line-height: 1;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.3));
        animation: pulseEmoji 2s infinite ease-in-out;
    }
    
    @keyframes pulseEmoji {
        0% { transform: scale(1); }
        50% { transform: scale(1.06); }
        100% { transform: scale(1); }
    }
    
    .emotion-label {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    
    .confidence-badge {
        display: inline-block;
        font-size: 1.25rem;
        font-weight: 600;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        background: rgba(0, 0, 0, 0.35);
        color: #FFFFFF;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 0.5rem;
    }

    .explanation-text {
        color: #E2E8F0;
        font-size: 1.05rem;
        margin-top: 1rem;
        background: rgba(0, 0, 0, 0.2);
        padding: 0.75rem 1rem;
        border-radius: 12px;
    }
    
    /* Quick Examples Chips */
    .quick-title {
        font-size: 0.9rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    /* Custom Progress Bar Styling */
    .prob-row {
        margin-bottom: 0.85rem;
    }
    
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.95rem;
        font-weight: 500;
        margin-bottom: 0.3rem;
        color: #E2E8F0;
    }
    
    .prob-track {
        height: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        overflow: hidden;
    }
    
    .prob-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease-in-out;
    }
    
    /* Footer */
    .footer-text {
        text-align: center;
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model_and_vectorizer():
    """Cached loader for model artifacts."""
    return load_model_artifacts()


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Architecture")
        st.markdown("""
        **FeelFinder** relies on a pre-trained scikit-learn Machine Learning pipeline.
        
        - **Algorithm**: Logistic Regression
        - **Vectorization**: CountVectorizer (Bag-of-Words)
        - **Test Accuracy**: `~89%`
        - **Classes**: 6 Emotion Categories
        """)
        
        st.divider()
        st.markdown("### 🏷️ Supported Emotions")
        for code, info in EMOTION_CONFIG.items():
            st.markdown(f"**{info['emoji']} {info['name']}** `(Label {code})`  \n*{info['description']}*")
        
        st.divider()
        st.markdown("### 🛡️ System & Security")
        files_ok, missing = check_model_files()
        if files_ok:
            st.success("✓ Model artifacts verified & loaded", icon="✅")
        else:
            st.error(f"⚠️ Missing files: {', '.join(missing)}", icon="❌")
            
        st.caption("FeelFinder NLP Emotion System v1.0")

    # Header / Hero
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">FeelFinder</div>
        <div class="hero-subtitle">AI-Powered Text Emotion Detection</div>
        <div class="badge-container">
            <span class="tech-badge">⚡ Natural Language Processing</span>
            <span class="tech-badge">🎯 Logistic Regression</span>
            <span class="tech-badge">📈 89% Accuracy</span>
            <span class="tech-badge">🔤 CountVectorizer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check model files before displaying interactive app
    files_ok, missing_files = check_model_files()
    if not files_ok:
        st.error(f"🚨 **Model files missing!**  \nThe application requires the following files in the project folder: `{', '.join(missing_files)}`", icon="⚠️")
        st.info("Please ensure `LogisticRegression_model.pkl` and `count_vectorizer.pkl` are located in the working directory.")
        return

    # Load model & vectorizer
    try:
        model, vectorizer = get_model_and_vectorizer()
    except Exception as e:
        st.error(f"❌ Error initializing machine learning engine: {str(e)}")
        return

    # Session State for Example Inputs
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    # Quick Example Buttons
    st.markdown("<div class='quick-title'>💡 Click an example sentence to quickly test:</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("😊 Dream Job", use_container_width=True, help="Test Joy prediction"):
            st.session_state.input_text = "I got my dream job today!"
    with col2:
        if st.button("😨 Scared Tomorrow", use_container_width=True, help="Test Fear prediction"):
            st.session_state.input_text = "I am really scared about tomorrow."
    with col3:
        if st.button("😠 So Angry", use_container_width=True, help="Test Anger prediction"):
            st.session_state.input_text = "I am so angry right now."
    with col4:
        if st.button("❤️ Love Family", use_container_width=True, help="Test Love prediction"):
            st.session_state.input_text = "I love spending time with my family."

    # Secondary row of examples
    col5, col6 = st.columns(2)
    with col5:
        if st.button("😢 Miserable & Lonely", use_container_width=True, help="Test Sadness prediction"):
            st.session_state.input_text = "I feel so lonely and miserable today."
    with col6:
        if st.button("😲 Wow Unbelievable", use_container_width=True, help="Test Surprise prediction"):
            st.session_state.input_text = "I can't believe what just happened, wow!"

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form / Text Area
    with st.form(key="emotion_form"):
        user_text = st.text_area(
            label="Enter text or paragraph for emotion analysis:",
            value=st.session_state.input_text,
            placeholder="Enter your text here...",
            height=140,
            key="user_text_input"
        )
        
        analyze_submitted = st.form_submit_button("🔍 Analyze Emotion", type="primary", use_container_width=True)

    # Execution flow
    if analyze_submitted:
        if not user_text or not user_text.strip():
            st.warning("⚠️ Please enter some text before analyzing.", icon="⚠️")
        else:
            with st.spinner("Analyzing text with CountVectorizer & Logistic Regression..."):
                time.sleep(0.15)  # Smooth transition effect
                result = predict_emotion(user_text, model, vectorizer)

            if not result.get("success"):
                st.error(f"❌ {result.get('error', 'Prediction failed.')}")
            else:
                # Main Results Section
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🎯 Detection Results")

                res_col1, res_col2 = st.columns([1.1, 1.2])

                with res_col1:
                    # Primary Card with Dynamic Emotion Color Accent
                    st.markdown(f"""
                    <div class="result-card" style="background: {result['gradient']}; border: 1px solid {result['border_color']};">
                        <div class="emotion-emoji">{result['emoji']}</div>
                        <div style="font-size: 0.85rem; letter-spacing: 0.1em; color: rgba(255,255,255,0.7); text-transform: uppercase; font-weight: 600;">Detected Emotion</div>
                        <div class="emotion-label">{result['emotion']}</div>
                        <div class="confidence-badge">Confidence: {result['confidence']}%</div>
                        <div class="explanation-text">{result['explanation']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with res_col2:
                    # Probability Distribution Breakdown Card
                    st.markdown("""
                    <div class="custom-card">
                        <div style="font-size: 1.15rem; font-weight: 600; color: #F8FAFC; margin-bottom: 1rem;">
                            📊 Probability Breakdown Across Emotions
                        </div>
                    """, unsafe_allow_html=True)

                    # Sorted probabilities
                    probs = result["probabilities"]
                    # Sort by probability descending
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1]["probability"], reverse=True)

                    for emotion_name, prob_data in sorted_probs:
                        pct = prob_data["percentage"]
                        emoji = prob_data["emoji"]
                        color = prob_data["color"]
                        
                        st.markdown(f"""
                        <div class="prob-row">
                            <div class="prob-header">
                                <span>{emoji} {emotion_name}</span>
                                <span>{pct}%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill" style="width: {pct}%; background-color: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-text">
        FeelFinder Machine Learning Portfolio Application • Built with Python, Streamlit & Scikit-Learn • ~89% Test Accuracy
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
