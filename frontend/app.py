import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="T20 Score Predictor 🏏",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Cricket Theme
st.markdown("""
<style>
    /* Main Background - Dark Navy */
    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Title Styling */
    .main-title {
        text-align: center;
        color: #FF6B35;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #E8E8E8;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 30px;
    }
    
    /* Cricket Ball Animation */
    .cricket-ball {
        font-size: 4rem;
        text-align: center;
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    /* Score Display */
    .score-display {
        text-align: center;
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 2px solid #FF6B35;
        box-shadow: 0 0 30px rgba(255,107,53,0.2);
    }
    
    .predicted-score {
        font-size: 4.5rem;
        font-weight: bold;
        color: #FF6B35;
        text-shadow: 0 0 20px rgba(255,107,53,0.4);
        margin: 10px 0;
    }
    
    .score-range {
        font-size: 1.4rem;
        color: #4ECDC4;
        margin-top: 10px;
    }
    
    /* Input Labels */
    .stNumberInput label, .stSlider label {
        color: #E8E8E8 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
    }
    
    /* Stadium Lights */
    .stadium-lights {
        text-align: center;
        font-size: 1.8rem;
        margin: 15px 0;
        letter-spacing: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(145deg, #FF6B35, #FF8C42);
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: bold;
        padding: 15px 40px;
        border-radius: 30px;
        border: none;
        box-shadow: 0 5px 25px rgba(255,107,53,0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 35px rgba(255,107,53,0.6);
        background: linear-gradient(145deg, #FF8C42, #FF6B35);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #4ECDC4 !important;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B8B8D1 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #95E879 !important;
    }
    
    /* Markdown Headers */
    h3 {
        color: #FF6B35 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6B6B8D;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #2a2a5a;
    }
    
    .footer p {
        margin: 5px 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #FF6B35 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2a2a5a !important;
    }
    
    /* Number Input */
    .stNumberInput input {
        background-color: #1e1e3f !important;
        color: #E8E8E8 !important;
        border: 1px solid #3a3a6a !important;
        border-radius: 8px !important;
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] {
        background: #2a2a5a !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with Cricket Theme
st.markdown('<div class="cricket-ball">🏏</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">T20 SCORE PREDICTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">⚡ Predict the Final Score Like a Pro! ⚡</p>', unsafe_allow_html=True)

# Stadium Lights Effect
st.markdown('<div class="stadium-lights">✦ ✦ ✦ ✦ ✦</div>', unsafe_allow_html=True)

# API URL
API_URL = "http://localhost:8000"

# Check API Status
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Input Section
st.markdown("---")
st.markdown("### 📊 Match Situation")

col1, col2 = st.columns(2)

with col1:
    current_score = st.number_input(
        "🏏 Current Score",
        min_value=0,
        max_value=400,
        value=85,
        step=1,
        help="Enter the current score of the batting team"
    )
    
    overs_completed = st.number_input(
        "⚾ Overs Completed",
        min_value=0.1,
        max_value=19.6,
        value=10.0,
        step=0.1,
        format="%.1f",
        help="Enter overs completed (e.g., 10.3 for 10 overs 3 balls)"
    )

with col2:
    wickets_left = st.slider(
        "🎯 Wickets Left",
        min_value=0,
        max_value=10,
        value=8,
        help="Select number of wickets remaining"
    )
    
    last_five_runs = st.number_input(
        "🔥 Last 5 Overs Runs",
        min_value=0,
        max_value=150,
        value=45,
        step=1,
        help="Runs scored in the last 5 overs (30 balls)"
    )

# Display Current Match Stats
st.markdown("---")
st.markdown("### 📈 Current Match Stats")

# Calculate derived stats for display
over_part = int(overs_completed)
ball_part = int(round((overs_completed - over_part) * 10))
if ball_part > 6:
    ball_part = 6
balls_bowled = (over_part * 6) + ball_part
balls_left = max(0, 120 - balls_bowled)
crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

with stat_col1:
    st.metric(label="🏏 Score", value=f"{current_score}")

with stat_col2:
    st.metric(label="⚾ Balls Left", value=f"{balls_left}")

with stat_col3:
    st.metric(label="🎯 Wickets", value=f"{wickets_left}")

with stat_col4:
    st.metric(label="📊 CRR", value=f"{crr:.2f}")

st.markdown("---")

# Predict Button
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button("🎯 PREDICT SCORE 🎯", use_container_width=True)

# Prediction Logic
if predict_button:
    # Validate overs format
    if ball_part > 6:
        st.error("❌ Invalid overs format! Ball number cannot exceed 6 (e.g., use 10.6 not 10.7)")
    else:
        # Check API
        if not check_api_status():
            st.error("⚠️ API is not running! Please start the FastAPI server first.")
            st.code("cd fast_api && uvicorn main:app --reload --port 8000", language="bash")
        else:
            # Make prediction request
            with st.spinner("🏏 Analyzing match situation..."):
                try:
                    payload = {
                        "current_score": current_score,
                        "overs": overs_completed,
                        "wickets_left": wickets_left,
                        "last_five_overs_runs": last_five_runs
                    }
                    
                    response = requests.post(
                        f"{API_URL}/predict",
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Success Animation
                        st.balloons()
                        
                        # Display Prediction
                        st.markdown("---")
                        st.markdown("### 🎉 PREDICTION RESULT 🎉")
                        
                        # Main Score Display
                        st.markdown(f"""
                        <div class="score-display">
                            <p style="color: #888; margin: 0; font-size: 1.2rem;">PREDICTED FINAL SCORE</p>
                            <p class="predicted-score">{result['predicted_score']}</p>
                            <p class="score-range">📊 Range: {result['score_range']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional Stats
                        res_col1, res_col2, res_col3 = st.columns(3)
                        
                        runs_to_add = result['predicted_score'] - current_score
                        overs_left = balls_left / 6
                        required_rate = runs_to_add / overs_left if overs_left > 0 else 0
                        
                        with res_col1:
                            st.metric(
                                label="🎯 Runs to Add",
                                value=f"{runs_to_add}",
                                delta=f"+{runs_to_add} runs"
                            )
                        
                        with res_col2:
                            st.metric(
                                label="⚾ Overs Left",
                                value=f"{overs_left:.1f}"
                            )
                        
                        with res_col3:
                            st.metric(
                                label="📈 Req. Rate",
                                value=f"{required_rate:.2f}",
                                delta=f"{required_rate - crr:+.2f}" if crr > 0 else None
                            )
                        
                        # Insight Message
                        st.markdown("---")
                        if required_rate > crr + 2:
                            st.warning("⚡ The batting team needs to accelerate!")
                        elif required_rate < crr:
                            st.success("🎯 Great momentum! Keep it going!")
                        else:
                            st.info("📊 Steady innings - maintain the run rate!")
                        
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Prediction failed: {error_detail}")
                        
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to API. Make sure the server is running.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🏏 T20 Score Predictor | Built with ❤️ using Machine Learning</p>
    <p>Powered by XGBoost & FastAPI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with Info
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Enter Current Score** - The runs scored so far
    2. **Enter Overs Completed** - Format: 10.3 means 10 overs 3 balls
    3. **Select Wickets Left** - Remaining wickets (0-10)
    4. **Enter Last 5 Overs Runs** - Runs in recent 30 balls
    5. **Click Predict** - Get the predicted final score!
    """)
    
    st.markdown("---")
    st.markdown("## 🔧 API Status")
    if check_api_status():
        st.success("✅ API is Online")
    else:
        st.error("❌ API is Offline")
        st.markdown("Start the API with:")
        st.code("uvicorn fast_api.main:app --reload", language="bash")
    
    st.markdown("---")
    st.markdown("## 📊 Model Info")
    st.markdown("""
    - **Algorithm**: XGBoost Regressor
    - **Features**: 5 input features
    - **Trained on**: International T20 matches
    """)
