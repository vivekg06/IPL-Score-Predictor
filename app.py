import streamlit as st
import numpy as np
from keras.models import load_model
import pickle
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="IPL Score Predictor Pro",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS (Modern Dark Theme & Glassmorphism) ---
st.markdown(
    """
<style>
    /* Global Font & Reset */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        font-size: 18px;
        letter-spacing: 0.3px;
    }
    
    /* Liquid Background */
    .stApp {
        background-color: #050505;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
        background-attachment: fixed;
        color: #E0E0E0;
    }
    
    /* Liquid Glass Container */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08); /* Subtle border */
        border-radius: 24px;
        padding: 40px; /* More negative space */
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15); /* Highlight on hover */
    }

    /* Inputs - Minimalist */
    .stTextInput > div > div, .stNumberInput > div > div, .stSelectbox > div > div {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 0 !important;
        color: white !important;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div:focus-within, .stNumberInput > div > div:focus-within, .stSelectbox > div > div:focus-within {
        border-bottom-color: #6C63FF !important; /* Accent color */
        box-shadow: none !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] span {
        color: white !important;
        font-weight: 400;
        font-size: 1.1rem;
    }
    
    label {
        color: #A0A0A0 !important;
        font-weight: 300 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    .header-title {
        font-size: 4rem;
        background: linear-gradient(135deg, #FFFFFF 0%, #A5A5A5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #888;
        font-weight: 300;
        text-align: center;
        margin-bottom: 60px; /* Space out header */
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        color: #888;
        font-weight: 400;
        padding-bottom: 15px;
    }

    .stTabs [aria-selected="true"] {
        color: #fff !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #fff;
    }

    /* Prediction Box - Hero Element */
    .prediction-box {
        background: radial-gradient(circle at top left, rgba(108, 99, 255, 0.15), transparent 70%);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 30px;
        padding: 3rem;
        text-align: center;
        margin-top: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .prediction-label {
        font-size: 1rem;
        color: #A0A0A0;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }

    .prediction-score {
        font-size: 5rem;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 0 30px rgba(108, 99, 255, 0.5); /* Glowing text */
        line-height: 1;
    }

    /* Button - Liquid Gradient */
    .stButton > button {
        background: linear-gradient(90deg, #2d2d2d, #1a1a1a) !important; /* Dark base */
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 50px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        position: relative;
        overflow: hidden;
    }
    
    /* "Liquid" hover effect */
    .stButton > button::before {
        content: "";
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #6C63FF, #4E44CE) !important; /* Accent on hover */
        border-color: #6C63FF !important;
        box-shadow: 0 10px 30px rgba(108, 99, 255, 0.3) !important;
        transform: scale(1.02);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 0 !important;
    }
    div[data-testid="stExpander"] summary:hover {
        color: #6C63FF !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #050505;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- Cached Model Loading for Performance ---
@st.cache_resource
def load_prediction_model():
    """Load the trained model with caching for faster subsequent loads."""
    return load_model("ipl_score_model.keras")


@st.cache_resource
def load_encoders():
    """Load label encoders with caching."""
    with open("team_encoder.pkl", "rb") as f:
        team_enc = pickle.load(f)
    with open("venue_encoder.pkl", "rb") as f:
        venue_enc = pickle.load(f)
    return team_enc, venue_enc


# Load model and encoders
model = load_prediction_model()
team_encoder, venue_encoder = load_encoders()

# --- Active IPL Teams (2024-2026) ---
# Mapping old team names to current names for display
TEAM_NAME_MAPPING = {
    "Chennai Super Kings": "Chennai Super Kings",
    "Mumbai Indians": "Mumbai Indians",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Rajasthan Royals": "Rajasthan Royals",
    "Delhi Daredevils": "Delhi Capitals",  # Renamed in 2019
    "Kings XI Punjab": "Punjab Kings",  # Renamed in 2021
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
}

# New teams (not in old data - will use closest match for encoding)
NEW_TEAMS = ["Gujarat Titans", "Lucknow Super Giants"]

# Active teams for display
ACTIVE_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Rajasthan Royals",
    "Delhi Capitals",
    "Punjab Kings",
    "Sunrisers Hyderabad",
    "Gujarat Titans",
    "Lucknow Super Giants",
]

# Team Icons/Logos - Official IPL Sources (Stable)
TEAM_ICONS = {
    "Chennai Super Kings": "https://scores.iplt20.com/ipl/teamlogos/CSK.png",
    "Mumbai Indians": "https://scores.iplt20.com/ipl/teamlogos/MI.png",
    "Royal Challengers Bangalore": "https://scores.iplt20.com/ipl/teamlogos/RCB.png",
    "Kolkata Knight Riders": "https://scores.iplt20.com/ipl/teamlogos/KKR.png",
    "Rajasthan Royals": "https://scores.iplt20.com/ipl/teamlogos/RR.png",
    "Delhi Capitals": "https://scores.iplt20.com/ipl/teamlogos/DC.png",
    "Punjab Kings": "https://scores.iplt20.com/ipl/teamlogos/PBKS.png",
    "Sunrisers Hyderabad": "https://scores.iplt20.com/ipl/teamlogos/SRH.png",
    "Gujarat Titans": "https://scores.iplt20.com/ipl/teamlogos/GT.png",
    "Lucknow Super Giants": "https://scores.iplt20.com/ipl/teamlogos/LSG.png",
}

ACTIVE_VENUES = [
    "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk",
    "Wankhede Stadium",
    "Eden Gardens",
    "Narendra Modi Stadium",
    "Arun Jaitley Stadium",
    "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium, Uppal",
    "Punjab Cricket Association IS Bindra Stadium",
]

# Team Name Mappings (Old -> New or Standard)
TEAM_NAME_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Deccan Chargers": "Sunrisers Hyderabad",  # Franchise change
    "Gujarat Lions": "Gujarat Titans",  # Spiritual successor? maybe not strictly but similar region
    "Pune Warriors": "Rising Pune Supergiant",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",  # Normalize
}

# Display Name -> Encoder Name Mapping
# Since our model is trained on diverse names (some old, some new), we need to handle this carefully.
# However, ideally we should have normalized names in data preprocessing.
# Given we didn't, we might need to map 'Display' to 'Encoder' name if they differ.
# But for now, we assume standard names match encoder labels for active teams.
# Old names not in ACTIVE_TEAMS won't be selectable but might be in encoder.

DISPLAY_TO_ENCODER = {
    # If the user selects "Delhi Capitals", we pass "Delhi Capitals" (which is in encoder).
    # If the user selects "Punjab Kings", we pass "Punjab Kings".
    # We remove the hacky mappings.
}

# Venue Mapping
VENUE_TO_ENCODER = {
    # Standardize venue names if needed.
    # The encoder uses names from CSV.
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chepauk",
    "Wankhede Stadium": "Wankhede Stadium",
    "Eden Gardens": "Eden Gardens",
    "Narendra Modi Stadium": "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",  # Verify exact string in CSV
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium, Uppal",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium, Mohali",
}

# Use active teams and venues for UI
teams = ACTIVE_TEAMS
venues = ACTIVE_VENUES

import scipy.stats as stats


# --- Helper Functions ---
def calculate_win_probability(predicted_score, target_score, runs_std_dev=20):
    """
    Calculate win probability for the chasing team.
    Assumes the prediction error follows a normal distribution.
    runs_std_dev: Standard deviation of the model's prediction errors (approx 20 runs)
    """
    # Probability that the chasing team scores > target
    # This is 1 - CDF(target) on the distribution N(predicted, std_dev)
    win_prob = 1 - stats.norm.cdf(target_score, loc=predicted_score, scale=runs_std_dev)
    return win_prob


# --- Initialize Session State ---
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --- Header ---
st.markdown(
    '<h1 class="main-header">🏏 IPL Score Predictor</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="sub-header">AI-Powered Match Score Prediction</p>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header Section
# ------------------------------------------------------------------
st.markdown(
    '<div class="header-title">🏏 IPL Score Predictor Pro</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="header-subtitle">Advanced AI Analytics & Match Simulation</div>',
    unsafe_allow_html=True,
)

# --- Tabs for Modes ---
tab1, tab2 = st.tabs(["🚀 1st Innings Score", "🎯 2nd Innings Analysis"])

# ==========================================
# TAB 1: 1st Innings Score Prediction
# ==========================================
with tab1:
    # Row 1: Team Selection & Venue
    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 0.2, 1])

        with col1:
            st.markdown("### 🏏 Batting Team")
            batting_team = st.selectbox(
                "Select Team", teams, key="batting_t1", label_visibility="collapsed"
            )
            if batting_team in TEAM_ICONS:
                st.image(TEAM_ICONS[batting_team], width=100)

        with col2:
            st.markdown(
                "<h2 style='text-align: center; padding-top: 50px;'>VS</h2>",
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("### 🎯 Bowling Team")
            bowling_team = st.selectbox(
                "Select Team",
                teams,
                key="bowling_t1",
                index=1,
                label_visibility="collapsed",
            )
            if bowling_team in TEAM_ICONS:
                st.image(TEAM_ICONS[bowling_team], width=100)

        st.markdown("---")
        venue = st.selectbox("🏟️ Select Venue", venues, key="venue_t1")
        st.markdown("</div>", unsafe_allow_html=True)

    # Validation
    if batting_team == bowling_team:
        st.error("⚠️ Teams must be different!")

    # Row 2: Match Stats (Glass Container)
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Live Match Stats")

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        runs = st.number_input(
            "Runs", min_value=0, max_value=400, value=0, key="runs_t1"
        )
    with col_s2:
        wickets = st.number_input(
            "Wickets", min_value=0, max_value=9, value=0, key="wickets_t1"
        )
    with col_s3:
        completed_overs = st.number_input(
            "Overs Done", min_value=0, max_value=19, value=0, key="overs_t1"
        )
    with col_s4:
        balls = st.number_input(
            "Balls (Current)", min_value=0, max_value=5, value=0, key="balls_t1"
        )

    overs = completed_overs + balls / 6.0

    # Last 5 Overs
    if overs >= 5:
        st.markdown("#### Powerplay / Recent Form")
        col_l5_1, col_l5_2 = st.columns(2)
        with col_l5_1:
            runs_last_5 = st.number_input(
                "Runs (Last 5 Ov)",
                min_value=0,
                max_value=100,
                value=0,
                key="runs_l5_t1",
            )
        with col_l5_2:
            wickets_last_5 = st.number_input(
                "Wickets (Last 5 Ov)",
                min_value=0,
                max_value=10,
                value=0,
                key="wickets_l5_t1",
            )
    else:
        runs_last_5, wickets_last_5 = 0, 0
        st.info("Wait for 5 overs to unlock advanced metrics")

    st.markdown("</div>", unsafe_allow_html=True)

    # Stats Summary
    col_m1, col_m2, col_m3 = st.columns(3)
    crr = runs / overs if overs > 0 else 0
    with col_m1:
        st.metric("Current Run Rate", f"{crr:.2f}")
    with col_m2:
        st.metric("Projected Score (CRR)", f"{int(crr * 20)}")
    with col_m3:
        st.metric("Wickets In Hand", f"{10 - wickets}")

    # Predict Button
    if st.button("🚀 PREDICT SCORE", use_container_width=True, key="btn_p1"):

        # Encode inputs
        bat_team_for_encoder = DISPLAY_TO_ENCODER.get(batting_team, batting_team)
        bowl_team_for_encoder = DISPLAY_TO_ENCODER.get(bowling_team, bowling_team)
        venue_for_encoder = VENUE_TO_ENCODER.get(venue, venue)

        bat_team_encoded = team_encoder.transform([bat_team_for_encoder])[0]
        bowl_team_encoded = team_encoder.transform([bowl_team_for_encoder])[0]
        venue_encoded = venue_encoder.transform([venue_for_encoder])[0]

        # Prepare features
        input_features = np.array(
            [
                [
                    bat_team_encoded,
                    bowl_team_encoded,
                    venue_encoded,
                    runs,
                    wickets,
                    overs,
                    runs_last_5,
                    wickets_last_5,
                ]
            ]
        )

        # Make prediction
        predicted_score = model.predict(input_features, verbose=0)[0][0]

        # Calculate score range
        score_low = int(predicted_score * 0.92)
        score_high = int(predicted_score * 1.08)

        if predicted_score < runs:
            predicted_score = runs + 10
            score_low = int(predicted_score * 0.92)
            score_high = int(predicted_score * 1.08)

        # Display prediction
        st.markdown(
            f"""
        <div class="prediction-box">
            <div class="prediction-label">PREDICTED FINAL SCORE</div>
            <div class="prediction-score">{predicted_score:.0f}</div>
            <div class="prediction-range" style="color: #AAA;">Expected Range: {score_low} - {score_high}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Save to history
        prediction_record = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": "1st Innings",
            "batting": batting_team,
            "bowling": bowling_team,
            "current": f"{runs}/{wickets} ({overs:.1f} ov)",
            "predicted": int(predicted_score),
            "range": f"{score_low}-{score_high}",
        }
        st.session_state.prediction_history.insert(0, prediction_record)
        st.session_state.prediction_history = st.session_state.prediction_history[:10]


# ==========================================
# TAB 2: 2nd Innings Target Chase (Chase Master)
# ==========================================
with tab2:
    # Row 1: Target & Teams
    with st.container():
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)

        # Target Score Input (Prominent)
        st.markdown(
            "<h3 style='text-align: center; margin-bottom: 20px;'>🏆 Target to Chase</h3>",
            unsafe_allow_html=True,
        )
        col_t_1, col_t_2, col_t_3 = st.columns([1, 2, 1])
        with col_t_2:
            target_score = st.number_input(
                "Target Score",
                min_value=50,
                max_value=300,
                value=180,
                step=1,
                key="target_t2",
                label_visibility="collapsed",
            )

        st.markdown("---")

        col1_t2, col2_t2, col3_t2 = st.columns([1, 0.2, 1])
        with col1_t2:
            st.markdown("### 🏏 Chasing Team")
            chasing_team = st.selectbox(
                "Select Team",
                teams,
                index=1,
                key="batting_t2",
                label_visibility="collapsed",
            )
            if chasing_team in TEAM_ICONS:
                st.image(TEAM_ICONS[chasing_team], width=100)

        with col2_t2:
            st.markdown(
                "<h2 style='text-align: center; padding-top: 50px;'>VS</h2>",
                unsafe_allow_html=True,
            )

        with col3_t2:
            st.markdown("### 🎯 Defending Team")
            defending_team = st.selectbox(
                "Select Team",
                teams,
                index=0,
                key="bowling_t2",
                label_visibility="collapsed",
            )
            if defending_team in TEAM_ICONS:
                st.image(TEAM_ICONS[defending_team], width=100)

        st.markdown("---")
        venue_t2 = st.selectbox("🏟️ Select Venue", venues, key="venue_t2")
        st.markdown("</div>", unsafe_allow_html=True)

    if chasing_team == defending_team:
        st.error("⚠️ Teams should be different!")

    # Row 2: Chase Stats (Glass Container)
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Live Information")

    col_stats1_t2, col_stats2_t2, col_stats3_t2, col_stats4_t2 = st.columns(4)
    with col_stats1_t2:
        runs_t2 = st.number_input(
            "Current Runs",
            min_value=0,
            max_value=target_score + 6,
            step=1,
            value=0,
            key="runs_t2",
        )
    with col_stats2_t2:
        wickets_t2 = st.number_input(
            "Wickets Down", min_value=0, max_value=10, step=1, value=0, key="wickets_t2"
        )
    with col_stats3_t2:
        completed_overs_t2 = st.number_input(
            "Overs Done", min_value=0, max_value=19, step=1, value=0, key="overs_t2"
        )
    with col_stats4_t2:
        balls_t2 = st.number_input(
            "Balls (Current)", min_value=0, max_value=5, step=1, value=0, key="balls_t2"
        )

    overs_t2 = completed_overs_t2 + balls_t2 / 6.0

    # Basic Calculations
    runs_needed = target_score - runs_t2
    balls_left = 120 - int(overs_t2 * 6)
    required_rr = (runs_needed / balls_left) * 6 if balls_left > 0 else 0

    # Recent Form
    if overs_t2 >= 5:
        st.markdown("#### Powerplay / Recent Form")
        col_l5_1_t2, col_l5_2_t2 = st.columns(2)
        with col_l5_1_t2:
            runs_last_5_t2 = st.number_input(
                "Runs (Last 5)", min_value=0, max_value=150, value=0, key="runs_l5_t2"
            )
        with col_l5_2_t2:
            wickets_last_5_t2 = st.number_input(
                "Wickets (Last 5)",
                min_value=0,
                max_value=10,
                value=0,
                key="wickets_l5_t2",
            )
    else:
        runs_last_5_t2, wickets_last_5_t2 = 0, 0
        st.info("Unlock advanced metrics after 5 overs")

    st.markdown("</div>", unsafe_allow_html=True)

    # Chase Equation
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("Runs Needed", f"{runs_needed}")
    with col_d2:
        st.metric("Balls Remaining", f"{balls_left}")
    with col_d3:
        st.metric("Required Rate", f"{required_rr:.2f}")

    # Prediction Button
    if st.button("🔮 ANALYZE CHASE", use_container_width=True, key="btn_t2"):

        # Encode inputs
        bat_team_encoded = team_encoder.transform(
            [DISPLAY_TO_ENCODER.get(chasing_team, chasing_team)]
        )[0]
        bowl_team_encoded = team_encoder.transform(
            [DISPLAY_TO_ENCODER.get(defending_team, defending_team)]
        )[0]
        venue_encoded = venue_encoder.transform(
            [VENUE_TO_ENCODER.get(venue_t2, venue_t2)]
        )[0]

        input_features = np.array(
            [
                [
                    bat_team_encoded,
                    bowl_team_encoded,
                    venue_encoded,
                    runs_t2,
                    wickets_t2,
                    overs_t2,
                    runs_last_5_t2,
                    wickets_last_5_t2,
                ]
            ]
        )

        # Predict likely final score if they continue playing normally
        predicted_final_score = model.predict(input_features, verbose=0)[0][0]

        # Calculate Win Probability
        win_prob = calculate_win_probability(predicted_final_score, target_score)
        win_percentage = int(win_prob * 100)

        # Display Results
        st.markdown(
            f"""
        <div class="prediction-box">
            <div class="prediction-label">WIN PROBABILITY</div>
            <div class="prediction-score" style="color: {'#4caf50' if win_percentage > 50 else '#ff5252'};">{win_percentage}%</div>
            <div class="prediction-range">
                {'🟢 Chase Possible' if win_percentage > 50 else '🔴 Tough Chase'}
            </div>
            <div style="margin-top: 15px; font-size: 0.9rem; color: #aaa;">
                Projected Final Score: {int(predicted_final_score)}
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # --- Worm Graph Visualization ---
        st.markdown("### 📈 Chase Trajectory")

        import pandas as pd

        chart_df = pd.DataFrame(
            {
                "Overs": [0, overs_t2, 20],
                "Projected Runs": [0, runs_t2, predicted_final_score],
                "Target Pace": [0, runs_t2, target_score],
            }
        )

        st.line_chart(chart_df.set_index("Overs"))

        # Save to history
        prediction_record = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "mode": "2nd Innings (Chase)",
            "batting": chasing_team,
            "bowling": defending_team,
            "current": f"{runs_t2}/{wickets_t2} ({overs_t2:.1f} ov)",
            "predicted": f"{win_percentage}% Win Prob",
            "range": f"Target: {target_score}",
        }
        st.session_state.prediction_history.insert(0, prediction_record)
        st.session_state.prediction_history = st.session_state.prediction_history[:10]


# --- Prediction History ---
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("### 📜 Recent Predictions")

    for i, record in enumerate(st.session_state.prediction_history[:5]):
        label = f"{record['timestamp']} | {record['batting']} vs {record['bowling']}"
        with st.expander(label, expanded=(i == 0)):
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.write(f"**Status:** {record['current']}")
            with col_h2:
                st.write(f"**Pred:** {record['predicted']}")
            with col_h3:
                st.write(f"**Info:** {record['range']}")

# --- Clear History Button ---
if st.session_state.prediction_history:
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()

# --- Footer ---
st.markdown("---")
st.markdown(
    """
<div class="footer">
    <p>🏏 <strong>IPL Score Predictor Pro</strong> | Built by <strong>SUJAL MEENA</strong></p>
    <p style="font-size: 0.8rem; color: #555;">Powered by Advanced AI & Streamlit</p>
    <p style="font-size: 1.2rem; color: white; margin-top: 10px; font-weight: bold;">Note: Enable Dark Mode for the best-looking interface.</p>
</div>
""",
    unsafe_allow_html=True,
)
