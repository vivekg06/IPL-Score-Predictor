# 🏏 IPL Score Prediction App

An AI-powered web application that predicts the final score of IPL cricket matches based on current match statistics using Deep Learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ipl-score-prediction.streamlit.app/)

> **Note:** Replace `https://ipl-score-prediction.streamlit.app/` with your actual deployment URL in the README if different.

## ✨ Features

- **🎯 Real-time Score Prediction** - Predict final scores based on current match situation
- **📊 Score Range Visualization** - Get confidence intervals for predictions
- **📈 Live Stats** - View current run rate, balls remaining, and wickets left
- **📜 Prediction History** - Track your previous predictions in the session
- **🎨 Modern UI** - Beautiful, responsive dark-themed interface
- **⚡ Fast Loading** - Cached model for instant predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone/Download the project**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`

## 📁 Project Structure

```
IPL Score prediction/
├── app.py                    # Main Streamlit application
├── ipl_score_prediction.py   # Model training script
├── ipl_score_model.keras     # Trained neural network model
├── team_encoder.pkl          # Team name encoder
├── venue_encoder.pkl         # Venue name encoder
├── ipl.csv                   # Training dataset
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧠 How It Works

1. **Input Match Data**: Select batting team, bowling team, venue, and current match stats
2. **AI Processing**: The neural network processes 8 features to make predictions
3. **Prediction Output**: Get the predicted final score with a confidence range

### Model Architecture

- **Input Layer**: 8 features (teams, venue, runs, wickets, overs, last 5 overs stats)
- **Hidden Layers**: 128 → 64 neurons with ReLU activation
- **Output Layer**: 1 neuron for score regression

## 🔧 Retraining the Model

If you want to retrain the model with updated data:

```bash
python ipl_score_prediction.py
```

This will:
- Load the IPL dataset
- Train a new neural network
- Save the model and encoders

## 📊 Features Used for Prediction

| Feature | Description |
|---------|-------------|
| Batting Team | The team currently batting |
| Bowling Team | The team currently bowling |
| Venue | Stadium where match is being played |
| Current Runs | Runs scored so far |
| Wickets | Wickets fallen |
| Overs | Overs bowled |
| Runs (Last 5) | Runs in last 5 overs |
| Wickets (Last 5) | Wickets in last 5 overs |

## 🤝 Contributors

- **VIVEK GUPTA**

## 📝 License

This project is for educational purposes.

---

<p align="center">This project is made by SUJAL MEENA using Streamlit & TensorFlow</p>
