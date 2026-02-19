# ipl_score_prediction.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import pickle


# --- Step 1: Load dataset ---
df = pd.read_csv("ipl.csv")
print("✅ Dataset loaded successfully!")

# --- Step 2: Drop unnecessary columns ---
columns_to_drop = ["date", "mid", "batsman", "bowler", "striker", "non-striker"]
df = df.drop(columns=columns_to_drop, errors="ignore")

# --- Step 3: Encode categorical columns using separate encoders ---
team_encoder = LabelEncoder()
venue_encoder = LabelEncoder()

df["bat_team"] = team_encoder.fit_transform(df["bat_team"])
df["bowl_team"] = team_encoder.transform(df["bowl_team"])  # same encoder
df["venue"] = venue_encoder.fit_transform(df["venue"])

# --- Save encoders for use in app.py ---
with open("team_encoder.pkl", "wb") as f:
    pickle.dump(team_encoder, f)

with open("venue_encoder.pkl", "wb") as f:
    pickle.dump(venue_encoder, f)

# --- Step 4: Define features (X) and target (y) ---
X = df.drop("total", axis=1)  # should include venue
y = df["total"]

print("\n✅ Columns used for training:")
print(list(X.columns))

print(f"\n📐 Input dimension (should be 8): {X.shape[1]}")

# --- Step 5: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# --- Step 6: Build the neural network ---
model = Sequential(
    [
        Dense(128, input_dim=X_train.shape[1], activation="relu"),
        Dense(64, activation="relu"),
        Dense(1),  # output layer for regression
    ]
)

model.compile(loss="mean_squared_error", optimizer="adam")

# --- Step 7: Train the model ---
history = model.fit(
    X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1
)
print("✅ Model training complete!")

# --- Step 8: Evaluate model ---
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
print(f"📉 Mean Squared Error on test data: {mse:.2f}")

# --- Step 9: Save the trained model ---
model.save("ipl_score_model.keras")

print("💾 Model saved as 'ipl_score_model.keras")

# --- Step 10: Plot training loss ---
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --- Step 11: Plot actual vs predicted scores ---
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Total Score")
plt.ylabel("Predicted Total Score")
plt.title("Actual vs Predicted Total Score")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.show()

print("🎉 All done! Your model is now ready to use with venue included.")
