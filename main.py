import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ✅ Create folders automatically
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# STEP 1: Create Dataset (Simulation)
data = pd.DataFrame({
    "temperature": [60, 80, 55, 90, 70, 95, 50, 85],
    "vibration": [20, 40, 15, 50, 25, 60, 10, 45],
    "pressure": [30, 70, 25, 80, 40, 85, 20, 75],
    "failure": [0, 1, 0, 1, 0, 1, 0, 1]
})

print("Dataset Preview:\n", data)

# STEP 2: Features & Target
X = data[["temperature", "vibration", "pressure"]]
y = data["failure"]

# STEP 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# STEP 4: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# STEP 5: Predictions
y_pred = model.predict(X_test)

# STEP 6: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# STEP 7: Save Model
joblib.dump(model, "models/model.pkl")

# STEP 8: Graph
plt.plot(data["temperature"], label="Temperature")
plt.plot(data["vibration"], label="Vibration")
plt.plot(data["pressure"], label="Pressure")
plt.legend()
plt.title("Sensor Data Analysis")
plt.savefig("outputs/graph.png")
plt.show()

# STEP 9: Final Prediction
new_data = np.array([[85, 50, 80]])
prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\n⚠️ ALERT: Machine Failure Likely!")
else:
    print("\n✅ Machine is Safe")