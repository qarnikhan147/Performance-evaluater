import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# load model / scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# load or create test split
if os.path.exists("X_test.npy") and os.path.exists("y_test.npy"):
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")
else:
    df = pd.read_excel("dataset.xlsx")
    X = df.drop("result", axis=1)
    y = df["result"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

# prediction and evaluation
y_pred = model.predict(scaler.transform(X_test))
cm = confusion_matrix(y_test, y_pred)

plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Fail", "Pass"])
plt.yticks([0, 1], ["Fail", "Pass"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
print("Confusion matrix saved to confusion_matrix.png")