import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load model & test data
model = joblib.load("models/svm_rice.pkl")
X_test, y_test = joblib.load("data_split/test_data.pkl")

print("Evaluating...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}%")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

os.makedirs("results", exist_ok=True)
with open("results/evaluation.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

print("Evaluation saved to results/evaluation.txt")

# Confusion Matrix Plot
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (SVM + HOG)")
plt.colorbar()

labels_sorted = np.unique(y_test)
tick_marks = np.arange(len(labels_sorted))

plt.xticks(tick_marks, labels_sorted, rotation=45)
plt.yticks(tick_marks, labels_sorted)

thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i,j]),
                 ha="center", va="center",
                 color="white" if cm[i,j] > thresh else "black")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

print("Done!")
