import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from preprocessing import load_images

DATASET_PATH = "data/rice_leaf_diseases"

print("Loading images...")
X, y = load_images(DATASET_PATH)

features = []
for img in X:
    fd = hog(img, orientations=9, pixels_per_cell=(8,8), 
            cells_per_block=(2,2), channel_axis=-1)
    features.append(fd)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42
)

print("Training SVM...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

print("Saving model...")
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_rice.pkl")

print("Done!")
