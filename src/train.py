import os
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from skimage.feature import hog
from preprocessing import load_images

DATASET_PATH = "data/rice_leaf_diseases"

print("Loading images...")
X, y = load_images(DATASET_PATH)

print("Extracting HOG features...")
features = []
for img in X:
    fd = hog(
        img,
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        channel_axis=-1
    )
    features.append(fd)

features = np.array(features)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.20, random_state=42, shuffle=True
)

print("Training SVM...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/svm_rice.pkl")

os.makedirs("data_split", exist_ok=True)
joblib.dump((X_train, y_train), "data_split/train_data.pkl")
joblib.dump((X_test, y_test), "data_split/test_data.pkl")

print("Training completed successfully!")
