import cv2
import os
import numpy as np

def load_images(folder):
    images = []
    labels = []

    for label_name in os.listdir(folder):
        label_folder = os.path.join(folder, label_name)

        if not os.path.isdir(label_folder):
            continue

        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            images.append(img)
            labels.append(label_name)

    return np.array(images), np.array(labels)
