import os
import cv2
import pandas as pd
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths
train_dir = 'dataset/train'
sample_submission_path = 'sample_submission.csv'

# Load class names
class_names = sorted(os.listdir(train_dir))
print("Classes:", class_names)

# Extract HOG features and labels
features, labels = [], []
for label, class_name in enumerate(class_names):
    print(f"Processing class: {class_name}")
    class_folder = os.path.join(train_dir, class_name)
    for img_file in os.listdir(class_folder):
        print(f"Processing image: {img_file}")
        img_path = os.path.join(class_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = hog(gray,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   visualize=False)
        features.append(feat)
        labels.append(label)
    print(f"Processed {len(features)} images for class {class_name}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

# Train SVM classifier
classifier = SVC(kernel='rbf', C=1.0, random_state=42)
classifier.fit(X_train, y_train)
print("Training completed.")

# Evaluate on the local test split
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Validation accuracy: {acc:.4f}")

