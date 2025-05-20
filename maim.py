import os
import cv2
import pandas as pd
from skimage.feature import hog
from sklearn.svm import SVC

# Paths to the dataset dirs and submission file
train_dir = 'dataset/train'
test_dir = 'dataset/test'
sample_submission_path = 'sample_submission.csv'

# Load class names (5 classes: 'daisy', 'dandelion', 'rose', 'sunflower', 'tulip')
class_names = sorted(os.listdir(train_dir))
print("Found classes:", class_names)

# Prepare training data
features = []
labels = []
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(train_dir, class_name)
    for img_file in os.listdir(class_folder):
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

# Train the SVM classifier with RBF kernel
classifier = SVC(kernel='rbf', C=1.0, random_state=42)
classifier.fit(features, labels)
print("Training completed.")

# Load sample submission and predict on test set
submission = pd.read_csv(sample_submission_path)
predictions = []
for img_id in submission['id']:
    img_path = os.path.join(test_dir, img_id)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = hog(gray,
               orientations=9,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               visualize=False)
    pred_idx = classifier.predict([feat])[0]
    predictions.append(class_names[pred_idx])

submission['label'] = predictions
submission.to_csv('submission.csv', index=False)
print("Written predictions to submission.csv")
