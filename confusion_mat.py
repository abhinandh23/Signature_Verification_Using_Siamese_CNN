from keras.models import load_model
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import preprocessing
# Import the module so the decorator runs and abs_fn is registered
import Siamese_CNN_Model

model = load_model("signature_verification_model.h5", safe_mode=False)
test_df = pd.read_csv("./sign_data/test_data.csv")
img_size = (128, 128)
image_dir = './sign_data_processed/test'

test_img1 = []
test_img2 = []
test_labels = test_df['forged'].values  # ground truth

for f1, f2 in zip(test_df['file1'], test_df['file2']):
    path1 = f"{image_dir}/{f1}"
    path2 = f"{image_dir}/{f2}"

    # Load grayscale
    i1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    # Resize + normalize
    i1 = cv2.resize(i1, img_size) / 255.0
    i2 = cv2.resize(i2, img_size) / 255.0

    test_img1.append(i1)
    test_img2.append(i2)

# Convert to tensors
test_img1 = np.array(test_img1).reshape(-1, 128, 128, 1)
test_img2 = np.array(test_img2).reshape(-1, 128, 128, 1)
preds = model.predict([test_img1, test_img2])
binary_preds = (preds > 0.5).astype(int)
cm = confusion_matrix(test_labels, binary_preds)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(test_labels, binary_preds))
