from keras.models import load_model
import preprocessing
import cv2 as cv
import numpy as np
# Import the module so the decorator runs and abs_fn is registered
import Siamese_CNN_Model

# Load the model (allow Lambda deserialization)
model = load_model("./signature_verification_model.h5", safe_mode=False)

# Load images
sig1 = cv.imread(r"sign_data_processed\sign_data\test\050\01_050.png")
sig2 = cv.imread(r"sign_data_processed\sign_data\test\050_forg\01_0125050.PNG")


# Preprocess images
sig1_preprocessed = preprocessing.preprocess(sig1)  # Ensure this outputs a (height, width, channels) array
sig2_preprocessed = preprocessing.preprocess(sig2)

# Resize and expand dimensions if needed
sig1_preprocessed = np.expand_dims(sig1_preprocessed, axis=0)
sig2_preprocessed = np.expand_dims(sig2_preprocessed, axis=0)

# Stack images for Siamese model input
# Check if your model expects [img1, img2] as a list or a numpy array with two images
images_pair = [sig1_preprocessed, sig2_preprocessed]

# Make prediction
prediction = model.predict(images_pair)

print("I'm:", prediction[0][0]*100,"% sure that second image is forged ")
