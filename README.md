# Signature_Verification_Using_Siamese_CNN
Signature verification is essential in banking, legal documentation, and secure access 
systems. Manual verification is slow  and error-prone, so I have developedd an 
automated verification system using a Siamese Convolutional Neural Network (CNN). 
Instead of learning hundreds of thosands of signatures my model just compares if two 
given signatures are similar  by passing them through twin CNN branches with shared 
weights. Each branch generates a 4096-dimensional feature vector, and the distance 
between these vectors is used to predict whether the signatures match or if one is forged. 
The system was trained on preprocessed signature pairs using binary cross-entropy loss 
and the Adam optimizer with exponential learning-rate decay. Results show the model 
performs well in distinguishing genuine signatures from forgeries, showingthe strength 
of metric learning for this task. Unlike traditional classification models, the Siamese 
approach scales to new users without retraining, making it practical for banking and 
legal verification. Future improvements may include dynamic signature analysis, 
triplet-loss training, and deployment as a mobile verification application
