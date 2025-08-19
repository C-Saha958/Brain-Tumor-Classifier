Brain Tumor Classification using SVM
Project Overview

This project classifies brain tumors from MRI scans into four categories:

Glioma Tumor

Meningioma Tumor

Pituitary Tumor

No Tumor

It uses Support Vector Machine (SVM) with feature extraction techniques such as Histogram of Oriented Gradients (HOG) and Sobel edge detection. Principal Component Analysis (PCA) is applied for dimensionality reduction, and hyperparameter tuning optimizes the SVM performance.

Dataset

The dataset contains MRI scans of brain tumors.

Training and testing images are organized into folders based on tumor type.

Dataset path example: /content/Brain-Tumor-Classification-DataSet/Training

Features & Methods

Image Preprocessing:

Grayscale conversion for reduced complexity.

Histogram equalization to enhance image contrast.

Feature Extraction:

HOG (Histogram of Oriented Gradients) for shape-based features.

Sobel edge detection for texture information.

Dimensionality Reduction:

PCA (Principal Component Analysis) reduces feature space to improve efficiency.

Model Training:

SVM classifier with RBF kernel.

Hyperparameter tuning using GridSearchCV.

Class imbalance handled via class weights.

Evaluation Metrics:

Accuracy

Classification report

Confusion matrix

ROC curves for each class

Usage Instructions
1. Load the dataset
X, y = load_data('/content/Brain-Tumor-Classification-DataSet/Training')

2. Train the model

Model uses SVM with PCA-transformed features.

Hyperparameter tuning automatically finds the best SVM parameters.

3. Save the trained model
import joblib
joblib.dump(svm_classifier, "svm_model.pkl")
joblib.dump(pca, "pca_transform.pkl")
joblib.dump(scaler, "scaler.pkl")

4. Predict on new images
from prediction import predict_tumor
image_path = "Testing/no_tumor/image(100).jpg"
predicted_tumor = predict_tumor(image_path)
print(f"Predicted Tumor Type: {predicted_tumor}")

Evaluation & Visualization

ROC curves for each class to visualize sensitivity vs. specificity.

Confusion matrix to identify misclassifications.

Requirements

Python 3.x

Libraries: numpy, opencv-python, scikit-image, scikit-learn, joblib, matplotlib, seaborn, tensorflow

Conclusion

This project successfully classifies brain tumors from MRI scans with high accuracy using SVM. Feature extraction and PCA improve model performance, and the saved model can be used for predicting tumor type on new images.
