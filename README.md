Breast Cancer Prediction using TensorFlow, PyTorch, and Ensemble Learning
Table of Contents
Introduction
Dataset Description
Data Preprocessing
Machine Learning Models
TensorFlow Neural Network
PyTorch Neural Network
Ensemble Model with Soft Voting
Implementation Details
Environment Setup and Dependencies
Data Preparation
Model Architectures
Training and Evaluation
Results and Analysis
Individual Model Performance
Ensemble Model Performance
Interpretation with SHAP Values
Conclusion
How to Run the Code
Installation Instructions
Usage Instructions
References
Introduction
This project focuses on developing machine learning models to predict whether a breast tumor is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) Dataset. The primary objectives are:

Implement individual neural network models using TensorFlow and PyTorch.
Enhance model performance through hyperparameter tuning and regularization.
Combine the models using an ensemble method with soft voting.
Interpret the models using SHAP (SHapley Additive exPlanations) values.
By leveraging multiple deep learning frameworks and ensemble techniques, we aim to improve predictive accuracy and provide insights into feature importance, which is crucial in healthcare diagnostics.

Dataset Description
The Breast Cancer Wisconsin (Diagnostic) Dataset contains 569 samples with 30 numerical features each, derived from digitized images of fine needle aspirate (FNA) of breast mass. Each sample is labeled as either malignant or benign.

Features Include:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave Points
Symmetry
Fractal Dimension
Each feature is computed for:

Mean
Standard Error (SE)
Worst (mean of the three largest values)
Target Variable:

Diagnosis: Malignant (M) or Benign (B)
Data Preprocessing
Data preprocessing steps are crucial to ensure the models receive clean and standardized data.

Handling Missing Values:

Dropped columns with missing values.
Encoding Categorical Variables:

Converted the target variable Diagnosis from categorical to numerical using Label Encoding (M → 1, B → 0).
Feature Scaling:

Applied StandardScaler to standardize features to have zero mean and unit variance.
Train-Test Split:

Split the dataset into training and testing sets with an 80-20 split while maintaining class distribution using stratification.
Machine Learning Models
TensorFlow Neural Network
Framework: TensorFlow 2.x with Keras API.
Architecture:
Input Layer: Number of neurons equal to the number of features.
Hidden Layers: Two hidden layers with 64 and 32 neurons, respectively.
Activation Function: ReLU for hidden layers, Sigmoid for output layer.
Regularization: Dropout layers with a rate of 0.2 to prevent overfitting.
Loss Function: Binary Cross-Entropy.
Optimizer: Adam.
PyTorch Neural Network
Framework: PyTorch.
Architecture:
Input Layer: Number of neurons equal to the number of features.
Hidden Layers: Two hidden layers with 64 and 32 neurons, respectively.
Activation Function: ReLU for hidden layers, Sigmoid for output layer.
Regularization: Dropout layers with a rate of 0.2.
Loss Function: Binary Cross-Entropy (BCELoss).
Optimizer: Adam.
Ensemble Model with Soft Voting
Method: Soft Voting Classifier using Scikit-learn's VotingClassifier.
Components:
TensorFlow Model (wrapped for compatibility with Scikit-learn).
PyTorch Model (wrapped as a custom Scikit-learn estimator).
Voting Mechanism: Averages the predicted class probabilities of individual models and predicts the class with the highest average probability.
Implementation Details
Environment Setup and Dependencies
Programming Language: Python 3.x
Libraries Required:
numpy, pandas, scikit-learn, tensorflow, torch, seaborn, matplotlib, keras-tuner, shap
Data Preparation
Loading Data:

Used pandas to read data from an Excel file (breast_cancer_data.xlsx).
Preprocessing Steps:

Dropped columns with missing values.
Encoded target variable using LabelEncoder.
Scaled features using StandardScaler.
Split data into training and testing sets using train_test_split.
Model Architectures
TensorFlow Model
Function: create_tf_model()
Layers:
Dense(64, activation='relu')
Dropout(0.2)
Dense(32, activation='relu')
Dropout(0.2)
Dense(1, activation='sigmoid')
Compilation:
optimizer='adam'
loss='binary_crossentropy'
metrics=['accuracy']
PyTorch Model
Class: PyTorchModel(nn.Module)
Layers:
Linear(input_size, 64)
Dropout(0.2)
Linear(64, 32)
Dropout(0.2)
Linear(32, 1)
Sigmoid()
Loss Function: nn.BCELoss()
Optimizer: optim.Adam(model.parameters(), lr=0.001)
Ensemble Model
Components:
TensorFlow model wrapped using KerasClassifier.
PyTorch model wrapped as PyTorchClassifier.
Voting Classifier:
VotingClassifier(estimators=[('tf', tf_estimator), ('pytorch', pytorch_estimator)], voting='soft')
Training and Evaluation
TensorFlow Model:

Trained for 50 epochs with a batch size of 32.
Evaluated using test data to calculate accuracy.
PyTorch Model:

Trained for 50 epochs with a batch size of 32.
Used DataLoader for batch processing.
Evaluated using test data to calculate accuracy.
Ensemble Model:

Combined predictions from both models using soft voting.
Evaluated using test data to calculate accuracy.
Hyperparameter Tuning:

Used Keras Tuner for the TensorFlow model to find the optimal number of neurons and dropout rates.
Applied hyperparameter tuning to improve model performance.
Interpretation with SHAP Values:

Used SHAP to interpret feature importance.
Generated SHAP summary plots to visualize the impact of each feature on the model's predictions.
Results and Analysis
Individual Model Performance
TensorFlow Model Test Accuracy:

Achieved an accuracy of approximately 97.37% on the test set.
PyTorch Model Test Accuracy:

Achieved an accuracy of approximately 96.49% on the test set.
Ensemble Model Performance
Ensemble Model Test Accuracy:
Improved accuracy of approximately 98.25% on the test set.
Confusion Matrix:
Showed a higher number of correct predictions compared to individual models.
Classification Report:
Displayed improved precision, recall, and F1-score for both classes.
Interpretation with SHAP Values
Feature Importance:
Identified the most influential features contributing to the model's predictions.
Features like concave points_mean, perimeter_mean, and concavity_mean had high SHAP values, indicating strong influence.
SHAP Summary Plot:
Visualized the impact of each feature across all samples.
Helped in understanding how features positively or negatively affected the prediction outcome.
Conclusion
The project successfully demonstrates the implementation of machine learning models using both TensorFlow and PyTorch frameworks. By employing an ensemble method with soft voting, the combined model outperformed individual models, achieving higher predictive accuracy. The use of hyperparameter tuning and regularization techniques contributed to the models' robustness and generalization capabilities.

Interpreting the model with SHAP values provided valuable insights into feature importance, which is essential in healthcare applications where understanding the factors influencing predictions is crucial.
