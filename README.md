# Credit-Card-Fraud-Detection-using-machine-learning
Credit Card Fraud Detection
Project Overview
This project aims to detect fraudulent credit card transactions using a machine learning model. The dataset used in this project contains transactions made by credit card holders, with a focus on differentiating between normal and fraudulent transactions.

Dataset
The dataset used in this project is the Kaggle Credit Card Fraud Detection dataset. It contains 284,807 transactions, each with 30 features including Time, Amount, and 28 anonymized features labeled V1 to V28. The Class column is the target variable, where 0 represents a normal transaction and 1 represents a fraudulent one.

Number of Instances: 284,807
Number of Features: 31 (including the target variable)
Number of Normal Transactions: 284,315 (99.83%)
Number of Fraudulent Transactions: 492 (0.17%)
Data Preprocessing
The dataset is highly imbalanced, with fraudulent transactions constituting only 0.17% of all transactions. To address this, undersampling was used to create a balanced dataset where the number of normal and fraudulent transactions is equal.

Steps Involved:
Loading the Dataset: The dataset was loaded into a Pandas DataFrame.
Exploratory Data Analysis (EDA): Basic statistics and distributions of the dataset were analyzed.
Balancing the Dataset: Undersampling was performed to create a balanced dataset with an equal number of normal and fraudulent transactions (492 each).
Feature Selection: All features except the target variable (Class) were selected for model training.
Data Splitting: The balanced dataset was split into features (X) and target (Y).
Model Building
A Logistic Regression model was trained on the balanced dataset to classify transactions as normal or fraudulent.

Steps Involved:
Splitting the Data: The dataset was split into training and testing sets.
Training the Model: A Logistic Regression model was trained on the training data.
Model Evaluation: The model's performance was evaluated using accuracy, confusion matrix, and classification report.
Model Performance
Accuracy: Achieved a high accuracy on the test data.
Confusion Matrix: Shows the number of correct and incorrect predictions.
Classification Report: Provides precision, recall, F1-score, and support for each class.
Conclusion
This project demonstrates how to detect fraudulent credit card transactions using machine learning. Although the dataset was highly imbalanced, undersampling helped in creating a balanced dataset, allowing the model to learn effectively. The Logistic Regression model provided satisfactory results in identifying fraudulent transactions.

Installation
To run this project, you need to have Python installed along with the following libraries:

numpy
pandas
scikit-learn
You can install the required libraries using pip:

bash
Copy code
pip install numpy pandas scikit-learn
Usage
To use this project, clone the repository and run the credit_card_fraud_detection.py script:

bash
Copy code
git clone https://github.com/renaiah/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python credit_card_fraud_detection.py
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
The dataset used in this project was provided by Kaggle.
The project was developed using open-source libraries like Pandas, NumPy, and Scikit-learn.
