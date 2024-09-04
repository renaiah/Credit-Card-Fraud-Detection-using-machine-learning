
# Credit Card Fraud Detection

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)



## Introduction
The Credit Card Fraud Detection project aims to identify fraudulent credit card transactions using machine learning techniques. By analyzing patterns in transaction data, the model is designed to differentiate between legitimate and fraudulent transactions, thereby helping financial institutions reduce the risk of fraud. This project uses a dataset with imbalanced classes, where fraudulent transactions are much rarer than legitimate ones.

## Features
Fraud Detection 
- Classifies credit card transactions as fraudulent or legitimate.
Data Handling
- Balances the dataset using undersampling to handle class imbalance.
Model Performance
- Achieves high accuracy, precision, recall, and F1-score on the test data.
Visualization
- Provides a confusion matrix and classification report for performance evaluation.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/renaiah/credit-card-fraud-detection.git

    ```
2. Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```
3. Run the script:
    ```bash
    python credit_card_fraud_detection.py
    ```
4. Install the required libraries:
    ```bash
    pip install numpy pandas scikit-learn
    ```

## Usage
To use the Credit Card Fraud Detection model:

1. Ensure the dataset (creditcard.csv) is in the project directory.
2. Run the Streamlit app:
    ```bash
    python credit_card_fraud_detection.py
    ```
3. The script will preprocess the data, train the model, and display the evaluation metrics.

## Model Training
The model is trained using TensorFlow and Keras. Key details about the training process:

- **Model:** Logestic Regression
- **Dataset:** 284,807 transactions in data from Kaggle
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn
- **Train Accuracy:** ~94%
- **Validation Accuracy:** ~91%

### Data Preparation
The dataset is highly imbalanced, so undersampling is used to balance the number of fraudulent and legitimate transactions. The data is split into training and testing sets.


### Training Process
The Logistic Regression model is trained on the balanced dataset using Scikit-learn. The training process includes:
- Splitting: Data is split into training and test sets.
- Training: The model is trained on the training data.
- Evaluation: The model's performance is evaluated using accuracy, precision, recall, and F1-score on the test data.

### Visualization
The script includes visualizations such as a confusion matrix and a classification report to provide insights into the model's performance.

## Future Scope
This project currently uses Logistic Regression for fraud detection. Future improvements could include:

- Model Expansion: Exploring more complex models like Random Forest, Gradient Boosting, or Neural Networks for better accuracy.
- Data Augmentation: Implementing techniques like SMOTE for better handling of imbalanced datasets.
- Real-Time Implementation: Developing a real-time fraud detection system that can be integrated into financial transaction systems.
- Feature Engineering: Creating new features from existing data to enhance model performance.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes. Suggestions for improving the model or the codebase are highly appreciated.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
