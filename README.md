# Bayesian Network for Breast Cancer Diagnosis

## Overview
This project implements a Bayesian Network model for diagnosing breast cancer based on various features such as age, menopause status, tumor characteristics, and treatment factors. The Bayesian Network is constructed using the pgmpy library in Python, and the Expectation-Maximization (EM) algorithm is employed to estimate the model parameters.

## Features
- Constructs a Bayesian Network model for breast cancer diagnosis.
- Utilizes the Expectation-Maximization (EM) algorithm for parameter estimation.
- Evaluates the model performance using accuracy as the primary metric.
- Provides visualization of the Bayesian Network structure.
- Performs data preprocessing tasks including splitting data into training and testing sets.

## Dependencies
- pandas
- numpy
- pgmpy
- matplotlib
- networkx
- sklearn
- seaborn (optional, for visualization)

## Usage
1. **Setup**: Ensure all dependencies are installed. You can install them via pip:
    ```
    pip install pandas numpy pgmpy matplotlib networkx scikit-learn seaborn
    ```

2. **Data Preparation**: 
   - The breast cancer dataset should be provided in a CSV file named `breast-cancer.csv`.
   - Ensure the dataset contains relevant features such as age, menopause, node_caps, inv_nodes, tumor_size, deg_malig, irradiat, and class labels.

3. **Model Construction**:
   - Define the Bayesian Network structure based on the relationships between features.
   - Instantiate the ExpectationMaximization estimator with the defined model and input data.

4. **Model Training**:
   - Fit the ExpectationMaximization estimator to estimate the model parameters using the provided dataset.

5. **Model Evaluation**:
   - Split the dataset into training and testing sets.
   - Use the trained model to make predictions on the test data.
   - Evaluate the model performance using metrics such as accuracy, precision, recall, and F1-score.

6. **Visualization**:
   - Visualize the Bayesian Network structure to understand the relationships between features.

## Phases of Model Development

This project follows a two-phase approach to develop and evaluate the Bayesian Network model for breast cancer diagnosis:

1. **Initial Model Development**: 
   - In the first phase, the model is constructed using all available data.
   - The Bayesian Network structure is defined based on prior knowledge and domain expertise.
   - The Expectation-Maximization (EM) algorithm is applied to estimate the model parameters using the entire dataset.
   - The model is evaluated for its performance on the complete dataset to assess its effectiveness in predicting breast cancer outcomes.

2. **Data Splitting and Testing**: 
   - In the second phase, the dataset is split into training and testing sets.
   - The Bayesian Network model is retrained using only the training data to avoid overfitting.
   - The trained model is then evaluated using the testing data to provide a more realistic estimate of its performance on unseen data.
   - Performance metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's predictive ability and generalization capability.

By adopting this phased approach, the project aims to ensure the robustness and reliability of the developed Bayesian Network model for breast cancer diagnosis.

