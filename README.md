# Midterm: Deep Learning End-to-End Pipeline

## 1. Purpose of the Repository
This repository contains the code and resources for the midterm assignment of the Deep Learning class. It demonstrates end-to-end data pipelines, focusing on building, training, and evaluating Deep Neural Networks for both classification and regression tasks.

## 2. Brief Overview of the Project
This repository covers two practical tasks applying Deep Learning architectures to complex datasets:
*   **Task 1: Fraud Detection (Classification)** - Predicting the probability of an online transaction being fraudulent using a Sequential Neural Network.
*   **Task 2: Song Year Prediction (Regression)** - Predicting the release year of songs based on continuous audio features using a Deep Learning regression model.

## 3. Description of the Models and Matrix Results Used
*   **Classification (Fraud Detection):** 
    *   **Model:** Sequential Deep Neural Network (Dense layers with Dropout for regularization, Sigmoid activation for binary output).
    *   **Metrics:** ROC-AUC Score, Binary Crossentropy Loss.
*   **Regression (Song Year Prediction):**
    *   **Model:** Sequential Deep Neural Network (Multiple Dense layers with ReLU activation, Linear activation for continuous output).
    *   **Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R²).

## 4. How to Navigate Through the GitHub/Notebooks
1. Ensure you have the required datasets placed in the same directory as the notebooks (datasets are not uploaded due to GitHub size restrictions).
    *   `train_transaction.csv` & `test_transaction.csv` (for Task 1)
    *   `midterm-regresi-dataset.csv` (for Task 2)
2. Install required dependencies: `pip install pandas numpy scikit-learn tensorflow`
3. Open the `.ipynb` files (`Fraud_Detection.ipynb`, `Song_Year_Prediction.ipynb`) in Jupyter Notebook or JupyterLab.
4. Run the cells from top to bottom. Note: The notebooks contain both ML and DL models; focus on the Deep Learning sections for this specific repository evaluation.
5. The `task1_submission.csv` contains the final predicted probabilities for the test dataset.

## 5. Identification
*   **Name:** Rakha Primindra Danuatmaja
*   **Class:** TK-46-GAB
*   **NIM:** 1103223001
