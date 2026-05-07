# MNIST Image Classification with Logistic Regression⭐

This project implements an image classification model for the MNIST dataset using Logistic Regression. The goal is to classify handwritten digits (0-9) based on their pixel data.

## Dataset

The dataset used is `MNIST-10000-784.csv`, which contains 10,000 samples of handwritten digits, each represented by 784 features (28x28 pixels).

## Libraries Used

*   **pandas**: For data manipulation and reading the CSV file.
*   **scikit-learn**: For machine learning model implementation, data splitting, preprocessing, and evaluation.

## Project Structure

The main logic is contained within a single Python script.

## Steps

1.  **Load Data**: The `MNIST-10000-784.csv` file is loaded into a pandas DataFrame.
2.  **Feature and Label Separation**: The features (pixel values) are separated from the target labels (digits 0-9).
3.  **Data Splitting**: The dataset is split into training and testing sets (80% train, 20% test) with stratification to ensure an equal distribution of classes in both sets.
4.  **Model Building**: A pipeline is created using `scikit-learn` that includes:
    *   `StandardScaler`: To standardize the features by removing the mean and scaling to unit variance.
    *   `LogisticRegression`: A linear model for classification. Key parameters:
        *   `max_iter=200`: Increased iterations for convergence.
        *   `n_jobs=-1`: Utilizes all available CPU cores for faster computation.
        *   `C=0.5`: Inverse of regularization strength; smaller values specify stronger regularization.
        *   `solver="saga"`: An optimization algorithm that supports L1/L2 regularization and works well on large datasets.
5.  **Model Training**: The `LogisticRegression` model is trained on the training data (`X_train`, `y_train`).
6.  **Model Testing**:
    *   Predictions are made on the test set (`X_test`).
    *   The accuracy of the model is calculated using `accuracy_score`.
    *   A detailed `classification_report` is printed, showing precision, recall, F1-score, and support for each class.

## Results

The script outputs the accuracy score and the classification report for the trained model on the test data.
