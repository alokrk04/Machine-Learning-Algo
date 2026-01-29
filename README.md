# Machine-Learning-Algo
A comprehensive collection of machine learning algorithms implemented in Python. Includes data preprocessing, supervised learning (Regression, Classification), and unsupervised learning techniques applied to datasets like Titanic, Mall Customers, and Pima Diabetes.


# Machine Learning Algorithms Portfolio 🧠💻

## 📌 Overview
This repository contains a collection of Jupyter Notebooks demonstrating the implementation of various **Machine Learning algorithms** using Python. It serves as a practical guide to understanding key concepts in data science, including data preprocessing, supervised learning, and unsupervised learning.

The project utilizes popular libraries such as **Scikit-Learn**, **Statsmodels**, **Pandas**, and **Matplotlib** to analyze real-world datasets and visualize results.

## 📂 Algorithms & Techniques Covered

### 1. Data Preprocessing & EDA 🧹
* **Handling Missing Values**: Imputation techniques (e.g., median filling on the Titanic dataset).
* **Feature Engineering**: Creating new features like `Travelalone` and binning.
* **Categorical Encoding**: One-Hot Encoding using `pd.get_dummies`.
* **Feature Scaling**: Implementation of `MinMaxScaler` and `StandardScaler`.

### 2. Supervised Learning 📉
#### Regression
* **Linear Regression**: Predicting Sales based on TV, Radio, and Newspaper advertising budgets.
* **Polynomial Regression**: Modeling non-linear relationships.
* **Support Vector Regression (SVR)**: Implementing SVR with RBF kernels on synthetic data.

#### Classification
* **Logistic Regression**: Binary classification on the **Pima Indians Diabetes** dataset.
* **K-Nearest Neighbors (KNN)**: Classification with confusion matrix evaluation.
* **Naive Bayes**: Implementation of `GaussianNB` for probabilistic classification.
* **Decision Trees**: Building interpretable classification trees.
* **Perceptron**: A custom implementation of the Perceptron algorithm from scratch.

### 3. Unsupervised Learning 🔍
* **Hierarchical Clustering**: Customer segmentation using the **Mall Customers** dataset.
* **Nearest Neighbors**: Finding spatial neighbors using `NearestNeighbors`.

### 4. Model Evaluation & Selection ✅
* **Metrics**: Confusion Matrix, Classification Report (Precision, Recall, F1-Score), Accuracy Score.
* **Cross-Validation**: Implementing `KFold` cross-validation (e.g., on the Breast Cancer dataset).

## 📊 Datasets Used
The following datasets are analyzed in this repository:
* **Titanic Dataset** (`tested.csv`): For preprocessing and survival prediction.
* **Advertising Dataset** (`advertising.csv`): For linear regression analysis.
* **Pima Indians Diabetes**: For binary classification tasks.
* **Mall Customers** (`Mall_Customers.csv`): For clustering and segmentation.
* **Salary Dataset** (`Salary.csv`): For simple linear regression.
* **Breast Cancer Dataset** (Scikit-learn built-in): For cross-validation.

## 🛠️ Technologies & Libraries
* **Python 3.x**
* **Pandas & NumPy**: Data manipulation and numerical operations.
* **Matplotlib & Seaborn**: Data visualization.
* **Scikit-Learn**: Machine learning models and preprocessing.
* **Statsmodels**: Statistical modeling (OLS Regression).


## 📈 Visualizations
The notebooks include various visualizations such as:
* Regression lines vs. actual data points.
* Dendrograms for Hierarchical Clustering.
* Confusion Matrices for classification performance.
* Scatter plots for customer segmentation.

