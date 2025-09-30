# Task 6: K-Nearest Neighbors (KNN) Classification

This project is a submission for Task 6 of the Elevate Labs AI/ML Internship. [cite_start]The objective is to implement the K-Nearest Neighbors (KNN) algorithm for a classification problem. [cite: 3, 4]

## Project Objective

The goal is to build a KNN classifier to predict whether a user will purchase a product based on their age and estimated salary. [cite_start]The project involves data preprocessing, finding an optimal value for K, training the model, evaluating its performance, and visualizing the decision boundaries. [cite: 4, 10]

## Dataset

The project uses the **Social Network Ads** dataset, which contains information about users on a social network, including their age, salary, and whether they purchased a particular product.

- **Source**: [Kaggle](https://www.kaggle.com/datasets/rakeshrau/social-network-ads)
- **Features Used**: `Age`, `EstimatedSalary`
- **Target Variable**: `Purchased` (0 = No, 1 = Yes)

## Directory Structure

```
.
├── .github/              # Contains GitHub action workflows and templates
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
├── data/                 # Contains the dataset
│   └── Social_Network_Ads.csv
├── notebooks/            # Contains the main Jupyter Notebook
│   └── knn_classification_analysis.ipynb
├── visualizations/       # Stores output plots and images
│   ├── confusion_matrix.png
│   ├── elbow_method.png
│   └── knn_decision_boundary.png
├── .gitignore            # Specifies files to be ignored by Git
└── README.md             # Project documentation (this file)
```

## Methodology

The project workflow is as follows:
1.  **Data Loading & Exploration**: The dataset is loaded, and an initial analysis is performed to understand its structure and statistics.
2.  **Preprocessing**:
    - The data is split into training (75%) and testing (25%) sets.
    - **Feature Scaling** is applied using `StandardScaler` to normalize the `Age` and `EstimatedSalary` features. [cite_start]This is critical for KNN, as it's a distance-based algorithm. [cite: 6]
3.  [cite_start]**Finding Optimal K**: The **Elbow Method** is implemented to find the most suitable value for K by plotting the error rate for K values from 1 to 39. The "elbow" point indicates the optimal K. [cite: 8]
4.  [cite_start]**Model Training**: A `KNeighborsClassifier` is trained on the scaled training data using the optimal K value (K=5) and Euclidean distance. [cite: 7]
5.  **Evaluation**: The model's performance is evaluated on the test set using:
    - **Accuracy Score**: Overall correct predictions.
    - **Confusion Matrix**: To understand true/false positives and negatives.
    - [cite_start]**Classification Report**: Providing precision, recall, and F1-score. [cite: 9]
6.  [cite_start]**Visualization**: The decision boundary of the trained classifier is plotted to visually represent how the model separates the two classes. [cite: 10]

## Key Results

### Optimal K Value
The Elbow Method plot indicated that **K=5** is an optimal choice, as the error rate stabilizes around this point.

![Elbow Method Plot](visualizations/elbow_method.png)

### Model Performance
- **Accuracy**: **93.00%** on the test set.
- **Confusion Matrix**:
  ![Confusion Matrix](visualizations/confusion_matrix.png)
- **Classification Report**: The model shows strong precision and recall for both classes.

### Decision Boundary
The following plot visualizes how the KNN model partitions the feature space to classify users.

![KNN Decision Boundary](visualizations/knn_decision_boundary.png)

## Execution Instructions

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/Kool-K/knn-classifier-social-network-ads.git](https://github.com/Kool-K/knn-classifier-social-network-ads.git)
    cd root_folder_name
    ```
2.  **Create a virtual environment and install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
4.  Open the `notebooks/knn_classification_analysis.ipynb` file and run the cells.

---