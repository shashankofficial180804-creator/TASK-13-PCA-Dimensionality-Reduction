# TASK-13-PCA-Dimensionality-Reduction
PCA Dimensionality Reduction on MNIST dataset using Scikit-learn. Includes feature scaling, explained variance analysis, dimensionality reduction, logistic regression accuracy comparison, and 2D PCA visualization to understand variance–accuracy trade-offs.

📌 Task 13: PCA – Dimensionality Reduction
🔍 Objective

To understand how Principal Component Analysis (PCA) reduces dimensionality while preserving maximum variance and to analyze its effect on classification accuracy.

🛠 Tools & Libraries

Python

Scikit-learn

NumPy

Matplotlib

📂 Dataset

Primary Dataset: MNIST handwritten digits dataset (from sklearn.datasets.fetch_openml)

70,000 grayscale digit images (28×28 pixels)

🚀 Steps Performed

Loaded MNIST dataset and flattened images into feature vectors

Standardized features using StandardScaler

Applied PCA with 2, 10, 30, 50 components

Tracked and plotted explained variance ratio

Selected optimal components using cumulative variance

Transformed dataset into reduced dimensions

Trained Logistic Regression on:

Original dataset

PCA-reduced dataset

Compared accuracy scores

Visualized 2D PCA scatter plot

📊 Deliverables

✔ Explained variance plot
✔ Reduced datasets
✔ Accuracy comparison report

🎯 Final Outcome

Understanding of feature compression, variance trade-off, and model performance after dimensionality reduction.
