import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load MNIST Dataset
X, y = fetch_openml(
    'mnist_784',
    version=1,
    return_X_y=True,
    as_frame=False
)

# Convert labels to integers
y = y.astype(int)

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. PCA Explained Variance Study
components = [2, 10, 30, 50]
explained_variance = []

for n in components:
    pca = PCA(n_components=n, random_state=42)
    pca.fit(X_train_scaled)
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(components, explained_variance, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance Analysis")
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. PCA with 50 Components
pca_50 = PCA(n_components=50, random_state=42)
X_train_pca = pca_50.fit_transform(X_train_scaled)
X_test_pca = pca_50.transform(X_test_scaled)

# 6. Logistic Regression (Original Data)
lr_original = LogisticRegression(max_iter=1000)
lr_original.fit(X_train_scaled, y_train)

y_pred_original = lr_original.predict(X_test_scaled)
acc_original = accuracy_score(y_test, y_pred_original)

# 7. Logistic Regression (PCA Data)
lr_pca = LogisticRegression(max_iter=1000)
lr_pca.fit(X_train_pca, y_train)

y_pred_pca = lr_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"Accuracy without PCA: {acc_original:.4f}")
print(f"Accuracy with PCA (50 components): {acc_pca:.4f}")

# 8. 2D PCA Visualization
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_train_scaled)

plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=y_train,
    cmap='tab10',
    s=6,
    alpha=0.7
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Visualization of MNIST")
plt.legend(
    *scatter.legend_elements(),
    title="Digit",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()
