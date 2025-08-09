# src/train.py
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.datasets import mnist
import seaborn as sns
import textwrap

# 1️⃣ Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(len(X_train), -1).astype("float32") / 255.0
X_test = X_test.reshape(len(X_test), -1).astype("float32") / 255.0

# 2️⃣ Train KNN model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3️⃣ Save the model
joblib.dump(model, "models/knn_pipeline.joblib")
print("✅ Model saved at models/knn_pipeline.joblib")

# 4️⃣ Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc:.4f}")

# 5️⃣ Classification report
report = classification_report(y_test, y_pred)
print("\n📊 Classification Report:\n", report)

# 6️⃣ Normalized Confusion Matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')
fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format=".2f", colorbar=True)
plt.title("Normalized Confusion Matrix")
plt.savefig("normalized_confusion_matrix.png")
print("📁 Saved confusion matrix → normalized_confusion_matrix.png")

# 7️⃣ Show some real test samples with predictions
fig_samples, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("real_test_samples.png")
print("📁 Saved test samples → real_test_samples.png")

# 8️⃣ Fancy figure with code snippet
code_text = textwrap.dedent("""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=3))
    ])
    model.fit(X_train, y_train)
""")

fig_code, ax_code = plt.subplots(figsize=(6, 4))
ax_code.axis("off")
ax_code.text(0, 1, "Model Setup & Training Code:", fontsize=12, fontweight="bold", va="top")
ax_code.text(0, 0.95, code_text, fontsize=10, family="monospace", va="top")
plt.savefig("model_code_snippet.png")
print("📁 Saved code snippet → model_code_snippet.png")
