from keras.datasets import mnist
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("models/knn_pipeline.joblib")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig, axes = plt.subplots(2,5, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    img = X_test[i]
    ax.imshow(img, cmap="gray")
    pred = model.predict((img.reshape(1,-1).astype(float)/255.0))[0]
    ax.set_title(f"Pred: {int(pred)}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions.png")
print("Saved sample_predictions.png")
