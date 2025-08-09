import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import joblib
import os

# === Load trained model ===
MODEL_PATH = os.path.join("models", "knn_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# === Segment digits from saved canvas image ===
def segment_digits(image_path, gap_threshold=5):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.point(lambda x: 255 if x > 50 else 0)
    arr = np.array(img)

    col_sum = np.sum(arr, axis=0)
    digit_regions, in_digit, start = [], False, 0

    for i, val in enumerate(col_sum):
        if val > 0 and not in_digit:
            in_digit, start = True, i
        elif val <= 0 and in_digit:
            in_digit = False
            if i - start > gap_threshold:
                digit_regions.append((start, i))

    if in_digit:
        digit_regions.append((start, len(col_sum)))

    digits = []
    for (x_start, x_end) in digit_regions:
        digit_img = arr[:, x_start:x_end]
        pil_digit = Image.fromarray(digit_img)

        # Resize keeping aspect ratio, then center
        pil_digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
        new_img = Image.new('L', (28, 28), color=0)
        x_offset = (28 - pil_digit.width) // 2
        y_offset = (28 - pil_digit.height) // 2
        new_img.paste(pil_digit, (x_offset, y_offset))

        digits.append(new_img)

    return digits


# === Tkinter Canvas App ===
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Digit Recognizer")

        self.canvas = tk.Canvas(root, width=300, height=100, bg='white')
        self.canvas.pack()

        self.image = Image.new("RGB", (300, 100), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        tk.Button(root, text="Predict", command=self.predict).pack()
        tk.Button(root, text="Clear", command=self.clear).pack()
        self.result_label = tk.Label(root, text="", font=("Helvetica", 18))
        self.result_label.pack()

    def paint(self, event):
        x1, y1 = (event.x - 6), (event.y - 6)
        x2, y2 = (event.x + 6), (event.y + 6)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 300, 100], fill="white")
        self.result_label.config(text="")

    def predict(self):
        img_path = "temp_digits.png"
        self.image.save(img_path)

        digits = segment_digits(img_path)
        predictions = []

        for d in digits:
            arr = np.array(d).reshape(1, -1) / 255.0
            pred = model.predict(arr)[0]
            predictions.append(str(int(pred)))

        self.result_label.config(text="".join(predictions))

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
