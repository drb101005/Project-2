import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import joblib
import numpy as np
from src.preprocess import prepare_image_from_pil

MODEL_PATH = "models/knn_pipeline.joblib"
model = joblib.load(MODEL_PATH)

CANVAS_SIZE = 200
IMG_SIZE = 28

class App:
    def __init__(self, root):
        self.root = root
        root.title("MNIST Digit Recognizer")
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.draw = ImageDraw.Draw(self.image)

        tk.Button(root, text="Predict", command=self.predict).grid(row=1, column=0)
        tk.Button(root, text="Clear", command=self.clear).grid(row=1, column=1)
        self.label = tk.Label(root, text="Draw a digit", font=("Arial", 16))
        self.label.grid(row=1, column=2)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1,y1,x2,y2, fill="black", width=0)
        self.draw.ellipse([x1,y1,x2,y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,CANVAS_SIZE,CANVAS_SIZE], fill=255)
        self.label.config(text="Draw a digit")

    def predict(self):
        # prepare PIL image and feed to model (in-memory)
        pil = self.image.copy()
        # center/resize like preprocessing
        pred_input = prepare_image_from_pil(pil)
        pred = model.predict(pred_input)[0]
        self.label.config(text=f"Predicted: {int(pred)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
