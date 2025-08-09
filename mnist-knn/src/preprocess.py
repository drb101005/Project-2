from PIL import Image, ImageOps
import numpy as np

def prepare_image_from_pil(pil_img):
    """
    Input: PIL Image (any size)
    Output: np array shape (1,784) normalized 0..1, MNIST-style
    """
    img = pil_img.convert("L")
    # Invert so digit is white on black like MNIST (if necessary)
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    # Resize keeping aspect (thumbnail) and center in 28x28
    img = img.resize((28,28), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr.reshape(1, -1)

def prepare_image(image_path):
    pil_img = Image.open(image_path)
    return prepare_image_from_pil(pil_img)
