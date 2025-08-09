import argparse
import joblib
from src.preprocess import prepare_image

def main(model_path, image_path):
    model = joblib.load(model_path)
    X = prepare_image(image_path)
    pred = model.predict(X)
    print("Predicted digit:", int(pred[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/knn_pipeline.joblib")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    main(args.model, args.image)
